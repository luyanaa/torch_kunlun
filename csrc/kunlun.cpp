#include "kunlun.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

#include "xpu/refactor/context/newcontext.h"

#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace torch_kunlun {

void check_xpu(int ret, const char* what) {
    TORCH_CHECK(ret == 0, what, " failed: ret=", ret, " (", xpu_strerror(ret), ")");
}

namespace {

// A K200 PD exposes 4 clusters and 4 SDNN engines, and XDNN rejects a larger
// launch (Context::set_ncluster returns -1).
constexpr int kMaxClusters = 4;
constexpr int kMaxSdnn = 4;

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) return fallback;
    return static_cast<int>(parsed);
}

std::atomic<int>& cluster_budget_ref() {
    static std::atomic<int> budget{
        std::min(kMaxClusters, std::max(1, env_int("KUNLUN_NCLUSTER", kMaxClusters)))};
    return budget;
}

std::atomic<int>& sdnn_budget_ref() {
    static std::atomic<int> budget{
        std::min(kMaxSdnn, std::max(1, env_int("KUNLUN_NSDNN", kMaxSdnn)))};
    return budget;
}

bool auto_cluster_policy() {
    static const bool enabled = [] {
        const char* mode = std::getenv("KUNLUN_CLUSTER_POLICY");
        return mode != nullptr && std::strcmp(mode, "auto") == 0;
    }();
    return enabled;
}

// Small launches leave three clusters idle instead of paying the
// cluster-cooperative launch cost.
constexpr int64_t kAutoSmallWork = 1LL << 22;

void apply_cluster_policy(DeviceContext& dc, int64_t work_elems) {
    int ncluster = cluster_budget_ref().load();
    int nsdnn = sdnn_budget_ref().load();
    if (auto_cluster_policy() && work_elems > 0 && work_elems < kAutoSmallWork) {
        ncluster = 1;
        nsdnn = 1;
    }
    // The vendor calls may reallocate the context workspace, so only issue
    // them when the requested policy actually changed.
    if (dc.active_ncluster != ncluster) {
        if (dc.ctx->set_ncluster(ncluster) != 0) {
            ncluster = kMaxClusters;
            dc.ctx->set_ncluster(ncluster);
        }
        dc.active_ncluster = ncluster;
    }
    if (dc.active_nsdnn != nsdnn) {
        if (dc.ctx->set_nsdnn(nsdnn) != 0) {
            nsdnn = kMaxSdnn;
            dc.ctx->set_nsdnn(nsdnn);
        }
        dc.active_nsdnn = nsdnn;
    }
}

}  // namespace

int set_cluster_budget(int nclusters, int nsdnn) {
    TORCH_CHECK(nclusters >= 1 && nclusters <= kMaxClusters,
                "kunlun cluster budget must be 1..", kMaxClusters, ", got ", nclusters);
    TORCH_CHECK(nsdnn >= 1 && nsdnn <= kMaxSdnn,
                "kunlun SDNN budget must be 1..", kMaxSdnn, ", got ", nsdnn);
    cluster_budget_ref().store(nclusters);
    sdnn_budget_ref().store(nsdnn);
    return nclusters;
}

int cluster_budget() {
    return cluster_budget_ref().load();
}

int sdnn_budget() {
    return sdnn_budget_ref().load();
}

namespace {

// The loaded libxpuapi carries a literal XDNN_VERSION_<version> tag.  The tag
// lives in whichever module owns the XDNN symbols: the vendor .so when linked
// dynamically, or this extension when libxpuapi is linked statically.
std::string scan_xdnn_version(const char* path) {
    if (path == nullptr || *path == '\0') return {};
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    static constexpr char kNeedle[] = "XDNN_VERSION_";
    constexpr size_t kNeedleLen = sizeof(kNeedle) - 1;
    constexpr size_t kChunk = 1U << 20;
    std::vector<char> buf(kChunk);
    std::string carry;
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto got = static_cast<size_t>(in.gcount());
        if (got == 0) break;
        std::string chunk = carry + std::string(buf.data(), got);
        const size_t pos = chunk.find(kNeedle);
        if (pos != std::string::npos) {
            size_t end = pos + kNeedleLen;
            while (end < chunk.size()) {
                const char c = chunk[end];
                const bool ident = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                                   (c >= 'A' && c <= 'Z') || c == '.' || c == '-' ||
                                   c == '_';
                if (!ident) break;
                ++end;
            }
            return chunk.substr(pos + kNeedleLen, end - (pos + kNeedleLen));
        }
        carry = chunk.size() > kNeedleLen ? chunk.substr(chunk.size() - kNeedleLen) : chunk;
    }
    return {};
}

// dladdr(&create_context) resolves to this module's PLT slot when XDNN is
// linked dynamically, so the version tag in libxpuapi would be missed.  Scan
// the loaded objects whose names look like XDNN instead.
int scan_loaded_xdnn(struct dl_phdr_info* info, size_t, void* data) {
    auto* found = static_cast<std::string*>(data);
    if (!found->empty()) return 1;
    if (info->dlpi_name == nullptr || *info->dlpi_name == '\0') return 0;
    const std::string path = info->dlpi_name;
    const auto slash = path.find_last_of('/');
    const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    if (base.find("xpuapi") == std::string::npos && base.find("xdnn") == std::string::npos) {
        return 0;
    }
    *found = scan_xdnn_version(path.c_str());
    return found->empty() ? 0 : 1;
}

}  // namespace

std::string xdnn_version() {
    static const std::string version = [] {
        const char* force = std::getenv("KUNLUN_XDNN_VERSION");
        if (force != nullptr && *force != '\0') return std::string(force);
        std::string found;
        dl_iterate_phdr(&scan_loaded_xdnn, &found);
        if (!found.empty()) return found;
        Dl_info info;
        if (dladdr(reinterpret_cast<void*>(&baidu::xpu::api::create_context), &info) != 0) {
            found = scan_xdnn_version(info.dli_fname);
            if (!found.empty()) return found;
        }
        if (dladdr(reinterpret_cast<void*>(&torch_kunlun::device_count), &info) != 0) {
            found = scan_xdnn_version(info.dli_fname);
        }
        return found;
    }();
    return version;
}

void check_xdnn_runtime() {
    if (std::getenv("KUNLUN_SKIP_XDNN_VERSION_CHECK") != nullptr) return;
    const std::string version = xdnn_version();
    const char* require = std::getenv("KUNLUN_XDNN_REQUIRE");
    if (require != nullptr && *require != '\0') {
        TORCH_CHECK(version.find(require) != std::string::npos,
                    "kunlun: loaded XDNN version '", version,
                    "' does not contain KUNLUN_XDNN_REQUIRE='", require, "'");
        return;
    }
    if (version.empty()) {
        std::fprintf(stderr,
                     "kunlun: could not detect the loaded XDNN version; this build targets "
                     "XDNN 2.0.x headers, so verify libxpuapi matches\n");
        return;
    }
    if (std::getenv("KUNLUN_XDNN_ALLOW_MISMATCH") != nullptr) return;
    // 2.4 changed argument lists for max_pool2d/adaptive_avg_pool2d/concat, so
    // a 2.0-compiled extension calling a 2.4 runtime is not ABI-safe.
    if (version.rfind("2.0.", 0) != 0) {
        std::fprintf(stderr,
                     "kunlun: WARNING loaded XDNN %s, but this extension was compiled against "
                     "XDNN 2.0.x headers; XDNN calls may be ABI-mismatched. "
                     "Rebuild against the matching headers, or set "
                     "KUNLUN_XDNN_ALLOW_MISMATCH=1 to silence.\n",
                     version.c_str());
    }
}

namespace {

std::mutex& pin_mu() {
    static std::mutex m;
    return m;
}

std::map<uintptr_t, size_t>& pin_allocs() {
    static std::map<uintptr_t, size_t> allocs;
    return allocs;
}

void register_pin(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(pin_mu());
    pin_allocs()[reinterpret_cast<uintptr_t>(ptr)] = size;
}

void unregister_pin(void* ptr) {
    std::lock_guard<std::mutex> lock(pin_mu());
    pin_allocs().erase(reinterpret_cast<uintptr_t>(ptr));
}

class PinnedHostAllocator final : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t n) override {
        if (n == 0) {
            return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::CPU));
        }
        void* ptr = nullptr;
        const int ret = xpu_host_alloc(&ptr, n, 0);
        TORCH_CHECK(ret == 0 && ptr != nullptr,
                    "kunlun pinned host allocation of ", n, " bytes failed: ret=", ret);
        register_pin(ptr, n);
        return c10::DataPtr(ptr, ptr, &PinnedHostAllocator::deleter,
                            c10::Device(c10::DeviceType::CPU));
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &PinnedHostAllocator::deleter;
    }

private:
    static void deleter(void* ptr) {
        if (ptr == nullptr) return;
        unregister_pin(ptr);
        xpu_host_free(ptr);
    }
};

}  // namespace

bool is_pinned_ptr(const void* ptr) {
    if (ptr == nullptr) return false;
    std::lock_guard<std::mutex> lock(pin_mu());
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    auto it = pin_allocs().upper_bound(addr);
    if (it == pin_allocs().begin()) return false;
    --it;
    return addr < it->first + it->second;
}

c10::Allocator* get_pinned_allocator() {
    static PinnedHostAllocator allocator;
    return &allocator;
}

int device_count() {
    int n = 0;
    if (xpu_device_count(&n) != 0) {
        return 0;
    }
    return n;
}

int current_device() {
    int device = 0;
    check_xpu(xpu_current_device(&device), "xpu_current_device");
    return device;
}

void set_device(int device) {
    check_xpu(xpu_set_device(device), "xpu_set_device");
}

void synchronize() {
    synchronize_all();
}

namespace {

enum class FallbackPolicy { Default, Error, Warn, Silent };

// Unset means "no generic fallback, but deliberate host-staged kernels are
// allowed with a notice".  A generic CPU hop hides missing device coverage and
// moves activations across the host link, so it must be opt-in.
FallbackPolicy fallback_policy() {
    static const FallbackPolicy policy = [] {
        const char* mode = std::getenv("KUNLUN_CPU_FALLBACK");
        const std::string value = mode == nullptr ? "" : mode;
        if (value == "silent") return FallbackPolicy::Silent;
        if (value == "warn") return FallbackPolicy::Warn;
        if (value == "error") return FallbackPolicy::Error;
        return FallbackPolicy::Default;
    }();
    return policy;
}

// Prints at most once per operator so a training loop does not flood stderr.
bool first_notice(const char* op) {
    static std::mutex mu;
    static std::vector<std::string> seen;
    std::lock_guard<std::mutex> lock(mu);
    const std::string name = op == nullptr ? "" : op;
    if (std::find(seen.begin(), seen.end(), name) != seen.end()) {
        return false;
    }
    seen.push_back(name);
    return true;
}

}  // namespace

void note_cpu_fallback(const char* op) {
    static std::atomic<size_t> count{0};
    const FallbackPolicy policy = fallback_policy();
    TORCH_CHECK(policy != FallbackPolicy::Error,
                "kunlun operator '", op,
                "' is implemented by staging tensors through the CPU, which "
                "KUNLUN_CPU_FALLBACK=error rejects.  Unset the variable (or use "
                "warn/silent) to allow deliberate host-staged kernels.");
    const size_t n = ++count;
    if (policy != FallbackPolicy::Silent && first_notice(op)) {
        std::fprintf(stderr,
                     "kunlun host-staged implementation #%zu: %s "
                     "(KUNLUN_CPU_FALLBACK=silent hides this)\n",
                     n, op);
    }
}

void check_generic_fallback(const char* op) {
    const FallbackPolicy policy = fallback_policy();
    if (policy == FallbackPolicy::Silent) {
        return;
    }
    if (policy == FallbackPolicy::Warn) {
        std::fprintf(stderr, "kunlun CPU fallback: %s\n", op);
        return;
    }
    TORCH_CHECK(false,
                "kunlun has no device implementation for '", op,
                "'; the generic CPU fallback is disabled by default. Set "
                "KUNLUN_CPU_FALLBACK=warn (or silent) to allow it.");
}

namespace {

struct StreamHandle {
    XPUStream stream = nullptr;
    int device = 0;
    bool owned = false;
};

struct EventHandle {
    XPUEvent event = nullptr;
};

std::mutex& stream_mu() {
    static std::mutex m;
    return m;
}

std::vector<StreamHandle*>& all_streams() {
    static std::vector<StreamHandle*> streams;
    return streams;
}

std::vector<StreamHandle*>& default_streams() {
    static std::vector<StreamHandle*> defaults;
    return defaults;
}

thread_local std::vector<StreamHandle*> current_streams;

StreamHandle* default_stream_handle(int device) {
    std::lock_guard<std::mutex> lock(stream_mu());
    if (default_streams().size() <= static_cast<size_t>(device)) {
        default_streams().resize(device + 1, nullptr);
    }
    if (default_streams()[device] == nullptr) {
        default_streams()[device] = new StreamHandle{nullptr, device, false};
    }
    return default_streams()[device];
}

StreamHandle* current_stream_handle(int device) {
    if (current_streams.size() <= static_cast<size_t>(device)) {
        current_streams.resize(device + 1, nullptr);
    }
    if (current_streams[device] == nullptr) {
        current_streams[device] = default_stream_handle(device);
    }
    return current_streams[device];
}

StreamHandle* stream_from_id(uint64_t id) {
    return reinterpret_cast<StreamHandle*>(id);
}

EventHandle* event_from_id(uint64_t id) {
    return reinterpret_cast<EventHandle*>(id);
}

}  // namespace

uint64_t current_stream_id(int device) {
    return reinterpret_cast<uint64_t>(current_stream_handle(device));
}

uint64_t default_stream_id(int device) {
    return reinterpret_cast<uint64_t>(default_stream_handle(device));
}

uint64_t create_stream(int device) {
    int old_device = 0;
    check_xpu(xpu_current_device(&old_device), "xpu_current_device");
    check_xpu(xpu_set_device(device), "xpu_set_device(create_stream)");
    XPUStream stream = nullptr;
    check_xpu(xpu_stream_create(&stream), "xpu_stream_create");
    check_xpu(xpu_set_device(old_device), "xpu_set_device(create_stream restore)");
    auto* handle = new StreamHandle{stream, device, true};
    {
        std::lock_guard<std::mutex> lock(stream_mu());
        all_streams().push_back(handle);
    }
    return reinterpret_cast<uint64_t>(handle);
}

void destroy_stream(uint64_t stream_id) {
    auto* handle = stream_from_id(stream_id);
    if (handle == nullptr || !handle->owned) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(stream_mu());
        auto& streams = all_streams();
        streams.erase(std::remove(streams.begin(), streams.end(), handle), streams.end());
    }
    if (current_streams.size() > static_cast<size_t>(handle->device) &&
        current_streams[handle->device] == handle) {
        current_streams[handle->device] = default_stream_handle(handle->device);
    }
    int old_device = 0;
    const bool have_device = xpu_current_device(&old_device) == 0;
    xpu_set_device(handle->device);
    // Drain first: destroying a stream with in-flight work would invalidate
    // launches that already captured it.
    xpu_wait(handle->stream);
    xpu_stream_destroy(handle->stream);
    if (have_device) {
        xpu_set_device(old_device);
    }
    delete handle;
}

void set_current_stream(uint64_t stream_id) {
    auto* handle = stream_from_id(stream_id);
    TORCH_CHECK(handle != nullptr, "invalid kunlun stream id");
    if (current_streams.size() <= static_cast<size_t>(handle->device)) {
        current_streams.resize(handle->device + 1, nullptr);
    }
    current_streams[handle->device] = handle;
}

void synchronize_stream(uint64_t stream_id) {
    auto* handle = stream_from_id(stream_id);
    TORCH_CHECK(handle != nullptr, "invalid kunlun stream id");
    int old_device = 0;
    const bool have_device = xpu_current_device(&old_device) == 0;
    xpu_set_device(handle->device);
    const int ret = xpu_wait(handle->stream);
    if (have_device) {
        xpu_set_device(old_device);
    }
    check_xpu(ret, "xpu_wait(stream)");
}

XPUStream current_stream_native(int device) {
    return current_stream_handle(device)->stream;
}

void synchronize_all() {
    std::vector<StreamHandle*> handles;
    {
        std::lock_guard<std::mutex> lock(stream_mu());
        handles = all_streams();
        for (auto* h : default_streams()) {
            if (h != nullptr) {
                handles.push_back(h);
            }
        }
    }
    int old_device = 0;
    const bool have_device = xpu_current_device(&old_device) == 0;
    for (auto* h : handles) {
        // xpu_wait interprets the handle in the current device's context.
        xpu_set_device(h->device);
        xpu_wait(h->stream);
    }
    if (have_device) {
        xpu_set_device(old_device);
    }
}

uint64_t event_create() {
    auto* handle = new EventHandle();
    check_xpu(xpu_event_create(&handle->event), "xpu_event_create");
    return reinterpret_cast<uint64_t>(handle);
}

void event_destroy(uint64_t event_id) {
    auto* handle = event_from_id(event_id);
    if (handle == nullptr) {
        return;
    }
    if (handle->event != nullptr) {
        xpu_event_destroy(handle->event);
    }
    delete handle;
}

void event_record(uint64_t event_id, uint64_t stream_id) {
    auto* event = event_from_id(event_id);
    auto* stream = stream_from_id(stream_id);
    TORCH_CHECK(event != nullptr && stream != nullptr, "invalid kunlun event/stream");
    check_xpu(xpu_event_record(event->event, stream->stream), "xpu_event_record");
}

void stream_wait_event(uint64_t stream_id, uint64_t event_id) {
    auto* stream = stream_from_id(stream_id);
    auto* event = event_from_id(event_id);
    TORCH_CHECK(event != nullptr && stream != nullptr, "invalid kunlun event/stream");
    check_xpu(xpu_stream_wait_event(stream->stream, event->event), "xpu_stream_wait_event");
}

void event_synchronize(uint64_t event_id) {
    auto* event = event_from_id(event_id);
    TORCH_CHECK(event != nullptr, "invalid kunlun event id");
    check_xpu(xpu_event_wait(event->event), "xpu_event_wait");
}

bool event_query(uint64_t event_id) {
    // XPURT has no non-blocking event query; this is intentionally a wait.
    auto* event = event_from_id(event_id);
    if (event == nullptr || event->event == nullptr) {
        return true;
    }
    // XPURT has no non-blocking query; wait here and report completed.
    check_xpu(xpu_event_wait(event->event), "xpu_event_wait");
    return true;
}

DeviceContext::DeviceContext(int device_) : device(device_), ctx(nullptr) {
    xpu_set_device(device);
    ctx = baidu::xpu::api::create_context();
    TORCH_CHECK(ctx != nullptr, "xdnn::create_context failed for device ", device);
    // Ignore a rejection so the runtime can still be used on a partition with
    // fewer clusters; ContextGuard re-applies the policy per launch.
    const int ncluster = cluster_budget_ref().load();
    const int nsdnn = sdnn_budget_ref().load();
    active_ncluster = ctx->set_ncluster(ncluster) == 0 ? ncluster : 0;
    active_nsdnn = ctx->set_nsdnn(nsdnn) == 0 ? nsdnn : 0;
    ctx->set_stream(nullptr);
}

DeviceContext::~DeviceContext() {
    if (ctx != nullptr) {
        baidu::xpu::api::destroy_context(ctx);
        ctx = nullptr;
    }
}

DeviceContext& get_context(int device) {
    static std::mutex init_mu;
    // Intentionally leaked: XDNN/XPURT teardown order at process exit is not
    // guaranteed, and destroying contexts after the runtime can crash.
    static std::vector<DeviceContext*> contexts;

    std::lock_guard<std::mutex> lock(init_mu);
    if (device < 0) {
        device = current_device();
    }
    TORCH_CHECK(device < device_count(), "invalid kunlun device index ", device);
    if (contexts.size() <= static_cast<size_t>(device)) {
        contexts.resize(device + 1);
    }
    if (!contexts[device]) {
        contexts[device] = new DeviceContext(device);
    }
    return *contexts[device];
}

ContextGuard::ContextGuard(int device, int64_t work_elems)
    : dc(get_context(device)), lock_(dc.mu) {
    xpu_set_device(dc.device);
    XPUStream want = current_stream_native(dc.device);
    if (dc.active_stream != nullptr && dc.active_stream != want) {
        // The context owns one launch stream and a shared workspace, so a
        // launch on another stream must drain the previous one first.  Ops on
        // the same stream still pipeline.
        check_xpu(xpu_wait(dc.active_stream), "xpu_wait(context stream switch)");
    }
    dc.active_stream = want;
    dc.ctx->set_stream(want);
    apply_cluster_policy(dc, work_elems);
}

namespace {

struct KunlunGuardImpl final : c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

    c10::DeviceType type() const override {
        return static_type;
    }

    c10::Device exchangeDevice(c10::Device d) const override {
        const int old = current_device();
        set_device(d.index());
        return c10::Device(static_type, old);
    }

    c10::Device getDevice() const override {
        return c10::Device(static_type, current_device());
    }

    void setDevice(c10::Device d) const override {
        set_device(d.index());
    }

    void uncheckedSetDevice(c10::Device d) const noexcept override {
        xpu_set_device(d.index());
    }

    c10::Stream getStream(c10::Device d) const noexcept override {
        return c10::Stream(c10::Stream::UNSAFE, d,
                           static_cast<c10::StreamId>(current_stream_id(d.index())));
    }

    c10::Stream getDefaultStream(c10::Device d) const override {
        return c10::Stream(c10::Stream::UNSAFE, d,
                           static_cast<c10::StreamId>(default_stream_id(d.index())));
    }

    c10::Stream getStreamFromGlobalPool(c10::Device d, bool = false) const override {
        // A minimal per-device global pool: create once and reuse.  XPURT
        // streams are not reference-counted, so never destroy a pooled handle.
        static std::mutex pool_mu;
        static std::map<int, uint64_t> pool;
        uint64_t id = 0;
        {
            std::lock_guard<std::mutex> lock(pool_mu);
            auto it = pool.find(d.index());
            if (it != pool.end()) id = it->second;
        }
        if (id == 0) {
            const uint64_t created = create_stream(d.index());
            std::lock_guard<std::mutex> lock(pool_mu);
            auto [it, inserted] = pool.emplace(d.index(), created);
            if (!inserted) {
                // Another thread won the race; discard the extra handle.
                destroy_stream(created);
                id = it->second;
            } else {
                id = created;
            }
        }
        return c10::Stream(c10::Stream::UNSAFE, d,
                           static_cast<c10::StreamId>(id));
    }

    c10::Stream getNewStream(c10::Device d, int = 0) const override {
        return c10::Stream(c10::Stream::UNSAFE, d,
                           static_cast<c10::StreamId>(create_stream(d.index())));
    }

    c10::Stream exchangeStream(c10::Stream s) const noexcept override {
        const c10::Stream old = getStream(s.device());
        set_current_stream(static_cast<uint64_t>(s.id()));
        return old;
    }

    c10::DeviceIndex deviceCount() const noexcept override {
        return static_cast<c10::DeviceIndex>(device_count());
    }

    void synchronizeDevice(c10::DeviceIndex) const override {
        synchronize_all();
    }

    void synchronizeStream(const c10::Stream& stream) const override {
        synchronize_stream(static_cast<uint64_t>(stream.id()));
    }

    void record(void** event, const c10::Stream& stream, c10::DeviceIndex,
                c10::EventFlag) const override {
        if (*event == nullptr) {
            *event = reinterpret_cast<void*>(event_create());
        }
        event_record(reinterpret_cast<uint64_t>(*event),
                     static_cast<uint64_t>(stream.id()));
    }

    void block(void* event, const c10::Stream& stream) const override {
        stream_wait_event(static_cast<uint64_t>(stream.id()),
                          reinterpret_cast<uint64_t>(event));
    }

    bool queryEvent(void* event) const override {
        return event_query(reinterpret_cast<uint64_t>(event));
    }

    void synchronizeEvent(void* event) const override {
        event_synchronize(reinterpret_cast<uint64_t>(event));
    }

    void destroyEvent(void* event, c10::DeviceIndex) const noexcept override {
        event_destroy(reinterpret_cast<uint64_t>(event));
    }

    double elapsedTime(void*, void*, c10::DeviceIndex) const override {
        TORCH_CHECK(false, "kunlun: elapsedTime is not supported by XPURT");
        return 0.0;
    }

    void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const c10::Stream& stream) const override {
        record_data_ptr_on_stream(data_ptr, stream);
    }
};

struct KunlunHooksInterface final : at::PrivateUse1HooksInterface {
    bool hasPrimaryContext(c10::DeviceIndex) const override {
        return true;
    }

    void init() const override {}

    at::Device getDeviceFromPtr(void* ptr) const override {
        const int device = torch_kunlun::device_of_ptr(ptr);
        TORCH_CHECK(device >= 0, "kunlun getDeviceFromPtr: pointer is not owned by torch_kunlun allocator");
        return at::Device(at::DeviceType::PrivateUse1, device);
    }

    bool isPinnedPtr(const void* ptr) const override {
        return torch_kunlun::is_pinned_ptr(ptr);
    }

    c10::Allocator* getPinnedMemoryAllocator() const override {
        return torch_kunlun::get_pinned_allocator();
    }

    c10::DeviceIndex deviceCount() const override {
        return static_cast<c10::DeviceIndex>(torch_kunlun::device_count());
    }

    c10::DeviceIndex getCurrentDevice() const override {
        return static_cast<c10::DeviceIndex>(torch_kunlun::current_device());
    }

    void setCurrentDevice(c10::DeviceIndex device) const override {
        torch_kunlun::set_device(static_cast<int>(device));
    }

    c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override {
        const int old = torch_kunlun::current_device();
        torch_kunlun::set_device(static_cast<int>(device));
        return static_cast<c10::DeviceIndex>(old);
    }

    c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override {
        return exchangeDevice(device);
    }
};

}  // namespace

}  // namespace torch_kunlun

C10_REGISTER_GUARD_IMPL(PrivateUse1, torch_kunlun::KunlunGuardImpl);

namespace torch_kunlun {
namespace {

const bool kRegistered = [] {
    c10::register_privateuse1_backend("kunlun");
    c10::SetAllocator(c10::DeviceType::PrivateUse1, get_allocator());
    at::RegisterPrivateUse1HooksInterface(new KunlunHooksInterface());
    // File-only check: does not touch XPURT, so importing on a host without
    // the K200 driver stays safe.
    check_xdnn_runtime();
    return true;
}();

}  // namespace
}  // namespace torch_kunlun

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("device_count", &torch_kunlun::device_count);
    m.def("current_device", &torch_kunlun::current_device);
    m.def("set_device", &torch_kunlun::set_device);
    m.def("synchronize", &torch_kunlun::synchronize);
    m.def("memory_allocated", &torch_kunlun::memory_allocated);
    m.def("memory_reserved", &torch_kunlun::memory_reserved);
    m.def("adam_step", &torch_kunlun::adam_step);
    m.def("adamw_step", &torch_kunlun::adamw_step);
    m.def("empty_cache", &torch_kunlun::empty_cache);
    m.def("set_cluster_budget", &torch_kunlun::set_cluster_budget);
    m.def("cluster_budget", &torch_kunlun::cluster_budget);
    m.def("sdnn_budget", &torch_kunlun::sdnn_budget);
    m.def("xdnn_version", &torch_kunlun::xdnn_version);
    m.def("is_pinned_ptr", [](uint64_t ptr) {
        return torch_kunlun::is_pinned_ptr(reinterpret_cast<const void*>(ptr));
    });

    m.def("current_stream", &torch_kunlun::current_stream_id);
    // The raw XPUStream, as an integer, for callers that hand it straight to the
    // runtime (the Triton launcher) instead of through create_stream/set_stream.
    m.def("current_stream_native", [](int device) {
        return static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(torch_kunlun::current_stream_native(device)));
    });
    m.def("default_stream", &torch_kunlun::default_stream_id);
    m.def("create_stream", &torch_kunlun::create_stream);
    m.def("destroy_stream", &torch_kunlun::destroy_stream);
    m.def("set_stream", &torch_kunlun::set_current_stream);
    m.def("synchronize_stream", &torch_kunlun::synchronize_stream);
    m.def("event_create", &torch_kunlun::event_create);
    m.def("event_destroy", &torch_kunlun::event_destroy);
    m.def("event_record", &torch_kunlun::event_record);
    m.def("stream_wait_event", &torch_kunlun::stream_wait_event);
    m.def("event_synchronize", &torch_kunlun::event_synchronize);
    m.def("event_query", &torch_kunlun::event_query);

    m.def("xccl_get_unique_id", []() {
        return py::bytes(torch_kunlun::xccl_get_unique_id());
    });
    m.def("xccl_init_rank", [](int rank, int nranks, py::bytes uid, int device) {
        torch_kunlun::xccl_init_rank(rank, nranks, uid.cast<std::string>(), device);
    });
    m.def("xccl_destroy", &torch_kunlun::xccl_destroy);
    m.def("xccl_rank", &torch_kunlun::xccl_rank);
    m.def("xccl_nranks", &torch_kunlun::xccl_nranks);
    m.def("xccl_all_reduce", &torch_kunlun::xccl_all_reduce);
    m.def("xccl_broadcast", &torch_kunlun::xccl_broadcast);
    m.def("xccl_all_gather", &torch_kunlun::xccl_all_gather);
    m.def("xccl_reduce_scatter", &torch_kunlun::xccl_reduce_scatter);
    m.def("xccl_barrier", &torch_kunlun::xccl_barrier);

    m.def("get_device_properties", [](int device) {
        uint64_t memory = 0;
        uint64_t clusters = 0;
        uint64_t model = 0;
        xpu_device_get_attr(&memory, XPUATTR_MEM_MAIN_CAPACITY, device);
        xpu_device_get_attr(&clusters, XPUATTR_NUM_CLUSTER, device);
        xpu_device_get_attr(&model, XPUATTR_MODEL, device);

        py::dict props;
        props["name"] = "Kunlun K200";
        props["total_memory"] = memory;
        props["multi_processor_count"] = static_cast<int>(clusters) * 16;
        props["warp_size"] = 8;
        props["major"] = 1;
        props["minor"] = 0;
        props["model"] = static_cast<int>(model);
        return props;
    });
}
