#include "kunlun.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

#include "xpu/refactor/context/newcontext.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace torch_kunlun {

void check_xpu(int ret, const char* what) {
    TORCH_CHECK(ret == 0, what, " failed: ret=", ret, " (", xpu_strerror(ret), ")");
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
    XPUStream stream = nullptr;
    check_xpu(xpu_stream_create(&stream), "xpu_stream_create");
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
    xpu_stream_destroy(handle->stream);
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
    check_xpu(xpu_wait(handle->stream), "xpu_wait(stream)");
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
    for (auto* h : handles) {
        xpu_wait(h->stream);
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
    // K200: 4 clusters, 4 SDNN.  Ignore a rejection so the runtime can still
    // be used on a different partition.
    ctx->set_ncluster(4);
    ctx->set_nsdnn(4);
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
        return getStream(d);
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

    at::Device getDeviceFromPtr(void*) const override {
        return at::Device(at::DeviceType::PrivateUse1, current_device());
    }

    bool isPinnedPtr(const void*) const override {
        return false;
    }

    c10::Allocator* getPinnedMemoryAllocator() const override {
        TORCH_CHECK(false, "kunlun: pinned host memory is not implemented in the basic runtime");
        return nullptr;
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

    m.def("current_stream", &torch_kunlun::current_stream_id);
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
