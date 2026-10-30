#include "kunlun.h"

#include <c10/core/Stream.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch_kunlun {
namespace {

struct LiveBlock {
    int device;
    size_t size;        // block capacity (rounded, may exceed the request)
    size_t requested;   // bytes asked for by the caller
    uint64_t alloc_stream;
    std::vector<uint64_t> events;
};

struct FreeBlock {
    int device;
    void* ptr;
    size_t size;
    std::vector<uint64_t> events;
};

struct DeviceBudget {
    size_t capacity = 0;
    size_t budget = 0;
    size_t reserved = 0;
    size_t max_single = 0;
};

constexpr size_t kMiB = 1024ULL * 1024ULL;
constexpr size_t kGiB = 1024ULL * kMiB;

// K200 device pointers are 32-bit and the runtime's copy/launch length is a
// signed 32-bit value, so a single live allocation must stay below 2 GiB.
constexpr size_t kMaxSingleAlloc = 1ULL << 31;
constexpr size_t kRoundSize = 512;
constexpr size_t kMinSplit = 1ULL * kMiB;

size_t round_size(size_t n) {
    return (n + kRoundSize - 1) & ~(kRoundSize - 1);
}

size_t env_size(const char* name, size_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value) return fallback;
    return static_cast<size_t>(parsed);
}

double env_fraction(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || parsed <= 0.0) return fallback;
    return parsed;
}

class KunlunCachingAllocator final : public c10::Allocator {
public:
    static KunlunCachingAllocator& instance() {
        static KunlunCachingAllocator allocator;
        return allocator;
    }

    c10::DataPtr allocate(size_t n) override {
        int device = 0;
        check_xpu(xpu_current_device(&device), "xpu_current_device");
        const auto c10_device = c10::Device(c10::DeviceType::PrivateUse1, device);
        if (n == 0) {
            return c10::DataPtr(nullptr, c10_device);
        }
        const size_t need = round_size(n);
        TORCH_CHECK(need < kMaxSingleAlloc,
                    "kunlun single allocation must stay below 2 GiB (K200 uses "
                    "32-bit device pointers and int32 copy lengths), got ", n);

        uint64_t alloc_stream = 0;
        try {
            alloc_stream = current_stream_id(device);
        } catch (...) {
        }

        void* ptr = nullptr;
        size_t capacity = need;
        {
            std::lock_guard<std::mutex> lock(mu_);
            FreeBlock reused;
            reused.ptr = nullptr;
            while (find_best_fit_locked(device, need, reused)) {
                if (wait_events(reused.events, device)) {
                    break;
                }
                // The block was last used on a torn-down stream, so its events
                // cannot prove quiescence.  Drain every stream once, then free
                // it rather than return memory with an unknown lifetime.
                synchronize_all();
                xpu_set_device(device);
                if (xpu_free(reused.ptr) != 0) {
                    std::fprintf(stderr,
                                 "kunlun allocator: xpu_free(%p) failed while reclaiming a "
                                 "block from a torn-down stream\n",
                                 reused.ptr);
                }
                release_reserved_locked(device, reused.size);
                reused.ptr = nullptr;
            }
            if (reused.ptr != nullptr) {
                ptr = reused.ptr;
                capacity = reused.size;
                if (capacity >= need + kMinSplit) {
                    const auto base = reinterpret_cast<uintptr_t>(ptr);
                    insert_free_locked(FreeBlock{device, reinterpret_cast<void*>(base + need),
                                                 capacity - need, {}});
                    capacity = need;
                }
            } else {
                auto& budget = budget_for_locked(device);
                TORCH_CHECK(budget.reserved <= budget.budget &&
                                need <= budget.budget - budget.reserved,
                            "kunlun allocation guard: request ", n,
                            " bytes exceeds remaining device budget (device=", device,
                            ", reserved=", budget.reserved, ", budget=", budget.budget,
                            "). Set KUNLUN_MEMORY_FRACTION / KUNLUN_MEMORY_RESERVE_MB "
                            "or KUNLUN_MEMORY_LIMIT_MB if this limit is intentional.");
                TORCH_CHECK(budget.max_single == 0 || need <= budget.max_single,
                            "kunlun allocation guard: single allocation ", n,
                            " bytes exceeds KUNLUN_MAX_ALLOC_MB=",
                            budget.max_single / kMiB, " MiB; K200 can hang on "
                            "oversized xpu_malloc. Raise/disable explicitly if intended.");
                check_xpu(xpu_malloc(&ptr, need), "xpu_malloc");
                reserve_reserved_locked(device, need);
            }
            const bool live_ok =
                live_.emplace(ptr, LiveBlock{device, capacity, n, alloc_stream, {}}).second;
            const bool range_ok =
                live_ranges_
                    .emplace(std::make_pair(device, reinterpret_cast<uintptr_t>(ptr)),
                             capacity)
                    .second;
            TORCH_CHECK(live_ok && range_ok,
                        "kunlun allocator returned a pointer that is already live: ", ptr);
            allocated_ += capacity;
        }
        return c10::DataPtr(ptr, ptr, &KunlunCachingAllocator::deleter, c10_device);
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &KunlunCachingAllocator::deleter;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        // xpu_memcpy has no stream argument and runs on the default stream of
        // the current device, so select the destination's device, drain every
        // producer stream, and then wait for the copy itself.
        const int device = device_of_ptr(dest);
        int old_device = 0;
        const bool have_old = xpu_current_device(&old_device) == 0;
        if (device >= 0) {
            xpu_set_device(device);
        }
        synchronize_all();
        const int ret = xpu_memcpy(dest, src, count, XPU_DEVICE_TO_DEVICE);
        const int wait_ret = xpu_wait(nullptr);
        if (have_old) {
            xpu_set_device(old_device);
        }
        check_xpu(ret, "xpu_memcpy D2D");
        check_xpu(wait_ret, "xpu_wait(copy_data)");
    }

    void record_stream(const c10::DataPtr& data_ptr, const c10::Stream& stream) {
        void* ptr = data_ptr.get();
        if (ptr == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto it = live_.find(ptr);
        if (it == live_.end()) {
            return;
        }
        uint64_t event_id = 0;
        try {
            event_id = event_create();
            event_record(event_id, static_cast<uint64_t>(stream.id()));
        } catch (...) {
            if (event_id != 0) {
                event_destroy(event_id);
            }
            // Fallback: wait the stream rather than risk an early reuse.
            try {
                synchronize_stream(static_cast<uint64_t>(stream.id()));
            } catch (...) {
            }
            return;
        }
        it->second.events.push_back(event_id);
    }

    void release(void* ptr) {
        if (ptr == nullptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(mu_);
        auto it = live_.find(ptr);
        if (it == live_.end()) {
            // Not allocated by this allocator (host pointer, foreign device
            // pointer, or double free).  xpu_free would corrupt a heap we do
            // not own, so leak it and report instead.
            std::fprintf(stderr,
                         "kunlun allocator: release of unknown pointer %p ignored\n", ptr);
            return;
        }

        LiveBlock block = std::move(it->second);
        live_.erase(it);
        live_ranges_.erase(std::make_pair(block.device, reinterpret_cast<uintptr_t>(ptr)));
        if (allocated_ >= block.size) {
            allocated_ -= block.size;
        }

        // The block can still be referenced by the stream that allocated it
        // (typical when a tensor is freed on another thread) and by the stream
        // that frees it.  Wait for both before handing it back to the cache.
        uint64_t free_stream = 0;
        try {
            free_stream = current_stream_id(block.device);
        } catch (...) {
        }
        std::vector<uint64_t> streams;
        if (block.alloc_stream != 0) {
            streams.push_back(block.alloc_stream);
        }
        if (free_stream != 0 && free_stream != block.alloc_stream) {
            streams.push_back(free_stream);
        }
        bool complete = !streams.empty();
        for (uint64_t stream_id : streams) {
            uint64_t event_id = 0;
            try {
                event_id = event_create();
                event_record(event_id, stream_id);
                block.events.push_back(event_id);
            } catch (...) {
                if (event_id != 0) {
                    event_destroy(event_id);
                }
                complete = false;
            }
        }
        if (!complete) {
            // Event bookkeeping failed or the stream is unknown: wait every
            // stream once rather than risk an early reuse.
            synchronize_all();
        }

        insert_free_locked(FreeBlock{block.device, ptr, block.size, std::move(block.events)});
    }

    void empty_cache() {
        std::vector<FreeBlock> blocks;
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (auto& entry : free_) {
                blocks.push_back(std::move(entry.second));
            }
            free_.clear();
            for (auto& block : blocks) {
                release_reserved_locked(block.device, block.size);
            }
        }
        for (auto& block : blocks) {
            if (!wait_events(block.events, block.device, /*destroy=*/true)) {
                // At least one event referred to a torn-down stream.  Drain
                // everything once rather than free memory still in use.
                synchronize_all();
            }
            xpu_set_device(block.device);
            if (xpu_free(block.ptr) != 0) {
                std::fprintf(stderr, "kunlun empty_cache: xpu_free(%p) failed\n", block.ptr);
            }
        }
    }

    size_t allocated() const {
        std::lock_guard<std::mutex> lock(mu_);
        return allocated_;
    }

    size_t reserved() const {
        std::lock_guard<std::mutex> lock(mu_);
        return reserved_;
    }

    int device_of_ptr(void* ptr) const {
        if (ptr == nullptr) {
            return -1;
        }
        const auto addr = reinterpret_cast<uintptr_t>(ptr);
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto& entry : budgets_) {
            const int device = entry.first;
            auto it = live_ranges_.upper_bound(std::make_pair(device, addr));
            if (it == live_ranges_.begin()) {
                continue;
            }
            --it;
            if (it->first.first != device) {
                continue;
            }
            if (addr < it->first.second + it->second) {
                return device;
            }
        }
        return -1;
    }

private:
    KunlunCachingAllocator() = default;

    static void deleter(void* ptr) {
        KunlunCachingAllocator::instance().release(ptr);
    }

    // Pops the smallest free block of this device that can satisfy `need`.
    bool find_best_fit_locked(int device, size_t need, FreeBlock& out) {
        auto best = free_.end();
        for (auto it = free_.lower_bound(std::make_pair(device, uintptr_t{0}));
             it != free_.end() && it->first.first == device; ++it) {
            if (it->second.size < need) {
                continue;
            }
            if (best == free_.end() || it->second.size < best->second.size) {
                best = it;
            }
        }
        if (best == free_.end()) {
            return false;
        }
        out = std::move(best->second);
        free_.erase(best);
        return true;
    }

    // Inserts a free block and merges it with address-adjacent neighbours so
    // that a later large request can reuse the combined range.
    void insert_free_locked(FreeBlock block) {
        const int device = block.device;
        const uintptr_t base = reinterpret_cast<uintptr_t>(block.ptr);
        auto it = free_.emplace(std::make_pair(device, base), std::move(block)).first;

        auto next = std::next(it);
        if (next != free_.end() && next->first.first == device &&
            base + it->second.size == next->first.second) {
            it->second.size += next->second.size;
            it->second.events.insert(it->second.events.end(), next->second.events.begin(),
                                     next->second.events.end());
            free_.erase(next);
        }
        if (it != free_.begin()) {
            auto prev = std::prev(it);
            if (prev->first.first == device &&
                prev->first.second + prev->second.size == base) {
                prev->second.size += it->second.size;
                prev->second.events.insert(prev->second.events.end(),
                                           it->second.events.begin(),
                                           it->second.events.end());
                free_.erase(it);
            }
        }
    }

    // Returns true if all events completed and were destroyed.
    bool wait_events(std::vector<uint64_t>& events, int device, bool destroy = true) {
        xpu_set_device(device);
        bool ok = true;
        for (uint64_t event_id : events) {
            try {
                event_synchronize(event_id);
            } catch (...) {
                ok = false;
            }
            if (destroy) {
                try {
                    event_destroy(event_id);
                } catch (...) {
                }
            }
        }
        events.clear();
        return ok;
    }

    void reserve_reserved_locked(int device, size_t size) {
        budget_for_locked(device).reserved += size;
        reserved_ += size;
    }

    void release_reserved_locked(int device, size_t size) {
        auto it = budgets_.find(device);
        if (it != budgets_.end()) {
            it->second.reserved = it->second.reserved >= size ? it->second.reserved - size : 0;
        }
        reserved_ = reserved_ >= size ? reserved_ - size : 0;
    }

    DeviceBudget& budget_for_locked(int device) {
        auto it = budgets_.find(device);
        if (it != budgets_.end()) return it->second;
        DeviceBudget budget;
        uint64_t capacity = 0;
        if (xpu_device_get_attr(&capacity, XPUATTR_MEM_MAIN_CAPACITY, device) != 0 ||
            capacity == 0) {
            // Conservative fallback if the runtime cannot report capacity.
            capacity = 6 * kGiB;
        } else {
            // XRE versions differ: some report bytes, some report MB, and
            // some report GB.  Normalize the obvious small values.  A real
            // K200 PD is 8 GiB, so this does not mask a genuinely tiny pool.
            if (capacity <= 128) {
                capacity *= kGiB;
            } else if (capacity < (1ULL << 20)) {
                capacity *= kMiB;
            }
        }
        const size_t limit_mb = env_size("KUNLUN_MEMORY_LIMIT_MB", 0);
        if (limit_mb != 0) {
            capacity = std::min<uint64_t>(capacity, limit_mb * kMiB);
        }
        const double fraction = std::min(1.0, std::max(0.1, env_fraction(
            "KUNLUN_MEMORY_FRACTION", 0.85)));
        const size_t reserve = env_size("KUNLUN_MEMORY_RESERVE_MB", 256) * kMiB;
        budget.capacity = static_cast<size_t>(capacity);
        budget.budget = static_cast<size_t>(capacity * fraction);
        budget.budget = budget.budget > reserve ? budget.budget - reserve : 0;
        budget.max_single = std::min(env_size("KUNLUN_MAX_ALLOC_MB", 2048) * kMiB,
                                     kMaxSingleAlloc);
        if (budget.max_single > budget.budget) budget.max_single = budget.budget;
        return budgets_.emplace(device, budget).first->second;
    }

    mutable std::mutex mu_;
    // Address-ordered per device: key is (device, base address).
    std::map<std::pair<int, uintptr_t>, FreeBlock> free_;
    // Interval index of live blocks, also address-ordered per device.
    std::map<std::pair<int, uintptr_t>, size_t> live_ranges_;
    std::map<int, DeviceBudget> budgets_;
    std::unordered_map<void*, LiveBlock> live_;
    size_t allocated_ = 0;
    size_t reserved_ = 0;
};

}  // namespace

c10::Allocator* get_allocator() {
    return &KunlunCachingAllocator::instance();
}

size_t memory_allocated() {
    return KunlunCachingAllocator::instance().allocated();
}

size_t memory_reserved() {
    return KunlunCachingAllocator::instance().reserved();
}

int device_of_ptr(void* ptr) {
    return KunlunCachingAllocator::instance().device_of_ptr(ptr);
}

void empty_cache() {
    KunlunCachingAllocator::instance().empty_cache();
}

void record_data_ptr_on_stream(const c10::DataPtr& data_ptr, const c10::Stream& stream) {
    KunlunCachingAllocator::instance().record_stream(data_ptr, stream);
}

}  // namespace torch_kunlun
