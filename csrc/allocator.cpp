#include "kunlun.h"

#include <c10/core/Stream.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch_kunlun {
namespace {

struct LiveBlock {
    int device;
    size_t size;
    std::vector<uint64_t> events;
};

struct FreeBlock {
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
        // K200 device pointers are 32-bit; keep each allocation below 4 GiB.
        TORCH_CHECK(n < (1ULL << 32), "kunlun allocation must be <4 GiB, got ", n);

        void* ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu_);
            const auto key = std::make_pair(device, n);
            auto it = free_.find(key);
            while (it != free_.end() && !it->second.empty()) {
                FreeBlock block = std::move(it->second.back());
                it->second.pop_back();
                if (wait_events(block.events, device)) {
                    ptr = block.ptr;
                    break;
                }
                // The block was used on a torn-down stream; free it rather
                // than returning memory with an unknown lifetime.
                xpu_set_device(device);
                xpu_free(block.ptr);
                block_device_.erase(block.ptr);
                auto& budget = budget_for_locked(device);
                if (budget.reserved >= block.size) budget.reserved -= block.size;
                if (reserved_ >= block.size) reserved_ -= block.size;
            }
            if (ptr == nullptr) {
                auto& budget = budget_for_locked(device);
                TORCH_CHECK(budget.reserved <= budget.budget &&
                                n <= budget.budget - budget.reserved,
                            "kunlun allocation guard: request ", n,
                            " bytes exceeds remaining device budget (device=", device,
                            ", reserved=", budget.reserved, ", budget=", budget.budget,
                            "). Set KUNLUN_MEMORY_FRACTION / KUNLUN_MEMORY_RESERVE_MB "
                            "or KUNLUN_MEMORY_LIMIT_MB if this limit is intentional.");
                TORCH_CHECK(budget.max_single == 0 || n <= budget.max_single,
                            "kunlun allocation guard: single allocation ", n,
                            " bytes exceeds KUNLUN_MAX_ALLOC_MB=",
                            budget.max_single / kMiB, " MiB; K200 can hang on "
                            "oversized xpu_malloc. Raise/disable explicitly if intended.");
                check_xpu(xpu_malloc(&ptr, n), "xpu_malloc");
                budget.reserved += n;
                reserved_ += n;
            }
            live_[ptr] = LiveBlock{device, n, {}};
            block_device_[ptr] = device;
            allocated_ += n;
        }
        return c10::DataPtr(ptr, ptr, &KunlunCachingAllocator::deleter, c10_device);
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &KunlunCachingAllocator::deleter;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        check_xpu(xpu_memcpy(dest, src, count, XPU_DEVICE_TO_DEVICE), "xpu_memcpy D2D");
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
            // Not allocated by this allocator; do not silently leak.
            xpu_free(ptr);
            return;
        }

        LiveBlock block = std::move(it->second);
        live_.erase(it);
        if (allocated_ >= block.size) {
            allocated_ -= block.size;
        }

        // Record only the stream that frees the block.  Uses on other streams
        // are captured by record_stream()/recordDataPtrOnStream().
        uint64_t event_id = 0;
        try {
            event_id = event_create();
            event_record(event_id, current_stream_id(block.device));
            block.events.push_back(event_id);
        } catch (...) {
            if (event_id != 0) {
                event_destroy(event_id);
            }
            // Rare fallback: event creation failed, so wait all streams once.
            synchronize_all();
        }

        free_[{block.device, block.size}].push_back(
            FreeBlock{ptr, block.size, std::move(block.events)});
    }

    void empty_cache() {
        std::vector<FreeBlock> blocks;
        std::vector<int> devices;
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (auto& entry : free_) {
                for (auto& block : entry.second) {
                    blocks.push_back(std::move(block));
                    devices.push_back(entry.first.first);
                    block_device_.erase(block.ptr);
                }
            }
            free_.clear();
            for (size_t i = 0; i < blocks.size(); ++i) {
                auto& budget = budget_for_locked(devices[i]);
                if (budget.reserved >= blocks[i].size) {
                    budget.reserved -= blocks[i].size;
                }
                if (reserved_ >= blocks[i].size) reserved_ -= blocks[i].size;
            }
        }
        for (size_t i = 0; i < blocks.size(); ++i) {
            wait_events(blocks[i].events, devices[i], /*destroy=*/true);
            xpu_set_device(devices[i]);
            xpu_free(blocks[i].ptr);
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
        std::lock_guard<std::mutex> lock(mu_);
        auto it = block_device_.find(ptr);
        return it == block_device_.end() ? -1 : it->second;
    }

private:
    KunlunCachingAllocator() = default;

    static void deleter(void* ptr) {
        KunlunCachingAllocator::instance().release(ptr);
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
        budget.max_single = env_size("KUNLUN_MAX_ALLOC_MB", 1024) * kMiB;
        if (budget.max_single > budget.budget) budget.max_single = budget.budget;
        return budgets_.emplace(device, budget).first->second;
    }

    mutable std::mutex mu_;
    std::map<std::pair<int, size_t>, std::vector<FreeBlock>> free_;
    std::map<int, DeviceBudget> budgets_;
    std::unordered_map<void*, LiveBlock> live_;
    std::unordered_map<void*, int> block_device_;
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
