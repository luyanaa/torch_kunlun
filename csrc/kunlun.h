#pragma once

#include <torch/extension.h>
#include <xpu/runtime.h>

#include <cstdint>
#include <mutex>

namespace baidu {
namespace xpu {
namespace api {
struct Context;
}  // namespace api
}  // namespace xpu
}  // namespace baidu

namespace torch_kunlun {

void check_xpu(int ret, const char* what);
int device_count();
int current_device();
void set_device(int device);
void synchronize();
void synchronize_all();

// Stream ids are opaque handles (StreamHandle* / EventHandle* cast to uint64_t).
uint64_t current_stream_id(int device);
uint64_t default_stream_id(int device);
uint64_t create_stream(int device);
void destroy_stream(uint64_t stream_id);
void set_current_stream(uint64_t stream_id);
void synchronize_stream(uint64_t stream_id);
XPUStream current_stream_native(int device);

uint64_t event_create();
void event_destroy(uint64_t event_id);
void event_record(uint64_t event_id, uint64_t stream_id);
void stream_wait_event(uint64_t stream_id, uint64_t event_id);
void event_synchronize(uint64_t event_id);
bool event_query(uint64_t event_id);
void note_cpu_fallback(const char* op);
// Boxed fallback for operators with no PrivateUse1 implementation at all.
// Denied unless KUNLUN_CPU_FALLBACK=warn|silent.
void check_generic_fallback(const char* op);
void record_data_ptr_on_stream(const c10::DataPtr& data_ptr, const c10::Stream& stream);

// BKCL/XCCL collective wrappers (one context per process).
std::string xccl_get_unique_id();
void xccl_init_rank(int rank, int nranks, const std::string& unique_id, int device);
void xccl_destroy();
int xccl_rank();
int xccl_nranks();
void xccl_all_reduce(const at::Tensor& tensor, const std::string& op, int device);
void xccl_broadcast(const at::Tensor& tensor, int root, int device);
void xccl_all_gather(const at::Tensor& out, const at::Tensor& input, int device);
void xccl_reduce_scatter(const at::Tensor& out, const at::Tensor& input,
                         const std::string& op, int device);
void xccl_barrier(int device);

void adam_step(const at::Tensor& param, const at::Tensor& grad, at::Tensor& mom1, at::Tensor& mom2, double beta1_pow, double beta2_pow, double lr, double beta1, double beta2, double eps);
void adamw_step(const at::Tensor& param, const at::Tensor& grad, at::Tensor& mom1, at::Tensor& mom2, double beta1_pow, double beta2_pow, double lr, double beta1, double beta2, double eps, double weight_decay);

size_t memory_allocated();
size_t memory_reserved();
int device_of_ptr(void* ptr);
void empty_cache();
c10::Allocator* get_allocator();

// Pinned host memory.  xpu_host_alloc pages are directly DMA-able by XPURT,
// which is what DataLoader(pin_memory=True) and non_blocking H2D rely on.
bool is_pinned_ptr(const void* ptr);
c10::Allocator* get_pinned_allocator();

// XDNN cluster scheduling.  A K200 PD has 4 clusters and 4 SDNN engines; every
// launch uses at most `cluster_budget()` clusters (1..4).  The default is 4.
// Lower the budget when another engine (e.g. Triton with ncluster=2) must run
// on the same PD, or leave KUNLUN_CLUSTER_POLICY=auto to use one cluster for
// small launches.
int set_cluster_budget(int nclusters, int nsdnn);
int cluster_budget();
int sdnn_budget();

// XDNN library version actually loaded by this process.  The extension is
// compiled against one XDNN header set; a different libxpuapi at runtime is a
// silent-ABI-change risk, so the version is detected at import time.
std::string xdnn_version();
void check_xdnn_runtime();

// One XDNN context per device.  The context owns a launch stream and a shared
// workspace, so only one launch may be in flight at a time: the mutex
// serializes launches and a launch on a different stream drains the previous
// stream first.  Ops on the same stream still pipeline normally.
struct DeviceContext {
    int device;
    baidu::xpu::api::Context* ctx;
    std::mutex mu;
    XPUStream active_stream = nullptr;
    // Last accepted set_ncluster/set_nsdnn values; 0 forces a reapply.  Only
    // touched while mu is held, and the vendor call is skipped when unchanged
    // because it may resize the context workspace.
    int active_ncluster = 0;
    int active_nsdnn = 0;

    explicit DeviceContext(int device);
    ~DeviceContext();
};

DeviceContext& get_context(int device);

// Acquires the device context and applies the cluster policy.  `work_elems` is
// an optional work estimate (MACs for GEMM, elements otherwise) used by the
// "auto" cluster policy.
class ContextGuard {
public:
    explicit ContextGuard(int device, int64_t work_elems = 0);
    ~ContextGuard() = default;
    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;

    DeviceContext& dc;

private:
    std::unique_lock<std::mutex> lock_;
};

}  // namespace torch_kunlun
