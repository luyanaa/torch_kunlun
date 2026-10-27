#pragma once

#include <torch/extension.h>
#include <xpu/runtime.h>

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
void empty_cache();
c10::Allocator* get_allocator();

// One XDNN context per device.  The mutex serializes ops that share the
// context (XDNN context stream/workspace state is not thread-safe).
struct DeviceContext {
    int device;
    baidu::xpu::api::Context* ctx;
    std::mutex mu;

    explicit DeviceContext(int device);
    ~DeviceContext();
};

DeviceContext& get_context(int device);

}  // namespace torch_kunlun
