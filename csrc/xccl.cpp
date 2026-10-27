#include "kunlun.h"

#include <bkcl.h>

#include <cstring>
#include <mutex>
#include <string>

namespace torch_kunlun {
namespace {

std::mutex& xccl_mu() {
    static std::mutex mu;
    return mu;
}

BKCLContext_t& xccl_ctx() {
    static BKCLContext_t ctx = nullptr;
    return ctx;
}

int& xccl_rank_ref() {
    static int rank = -1;
    return rank;
}

int& xccl_nranks_ref() {
    static int nranks = 0;
    return nranks;
}

void check_bkcl(BKCLResult_t ret, const char* what) {
    TORCH_CHECK(ret == BKCL_SUCCESS, what, " failed: ", static_cast<int>(ret));
}

BKCLDataType to_bkcl_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat: return BKCL_FLOAT;
        case at::kHalf: return BKCL_FLOAT16;
        case at::kBFloat16: return BKCL_BFLOAT16;
        case at::kDouble: return BKCL_FLOAT64;
        case at::kInt: return BKCL_INT32;
        case at::kLong: return BKCL_INT64;
        case at::kByte:
        case at::kBool: return BKCL_UINT8;
        default:
            TORCH_CHECK(false, "kunlun xccl unsupported dtype ", dtype);
    }
}

BKCLOp to_bkcl_op(const std::string& op) {
    if (op == "sum" || op == "add") return BKCL_ADD;
    if (op == "prod" || op == "product" || op == "mul") return BKCL_PRODUCT;
    if (op == "min") return BKCL_MIN;
    if (op == "max") return BKCL_MAX;
    TORCH_CHECK(false, "kunlun xccl unsupported reduction op: ", op);
}

void check_tensor(const at::Tensor& t, const char* what) {
    TORCH_CHECK(t.device().type() == at::DeviceType::PrivateUse1,
                what, " requires a kunlun tensor");
    TORCH_CHECK(t.is_contiguous(), what, " requires a contiguous tensor");
}

}  // namespace

std::string xccl_get_unique_id() {
    BKCLUniqueId uid;
    check_bkcl(bkcl_get_unique_id(&uid), "bkcl_get_unique_id");
    return std::string(uid.id, BKCL_UNIQUE_ID_BYTES);
}

void xccl_init_rank(int rank, int nranks, const std::string& unique_id, int device) {
    // Single-node BKCL only.  The project does not bootstrap multi-node ranks.
    TORCH_CHECK(nranks > 0 && rank >= 0 && rank < nranks,
                "kunlun xccl requires 0 <= rank < nranks, got rank=", rank, " nranks=", nranks);
    TORCH_CHECK(unique_id.size() == BKCL_UNIQUE_ID_BYTES,
                "kunlun xccl unique id must be ", BKCL_UNIQUE_ID_BYTES, " bytes");
    std::lock_guard<std::mutex> lock(xccl_mu());
    if (xccl_ctx() != nullptr) {
        bkcl_destroy_context(xccl_ctx());
        xccl_ctx() = nullptr;
    }
    xpu_set_device(device);
    BKCLUniqueId uid;
    std::memcpy(uid.id, unique_id.data(), BKCL_UNIQUE_ID_BYTES);
    check_bkcl(bkcl_init_rank(&xccl_ctx(), rank, nranks, &uid), "bkcl_init_rank");
    xccl_rank_ref() = rank;
    xccl_nranks_ref() = nranks;
}

void xccl_destroy() {
    std::lock_guard<std::mutex> lock(xccl_mu());
    if (xccl_ctx() != nullptr) {
        bkcl_destroy_context(xccl_ctx());
        xccl_ctx() = nullptr;
    }
    xccl_rank_ref() = -1;
    xccl_nranks_ref() = 0;
}

int xccl_rank() {
    return xccl_rank_ref();
}

int xccl_nranks() {
    return xccl_nranks_ref();
}

void xccl_all_reduce(const at::Tensor& tensor, const std::string& op, int device) {
    check_tensor(tensor, "all_reduce");
    TORCH_CHECK(xccl_ctx() != nullptr, "kunlun xccl is not initialized");
    xpu_set_device(device);
    void* ptr = tensor.data_ptr();
    check_bkcl(bkcl_all_reduce(xccl_ctx(), ptr, ptr, tensor.numel(),
                               to_bkcl_dtype(tensor.scalar_type()), to_bkcl_op(op),
                               current_stream_native(device)),
               "bkcl_all_reduce");
}

void xccl_broadcast(const at::Tensor& tensor, int root, int device) {
    check_tensor(tensor, "broadcast");
    TORCH_CHECK(xccl_ctx() != nullptr, "kunlun xccl is not initialized");
    xpu_set_device(device);
    void* ptr = tensor.data_ptr();
    const void* send = (xccl_rank_ref() == root) ? ptr : nullptr;
    check_bkcl(bkcl_broadcast(xccl_ctx(), send, ptr, tensor.numel(),
                              to_bkcl_dtype(tensor.scalar_type()), root,
                              current_stream_native(device)),
               "bkcl_broadcast");
}

void xccl_all_gather(const at::Tensor& out, const at::Tensor& input, int device) {
    check_tensor(out, "all_gather out");
    check_tensor(input, "all_gather input");
    TORCH_CHECK(out.scalar_type() == input.scalar_type(), "all_gather dtype mismatch");
    TORCH_CHECK(xccl_ctx() != nullptr, "kunlun xccl is not initialized");
    const int nranks = xccl_nranks_ref();
    TORCH_CHECK(out.numel() >= input.numel() * static_cast<int64_t>(nranks),
                "all_gather output is too small");
    xpu_set_device(device);
    check_bkcl(bkcl_all_gather(xccl_ctx(), input.data_ptr(), input.numel(),
                               out.data_ptr(), to_bkcl_dtype(input.scalar_type()),
                               current_stream_native(device)),
               "bkcl_all_gather");
}

void xccl_reduce_scatter(const at::Tensor& out, const at::Tensor& input,
                         const std::string& op, int device) {
    check_tensor(out, "reduce_scatter out");
    check_tensor(input, "reduce_scatter input");
    TORCH_CHECK(out.scalar_type() == input.scalar_type(), "reduce_scatter dtype mismatch");
    TORCH_CHECK(xccl_ctx() != nullptr, "kunlun xccl is not initialized");
    const int nranks = xccl_nranks_ref();
    TORCH_CHECK(input.numel() >= out.numel() * static_cast<int64_t>(nranks),
                "reduce_scatter input is too small");
    xpu_set_device(device);
    check_bkcl(bkcl_reduce_scatter(xccl_ctx(), input.data_ptr(), out.data_ptr(),
                                   out.numel(), to_bkcl_dtype(out.scalar_type()),
                                   to_bkcl_op(op), current_stream_native(device)),
               "bkcl_reduce_scatter");
}

void xccl_barrier(int device) {
    TORCH_CHECK(xccl_ctx() != nullptr, "kunlun xccl is not initialized");
    xpu_set_device(device);
    check_bkcl(bkcl_barrier(xccl_ctx()), "bkcl_barrier");
}

}  // namespace torch_kunlun
