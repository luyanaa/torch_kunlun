#include "kunlun.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include "xpu/refactor/math.h"
#include "xpu/refactor/nn.h"
#include "xpu/refactor/nn_activation.h"
#include "xpu/refactor/nn_pool.h"

#include <optional>
#include <string>
#include <vector>

namespace torch_kunlun {
namespace {

using baidu::xpu::api::Context;

struct ContextGuard {
    DeviceContext& dc;
    std::unique_lock<std::mutex> lock;

    explicit ContextGuard(int device)
        : dc(get_context(device)), lock(dc.mu) {
        xpu_set_device(dc.device);
        dc.ctx->set_stream(current_stream_native(dc.device));
    }
};

void check_tensor(const at::Tensor& t, const char* what) {
    TORCH_CHECK(t.device().type() == at::DeviceType::PrivateUse1,
                what, " requires a kunlun tensor");
    TORCH_CHECK(t.is_contiguous(), what, " requires a contiguous tensor");
}

void check_same_dtype(const at::Tensor& a, const at::Tensor& b, const char* what) {
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), what, " dtype mismatch");
}

template <typename T>
const T* cptr(const at::Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}

template <typename T>
T* mptr(at::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

std::vector<int64_t> pair(at::IntArrayRef value, int64_t fallback = 0) {
    if (value.empty()) {
        return {fallback, fallback};
    }
    if (value.size() == 1) {
        return {value[0], value[0]};
    }
    TORCH_CHECK(value.size() == 2, "expected 1 or 2 element array");
    return {value[0], value[1]};
}

int64_t normalize_dim(int64_t dim, int64_t rank) {
    if (dim < 0) {
        dim += rank;
    }
    TORCH_CHECK(dim >= 0 && dim < rank, "dim out of range");
    return dim;
}

at::Tensor to_float(const at::Tensor& t, const char* what) {
    check_tensor(t, what);
    if (t.scalar_type() == at::kFloat) {
        return t;
    }
    TORCH_CHECK(t.scalar_type() == at::kHalf, what, " must be float16/float32");
    auto out = at::empty(t.sizes(), t.options().dtype(at::kFloat));
    ContextGuard guard(t.device().index());
    const int ret = baidu::xpu::api::cast<float16, float>(
        guard.dc.ctx, cptr<float16>(t), mptr<float>(out), t.numel());
    TORCH_CHECK(ret == 0, what, " float conversion failed: ret=", ret);
    return out;
}

template <typename T>
at::Tensor softmax_impl(const at::Tensor& self, int64_t dim) {
    check_tensor(self, "softmax");
    auto out = at::empty_like(self);
    const int64_t axis = normalize_dim(dim, self.dim());
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::softmax<T>(
        guard.dc.ctx, cptr<T>(self), mptr<T>(out), self.sizes().vec(), axis);
    TORCH_CHECK(ret == 0, "xdnn::softmax failed: ret=", ret);
    return out;
}

template <typename T, typename TW>
at::Tensor layer_norm_impl(const at::Tensor& input, at::IntArrayRef normalized_shape,
                           const std::optional<at::Tensor>& weight,
                           const std::optional<at::Tensor>& bias, double eps) {
    check_tensor(input, "layer_norm");
    int64_t n = 1;
    for (int64_t s : normalized_shape) {
        n *= s;
    }
    TORCH_CHECK(n > 0 && input.numel() % n == 0, "layer_norm normalized_shape mismatch");
    const int64_t m = input.numel() / n;

    constexpr bool kTWIsHalf = std::is_same<TW, float16>::value;
    constexpr bool kTWIsFloat = std::is_same<TW, float>::value;
    at::Tensor gamma;
    at::Tensor beta;
    if (weight.has_value()) {
        TORCH_CHECK(weight->numel() == n, "layer_norm weight size mismatch");
        gamma = weight->is_contiguous() ? *weight : weight->contiguous();
        TORCH_CHECK(gamma.scalar_type() == (kTWIsHalf ? at::kHalf : at::kFloat),
                    "layer_norm weight dtype mismatch");
    } else {
        TORCH_CHECK(kTWIsFloat, "layer_norm without weight requires TW=float");
        gamma = at::ones({n}, input.options().dtype(at::kFloat));
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() == n, "layer_norm bias size mismatch");
        beta = bias->is_contiguous() ? *bias : bias->contiguous();
        TORCH_CHECK(beta.scalar_type() == (kTWIsHalf ? at::kHalf : at::kFloat),
                    "layer_norm bias dtype mismatch");
    } else {
        TORCH_CHECK(kTWIsFloat, "layer_norm without bias requires TW=float");
        beta = at::zeros({n}, input.options().dtype(at::kFloat));
    }

    auto out = at::empty_like(input);
    auto mean = at::empty({m}, input.options().dtype(at::kFloat));
    auto var = at::empty({m}, input.options().dtype(at::kFloat));
    ContextGuard guard(input.device().index());
    const int ret = baidu::xpu::api::layer_norm<T, TW>(
        guard.dc.ctx, cptr<T>(input), mptr<T>(out), m, n, static_cast<float>(eps),
        cptr<TW>(gamma), cptr<TW>(beta), mptr<float>(mean), mptr<float>(var),
        /*is_rstd=*/false);
    TORCH_CHECK(ret == 0, "xdnn::layer_norm failed: ret=", ret);
    return out;
}

template <typename T>
at::Tensor gelu_impl(const at::Tensor& self, const std::string& approximate) {
    check_tensor(self, "gelu");
    auto out = at::empty_like(self);
    ContextGuard guard(self.device().index());
    int ret = 0;
    if (approximate == "tanh") {
        ret = baidu::xpu::api::approximate_gelu<T>(
            guard.dc.ctx, cptr<T>(self), mptr<T>(out), self.numel());
    } else {
        ret = baidu::xpu::api::gelu<T>(
            guard.dc.ctx, cptr<T>(self), mptr<T>(out), self.numel());
    }
    TORCH_CHECK(ret == 0, "xdnn::gelu failed: ret=", ret);
    return out;
}

template <typename T, typename TID>
at::Tensor embedding_impl(const at::Tensor& weight, const at::Tensor& indices,
                          int64_t padding_idx) {
    check_tensor(weight, "embedding weight");
    check_tensor(indices, "embedding indices");
    TORCH_CHECK(weight.dim() == 2, "embedding weight must be 2-D");
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                "embedding indices must be int32/int64");

    std::vector<int64_t> out_shape = indices.sizes().vec();
    out_shape.push_back(weight.size(1));
    auto out = at::empty(out_shape, weight.options());
    ContextGuard guard(weight.device().index());
    const int ret = baidu::xpu::api::embedding<T, TID>(
        guard.dc.ctx, cptr<T>(weight), cptr<TID>(indices), mptr<T>(out),
        weight.size(0), indices.numel(), weight.size(1), padding_idx);
    TORCH_CHECK(ret == 0, "xdnn::embedding failed: ret=", ret);
    return out;
}

template <typename T>
at::Tensor cat_impl(const std::vector<at::Tensor>& tensors, int64_t dim) {
    TORCH_CHECK(!tensors.empty(), "cat got an empty tensor list");
    const auto& first = tensors.front();
    check_tensor(first, "cat");
    dim = normalize_dim(dim, first.dim());
    std::vector<int64_t> out_shape = first.sizes().vec();
    out_shape[dim] = 0;
    std::vector<const T*> ptrs;
    std::vector<std::vector<int64_t>> shapes;
    for (const auto& t : tensors) {
        check_tensor(t, "cat");
        check_same_dtype(first, t, "cat");
        TORCH_CHECK(t.dim() == first.dim(), "cat rank mismatch");
        for (int64_t d = 0; d < t.dim(); ++d) {
            if (d != dim) {
                TORCH_CHECK(t.size(d) == first.size(d), "cat shape mismatch");
            }
        }
        out_shape[dim] += t.size(dim);
        ptrs.push_back(cptr<T>(t));
        shapes.push_back(t.sizes().vec());
    }
    auto out = at::empty(out_shape, first.options());
    ContextGuard guard(first.device().index());
    const int ret = baidu::xpu::api::concat<T>(guard.dc.ctx, ptrs, mptr<T>(out), shapes, dim);
    TORCH_CHECK(ret == 0, "xdnn::concat failed: ret=", ret);
    return out;
}

template <typename T, typename TGEMM>
at::Tensor bmm_impl(const at::Tensor& a, const at::Tensor& b) {
    check_tensor(a, "bmm");
    check_tensor(b, "bmm");
    check_same_dtype(a, b, "bmm");
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "bmm requires 3-D tensors");
    const int64_t batch = a.size(0);
    const int64_t m = a.size(1);
    const int64_t k = a.size(2);
    const int64_t n = b.size(2);
    TORCH_CHECK(b.size(0) == batch && b.size(1) == k, "bmm shape mismatch");

    auto out = at::empty({batch, m, n}, a.options());
    ContextGuard guard(a.device().index());
    for (int64_t i = 0; i < batch; ++i) {
        const int ret = baidu::xpu::api::fc<T, T, T, TGEMM>(
            guard.dc.ctx, cptr<T>(a) + i * m * k, cptr<T>(b) + i * k * n,
            mptr<T>(out) + i * m * n, m, n, k, false, false,
            nullptr, nullptr, nullptr);
        TORCH_CHECK(ret == 0, "xdnn::fc (bmm) failed: ret=", ret);
    }
    return out;
}

template <typename T>
std::tuple<at::Tensor, at::Tensor> max_pool2d_impl(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    check_tensor(self, "max_pool2d");
    TORCH_CHECK(self.dim() == 4, "max_pool2d requires 4-D input");
    TORCH_CHECK(!ceil_mode, "kunlun max_pool2d does not support ceil_mode=True");
    const auto k = pair(kernel_size);
    const auto s = stride.empty() ? k : pair(stride);
    const auto p = pair(padding);
    const auto d = pair(dilation, 1);
    const int64_t n = self.size(0), c = self.size(1), h = self.size(2), w = self.size(3);
    const int64_t oh = (h + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1;
    const int64_t ow = (w + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1;
    TORCH_CHECK(oh > 0 && ow > 0, "max_pool2d output shape is non-positive");

    auto out = at::empty({n, c, oh, ow}, self.options());
    auto indices = at::empty({n, c, oh, ow}, self.options().dtype(at::kInt));
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::max_pool2d<T, int>(
        guard.dc.ctx, cptr<T>(self), mptr<T>(out), mptr<int>(indices),
        n, c, h, w, k, s, p, /*is_nchw=*/true, nullptr, nullptr,
        /*xpu1_pad=*/true, /*ld_out_c=*/-1);
    TORCH_CHECK(ret == 0, "xdnn::max_pool2d failed: ret=", ret);
    return {out, indices};
}

template <typename T>
at::Tensor adaptive_avg_pool2d_impl(const at::Tensor& self, at::IntArrayRef output_size) {
    check_tensor(self, "adaptive_avg_pool2d");
    TORCH_CHECK(self.dim() == 4, "adaptive_avg_pool2d requires 4-D input");
    const auto os = pair(output_size);
    const int64_t n = self.size(0), c = self.size(1), h = self.size(2), w = self.size(3);
    auto out = at::empty({n, c, os[0], os[1]}, self.options());
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::adaptive_avg_pool2d<T>(
        guard.dc.ctx, cptr<T>(self), mptr<T>(out), n, c, h, w, os[0], os[1],
        /*is_nchw=*/true, nullptr, nullptr);
    TORCH_CHECK(ret == 0, "xdnn::adaptive_avg_pool2d failed: ret=", ret);
    return out;
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_infer_impl(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var, double eps) {
    check_tensor(input, "native_batch_norm");
    TORCH_CHECK(input.dim() == 4, "native_batch_norm (K200 basic) requires 4-D input");
    const int64_t n = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);

    at::Tensor gamma;
    at::Tensor beta;
    if (weight.has_value()) {
        gamma = to_float(*weight, "native_batch_norm weight");
    } else {
        gamma = at::ones({c}, input.options().dtype(at::kFloat));
    }
    if (bias.has_value()) {
        beta = to_float(*bias, "native_batch_norm bias");
    } else {
        beta = at::zeros({c}, input.options().dtype(at::kFloat));
    }
    TORCH_CHECK(running_mean.has_value() && running_var.has_value(),
                "native_batch_norm requires running_mean/running_var for inference");
    auto mean = to_float(*running_mean, "native_batch_norm running_mean");
    auto var = to_float(*running_var, "native_batch_norm running_var");
    TORCH_CHECK(mean.numel() == c && var.numel() == c, "native_batch_norm running stats size mismatch");

    auto out = at::empty_like(input);
    ContextGuard guard(input.device().index());
    const int ret = baidu::xpu::api::batch_norm_infer<T>(
        guard.dc.ctx, cptr<T>(input), mptr<T>(out), n, c, h, w,
        static_cast<float>(eps), cptr<float>(gamma), cptr<float>(beta),
        cptr<float>(mean), cptr<float>(var), /*is_nchw=*/true);
    TORCH_CHECK(ret == 0, "xdnn::batch_norm_infer failed: ret=", ret);

    auto save_mean = at::empty({0}, input.options().dtype(at::kFloat));
    auto save_invstd = at::empty({0}, input.options().dtype(at::kFloat));
    return {out, save_mean, save_invstd};
}

// ---------------------------------------------------------------------------
// Public ATen entry points
// ---------------------------------------------------------------------------

at::Tensor softmax(const at::Tensor& self, int64_t dim, std::optional<at::ScalarType>) {
    if (self.scalar_type() == at::kHalf) return softmax_impl<float16>(self, dim);
    if (self.scalar_type() == at::kFloat) return softmax_impl<float>(self, dim);
    TORCH_CHECK(false, "kunlun softmax only supports float16/float32");
}

at::Tensor _softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
    TORCH_CHECK(!half_to_float, "kunlun _softmax half_to_float=True is unsupported");
    return ::torch_kunlun::softmax(self, dim, std::nullopt);
}

at::Tensor layer_norm(const at::Tensor& input, at::IntArrayRef normalized_shape,
                      const std::optional<at::Tensor>& weight,
                      const std::optional<at::Tensor>& bias, double eps, bool) {
    if (input.scalar_type() == at::kHalf) {
        const bool weight_f32 = weight.has_value() && weight->scalar_type() == at::kFloat;
        if (weight_f32) {
            return layer_norm_impl<float16, float>(input, normalized_shape, weight, bias, eps);
        }
        return layer_norm_impl<float16, float16>(input, normalized_shape, weight, bias, eps);
    }
    if (input.scalar_type() == at::kFloat) {
        return layer_norm_impl<float, float>(input, normalized_shape, weight, bias, eps);
    }
    TORCH_CHECK(false, "kunlun layer_norm only supports float16/float32");
}

at::Tensor gelu(const at::Tensor& self, c10::string_view approximate) {
    const std::string approx(approximate);
    if (self.scalar_type() == at::kHalf) return gelu_impl<float16>(self, approx);
    if (self.scalar_type() == at::kFloat) return gelu_impl<float>(self, approx);
    TORCH_CHECK(false, "kunlun gelu only supports float16/float32");
}

at::Tensor embedding(const at::Tensor& weight, const at::Tensor& indices,
                     int64_t padding_idx, bool, bool) {
    if (indices.scalar_type() == at::kInt) {
        if (weight.scalar_type() == at::kHalf) {
            return embedding_impl<float16, int>(weight, indices, padding_idx);
        }
        if (weight.scalar_type() == at::kFloat) {
            return embedding_impl<float, int>(weight, indices, padding_idx);
        }
    } else if (indices.scalar_type() == at::kLong) {
        if (weight.scalar_type() == at::kHalf) {
            return embedding_impl<float16, int64_t>(weight, indices, padding_idx);
        }
        if (weight.scalar_type() == at::kFloat) {
            return embedding_impl<float, int64_t>(weight, indices, padding_idx);
        }
    }
    TORCH_CHECK(false, "kunlun embedding supports float16/float32 weight and int32/int64 indices");
}

at::Tensor cat(at::ITensorListRef tensors, int64_t dim) {
    std::vector<at::Tensor> list;
    for (const auto& t : tensors) {
        list.push_back(t);
    }
    TORCH_CHECK(!list.empty(), "cat got an empty tensor list");
    if (list.front().scalar_type() == at::kHalf) return cat_impl<float16>(list, dim);
    if (list.front().scalar_type() == at::kFloat) return cat_impl<float>(list, dim);
    TORCH_CHECK(false, "kunlun cat only supports float16/float32");
}

at::Tensor bmm(const at::Tensor& self, const at::Tensor& mat2) {
    if (self.scalar_type() == at::kHalf) return bmm_impl<float16, short>(self, mat2);
    if (self.scalar_type() == at::kFloat) return bmm_impl<float, float>(self, mat2);
    TORCH_CHECK(false, "kunlun bmm only supports float16/float32");
}

std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    if (self.scalar_type() == at::kHalf) {
        return max_pool2d_impl<float16>(self, kernel_size, stride, padding, dilation, ceil_mode);
    }
    if (self.scalar_type() == at::kFloat) {
        return max_pool2d_impl<float>(self, kernel_size, stride, padding, dilation, ceil_mode);
    }
    TORCH_CHECK(false, "kunlun max_pool2d only supports float16/float32");
}

at::Tensor adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size) {
    if (self.scalar_type() == at::kHalf) {
        return adaptive_avg_pool2d_impl<float16>(self, output_size);
    }
    if (self.scalar_type() == at::kFloat) {
        return adaptive_avg_pool2d_impl<float>(self, output_size);
    }
    TORCH_CHECK(false, "kunlun adaptive_avg_pool2d only supports float16/float32");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var, bool training, double, double eps) {
    TORCH_CHECK(!training, "kunlun native_batch_norm training=True is not implemented "
                           "in the basic runtime; use eval()/inference");
    if (input.scalar_type() == at::kHalf) {
        return batch_norm_infer_impl<float16>(input, weight, bias, running_mean, running_var, eps);
    }
    if (input.scalar_type() == at::kFloat) {
        return batch_norm_infer_impl<float>(input, weight, bias, running_mean, running_var, eps);
    }
    TORCH_CHECK(false, "kunlun native_batch_norm only supports float16/float32");
}

at::Tensor dropout(const at::Tensor& input, double, bool train) {
    TORCH_CHECK(!train, "kunlun dropout training=True is not implemented "
                        "(no XPU RNG in the basic runtime)");
    return input;
}

}  // namespace
}  // namespace torch_kunlun

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("softmax.int", TORCH_FN(torch_kunlun::softmax));
    m.impl("_softmax", TORCH_FN(torch_kunlun::_softmax));
    m.impl("layer_norm", TORCH_FN(torch_kunlun::layer_norm));
    m.impl("gelu", TORCH_FN(torch_kunlun::gelu));
    m.impl("embedding", TORCH_FN(torch_kunlun::embedding));
    m.impl("cat", TORCH_FN(torch_kunlun::cat));
    m.impl("bmm", TORCH_FN(torch_kunlun::bmm));
    m.impl("max_pool2d_with_indices", TORCH_FN(torch_kunlun::max_pool2d_with_indices));
    m.impl("adaptive_avg_pool2d", TORCH_FN(torch_kunlun::adaptive_avg_pool2d));
    m.impl("native_batch_norm", TORCH_FN(torch_kunlun::native_batch_norm));
    m.impl("dropout", TORCH_FN(torch_kunlun::dropout));
}
