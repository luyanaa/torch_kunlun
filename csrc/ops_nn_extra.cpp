#include "kunlun.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include "xpu/refactor/math.h"
#include "xpu/refactor/nn.h"
#include "xpu/refactor/nn_pool.h"

#include <optional>
#include <vector>

namespace torch_kunlun {
namespace {

using baidu::xpu::api::Context;

struct ContextGuard {
    DeviceContext& dc;
    std::unique_lock<std::mutex> lock;
    explicit ContextGuard(int device) : dc(get_context(device)), lock(dc.mu) {
        xpu_set_device(dc.device);
        dc.ctx->set_stream(current_stream_native(dc.device));
    }
};

void check_tensor(const at::Tensor& t, const char* what) {
    TORCH_CHECK(t.device().type() == at::DeviceType::PrivateUse1,
                what, " requires a kunlun tensor");
    TORCH_CHECK(t.is_contiguous(), what, " requires a contiguous tensor");
}

template <typename T> const T* cptr(const at::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <typename T> T* mptr(at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

std::vector<int64_t> triple(at::IntArrayRef value, int64_t fallback = 0) {
    if (value.empty()) return {fallback, fallback, fallback};
    if (value.size() == 1) return {value[0], value[0], value[0]};
    TORCH_CHECK(value.size() == 3, "expected 1 or 3 element array");
    return {value[0], value[1], value[2]};
}

at::Tensor to_float(const at::Tensor& t, const char* what) {
    check_tensor(t, what);
    if (t.scalar_type() == at::kFloat) return t;
    TORCH_CHECK(t.scalar_type() == at::kHalf, what, " must be float16/float32");
    auto out = at::empty(t.sizes(), t.options().dtype(at::kFloat));
    ContextGuard guard(t.device().index());
    const int ret = baidu::xpu::api::cast<float16, float>(guard.dc.ctx, cptr<float16>(t),
                                                          mptr<float>(out), t.numel());
    TORCH_CHECK(ret == 0, what, " cast failed");
    return out;
}

template <typename T, typename TW>
at::Tensor group_norm_impl(const at::Tensor& input, const std::optional<at::Tensor>& weight,
                           const std::optional<at::Tensor>& bias, int64_t groups, double eps) {
    check_tensor(input, "group_norm");
    TORCH_CHECK(input.dim() == 4, "group_norm requires NCHW");
    const int64_t n = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
    TORCH_CHECK(c % groups == 0, "group_norm channels not divisible by groups");
    auto gamma = weight.has_value() ? to_float(*weight, "group_norm weight")
                                    : at::ones({c}, input.options().dtype(at::kFloat));
    auto beta = bias.has_value() ? to_float(*bias, "group_norm bias")
                                 : at::zeros({c}, input.options().dtype(at::kFloat));
    auto out = at::empty_like(input);
    auto mean = at::empty({n * groups}, input.options().dtype(at::kFloat));
    auto var = at::empty({n * groups}, input.options().dtype(at::kFloat));
    ContextGuard guard(input.device().index());
    const int ret = baidu::xpu::api::group_norm<T>(
        guard.dc.ctx, cptr<T>(input), mptr<T>(out), n, c, h, w, groups,
        static_cast<float>(eps), cptr<float>(gamma), cptr<float>(beta),
        mptr<float>(mean), mptr<float>(var), true);
    TORCH_CHECK(ret == 0, "xdnn::group_norm failed: ret=", ret);
    return out;
}


at::Tensor conv1d(const at::Tensor& input, const at::Tensor& weight,
                  const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                  at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    check_tensor(input, "conv1d");
    check_tensor(weight, "conv1d");
    TORCH_CHECK(input.dim() == 3 && weight.dim() == 3, "conv1d requires NCL/OCL");
    const int64_t s = stride.empty() ? 1 : stride[0];
    const int64_t p = padding.empty() ? 0 : padding[0];
    const int64_t d = dilation.empty() ? 1 : dilation[0];
    auto in2 = input.unsqueeze(2).contiguous();
    auto w2 = weight.unsqueeze(2).contiguous();
    auto out2 = at::conv2d(in2, w2, bias,
                           std::vector<int64_t>{1, s},
                           std::vector<int64_t>{0, p},
                           std::vector<int64_t>{1, d}, groups);
    return out2.squeeze(2);
}

at::Tensor conv3d(const at::Tensor& input, const at::Tensor& weight,
                  const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                  at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    check_tensor(input, "conv3d");
    check_tensor(weight, "conv3d");
    TORCH_CHECK(input.dim() == 5 && weight.dim() == 5, "conv3d requires NCDHW/OCDHW");
    auto s = triple(stride, 1);
    auto p = triple(padding);
    auto d = triple(dilation, 1);
    const int64_t n = input.size(0), c = input.size(1), id = input.size(2),
                  ih = input.size(3), iw = input.size(4);
    const int64_t f = weight.size(0), kd = weight.size(2), kh = weight.size(3), kw = weight.size(4);
    const int64_t od = (id + 2 * p[0] - d[0] * (kd - 1) - 1) / s[0] + 1;
    const int64_t oh = (ih + 2 * p[1] - d[1] * (kh - 1) - 1) / s[1] + 1;
    const int64_t ow = (iw + 2 * p[2] - d[2] * (kw - 1) - 1) / s[2] + 1;
    TORCH_CHECK(od > 0 && oh > 0 && ow > 0, "conv3d output shape non-positive");
    auto out = at::empty({n, f, od, oh, ow}, input.options());
    ContextGuard guard(input.device().index());
    int ret = 0;
    if (input.scalar_type() == at::kHalf) {
        ret = baidu::xpu::api::conv3d<float16, float16, float16, short>(
            guard.dc.ctx, cptr<float16>(input), cptr<float16>(weight), mptr<float16>(out),
            n, c, id, ih, iw, f, {kd, kh, kw}, s, p, d, groups,
            nullptr, nullptr, nullptr, true);
    } else {
        ret = baidu::xpu::api::conv3d<float, float, float, float>(
            guard.dc.ctx, cptr<float>(input), cptr<float>(weight), mptr<float>(out),
            n, c, id, ih, iw, f, {kd, kh, kw}, s, p, d, groups,
            nullptr, nullptr, nullptr, true);
    }
    TORCH_CHECK(ret == 0, "xdnn::conv3d failed: ret=", ret);
    if (bias.has_value()) out = at::add(out, *bias);
    return out;
}

at::Tensor conv_transpose2d(const at::Tensor& input, const at::Tensor& weight,
                            const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                            at::IntArrayRef padding, at::IntArrayRef output_padding,
                            int64_t groups, at::IntArrayRef dilation) {
    check_tensor(input, "conv_transpose2d");
    check_tensor(weight, "conv_transpose2d");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "conv_transpose2d requires 4D");
    TORCH_CHECK(output_padding.empty() ||
                (output_padding[0] == 0 && (output_padding.size() < 2 || output_padding[1] == 0)),
                "kunlun conv_transpose2d only supports output_padding=0");
    const auto s = output_padding.empty() ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{1, 1};
    (void)s;
    auto st = std::vector<int64_t>{stride.empty() ? 1 : stride[0], stride.size() < 2 ? (stride.empty() ? 1 : stride[0]) : stride[1]};
    auto pd = std::vector<int64_t>{padding.empty() ? 0 : padding[0], padding.size() < 2 ? (padding.empty() ? 0 : padding[0]) : padding[1]};
    auto dl = std::vector<int64_t>{dilation.empty() ? 1 : dilation[0], dilation.size() < 2 ? (dilation.empty() ? 1 : dilation[0]) : dilation[1]};
    const int64_t n = input.size(0), yc = input.size(1), yh = input.size(2), yw = input.size(3);
    const int64_t xc = weight.size(1) * groups;
    const int64_t kh = weight.size(2), kw = weight.size(3);
    const int64_t xh = (yh - 1) * st[0] - 2 * pd[0] + dl[0] * (kh - 1) + 1;
    const int64_t xw = (yw - 1) * st[1] - 2 * pd[1] + dl[1] * (kw - 1) + 1;
    TORCH_CHECK(xh > 0 && xw > 0, "conv_transpose2d output shape non-positive");
    auto out = at::empty({n, xc, xh, xw}, input.options());
    ContextGuard guard(input.device().index());
    int ret = 0;
    if (input.scalar_type() == at::kHalf) {
        ret = baidu::xpu::api::conv2d_transpose<float16, float16, float16, short>(
            guard.dc.ctx, cptr<float16>(input), cptr<float16>(weight), mptr<float16>(out),
            n, yc, yh, yw, xc, {kh, kw}, st, pd, dl, groups, nullptr, nullptr, nullptr, true);
    } else {
        ret = baidu::xpu::api::conv2d_transpose<float, float, float, float>(
            guard.dc.ctx, cptr<float>(input), cptr<float>(weight), mptr<float>(out),
            n, yc, yh, yw, xc, {kh, kw}, st, pd, dl, groups, nullptr, nullptr, nullptr, true);
    }
    TORCH_CHECK(ret == 0, "xdnn::conv2d_transpose failed: ret=", ret);
    if (bias.has_value()) out = at::add(out, *bias);
    return out;
}
at::Tensor group_norm(const at::Tensor& input, int64_t groups,
                      const std::optional<at::Tensor>& weight,
                      const std::optional<at::Tensor>& bias, double eps, bool) {
    if (input.scalar_type() == at::kHalf) return group_norm_impl<float16, float>(input, weight, bias, groups, eps);
    if (input.scalar_type() == at::kFloat) return group_norm_impl<float, float>(input, weight, bias, groups, eps);
    TORCH_CHECK(false, "kunlun group_norm only supports float16/float32");
}

template <typename T>
at::Tensor instance_norm_impl(const at::Tensor& input, const std::optional<at::Tensor>& weight,
                              const std::optional<at::Tensor>& bias, double eps) {
    check_tensor(input, "instance_norm");
    TORCH_CHECK(input.dim() == 4, "instance_norm requires NCHW");
    const int64_t n = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
    auto gamma = weight.has_value() ? to_float(*weight, "instance_norm weight")
                                    : at::ones({c}, input.options().dtype(at::kFloat));
    auto beta = bias.has_value() ? to_float(*bias, "instance_norm bias")
                                 : at::zeros({c}, input.options().dtype(at::kFloat));
    auto out = at::empty_like(input);
    auto mean = at::empty({n * c}, input.options().dtype(at::kFloat));
    auto var = at::empty({n * c}, input.options().dtype(at::kFloat));
    ContextGuard guard(input.device().index());
    const int ret = baidu::xpu::api::instance_norm<T>(
        guard.dc.ctx, cptr<T>(input), mptr<T>(out), n, c, h, w, static_cast<float>(eps),
        cptr<float>(gamma), cptr<float>(beta), mptr<float>(mean), mptr<float>(var), true);
    TORCH_CHECK(ret == 0, "xdnn::instance_norm failed: ret=", ret);
    return out;
}

at::Tensor instance_norm(const at::Tensor& input, const std::optional<at::Tensor>& weight,
                         const std::optional<at::Tensor>& bias,
                         const std::optional<at::Tensor>& running_mean,
                         const std::optional<at::Tensor>& running_var, bool use_input_stats,
                         double momentum, double eps, bool) {
    TORCH_CHECK(use_input_stats, "kunlun instance_norm only supports use_input_stats=True");
    if (input.scalar_type() == at::kHalf) return instance_norm_impl<float16>(input, weight, bias, eps);
    if (input.scalar_type() == at::kFloat) return instance_norm_impl<float>(input, weight, bias, eps);
    TORCH_CHECK(false, "kunlun instance_norm only supports float16/float32");
}

template <typename T>
std::tuple<at::Tensor, at::Tensor> adaptive_max_pool2d_impl(const at::Tensor& self,
                                                            at::IntArrayRef output_size) {
    check_tensor(self, "adaptive_max_pool2d");
    TORCH_CHECK(self.dim() == 4, "adaptive_max_pool2d requires NCHW");
    const int64_t n = self.size(0), c = self.size(1), h = self.size(2), w = self.size(3);
    const int64_t oh = output_size[0];
    const int64_t ow = output_size.size() > 1 ? output_size[1] : output_size[0];
    auto out = at::empty({n, c, oh, ow}, self.options());
    auto indices = at::empty({n, c, oh, ow}, self.options().dtype(at::kInt));
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::adaptive_max_pool2d<T, int>(
        guard.dc.ctx, cptr<T>(self), mptr<T>(out), mptr<int>(indices),
        n, c, h, w, oh, ow, true);
    TORCH_CHECK(ret == 0, "xdnn::adaptive_max_pool2d failed: ret=", ret);
    return {out, indices};
}

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool2d(const at::Tensor& self,
                                                       at::IntArrayRef output_size) {
    if (self.scalar_type() == at::kHalf) return adaptive_max_pool2d_impl<float16>(self, output_size);
    if (self.scalar_type() == at::kFloat) return adaptive_max_pool2d_impl<float>(self, output_size);
    TORCH_CHECK(false, "kunlun adaptive_max_pool2d only supports float16/float32");
}

template <typename T>
at::Tensor avg_pool3d_impl(const at::Tensor& self, at::IntArrayRef kernel,
                           at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                           bool count_include_pad, std::optional<int64_t> divisor_override) {
    check_tensor(self, "avg_pool3d");
    TORCH_CHECK(self.dim() == 5, "avg_pool3d requires NCDHW");
    TORCH_CHECK(!ceil_mode, "kunlun avg_pool3d does not support ceil_mode=True");
    TORCH_CHECK(!divisor_override.has_value(), "kunlun avg_pool3d divisor_override unsupported");
    auto k = triple(kernel);
    auto s = stride.empty() ? k : triple(stride);
    auto p = triple(padding);
    const int64_t n = self.size(0), c = self.size(1), d = self.size(2), h = self.size(3), w = self.size(4);
    const int64_t od = (d + 2 * p[0] - k[0]) / s[0] + 1;
    const int64_t oh = (h + 2 * p[1] - k[1]) / s[1] + 1;
    const int64_t ow = (w + 2 * p[2] - k[2]) / s[2] + 1;
    TORCH_CHECK(od > 0 && oh > 0 && ow > 0, "avg_pool3d output shape non-positive");
    auto out = at::empty({n, c, od, oh, ow}, self.options());
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::avg_pool3d<T>(
        guard.dc.ctx, cptr<T>(self), mptr<T>(out), n, c, d, h, w, k, s, p,
        count_include_pad, true);
    TORCH_CHECK(ret == 0, "xdnn::avg_pool3d failed: ret=", ret);
    return out;
}


template <typename T, typename TW>
at::Tensor rms_norm_impl(const at::Tensor& input, int64_t m, int64_t n, double eps,
                         const at::Tensor& weight) {
    auto out = at::empty_like(input);
    auto var = at::empty({m}, input.options().dtype(at::kFloat));
    auto bias = at::zeros({n}, weight.options());
    ContextGuard guard(input.device().index());
    const int ret = baidu::xpu::api::rms_layer_norm<T, TW>(
        guard.dc.ctx, cptr<T>(input), mptr<T>(out), m, n, static_cast<float>(eps),
        cptr<TW>(weight), cptr<TW>(bias), mptr<float>(var), false);
    TORCH_CHECK(ret == 0, "xdnn::rms_layer_norm failed: ret=", ret);
    return out;
}

at::Tensor rms_norm(const at::Tensor& input, c10::SymIntArrayRef normalized_shape,
                    const std::optional<at::Tensor>& weight, std::optional<double> eps) {
    int64_t n = 1;
    for (auto s : normalized_shape) n *= s.expect_int();
    TORCH_CHECK(n > 0 && input.numel() % n == 0, "rms_norm shape mismatch");
    const int64_t m = input.numel() / n;
    auto gamma = weight.has_value() ? to_float(*weight, "rms_norm weight")
                                    : at::ones({n}, input.options().dtype(at::kFloat));
    const double e = eps.value_or(1e-5);
    if (input.scalar_type() == at::kHalf) {
        if (gamma.scalar_type() == at::kHalf) return rms_norm_impl<float16, float16>(input, m, n, e, gamma);
        return rms_norm_impl<float16, float>(input, m, n, e, gamma);
    }
    return rms_norm_impl<float, float>(input, m, n, e, gamma);
}
at::Tensor avg_pool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                      at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                      bool count_include_pad, std::optional<int64_t> divisor_override) {
    if (self.scalar_type() == at::kHalf) {
        return avg_pool3d_impl<float16>(self, kernel_size, stride, padding, ceil_mode,
                                        count_include_pad, divisor_override);
    }
    if (self.scalar_type() == at::kFloat) {
        return avg_pool3d_impl<float>(self, kernel_size, stride, padding, ceil_mode,
                                      count_include_pad, divisor_override);
    }
    TORCH_CHECK(false, "kunlun avg_pool3d only supports float16/float32");
}

}  // namespace
}  // namespace torch_kunlun

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("group_norm", TORCH_FN(torch_kunlun::group_norm));
    m.impl("instance_norm", TORCH_FN(torch_kunlun::instance_norm));
    m.impl("adaptive_max_pool2d", TORCH_FN(torch_kunlun::adaptive_max_pool2d));
    m.impl("avg_pool3d", TORCH_FN(torch_kunlun::avg_pool3d));
    m.impl("rms_norm", TORCH_FN(torch_kunlun::rms_norm));
    m.impl("conv1d", TORCH_FN(torch_kunlun::conv1d));
    m.impl("conv3d", TORCH_FN(torch_kunlun::conv3d));
    m.impl("conv_transpose2d", TORCH_FN(torch_kunlun::conv_transpose2d));
}
