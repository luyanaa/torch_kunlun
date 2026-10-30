#include "kunlun.h"

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "xpu/refactor/math.h"
#include "xpu/refactor/nn.h"
#include "xpu/refactor/nn_pool.h"

#include <optional>
#include <vector>

namespace torch_kunlun {
namespace {

using baidu::xpu::api::Context;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;

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
    const int ret = baidu::xpu::api::cast<float16, float>(
        guard.dc.ctx, cptr<float16>(t), mptr<float>(out), t.numel());
    TORCH_CHECK(ret == 0, what, " cast failed");
    return out;
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> group_forward(
    const at::Tensor& input, const at::Tensor& gamma, const at::Tensor& beta,
    int64_t groups, double eps) {
    const int64_t n = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
    auto out = at::empty_like(input);
    auto mean = at::empty({n * groups}, input.options().dtype(at::kFloat));
    auto var = at::empty({n * groups}, input.options().dtype(at::kFloat));
    ContextGuard guard(input.device().index());
    const int ret = baidu::xpu::api::group_norm<T>(
        guard.dc.ctx, cptr<T>(input), mptr<T>(out), n, c, h, w, groups,
        static_cast<float>(eps), cptr<float>(gamma), cptr<float>(beta),
        mptr<float>(mean), mptr<float>(var), true);
    TORCH_CHECK(ret == 0, "group_norm forward failed");
    return {out, mean, var};
}

class GroupNormFunction : public Function<GroupNormFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input, int64_t groups,
                              std::optional<at::Tensor> weight, std::optional<at::Tensor> bias,
                              double eps, bool cudnn) {
        const int64_t c = input.size(1);
        auto gamma = weight.has_value() ? to_float(*weight, "group_norm weight")
                                        : at::ones({c}, input.options().dtype(at::kFloat));
        auto beta = bias.has_value() ? to_float(*bias, "group_norm bias")
                                     : at::zeros({c}, input.options().dtype(at::kFloat));
        at::Tensor out, mean, var;
        if (input.scalar_type() == at::kHalf) {
            std::tie(out, mean, var) = group_forward<float16>(input, gamma, beta, groups, eps);
        } else {
            std::tie(out, mean, var) = group_forward<float>(input, gamma, beta, groups, eps);
        }
        ctx->save_for_backward({input, gamma, beta, mean, var, out});
        ctx->saved_data["groups"] = groups;
        ctx->saved_data["eps"] = eps;
        ctx->saved_data["has_weight"] = weight.has_value();
        ctx->saved_data["has_bias"] = bias.has_value();
        ctx->saved_data["weight_half"] = weight.has_value() && weight->scalar_type() == at::kHalf;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0]; auto gamma = saved[1]; auto beta = saved[2];
        auto mean = saved[3]; auto var = saved[4]; auto out = saved[5];
        auto grad = grad_outputs[0].contiguous();
        const int64_t groups = ctx->saved_data["groups"].toInt();
        const double eps = ctx->saved_data["eps"].toDouble();
        const int64_t n = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
        auto dx = at::empty_like(input);
        auto dgamma = at::empty({c}, input.options().dtype(at::kFloat));
        auto dbeta = at::empty({c}, input.options().dtype(at::kFloat));
        ContextGuard guard(input.device().index());
        int ret = 0;
        if (input.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::group_norm_grad<float16>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(out), cptr<float16>(grad), mptr<float16>(dx),
                n, c, h, w, groups, static_cast<float>(eps), cptr<float>(gamma),
                cptr<float>(beta), cptr<float>(mean), cptr<float>(var),
                mptr<float>(dgamma), mptr<float>(dbeta), true, false);
        } else {
            ret = baidu::xpu::api::group_norm_grad<float>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(out), cptr<float>(grad), mptr<float>(dx),
                n, c, h, w, groups, static_cast<float>(eps), cptr<float>(gamma),
                cptr<float>(beta), cptr<float>(mean), cptr<float>(var),
                mptr<float>(dgamma), mptr<float>(dbeta), true, false);
        }
        TORCH_CHECK(ret == 0, "group_norm backward failed");
        at::Tensor dweight, dbias_out;
        if (ctx->saved_data["has_weight"].toBool()) {
            if (ctx->saved_data["weight_half"].toBool()) {
                dweight = at::empty({c}, input.options());
                int r = baidu::xpu::api::cast<float, float16>(guard.dc.ctx, cptr<float>(dgamma),
                                                               mptr<float16>(dweight), c);
                TORCH_CHECK(r == 0, "group_norm dweight cast failed");
            } else dweight = dgamma;
        }
        if (ctx->saved_data["has_bias"].toBool()) {
            if (ctx->saved_data["weight_half"].toBool()) {
                dbias_out = at::empty({c}, input.options());
                int r = baidu::xpu::api::cast<float, float16>(guard.dc.ctx, cptr<float>(dbeta),
                                                               mptr<float16>(dbias_out), c);
                TORCH_CHECK(r == 0, "group_norm dbias cast failed");
            } else dbias_out = dbeta;
        }
        return {dx, at::Tensor(), dweight, dbias_out, at::Tensor(), at::Tensor()};
    }
};

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_forward(
    const at::Tensor& input, const at::Tensor& gamma, const at::Tensor& beta, double eps) {
    const int64_t n = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
    auto out = at::empty_like(input);
    auto mean = at::empty({n * c}, input.options().dtype(at::kFloat));
    auto var = at::empty({n * c}, input.options().dtype(at::kFloat));
    ContextGuard guard(input.device().index());
    const int ret = baidu::xpu::api::instance_norm<T>(
        guard.dc.ctx, cptr<T>(input), mptr<T>(out), n, c, h, w, static_cast<float>(eps),
        cptr<float>(gamma), cptr<float>(beta), mptr<float>(mean), mptr<float>(var), true);
    TORCH_CHECK(ret == 0, "instance_norm forward failed");
    return {out, mean, var};
}

class InstanceNormFunction : public Function<InstanceNormFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input,
                              std::optional<at::Tensor> weight, std::optional<at::Tensor> bias,
                              std::optional<at::Tensor> running_mean,
                              std::optional<at::Tensor> running_var, bool use_input_stats,
                              double momentum, double eps, bool cudnn) {
        TORCH_CHECK(use_input_stats, "kunlun instance_norm autograd requires use_input_stats=True");
        const int64_t c = input.size(1);
        auto gamma = weight.has_value() ? to_float(*weight, "instance_norm weight")
                                        : at::ones({c}, input.options().dtype(at::kFloat));
        auto beta = bias.has_value() ? to_float(*bias, "instance_norm bias")
                                     : at::zeros({c}, input.options().dtype(at::kFloat));
        at::Tensor out, mean, var;
        if (input.scalar_type() == at::kHalf) {
            std::tie(out, mean, var) = instance_forward<float16>(input, gamma, beta, eps);
        } else {
            std::tie(out, mean, var) = instance_forward<float>(input, gamma, beta, eps);
        }
        ctx->save_for_backward({input, gamma, beta, mean, var, out});
        ctx->saved_data["eps"] = eps;
        ctx->saved_data["has_weight"] = weight.has_value();
        ctx->saved_data["has_bias"] = bias.has_value();
        ctx->saved_data["weight_half"] = weight.has_value() && weight->scalar_type() == at::kHalf;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0]; auto gamma = saved[1]; auto beta = saved[2];
        auto mean = saved[3]; auto var = saved[4]; auto out = saved[5];
        auto grad = grad_outputs[0].contiguous();
        const double eps = ctx->saved_data["eps"].toDouble();
        const int64_t n = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
        auto dx = at::empty_like(input);
        auto dgamma = at::empty({c}, input.options().dtype(at::kFloat));
        auto dbeta = at::empty({c}, input.options().dtype(at::kFloat));
        ContextGuard guard(input.device().index());
        int ret = 0;
        if (input.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::instance_norm_grad<float16>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(grad), mptr<float16>(dx),
                cptr<float>(gamma), cptr<float>(mean), cptr<float>(var),
                mptr<float>(dgamma), mptr<float>(dbeta), n, c, h, w,
                static_cast<float>(eps), true);
        } else {
            ret = baidu::xpu::api::instance_norm_grad<float>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(grad), mptr<float>(dx),
                cptr<float>(gamma), cptr<float>(mean), cptr<float>(var),
                mptr<float>(dgamma), mptr<float>(dbeta), n, c, h, w,
                static_cast<float>(eps), true);
        }
        TORCH_CHECK(ret == 0, "instance_norm backward failed");
        at::Tensor dweight, dbias_out;
        if (ctx->saved_data["has_weight"].toBool()) {
            if (ctx->saved_data["weight_half"].toBool()) {
                dweight = at::empty({c}, input.options());
                int r = baidu::xpu::api::cast<float, float16>(guard.dc.ctx, cptr<float>(dgamma),
                                                               mptr<float16>(dweight), c);
                TORCH_CHECK(r == 0, "instance_norm dweight cast failed");
            } else dweight = dgamma;
        }
        if (ctx->saved_data["has_bias"].toBool()) {
            if (ctx->saved_data["weight_half"].toBool()) {
                dbias_out = at::empty({c}, input.options());
                int r = baidu::xpu::api::cast<float, float16>(guard.dc.ctx, cptr<float>(dbeta),
                                                               mptr<float16>(dbias_out), c);
                TORCH_CHECK(r == 0, "instance_norm dbias cast failed");
            } else dbias_out = dbeta;
        }
        return {dx, dweight, dbias_out, at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class AdaptiveMaxPool2dFunction : public Function<AdaptiveMaxPool2dFunction> {
public:
    static variable_list forward(AutogradContext* ctx, at::Tensor self,
                                 std::vector<int64_t> output_size) {
        const int64_t n = self.size(0), c = self.size(1), h = self.size(2), w = self.size(3);
        const int64_t oh = output_size[0];
        const int64_t ow = output_size.size() > 1 ? output_size[1] : output_size[0];
        auto out = at::empty({n, c, oh, ow}, self.options());
        auto indices = at::empty({n, c, oh, ow}, self.options().dtype(at::kInt));
        ContextGuard guard(self.device().index());
        int ret = 0;
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::adaptive_max_pool2d<float16, int>(
                guard.dc.ctx, cptr<float16>(self), mptr<float16>(out), mptr<int>(indices),
                n, c, h, w, oh, ow, true);
        } else {
            ret = baidu::xpu::api::adaptive_max_pool2d<float, int>(
                guard.dc.ctx, cptr<float>(self), mptr<float>(out), mptr<int>(indices),
                n, c, h, w, oh, ow, true);
        }
        TORCH_CHECK(ret == 0, "adaptive_max_pool2d forward failed");
        ctx->save_for_backward({self, out, indices});
        ctx->saved_data["oh"] = oh;
        ctx->saved_data["ow"] = ow;
        return {out, indices};
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto self = saved[0]; auto out = saved[1]; auto indices = saved[2];
        auto grad = grad_outputs[0].contiguous();
        const int64_t oh = ctx->saved_data["oh"].toInt();
        const int64_t ow = ctx->saved_data["ow"].toInt();
        auto dx = at::empty_like(self);
        ContextGuard guard(self.device().index());
        int ret = 0;
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::adaptive_max_pool2d_grad<float16, int>(
                guard.dc.ctx, cptr<float16>(self), cptr<float16>(out), cptr<int>(indices),
                cptr<float16>(grad), mptr<float16>(dx), self.size(0), self.size(1),
                self.size(2), self.size(3), oh, ow, true);
        } else {
            ret = baidu::xpu::api::adaptive_max_pool2d_grad<float, int>(
                guard.dc.ctx, cptr<float>(self), cptr<float>(out), cptr<int>(indices),
                cptr<float>(grad), mptr<float>(dx), self.size(0), self.size(1),
                self.size(2), self.size(3), oh, ow, true);
        }
        TORCH_CHECK(ret == 0, "adaptive_max_pool2d backward failed");
        return {dx, at::Tensor()};
    }
};

class AvgPool3dFunction : public Function<AvgPool3dFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self,
                              std::vector<int64_t> kernel, std::vector<int64_t> stride,
                              std::vector<int64_t> padding, bool ceil_mode,
                              bool count_include_pad, std::optional<int64_t> divisor_override) {
        TORCH_CHECK(!ceil_mode && !divisor_override.has_value(),
                    "kunlun avg_pool3d autograd does not support ceil/divisor_override");
        auto k = kernel; auto s = stride.empty() ? k : stride; auto p = padding;
        const int64_t n = self.size(0), c = self.size(1), d = self.size(2), h = self.size(3), w = self.size(4);
        const int64_t od = (d + 2 * p[0] - k[0]) / s[0] + 1;
        const int64_t oh = (h + 2 * p[1] - k[1]) / s[1] + 1;
        const int64_t ow = (w + 2 * p[2] - k[2]) / s[2] + 1;
        auto out = at::empty({n, c, od, oh, ow}, self.options());
        ContextGuard guard(self.device().index());
        int ret = 0;
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::avg_pool3d<float16>(guard.dc.ctx, cptr<float16>(self),
                mptr<float16>(out), n, c, d, h, w, k, s, p, count_include_pad, true);
        } else {
            ret = baidu::xpu::api::avg_pool3d<float>(guard.dc.ctx, cptr<float>(self),
                mptr<float>(out), n, c, d, h, w, k, s, p, count_include_pad, true);
        }
        TORCH_CHECK(ret == 0, "avg_pool3d forward failed");
        ctx->save_for_backward({self});
        ctx->saved_data["kernel"] = k;
        ctx->saved_data["stride"] = s;
        ctx->saved_data["padding"] = p;
        ctx->saved_data["cip"] = count_include_pad;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto self = ctx->get_saved_variables()[0];
        auto grad = grad_outputs[0].contiguous();
        auto k = ctx->saved_data["kernel"].toIntVector();
        auto s = ctx->saved_data["stride"].toIntVector();
        auto p = ctx->saved_data["padding"].toIntVector();
        const bool cip = ctx->saved_data["cip"].toBool();
        auto dx = at::empty_like(self);
        ContextGuard guard(self.device().index());
        int ret = 0;
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::avg_pool3d_grad<float16>(
                guard.dc.ctx, cptr<float16>(grad), mptr<float16>(dx), self.size(0),
                self.size(1), self.size(2), self.size(3), self.size(4), k, s, p, cip, true);
        } else {
            ret = baidu::xpu::api::avg_pool3d_grad<float>(
                guard.dc.ctx, cptr<float>(grad), mptr<float>(dx), self.size(0),
                self.size(1), self.size(2), self.size(3), self.size(4), k, s, p, cip, true);
        }
        TORCH_CHECK(ret == 0, "avg_pool3d backward failed");
        return {dx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor()};
    }
};


template <typename T, typename TW>
std::tuple<at::Tensor, at::Tensor> rms_forward(const at::Tensor& input, int64_t m,
                                               int64_t n, double eps,
                                               const at::Tensor& gamma) {
    auto out = at::empty_like(input);
    auto var = at::empty({m}, input.options().dtype(at::kFloat));
    auto beta = at::zeros({n}, gamma.options());
    ContextGuard guard(input.device().index());
    const int ret = baidu::xpu::api::rms_layer_norm<T, TW>(
        guard.dc.ctx, cptr<T>(input), mptr<T>(out), m, n, static_cast<float>(eps),
        cptr<TW>(gamma), cptr<TW>(beta), mptr<float>(var), false);
    TORCH_CHECK(ret == 0, "rms_norm forward failed");
    return {out, var};
}

class RmsNormFunction : public Function<RmsNormFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input,
                              std::vector<int64_t> normalized_shape,
                              std::optional<at::Tensor> weight, std::optional<double> eps) {
        int64_t n = 1;
        for (auto s : normalized_shape) n *= s;
        TORCH_CHECK(n > 0 && input.numel() % n == 0, "rms_norm shape mismatch");
        const int64_t m = input.numel() / n;
        auto gamma = weight.has_value() ? to_float(*weight, "rms_norm weight")
                                        : at::ones({n}, input.options().dtype(at::kFloat));
        const double e = eps.value_or(1e-5);
        at::Tensor out, var;
        if (input.scalar_type() == at::kHalf) {
            if (gamma.scalar_type() == at::kHalf) std::tie(out, var) = rms_forward<float16, float16>(input, m, n, e, gamma);
            else std::tie(out, var) = rms_forward<float16, float>(input, m, n, e, gamma);
        } else {
            std::tie(out, var) = rms_forward<float, float>(input, m, n, e, gamma);
        }
        ctx->save_for_backward({input, gamma, var});
        ctx->saved_data["m"] = m;
        ctx->saved_data["n"] = n;
        ctx->saved_data["eps"] = e;
        ctx->saved_data["has_weight"] = weight.has_value();
        ctx->saved_data["weight_half"] = weight.has_value() && weight->scalar_type() == at::kHalf;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0]; auto gamma = saved[1]; auto var = saved[2];
        auto grad = grad_outputs[0].contiguous();
        const int64_t m = ctx->saved_data["m"].toInt();
        const int64_t n = ctx->saved_data["n"].toInt();
        const double eps = ctx->saved_data["eps"].toDouble();
        auto dx = at::empty_like(input);
        auto dgamma = at::empty({n}, input.options().dtype(at::kFloat));
        auto dbeta = at::empty({n}, input.options().dtype(at::kFloat));
        ContextGuard guard(input.device().index());
        int ret = 0;
        if (input.scalar_type() == at::kHalf && gamma.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::rms_layer_norm_grad<float16, float16>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(grad), mptr<float16>(dx),
                m, n, static_cast<float>(eps), cptr<float16>(gamma), cptr<float>(var),
                mptr<float16>(dgamma), mptr<float16>(dbeta));
        } else if (input.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::rms_layer_norm_grad<float16, float>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(grad), mptr<float16>(dx),
                m, n, static_cast<float>(eps), cptr<float>(gamma), cptr<float>(var),
                mptr<float>(dgamma), mptr<float>(dbeta));
        } else {
            ret = baidu::xpu::api::rms_layer_norm_grad<float, float>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(grad), mptr<float>(dx),
                m, n, static_cast<float>(eps), cptr<float>(gamma), cptr<float>(var),
                mptr<float>(dgamma), mptr<float>(dbeta));
        }
        TORCH_CHECK(ret == 0, "rms_norm backward failed");
        at::Tensor dweight;
        if (ctx->saved_data["has_weight"].toBool()) {
            if (ctx->saved_data["weight_half"].toBool()) {
                dweight = at::empty({n}, input.options());
                int r = baidu::xpu::api::cast<float, float16>(guard.dc.ctx, cptr<float>(dgamma),
                                                               mptr<float16>(dweight), n);
                TORCH_CHECK(r == 0, "rms_norm dweight cast failed");
            } else dweight = dgamma;
        }
        return {dx, at::Tensor(), dweight, at::Tensor()};
    }
};

at::Tensor rms_norm_autograd(const at::Tensor& input, at::IntArrayRef normalized_shape,
                             const std::optional<at::Tensor>& weight,
                             std::optional<double> eps) {
    std::vector<int64_t> shape(normalized_shape.begin(), normalized_shape.end());
    return RmsNormFunction::apply(input, shape, weight, eps);
}


class Conv1dFunction : public Function<Conv1dFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input, at::Tensor weight,
                              std::optional<at::Tensor> bias, std::vector<int64_t> stride,
                              std::vector<int64_t> padding, std::vector<int64_t> dilation,
                              int64_t groups) {
        const int64_t s = stride.empty() ? 1 : stride[0];
        const int64_t p = padding.empty() ? 0 : padding[0];
        const int64_t d = dilation.empty() ? 1 : dilation[0];
        auto in2 = input.unsqueeze(2);
        auto w2 = weight.unsqueeze(2);
        auto out2 = at::conv2d(in2, w2, bias,
                               std::vector<int64_t>{1, s},
                               std::vector<int64_t>{0, p},
                               std::vector<int64_t>{1, d}, groups);
        auto out = out2.squeeze(2);
        ctx->save_for_backward({input, weight, bias.has_value() ? *bias : at::Tensor()});
        ctx->saved_data["s"] = s;
        ctx->saved_data["p"] = p;
        ctx->saved_data["d"] = d;
        ctx->saved_data["groups"] = groups;
        ctx->saved_data["has_bias"] = bias.has_value();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0]; auto weight = saved[1];
        auto grad = grad_outputs[0].unsqueeze(2).contiguous();
        const int64_t s = ctx->saved_data["s"].toInt();
        const int64_t p = ctx->saved_data["p"].toInt();
        const int64_t d = ctx->saved_data["d"].toInt();
        const int64_t groups = ctx->saved_data["groups"].toInt();
        auto in2 = input.unsqueeze(2);
        auto w2 = weight.unsqueeze(2);
        const int64_t n = input.size(0), c = input.size(1), w = input.size(2);
        const int64_t f = weight.size(0), k = weight.size(2);
        auto dx2 = at::empty_like(in2);
        auto dw2 = at::empty_like(w2);
        ContextGuard guard(input.device().index());
        int ret = 0;
        if (input.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::conv2d_grad<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(in2), cptr<float16>(w2), cptr<float16>(grad),
                mptr<float16>(dx2), mptr<float16>(dw2), n, c, 1, w, f, {1, k},
                {1, s}, {0, p}, {1, d}, groups, nullptr, nullptr, nullptr, nullptr,
                nullptr, true);
        } else {
            ret = baidu::xpu::api::conv2d_grad<float, float, float, float>(
                guard.dc.ctx, cptr<float>(in2), cptr<float>(w2), cptr<float>(grad),
                mptr<float>(dx2), mptr<float>(dw2), n, c, 1, w, f, {1, k},
                {1, s}, {0, p}, {1, d}, groups, nullptr, nullptr, nullptr, nullptr,
                nullptr, true);
        }
        TORCH_CHECK(ret == 0, "conv1d backward failed");
        auto dx = dx2.squeeze(2);
        auto dw = dw2.squeeze(2);
        at::Tensor gb;
        if (ctx->saved_data["has_bias"].toBool()) {
            gb = at::empty({f}, input.options());
            const std::vector<int64_t> dims{0, 2, 3};
            if (input.scalar_type() == at::kHalf) {
                int r = baidu::xpu::api::reduce_sum<float16>(guard.dc.ctx, cptr<float16>(grad),
                    mptr<float16>(gb), grad.sizes().vec(), dims);
                TORCH_CHECK(r == 0, "conv1d bias grad failed");
            } else {
                int r = baidu::xpu::api::reduce_sum<float>(guard.dc.ctx, cptr<float>(grad),
                    mptr<float>(gb), grad.sizes().vec(), dims);
                TORCH_CHECK(r == 0, "conv1d bias grad failed");
            }
        }
        return {dx, dw, gb, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor conv1d_autograd(const at::Tensor& input, const at::Tensor& weight,
                           const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    std::vector<int64_t> s(stride.begin(), stride.end());
    std::vector<int64_t> p(padding.begin(), padding.end());
    std::vector<int64_t> d(dilation.begin(), dilation.end());
    return Conv1dFunction::apply(input, weight, bias, s, p, d, groups);
}


class Conv3dFunction : public Function<Conv3dFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input, at::Tensor weight,
                              std::optional<at::Tensor> bias, std::vector<int64_t> stride,
                              std::vector<int64_t> padding, std::vector<int64_t> dilation,
                              int64_t groups) {
        auto out = at::conv3d(input, weight, bias, stride, padding, dilation, groups);
        ctx->save_for_backward({input, weight, bias.has_value() ? *bias : at::Tensor()});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["groups"] = groups;
        ctx->saved_data["has_bias"] = bias.has_value();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0]; auto weight = saved[1];
        auto grad = grad_outputs[0].contiguous();
        auto stride = ctx->saved_data["stride"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        auto dilation = ctx->saved_data["dilation"].toIntVector();
        const int64_t groups = ctx->saved_data["groups"].toInt();
        const int64_t n = input.size(0), c = input.size(1), d = input.size(2), h = input.size(3), w = input.size(4);
        const int64_t f = weight.size(0);
        auto dx = at::empty_like(input);
        auto dw = at::empty_like(weight);
        ContextGuard guard(input.device().index());
        int ret = 0;
        if (input.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::conv3d_grad<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(weight), cptr<float16>(grad),
                mptr<float16>(dx), mptr<float16>(dw), n, c, d, h, w, f,
                {weight.size(2), weight.size(3), weight.size(4)}, stride, padding, dilation,
                groups, nullptr, nullptr, nullptr, nullptr, nullptr, true);
        } else {
            ret = baidu::xpu::api::conv3d_grad<float, float, float, float>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(weight), cptr<float>(grad),
                mptr<float>(dx), mptr<float>(dw), n, c, d, h, w, f,
                {weight.size(2), weight.size(3), weight.size(4)}, stride, padding, dilation,
                groups, nullptr, nullptr, nullptr, nullptr, nullptr, true);
        }
        TORCH_CHECK(ret == 0, "conv3d backward failed");
        at::Tensor gb;
        if (ctx->saved_data["has_bias"].toBool()) {
            gb = at::empty({f}, input.options());
            const std::vector<int64_t> dims{0, 2, 3, 4};
            if (input.scalar_type() == at::kHalf) {
                ret = baidu::xpu::api::reduce_sum<float16>(guard.dc.ctx, cptr<float16>(grad),
                    mptr<float16>(gb), grad.sizes().vec(), dims);
            } else {
                ret = baidu::xpu::api::reduce_sum<float>(guard.dc.ctx, cptr<float>(grad),
                    mptr<float>(gb), grad.sizes().vec(), dims);
            }
            TORCH_CHECK(ret == 0, "conv3d bias grad failed");
        }
        return {dx, dw, gb, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor conv3d_autograd(const at::Tensor& input, const at::Tensor& weight,
                           const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    std::vector<int64_t> s(stride.begin(), stride.end());
    std::vector<int64_t> p(padding.begin(), padding.end());
    std::vector<int64_t> d(dilation.begin(), dilation.end());
    return Conv3dFunction::apply(input, weight, bias, s, p, d, groups);
}

class ConvTranspose2dFunction : public Function<ConvTranspose2dFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input, at::Tensor weight,
                              std::optional<at::Tensor> bias, std::vector<int64_t> stride,
                              std::vector<int64_t> padding, std::vector<int64_t> output_padding,
                              int64_t groups, std::vector<int64_t> dilation) {
        TORCH_CHECK(output_padding.empty() ||
                    (output_padding[0] == 0 && (output_padding.size() < 2 || output_padding[1] == 0)),
                    "kunlun conv_transpose2d only supports output_padding=0");
        auto out = at::conv_transpose2d(input, weight, bias, stride, padding,
                                        output_padding, groups, dilation);
        ctx->save_for_backward({input, weight, bias.has_value() ? *bias : at::Tensor()});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["groups"] = groups;
        ctx->saved_data["has_bias"] = bias.has_value();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0]; auto weight = saved[1];
        auto grad = grad_outputs[0].contiguous();
        auto stride = ctx->saved_data["stride"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        auto dilation = ctx->saved_data["dilation"].toIntVector();
        const int64_t groups = ctx->saved_data["groups"].toInt();
        const int64_t n = input.size(0), yc = input.size(1), yh = input.size(2), yw = input.size(3);
        const int64_t xc = weight.size(1) * groups;
        const int64_t xh = grad.size(2), xw = grad.size(3);
        auto dy = at::empty_like(input);
        auto dw = at::empty_like(weight);
        ContextGuard guard(input.device().index());
        int ret = 0;
        if (input.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::conv2d_transpose_grad<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(weight), cptr<float16>(grad),
                mptr<float16>(dy), mptr<float16>(dw), n, yc, yh, yw, xc, xh, xw,
                {weight.size(2), weight.size(3)}, stride, padding, dilation, groups,
                nullptr, nullptr, nullptr, nullptr, nullptr, true);
        } else {
            ret = baidu::xpu::api::conv2d_transpose_grad<float, float, float, float>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(weight), cptr<float>(grad),
                mptr<float>(dy), mptr<float>(dw), n, yc, yh, yw, xc, xh, xw,
                {weight.size(2), weight.size(3)}, stride, padding, dilation, groups,
                nullptr, nullptr, nullptr, nullptr, nullptr, true);
        }
        TORCH_CHECK(ret == 0, "conv_transpose2d backward failed");
        at::Tensor gb;
        if (ctx->saved_data["has_bias"].toBool()) {
            gb = at::empty({xc}, input.options());
            const std::vector<int64_t> dims{0, 2, 3};
            if (input.scalar_type() == at::kHalf) {
                ret = baidu::xpu::api::reduce_sum<float16>(guard.dc.ctx, cptr<float16>(grad),
                    mptr<float16>(gb), grad.sizes().vec(), dims);
            } else {
                ret = baidu::xpu::api::reduce_sum<float>(guard.dc.ctx, cptr<float>(grad),
                    mptr<float>(gb), grad.sizes().vec(), dims);
            }
            TORCH_CHECK(ret == 0, "conv_transpose2d bias grad failed");
        }
        return {dy, dw, gb, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor conv_transpose2d_autograd(const at::Tensor& input, const at::Tensor& weight,
                                     const std::optional<at::Tensor>& bias,
                                     at::IntArrayRef stride, at::IntArrayRef padding,
                                     at::IntArrayRef output_padding, int64_t groups,
                                     at::IntArrayRef dilation) {
    std::vector<int64_t> s(stride.begin(), stride.end());
    std::vector<int64_t> p(padding.begin(), padding.end());
    std::vector<int64_t> op(output_padding.begin(), output_padding.end());
    std::vector<int64_t> d(dilation.begin(), dilation.end());
    return ConvTranspose2dFunction::apply(input, weight, bias, s, p, op, groups, d);
}

at::Tensor group_norm_autograd(const at::Tensor& input, int64_t groups,
                               const std::optional<at::Tensor>& weight,
                               const std::optional<at::Tensor>& bias, double eps, bool cudnn) {
    return GroupNormFunction::apply(input, groups, weight, bias, eps, cudnn);
}
at::Tensor instance_norm_autograd(const at::Tensor& input,
                                  const std::optional<at::Tensor>& weight,
                                  const std::optional<at::Tensor>& bias,
                                  const std::optional<at::Tensor>& running_mean,
                                  const std::optional<at::Tensor>& running_var,
                                  bool use_input_stats, double momentum, double eps, bool cudnn) {
    return InstanceNormFunction::apply(input, weight, bias, running_mean, running_var,
                                       use_input_stats, momentum, eps, cudnn);
}
std::tuple<at::Tensor, at::Tensor> adaptive_max_pool2d_autograd(
    const at::Tensor& self, at::IntArrayRef output_size) {
    std::vector<int64_t> size(output_size.begin(), output_size.end());
    auto out = AdaptiveMaxPool2dFunction::apply(self, size);
    return {out[0], out[1]};
}
at::Tensor avg_pool3d_autograd(const at::Tensor& self, at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               bool ceil_mode, bool count_include_pad,
                               std::optional<int64_t> divisor_override) {
    std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
    std::vector<int64_t> s(stride.begin(), stride.end());
    std::vector<int64_t> p(padding.begin(), padding.end());
    return AvgPool3dFunction::apply(self, k, s, p, ceil_mode, count_include_pad,
                                    divisor_override);
}

}  // namespace
}  // namespace torch_kunlun

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("group_norm", TORCH_FN(torch_kunlun::group_norm_autograd));
    m.impl("instance_norm", TORCH_FN(torch_kunlun::instance_norm_autograd));
    m.impl("adaptive_max_pool2d", TORCH_FN(torch_kunlun::adaptive_max_pool2d_autograd));
    m.impl("avg_pool3d", TORCH_FN(torch_kunlun::avg_pool3d_autograd));
    m.impl("rms_norm", TORCH_FN(torch_kunlun::rms_norm_autograd));
    m.impl("conv1d", TORCH_FN(torch_kunlun::conv1d_autograd));
    m.impl("conv3d", TORCH_FN(torch_kunlun::conv3d_autograd));
    m.impl("conv_transpose2d", TORCH_FN(torch_kunlun::conv_transpose2d_autograd));
}
