#include "kunlun.h"

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "xpu/refactor/math.h"
#include "xpu/refactor/nn_activation.h"
#include "xpu/refactor/nn_pool.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace torch_kunlun {
namespace {

using baidu::xpu::api::Context;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;

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

template <typename T>
const T* cptr(const at::Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}

template <typename T>
T* mptr(at::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
at::Tensor scale_impl(const at::Tensor& x, double factor, double bias = 0.0) {
    auto out = at::empty_like(x);
    ContextGuard guard(x.device().index());
    const int ret = baidu::xpu::api::scale<T>(
        guard.dc.ctx, cptr<T>(x), mptr<T>(out), x.numel(), false,
        static_cast<float>(factor), static_cast<float>(bias));
    TORCH_CHECK(ret == 0, "xdnn::scale failed: ret=", ret);
    return out;
}

std::vector<int64_t> pair(at::IntArrayRef value, int64_t fallback = 0) {
    if (value.empty()) return {fallback, fallback};
    if (value.size() == 1) return {value[0], value[0]};
    TORCH_CHECK(value.size() == 2, "expected 1 or 2 element array");
    return {value[0], value[1]};
}

enum class UnaryCode {
    Abs, Neg, Exp, Log, Sqrt, Rsqrt, Sigmoid, Tanh, Silu, LeakyRelu,
    Elu, Softplus, Clamp, Square, Reciprocal, Sign, Ceil, Floor, Round,
    Log1p, Exp2, Erf, Erfinv, Selu, Celu, Mish, HardSwish, HardSigmoid,
    HardTanh, SoftShrink, Threshold
};

at::Tensor unary_forward(const at::Tensor& self, UnaryCode code,
                         double a, double b) {
    switch (code) {
        case UnaryCode::Abs: return at::abs(self);
        case UnaryCode::Neg: return at::neg(self);
        case UnaryCode::Exp: return at::exp(self);
        case UnaryCode::Log: return at::log(self);
        case UnaryCode::Sqrt: return at::sqrt(self);
        case UnaryCode::Rsqrt: return at::rsqrt(self);
        case UnaryCode::Sigmoid: return at::sigmoid(self);
        case UnaryCode::Tanh: return at::tanh(self);
        case UnaryCode::Silu: return at::silu(self);
        case UnaryCode::LeakyRelu: return at::leaky_relu(self, a);
        case UnaryCode::Elu: return at::elu(self, a, 1.0, 1.0);
        case UnaryCode::Softplus: return at::softplus(self, a, b);
        case UnaryCode::Clamp: return at::clamp(self, a, b);
        case UnaryCode::Square: return at::square(self);
        case UnaryCode::Reciprocal: return at::reciprocal(self);
        case UnaryCode::Sign: return at::sign(self);
        case UnaryCode::Ceil: return at::ceil(self);
        case UnaryCode::Floor: return at::floor(self);
        case UnaryCode::Round: return at::round(self);
        case UnaryCode::Log1p: return at::log1p(self);
        case UnaryCode::Exp2: return at::exp2(self);
        case UnaryCode::Erf: return at::erf(self);
        case UnaryCode::Erfinv: return at::erfinv(self);
        case UnaryCode::Selu: return at::selu(self);
        case UnaryCode::Celu: return at::celu(self, a);
        case UnaryCode::Mish: return at::mish(self);
        case UnaryCode::HardSwish: return at::hardswish(self);
        case UnaryCode::HardSigmoid: return at::hardsigmoid(self);
        case UnaryCode::HardTanh: return at::hardtanh(self, a, b);
        case UnaryCode::SoftShrink: return at::softshrink(self, a);
        case UnaryCode::Threshold: return at::threshold(self, a, b);
    }
}

class UnaryFunction : public Function<UnaryFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self,
                              int64_t code, double a, double b) {
        at::AutoDispatchBelowAutograd guard;
        auto out = unary_forward(self, static_cast<UnaryCode>(code), a, b);
        ctx->save_for_backward({self, out});
        ctx->saved_data["code"] = code;
        ctx->saved_data["a"] = a;
        ctx->saved_data["b"] = b;
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& self = saved[0];
        const at::Tensor& out = saved[1];
        auto grad = grad_outputs[0].contiguous();
        const auto code = static_cast<UnaryCode>(ctx->saved_data["code"].toInt());
        const double a = ctx->saved_data["a"].toDouble();
        const double b = ctx->saved_data["b"].toDouble();
        auto dx = at::empty_like(self);
        ContextGuard guard(self.device().index());
        if (self.scalar_type() == at::kHalf) {
            const auto* x = cptr<float16>(self);
            const auto* y = cptr<float16>(out);
            const auto* dy = cptr<float16>(grad);
            auto* d = mptr<float16>(dx);
            int ret = 0;
            switch (code) {
                case UnaryCode::Abs: ret = baidu::xpu::api::abs_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Neg: ret = baidu::xpu::api::neg<float16>(guard.dc.ctx, dy, d, self.numel()); break;
                case UnaryCode::Exp: ret = baidu::xpu::api::exp_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Log: {
                    at::AutoDispatchBelowAutograd no_grad;
                    dx = at::div(grad, self);
                    ret = 0;
                    break;
                }
                case UnaryCode::Sqrt: ret = baidu::xpu::api::sqrt_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Rsqrt: ret = baidu::xpu::api::rsqrt_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Sigmoid: ret = baidu::xpu::api::sigmoid_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Tanh: ret = baidu::xpu::api::tanh_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Silu: ret = baidu::xpu::api::swish_grad<float16>(guard.dc.ctx, x, dy, d, self.numel()); break;
                case UnaryCode::LeakyRelu: ret = baidu::xpu::api::leaky_relu_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel(), a); break;
                case UnaryCode::Elu: ret = baidu::xpu::api::elu_grad<float16>(guard.dc.ctx, x, dy, d, self.numel(), a); break;
                case UnaryCode::Softplus: ret = baidu::xpu::api::softplus_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel(), a, b); break;
                case UnaryCode::Clamp: ret = baidu::xpu::api::clip_grad<float16>(guard.dc.ctx, x, dy, d, self.numel(), static_cast<float16>(a), static_cast<float16>(b)); break;
                case UnaryCode::Square: ret = baidu::xpu::api::square_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Reciprocal: ret = baidu::xpu::api::reciprocal_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Log1p: {
                    at::AutoDispatchBelowAutograd no_grad;
                    dx = at::div(grad, at::add(self, 1.0));
                    ret = 0;
                    break;
                }
                case UnaryCode::Exp2: {
                    at::AutoDispatchBelowAutograd no_grad;
                    dx = at::mul(at::mul(grad, out), std::log(2.0));
                    ret = 0;
                    break;
                }
                case UnaryCode::Erf: {
                    at::AutoDispatchBelowAutograd no_grad;
                    auto x2 = at::mul(self, self);
                    dx = at::mul(at::mul(grad, at::exp(at::neg(x2))), 2.0 / std::sqrt(M_PI));
                    ret = 0;
                    break;
                }
                case UnaryCode::Erfinv: {
                    at::AutoDispatchBelowAutograd no_grad;
                    auto y2 = at::mul(out, out);
                    dx = at::mul(at::mul(grad, at::exp(y2)), std::sqrt(M_PI) / 2.0);
                    ret = 0;
                    break;
                }
                case UnaryCode::Selu: ret = baidu::xpu::api::selu_grad<float16>(guard.dc.ctx, x, dy, d, self.numel(), a, b); break;
                case UnaryCode::Celu: ret = baidu::xpu::api::celu_grad<float16>(guard.dc.ctx, x, dy, d, self.numel(), a); break;
                case UnaryCode::Mish: ret = baidu::xpu::api::mish_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel(), a); break;
                case UnaryCode::HardSwish: ret = baidu::xpu::api::hard_swish_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::HardSigmoid: ret = baidu::xpu::api::hard_sigmoid_grad<float16>(guard.dc.ctx, x, y, dy, d, self.numel(), a); break;
                case UnaryCode::HardTanh: ret = baidu::xpu::api::hard_tanh_grad<float16>(guard.dc.ctx, x, dy, d, self.numel(), a, b); break;
                case UnaryCode::SoftShrink: ret = baidu::xpu::api::soft_shrink_grad<float16>(guard.dc.ctx, x, dy, d, self.numel(), a); break;
                case UnaryCode::Threshold: ret = baidu::xpu::api::threshold_grad<float16>(guard.dc.ctx, x, dy, d, self.numel(), a); break;
                case UnaryCode::Sign:
                case UnaryCode::Ceil:
                case UnaryCode::Floor:
                case UnaryCode::Round: ret = baidu::xpu::api::constant<float16>(guard.dc.ctx, d, self.numel(), float16(0.0f)); break;
            }
            TORCH_CHECK(ret == 0, "unary backward failed: ret=", ret);
        } else {
            const auto* x = cptr<float>(self);
            const auto* y = cptr<float>(out);
            const auto* dy = cptr<float>(grad);
            auto* d = mptr<float>(dx);
            int ret = 0;
            switch (code) {
                case UnaryCode::Abs: ret = baidu::xpu::api::abs_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Neg: ret = baidu::xpu::api::neg<float>(guard.dc.ctx, dy, d, self.numel()); break;
                case UnaryCode::Exp: ret = baidu::xpu::api::exp_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Log: {
                    at::AutoDispatchBelowAutograd no_grad;
                    dx = at::div(grad, self);
                    ret = 0;
                    break;
                }
                case UnaryCode::Sqrt: ret = baidu::xpu::api::sqrt_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Rsqrt: ret = baidu::xpu::api::rsqrt_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Sigmoid: ret = baidu::xpu::api::sigmoid_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Tanh: ret = baidu::xpu::api::tanh_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Silu: ret = baidu::xpu::api::swish_grad<float>(guard.dc.ctx, x, dy, d, self.numel()); break;
                case UnaryCode::LeakyRelu: ret = baidu::xpu::api::leaky_relu_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel(), a); break;
                case UnaryCode::Elu: ret = baidu::xpu::api::elu_grad<float>(guard.dc.ctx, x, dy, d, self.numel(), a); break;
                case UnaryCode::Softplus: ret = baidu::xpu::api::softplus_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel(), a, b); break;
                case UnaryCode::Clamp: ret = baidu::xpu::api::clip_grad<float>(guard.dc.ctx, x, dy, d, self.numel(), static_cast<float>(a), static_cast<float>(b)); break;
                case UnaryCode::Square: ret = baidu::xpu::api::square_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Reciprocal: ret = baidu::xpu::api::reciprocal_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::Log1p: {
                    at::AutoDispatchBelowAutograd no_grad;
                    dx = at::div(grad, at::add(self, 1.0));
                    ret = 0;
                    break;
                }
                case UnaryCode::Exp2: {
                    at::AutoDispatchBelowAutograd no_grad;
                    dx = at::mul(at::mul(grad, out), std::log(2.0));
                    ret = 0;
                    break;
                }
                case UnaryCode::Erf: {
                    at::AutoDispatchBelowAutograd no_grad;
                    auto x2 = at::mul(self, self);
                    dx = at::mul(at::mul(grad, at::exp(at::neg(x2))), 2.0 / std::sqrt(M_PI));
                    ret = 0;
                    break;
                }
                case UnaryCode::Erfinv: {
                    at::AutoDispatchBelowAutograd no_grad;
                    auto y2 = at::mul(out, out);
                    dx = at::mul(at::mul(grad, at::exp(y2)), std::sqrt(M_PI) / 2.0);
                    ret = 0;
                    break;
                }
                case UnaryCode::Selu: ret = baidu::xpu::api::selu_grad<float>(guard.dc.ctx, x, dy, d, self.numel(), a, b); break;
                case UnaryCode::Celu: ret = baidu::xpu::api::celu_grad<float>(guard.dc.ctx, x, dy, d, self.numel(), a); break;
                case UnaryCode::Mish: ret = baidu::xpu::api::mish_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel(), a); break;
                case UnaryCode::HardSwish: ret = baidu::xpu::api::hard_swish_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel()); break;
                case UnaryCode::HardSigmoid: ret = baidu::xpu::api::hard_sigmoid_grad<float>(guard.dc.ctx, x, y, dy, d, self.numel(), a); break;
                case UnaryCode::HardTanh: ret = baidu::xpu::api::hard_tanh_grad<float>(guard.dc.ctx, x, dy, d, self.numel(), a, b); break;
                case UnaryCode::SoftShrink: ret = baidu::xpu::api::soft_shrink_grad<float>(guard.dc.ctx, x, dy, d, self.numel(), a); break;
                case UnaryCode::Threshold: ret = baidu::xpu::api::threshold_grad<float>(guard.dc.ctx, x, dy, d, self.numel(), a); break;
                case UnaryCode::Sign:
                case UnaryCode::Ceil:
                case UnaryCode::Floor:
                case UnaryCode::Round: ret = baidu::xpu::api::constant<float>(guard.dc.ctx, d, self.numel(), 0.0f); break;
            }
            TORCH_CHECK(ret == 0, "unary backward failed: ret=", ret);
        }
        return {dx, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

template <UnaryCode Code>
at::Tensor unary_autograd(const at::Tensor& self) {
    return UnaryFunction::apply(self, static_cast<int64_t>(Code), 0.0, 0.0);
}

at::Tensor leaky_relu_autograd(const at::Tensor& self, const at::Scalar& slope) {
    return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::LeakyRelu),
                                slope.toDouble(), 0.0);
}

at::Tensor elu_autograd(const at::Tensor& self, const at::Scalar& alpha,
                        const at::Scalar& scale, const at::Scalar& input_scale) {
    TORCH_CHECK(scale.toDouble() == 1.0 && input_scale.toDouble() == 1.0,
                "kunlun elu autograd only supports scale=1,input_scale=1");
    return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Elu),
                                alpha.toDouble(), 0.0);
}

at::Tensor softplus_autograd(const at::Tensor& self, const at::Scalar& beta,
                             const at::Scalar& threshold) {
    return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Softplus),
                                beta.toDouble(), threshold.toDouble());
}

at::Tensor clamp_autograd(const at::Tensor& self, const std::optional<at::Scalar>& min,
                          const std::optional<at::Scalar>& max) {
    const double lo = min.has_value() ? min->toDouble() : -std::numeric_limits<double>::infinity();
    const double hi = max.has_value() ? max->toDouble() : std::numeric_limits<double>::infinity();
    return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Clamp), lo, hi);
}

class DivFunction : public Function<DivFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor other) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::div(self, other);
        ctx->save_for_backward({self, other, out});
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& x = saved[0];
        const at::Tensor& y = saved[1];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(x);
        auto dy = at::empty_like(y);
        ContextGuard guard(x.device().index());
        int ret = 0;
        if (x.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::broadcast_div_grad<float16>(
                guard.dc.ctx, cptr<float16>(x), cptr<float16>(y),
                cptr<float16>(saved[2]), cptr<float16>(grad),
                mptr<float16>(dy), mptr<float16>(dx), x.sizes().vec(), y.sizes().vec());
        } else {
            ret = baidu::xpu::api::broadcast_div_grad<float>(
                guard.dc.ctx, cptr<float>(x), cptr<float>(y),
                cptr<float>(saved[2]), cptr<float>(grad),
                mptr<float>(dy), mptr<float>(dx), x.sizes().vec(), y.sizes().vec());
        }
        TORCH_CHECK(ret == 0, "div backward failed: ret=", ret);
        return {dx, dy};
    }
};

class DivScalarFunction : public Function<DivScalarFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Scalar other) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::div(self, other);
        ctx->saved_data["factor"] = 1.0 / other.toDouble();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto grad = grad_outputs[0];
        const double factor = ctx->saved_data["factor"].toDouble();
        if (grad.scalar_type() == at::kHalf) {
            return {scale_impl<float16>(grad, factor), at::Tensor()};
        }
        return {scale_impl<float>(grad, factor), at::Tensor()};
    }
};

class LogSoftmaxFunction : public Function<LogSoftmaxFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, int64_t dim,
                              std::optional<at::ScalarType> dtype) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::log_softmax(self, dim, dtype);
        ctx->save_for_backward({out});
        ctx->saved_data["dim"] = dim;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& out = saved[0];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(out);
        int64_t dim = ctx->saved_data["dim"].toInt();
        if (dim < 0) dim += out.dim();
        ContextGuard guard(out.device().index());
        int ret = 0;
        if (out.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::log_softmax_grad<float16>(
                guard.dc.ctx, cptr<float16>(out), cptr<float16>(grad),
                mptr<float16>(dx), out.sizes().vec(), dim);
        } else {
            ret = baidu::xpu::api::log_softmax_grad<float>(
                guard.dc.ctx, cptr<float>(out), cptr<float>(grad),
                mptr<float>(dx), out.sizes().vec(), dim);
        }
        TORCH_CHECK(ret == 0, "log_softmax backward failed: ret=", ret);
        return {dx, at::Tensor(), at::Tensor()};
    }
};

class AvgPool2dFunction : public Function<AvgPool2dFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self,
                              std::vector<int64_t> kernel, std::vector<int64_t> stride,
                              std::vector<int64_t> padding, bool ceil_mode,
                              bool count_include_pad, std::optional<int64_t> divisor_override) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::avg_pool2d(self, kernel, stride, padding, ceil_mode,
                                  count_include_pad, divisor_override);
        ctx->save_for_backward({self, out});
        ctx->saved_data["kernel"] = kernel;
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["count_include_pad"] = count_include_pad;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& self = saved[0];
        const at::Tensor& out = saved[1];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(self);
        auto kernel = ctx->saved_data["kernel"].toIntVector();
        auto stride = ctx->saved_data["stride"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        const bool cip = ctx->saved_data["count_include_pad"].toBool();
        ContextGuard guard(self.device().index());
        int ret = 0;
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::avg_pool2d_grad<float16>(
                guard.dc.ctx, cptr<float16>(self), cptr<float16>(out),
                cptr<float16>(grad), mptr<float16>(dx), self.size(0), self.size(1),
                self.size(2), self.size(3), kernel, stride, padding, cip, true);
        } else {
            ret = baidu::xpu::api::avg_pool2d_grad<float>(
                guard.dc.ctx, cptr<float>(self), cptr<float>(out),
                cptr<float>(grad), mptr<float>(dx), self.size(0), self.size(1),
                self.size(2), self.size(3), kernel, stride, padding, cip, true);
        }
        TORCH_CHECK(ret == 0, "avg_pool2d backward failed: ret=", ret);
        return {dx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor()};
    }
};


class StackFunction : public Function<StackFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, std::vector<at::Tensor> tensors, int64_t dim) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::stack(tensors, dim);
        std::vector<int64_t> sizes;
        for (const auto& t : tensors) sizes.push_back(t.size(dim < 0 ? dim + t.dim() + 1 : dim));
        ctx->saved_data["sizes"] = sizes;
        ctx->saved_data["dim"] = dim;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto grad = grad_outputs[0];
        auto sizes = ctx->saved_data["sizes"].toIntVector();
        int64_t dim = ctx->saved_data["dim"].toInt();
        if (dim < 0) dim += grad.dim();
        variable_list grads;
        int64_t offset = 0;
        for (int64_t size : sizes) {
            // stack inserts a singleton dimension; each gradient slice is
            // exactly one element along dim.
            grads.push_back(grad.narrow(dim, offset, 1).squeeze(dim));
            offset += 1;
        }
        return grads;
    }
};

at::Tensor stack_autograd(at::ITensorListRef tensors, int64_t dim) {
    std::vector<at::Tensor> list;
    for (const auto& t : tensors) list.push_back(t);
    return StackFunction::apply(list, dim);
}


std::vector<int64_t> normalize_dims(at::IntArrayRef dims, int64_t rank) {
    std::vector<int64_t> out;
    if (dims.empty()) {
        for (int64_t i = 0; i < rank; ++i) out.push_back(i);
        return out;
    }
    for (int64_t d : dims) {
        if (d < 0) d += rank;
        TORCH_CHECK(d >= 0 && d < rank, "dimension out of range");
        out.push_back(d);
    }
    return out;
}

std::vector<int64_t> keep_shape(at::IntArrayRef shape,
                                const std::vector<int64_t>& dims,
                                bool keepdim) {
    std::vector<int64_t> out;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
        const bool reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
        if (reduced) {
            if (keepdim) out.push_back(1);
        } else {
            out.push_back(shape[i]);
        }
    }
    return out;
}

at::Tensor reduce_grad(const at::Tensor& self, const at::Tensor& grad,
                       const std::vector<int64_t>& dims, bool keepdim) {
    auto shape = keep_shape(self.sizes(), dims, keepdim);
    auto g = keepdim ? grad : grad.reshape(shape);
    return g.expand(self.sizes()).contiguous();
}

class SumFunction : public Function<SumFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self,
                              std::vector<int64_t> dims, bool keepdim,
                              std::optional<at::ScalarType> dtype) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::sum(self, dims, keepdim, dtype);
        ctx->save_for_backward({self});
        ctx->saved_data["dims"] = dims;
        ctx->saved_data["keepdim"] = keepdim;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto self = ctx->get_saved_variables()[0];
        auto dims = ctx->saved_data["dims"].toIntVector();
        bool keepdim = ctx->saved_data["keepdim"].toBool();
        return {reduce_grad(self, grad_outputs[0], dims, keepdim), at::Tensor(),
                at::Tensor(), at::Tensor()};
    }
};

class MeanFunction : public Function<MeanFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self,
                              std::vector<int64_t> dims, bool keepdim,
                              std::optional<at::ScalarType> dtype) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::mean(self, dims, keepdim, dtype);
        ctx->save_for_backward({self});
        ctx->saved_data["dims"] = dims;
        ctx->saved_data["keepdim"] = keepdim;
        int64_t count = 1;
        for (int64_t d : dims) count *= self.size(d);
        ctx->saved_data["count"] = count;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto self = ctx->get_saved_variables()[0];
        auto dims = ctx->saved_data["dims"].toIntVector();
        bool keepdim = ctx->saved_data["keepdim"].toBool();
        const int64_t count = ctx->saved_data["count"].toInt();
        auto g = grad_outputs[0];
        if (self.scalar_type() == at::kHalf) {
            g = scale_impl<float16>(g, 1.0 / static_cast<double>(count));
        } else {
            g = scale_impl<float>(g, 1.0 / static_cast<double>(count));
        }
        return {reduce_grad(self, g, dims, keepdim), at::Tensor(),
                at::Tensor(), at::Tensor()};
    }
};

at::Tensor sum_autograd(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                        std::optional<at::ScalarType> dtype) {
    auto dims = normalize_dims(dim, self.dim());
    return SumFunction::apply(self, dims, keepdim, dtype);
}
at::Tensor sum_all_autograd(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
    return sum_autograd(self, at::IntArrayRef{}, false, dtype);
}
at::Tensor mean_autograd(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                         std::optional<at::ScalarType> dtype) {
    auto dims = normalize_dims(dim, self.dim());
    return MeanFunction::apply(self, dims, keepdim, dtype);
}
at::Tensor mean_all_autograd(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
    return mean_autograd(self, at::IntArrayRef{}, false, dtype);
}

class WhereFunction : public Function<WhereFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor condition,
                              at::Tensor self, at::Tensor other) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::where(condition, self, other);
        ctx->save_for_backward({condition, self, other});
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto condition = saved[0];
        auto self = saved[1];
        auto other = saved[2];
        auto grad = grad_outputs[0].contiguous();
        TORCH_CHECK(self.sizes().equals(other.sizes()),
                    "kunlun where autograd requires equal x/y shapes");
        auto dx = at::empty_like(self);
        auto dy = at::empty_like(other);
        ContextGuard guard(self.device().index());
        if (self.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::select_grad<float16>(
                guard.dc.ctx, cptr<bool>(condition), cptr<float16>(grad),
                mptr<float16>(dx), mptr<float16>(dy),
                condition.sizes().vec(), self.sizes().vec());
            TORCH_CHECK(ret == 0, "where backward failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::select_grad<float>(
                guard.dc.ctx, cptr<bool>(condition), cptr<float>(grad),
                mptr<float>(dx), mptr<float>(dy),
                condition.sizes().vec(), self.sizes().vec());
            TORCH_CHECK(ret == 0, "where backward failed: ret=", ret);
        }
        return {at::Tensor(), dx, dy};
    }
};

at::Tensor where_autograd(const at::Tensor& condition, const at::Tensor& self,
                          const at::Tensor& other) {
    return WhereFunction::apply(condition, self, other);
}

class MaskedFillFunction : public Function<MaskedFillFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor mask,
                              at::Scalar value) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::masked_fill(self, mask, value);
        ctx->save_for_backward({mask});
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto mask = ctx->get_saved_variables()[0];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(grad);
        ContextGuard guard(grad.device().index());
        if (grad.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::masked_fill_scalar<float16>(
                guard.dc.ctx, cptr<bool>(mask), cptr<float16>(grad), float16(0.0f),
                mptr<float16>(dx), mask.sizes().vec(), grad.sizes().vec());
            TORCH_CHECK(ret == 0, "masked_fill backward failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::masked_fill_scalar<float>(
                guard.dc.ctx, cptr<bool>(mask), cptr<float>(grad), 0.0f,
                mptr<float>(dx), mask.sizes().vec(), grad.sizes().vec());
            TORCH_CHECK(ret == 0, "masked_fill backward failed: ret=", ret);
        }
        return {dx, at::Tensor(), at::Tensor()};
    }
};

at::Tensor masked_fill_autograd(const at::Tensor& self, const at::Tensor& mask,
                                const at::Scalar& value) {
    return MaskedFillFunction::apply(self, mask, value);
}

class TriangleFunction : public Function<TriangleFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, int64_t diagonal,
                              bool upper) {
        at::AutoDispatchBelowAutograd guard;
        auto out = upper ? at::triu(self, diagonal) : at::tril(self, diagonal);
        ctx->saved_data["diagonal"] = diagonal;
        ctx->saved_data["upper"] = upper;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto grad = grad_outputs[0];
        const int64_t diagonal = ctx->saved_data["diagonal"].toInt();
        const bool upper = ctx->saved_data["upper"].toBool();
        at::AutoDispatchBelowAutograd guard;
        auto dx = upper ? at::triu(grad, diagonal) : at::tril(grad, diagonal);
        return {dx, at::Tensor(), at::Tensor()};
    }
};

at::Tensor triu_autograd(const at::Tensor& self, int64_t diagonal) {
    return TriangleFunction::apply(self, diagonal, true);
}
at::Tensor tril_autograd(const at::Tensor& self, int64_t diagonal) {
    return TriangleFunction::apply(self, diagonal, false);
}


class ExtremumFunction : public Function<ExtremumFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self,
                              std::vector<int64_t> dims, bool keepdim, bool is_max) {
        at::AutoDispatchBelowAutograd guard;
        auto out = is_max ? at::amax(self, dims, keepdim)
                          : at::amin(self, dims, keepdim);
        ctx->save_for_backward({self, out});
        ctx->saved_data["dims"] = dims;
        ctx->saved_data["keepdim"] = keepdim;
        ctx->saved_data["is_max"] = is_max;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto self = saved[0]; auto out = saved[1];
        auto dims = ctx->saved_data["dims"].toIntVector();
        bool keepdim = ctx->saved_data["keepdim"].toBool();
        auto shape = keep_shape(self.sizes(), dims, keepdim);
        auto grad = grad_outputs[0];
        auto g = (keepdim ? grad : grad.reshape(shape)).expand(self.sizes()).contiguous();
        auto o = (keepdim ? out : out.reshape(shape)).expand(self.sizes()).contiguous();
        auto mask = at::empty(self.sizes(), self.options().dtype(at::kBool));
        ContextGuard guard(self.device().index());
        int ret = 0;
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::broadcast_equal<float16>(
                guard.dc.ctx, cptr<float16>(self), cptr<float16>(o), mptr<bool>(mask),
                self.sizes().vec(), o.sizes().vec());
        } else {
            ret = baidu::xpu::api::broadcast_equal<float>(
                guard.dc.ctx, cptr<float>(self), cptr<float>(o), mptr<bool>(mask),
                self.sizes().vec(), o.sizes().vec());
        }
        TORCH_CHECK(ret == 0, "extremum backward mask failed: ret=", ret);
        auto zero = at::zeros_like(self);
        auto dx = at::where(mask, g, zero);
        return {dx, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor amax_autograd(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    auto dims = normalize_dims(dim, self.dim());
    return ExtremumFunction::apply(self, dims, keepdim, true);
}
at::Tensor amin_autograd(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    auto dims = normalize_dims(dim, self.dim());
    return ExtremumFunction::apply(self, dims, keepdim, false);
}

class LogSumExpFunction : public Function<LogSumExpFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self,
                              std::vector<int64_t> dims, bool keepdim) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::logsumexp(self, dims, keepdim);
        ctx->save_for_backward({self, out});
        ctx->saved_data["dims"] = dims;
        ctx->saved_data["keepdim"] = keepdim;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto self = saved[0]; auto out = saved[1];
        auto dims = ctx->saved_data["dims"].toIntVector();
        bool keepdim = ctx->saved_data["keepdim"].toBool();
        auto shape = keep_shape(self.sizes(), dims, keepdim);
        auto grad = grad_outputs[0];
        auto g = (keepdim ? grad : grad.reshape(shape)).expand(self.sizes()).contiguous();
        auto o = (keepdim ? out : out.reshape(shape)).expand(self.sizes()).contiguous();
        at::AutoDispatchBelowAutograd guard;
        auto shifted = at::sub(self, o);
        auto dx = at::mul(at::exp(shifted), g);
        return {dx, at::Tensor(), at::Tensor()};
    }
};

at::Tensor logsumexp_autograd(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    auto dims = normalize_dims(dim, self.dim());
    return LogSumExpFunction::apply(self, dims, keepdim);
}


class BinaryExtremumFunction : public Function<BinaryExtremumFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor other,
                              bool is_max) {
        at::AutoDispatchBelowAutograd guard;
        auto out = is_max ? at::maximum(self, other) : at::minimum(self, other);
        ctx->save_for_backward({self, other, out});
        ctx->saved_data["is_max"] = is_max;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto self = saved[0]; auto other = saved[1]; auto out = saved[2];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(self);
        auto dy = at::empty_like(other);
        ContextGuard guard(self.device().index());
        int ret = 0;
        if (self.scalar_type() == at::kHalf) {
            ret = ctx->saved_data["is_max"].toBool()
                ? baidu::xpu::api::broadcast_max_grad<float16>(
                      guard.dc.ctx, cptr<float16>(self), cptr<float16>(other), cptr<float16>(out),
                      cptr<float16>(grad), mptr<float16>(dy), mptr<float16>(dx),
                      self.sizes().vec(), other.sizes().vec())
                : baidu::xpu::api::broadcast_min_grad<float16>(
                      guard.dc.ctx, cptr<float16>(self), cptr<float16>(other), cptr<float16>(out),
                      cptr<float16>(grad), mptr<float16>(dy), mptr<float16>(dx),
                      self.sizes().vec(), other.sizes().vec());
        } else {
            ret = ctx->saved_data["is_max"].toBool()
                ? baidu::xpu::api::broadcast_max_grad<float>(
                      guard.dc.ctx, cptr<float>(self), cptr<float>(other), cptr<float>(out),
                      cptr<float>(grad), mptr<float>(dy), mptr<float>(dx),
                      self.sizes().vec(), other.sizes().vec())
                : baidu::xpu::api::broadcast_min_grad<float>(
                      guard.dc.ctx, cptr<float>(self), cptr<float>(other), cptr<float>(out),
                      cptr<float>(grad), mptr<float>(dy), mptr<float>(dx),
                      self.sizes().vec(), other.sizes().vec());
        }
        TORCH_CHECK(ret == 0, "maximum/minimum backward failed");
        return {dx, dy, at::Tensor()};
    }
};

at::Tensor maximum_autograd(const at::Tensor& self, const at::Tensor& other) {
    return BinaryExtremumFunction::apply(self, other, true);
}
at::Tensor minimum_autograd(const at::Tensor& self, const at::Tensor& other) {
    return BinaryExtremumFunction::apply(self, other, false);
}

at::Tensor scale_impl_dispatch(const at::Tensor& x, double factor) {
    if (x.scalar_type() == at::kHalf) return scale_impl<float16>(x, factor);
    return scale_impl<float>(x, factor);
}

class AddCMulFunction : public Function<AddCMulFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor a,
                              at::Tensor b, at::Scalar value) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::addcmul(self, a, b, value);
        ctx->save_for_backward({a, b});
        ctx->saved_data["value"] = value.toDouble();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto a = saved[0]; auto b = saved[1];
        auto grad = grad_outputs[0];
        const double value = ctx->saved_data["value"].toDouble();
        at::AutoDispatchBelowAutograd guard;
        auto da = scale_impl_dispatch(at::mul(grad, b), value);
        auto db = scale_impl_dispatch(at::mul(grad, a), value);
        return {grad, da, db, at::Tensor()};
    }
};

class AddCDivFunction : public Function<AddCDivFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor a,
                              at::Tensor b, at::Scalar value) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::addcdiv(self, a, b, value);
        ctx->save_for_backward({a, b});
        ctx->saved_data["value"] = value.toDouble();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto a = saved[0]; auto b = saved[1];
        auto grad = grad_outputs[0];
        const double value = ctx->saved_data["value"].toDouble();
        at::AutoDispatchBelowAutograd guard;
        auto da = scale_impl_dispatch(at::div(grad, b), value);
        auto b2 = at::mul(b, b);
        auto db = at::div(at::mul(grad, a), b2);
        db = scale_impl_dispatch(db, -value);
        return {grad, da, db, at::Tensor()};
    }
};

at::Tensor addcmul_autograd(const at::Tensor& self, const at::Tensor& a,
                           const at::Tensor& b, const at::Scalar& value) {
    return AddCMulFunction::apply(self, a, b, value);
}
at::Tensor addcdiv_autograd(const at::Tensor& self, const at::Tensor& a,
                           const at::Tensor& b, const at::Scalar& value) {
    return AddCDivFunction::apply(self, a, b, value);
}

at::Tensor log1p_autograd(const at::Tensor& self) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Log1p), 0.0, 0.0); }
at::Tensor exp2_autograd(const at::Tensor& self) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Exp2), 0.0, 0.0); }
at::Tensor erf_autograd(const at::Tensor& self) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Erf), 0.0, 0.0); }
at::Tensor erfinv_autograd(const at::Tensor& self) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Erfinv), 0.0, 0.0); }
at::Tensor selu_autograd(const at::Tensor& self) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Selu), 1.050701, 1.6732632); }
at::Tensor celu_autograd(const at::Tensor& self, const at::Scalar& alpha) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Celu), alpha.toDouble(), 0.0); }
at::Tensor mish_autograd(const at::Tensor& self) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Mish), 20.0, 0.0); }
at::Tensor hardswish_autograd(const at::Tensor& self) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::HardSwish), 0.0, 0.0); }
at::Tensor hardsigmoid_autograd(const at::Tensor& self) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::HardSigmoid), 1.0/6.0, 0.0); }
at::Tensor hardtanh_autograd(const at::Tensor& self, const at::Scalar& lo, const at::Scalar& hi) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::HardTanh), lo.toDouble(), hi.toDouble()); }
at::Tensor softshrink_autograd(const at::Tensor& self, const at::Scalar& lambd) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::SoftShrink), lambd.toDouble(), 0.0); }
at::Tensor threshold_autograd(const at::Tensor& self, const at::Scalar& thre, const at::Scalar& value) { return UnaryFunction::apply(self, static_cast<int64_t>(UnaryCode::Threshold), thre.toDouble(), value.toDouble()); }

at::Tensor div_autograd(const at::Tensor& self, const at::Tensor& other) {
    return DivFunction::apply(self, other);
}
at::Tensor div_scalar_autograd(const at::Tensor& self, const at::Scalar& other) {
    return DivScalarFunction::apply(self, other);
}
at::Tensor log_softmax_autograd(const at::Tensor& self, int64_t dim,
                                std::optional<at::ScalarType> dtype) {
    return LogSoftmaxFunction::apply(self, dim, dtype);
}
at::Tensor _log_softmax_autograd(const at::Tensor& self, int64_t dim, bool half_to_float) {
    TORCH_CHECK(!half_to_float, "kunlun _log_softmax half_to_float=True unsupported");
    return LogSoftmaxFunction::apply(self, dim, std::nullopt);
}
at::Tensor avg_pool2d_autograd(const at::Tensor& self, at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               bool ceil_mode, bool count_include_pad,
                               std::optional<int64_t> divisor_override) {
    std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
    std::vector<int64_t> s(stride.begin(), stride.end());
    std::vector<int64_t> p(padding.begin(), padding.end());
    return AvgPool2dFunction::apply(self, k, s, p, ceil_mode, count_include_pad,
                                    divisor_override);
}

}  // namespace
}  // namespace torch_kunlun

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("abs", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Abs>));
    m.impl("neg", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Neg>));
    m.impl("exp", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Exp>));
    m.impl("log", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Log>));
    m.impl("sqrt", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Sqrt>));
    m.impl("rsqrt", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Rsqrt>));
    m.impl("sigmoid", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Sigmoid>));
    m.impl("tanh", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Tanh>));
    m.impl("silu", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Silu>));
    m.impl("leaky_relu", TORCH_FN(torch_kunlun::leaky_relu_autograd));
    m.impl("elu", TORCH_FN(torch_kunlun::elu_autograd));
    m.impl("softplus", TORCH_FN(torch_kunlun::softplus_autograd));
    m.impl("clamp", TORCH_FN(torch_kunlun::clamp_autograd));
    m.impl("square", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Square>));
    m.impl("reciprocal", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Reciprocal>));
    m.impl("sign", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Sign>));
    m.impl("ceil", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Ceil>));
    m.impl("floor", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Floor>));
    m.impl("round", TORCH_FN(torch_kunlun::unary_autograd<torch_kunlun::UnaryCode::Round>));
    m.impl("log1p", TORCH_FN(torch_kunlun::log1p_autograd));
    m.impl("exp2", TORCH_FN(torch_kunlun::exp2_autograd));
    m.impl("erf", TORCH_FN(torch_kunlun::erf_autograd));
    m.impl("erfinv", TORCH_FN(torch_kunlun::erfinv_autograd));
    m.impl("selu", TORCH_FN(torch_kunlun::selu_autograd));
    m.impl("celu", TORCH_FN(torch_kunlun::celu_autograd));
    m.impl("mish", TORCH_FN(torch_kunlun::mish_autograd));
    m.impl("hardswish", TORCH_FN(torch_kunlun::hardswish_autograd));
    m.impl("hardsigmoid", TORCH_FN(torch_kunlun::hardsigmoid_autograd));
    m.impl("hardtanh", TORCH_FN(torch_kunlun::hardtanh_autograd));
    m.impl("softshrink", TORCH_FN(torch_kunlun::softshrink_autograd));
    m.impl("threshold", TORCH_FN(torch_kunlun::threshold_autograd));
    m.impl("div.Tensor", TORCH_FN(torch_kunlun::div_autograd));
    m.impl("div.Scalar", TORCH_FN(torch_kunlun::div_scalar_autograd));
    m.impl("log_softmax.int", TORCH_FN(torch_kunlun::log_softmax_autograd));
    m.impl("_log_softmax", TORCH_FN(torch_kunlun::_log_softmax_autograd));
    m.impl("avg_pool2d", TORCH_FN(torch_kunlun::avg_pool2d_autograd));
    m.impl("stack", TORCH_FN(torch_kunlun::stack_autograd));
    m.impl("sum.dim_IntList", TORCH_FN(torch_kunlun::sum_autograd));
    m.impl("sum", TORCH_FN(torch_kunlun::sum_all_autograd));
    m.impl("mean.dim", TORCH_FN(torch_kunlun::mean_autograd));
    m.impl("mean", TORCH_FN(torch_kunlun::mean_all_autograd));
    m.impl("where.self", TORCH_FN(torch_kunlun::where_autograd));
    m.impl("masked_fill.Scalar", TORCH_FN(torch_kunlun::masked_fill_autograd));
    m.impl("triu", TORCH_FN(torch_kunlun::triu_autograd));
    m.impl("tril", TORCH_FN(torch_kunlun::tril_autograd));
    m.impl("amax", TORCH_FN(torch_kunlun::amax_autograd));
    m.impl("amin", TORCH_FN(torch_kunlun::amin_autograd));
    m.impl("logsumexp", TORCH_FN(torch_kunlun::logsumexp_autograd));
    m.impl("maximum", TORCH_FN(torch_kunlun::maximum_autograd));
    m.impl("minimum", TORCH_FN(torch_kunlun::minimum_autograd));
    m.impl("addcmul", TORCH_FN(torch_kunlun::addcmul_autograd));
    m.impl("addcdiv", TORCH_FN(torch_kunlun::addcdiv_autograd));
}
