#include "kunlun.h"

#include <ATen/ATen.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "xpu/refactor/math.h"
#include "xpu/refactor/nn.h"
#include "xpu/refactor/nn_activation.h"

#include <optional>
#include <string>
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

template <typename T>
const T* cptr(const at::Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}

template <typename T>
T* mptr(at::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

std::vector<int64_t> all_dims(const at::Tensor& t) {
    std::vector<int64_t> dims(t.dim());
    for (int64_t i = 0; i < t.dim(); ++i) dims[i] = i;
    return dims;
}

template <typename T>
at::Tensor sub_impl(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
    TORCH_CHECK(alpha.toDouble() == 1.0, "kunlun sub only supports alpha=1");
    check_tensor(a, "sub");
    check_tensor(b, "sub");
    TORCH_CHECK(a.sizes().equals(b.sizes()), "kunlun sub requires equal shapes");
    auto out = at::empty_like(a);
    ContextGuard guard(a.device().index());
    const int ret = baidu::xpu::api::sub<T>(guard.dc.ctx, cptr<T>(a), cptr<T>(b),
                                            mptr<T>(out), a.numel());
    TORCH_CHECK(ret == 0, "xdnn::sub failed: ret=", ret);
    return out;
}

template <typename T>
at::Tensor mul_impl(const at::Tensor& a, const at::Tensor& b) {
    check_tensor(a, "mul");
    check_tensor(b, "mul");
    TORCH_CHECK(a.sizes().equals(b.sizes()), "kunlun mul requires equal shapes");
    auto out = at::empty_like(a);
    ContextGuard guard(a.device().index());
    const int ret = baidu::xpu::api::mul<T>(guard.dc.ctx, cptr<T>(a), cptr<T>(b),
                                            mptr<T>(out), a.numel());
    TORCH_CHECK(ret == 0, "xdnn::mul failed: ret=", ret);
    return out;
}

template <typename T>
at::Tensor scale_impl(const at::Tensor& x, double factor, double bias) {
    check_tensor(x, "scale");
    auto out = at::empty_like(x);
    ContextGuard guard(x.device().index());
    const int ret = baidu::xpu::api::scale<T>(
        guard.dc.ctx, cptr<T>(x), mptr<T>(out), x.numel(),
        /*bias_after_scale=*/false, static_cast<float>(factor), static_cast<float>(bias));
    TORCH_CHECK(ret == 0, "xdnn::scale failed: ret=", ret);
    return out;
}

template <typename T>
at::Tensor add_inplace_impl(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    check_tensor(self, "add_");
    check_tensor(other, "add_");
    TORCH_CHECK(self.sizes().equals(other.sizes()), "kunlun add_ requires equal shapes");
    ContextGuard guard(self.device().index());
    if (alpha.toDouble() == 1.0) {
        const int ret = baidu::xpu::api::add<T>(
            guard.dc.ctx, cptr<T>(self), cptr<T>(other), mptr<T>(self), self.numel());
        TORCH_CHECK(ret == 0, "xdnn::add (in-place) failed: ret=", ret);
    } else {
        auto scaled = scale_impl<T>(other, alpha.toDouble(), 0.0);
        const int ret = baidu::xpu::api::add<T>(
            guard.dc.ctx, cptr<T>(self), cptr<T>(scaled), mptr<T>(self), self.numel());
        TORCH_CHECK(ret == 0, "xdnn::add (in-place) failed: ret=", ret);
    }
    return self;
}

template <typename T>
at::Tensor mul_inplace_impl(at::Tensor& self, const at::Tensor& other) {
    check_tensor(self, "mul_");
    check_tensor(other, "mul_");
    TORCH_CHECK(self.sizes().equals(other.sizes()), "kunlun mul_ requires equal shapes");
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::mul<T>(
        guard.dc.ctx, cptr<T>(self), cptr<T>(other), mptr<T>(self), self.numel());
    TORCH_CHECK(ret == 0, "xdnn::mul (in-place) failed: ret=", ret);
    return self;
}

template <typename T>
at::Tensor mse_loss_impl(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
    check_tensor(self, "mse_loss");
    check_tensor(target, "mse_loss");
    TORCH_CHECK(self.sizes().equals(target.sizes()), "mse_loss shape mismatch");
    auto diff = at::empty_like(self);
    auto sq = at::empty_like(self);
    ContextGuard guard(self.device().index());
    int ret = baidu::xpu::api::sub<T>(guard.dc.ctx, cptr<T>(self), cptr<T>(target),
                                      mptr<T>(diff), self.numel());
    TORCH_CHECK(ret == 0, "mse_loss sub failed: ret=", ret);
    ret = baidu::xpu::api::mul<T>(guard.dc.ctx, cptr<T>(diff), cptr<T>(diff),
                                  mptr<T>(sq), self.numel());
    TORCH_CHECK(ret == 0, "mse_loss mul failed: ret=", ret);
    if (reduction == 0) {  // none
        return sq;
    }
    auto loss = at::empty({}, self.options());
    const auto dims = all_dims(sq);
    if (reduction == 1) {  // mean
        ret = baidu::xpu::api::reduce_mean<T>(guard.dc.ctx, cptr<T>(sq), mptr<T>(loss),
                                              sq.sizes().vec(), dims);
    } else {  // sum
        ret = baidu::xpu::api::reduce_sum<T>(guard.dc.ctx, cptr<T>(sq), mptr<T>(loss),
                                             sq.sizes().vec(), dims);
    }
    TORCH_CHECK(ret == 0, "mse_loss reduction failed: ret=", ret);
    return loss;
}

// ---------------------------------------------------------------------------
// Forward-only ops needed by the training slice
// ---------------------------------------------------------------------------

at::Tensor sub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    if (self.scalar_type() == at::kHalf) return sub_impl<float16>(self, other, alpha);
    if (self.scalar_type() == at::kFloat) return sub_impl<float>(self, other, alpha);
    TORCH_CHECK(false, "kunlun sub only supports float16/float32");
}

at::Tensor mul(const at::Tensor& self, const at::Tensor& other) {
    if (self.scalar_type() == at::kHalf) return mul_impl<float16>(self, other);
    if (self.scalar_type() == at::kFloat) return mul_impl<float>(self, other);
    TORCH_CHECK(false, "kunlun mul only supports float16/float32");
}

at::Tensor& add_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    if (self.scalar_type() == at::kHalf) {
        add_inplace_impl<float16>(self, other, alpha);
        return self;
    }
    if (self.scalar_type() == at::kFloat) {
        add_inplace_impl<float>(self, other, alpha);
        return self;
    }
    TORCH_CHECK(false, "kunlun add_ only supports float16/float32");
}

at::Tensor& mul_(at::Tensor& self, const at::Tensor& other) {
    if (self.scalar_type() == at::kHalf) {
        mul_inplace_impl<float16>(self, other);
        return self;
    }
    if (self.scalar_type() == at::kFloat) {
        mul_inplace_impl<float>(self, other);
        return self;
    }
    TORCH_CHECK(false, "kunlun mul_ only supports float16/float32");
}

at::Tensor mse_loss(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
    if (self.scalar_type() == at::kHalf) return mse_loss_impl<float16>(self, target, reduction);
    if (self.scalar_type() == at::kFloat) return mse_loss_impl<float>(self, target, reduction);
    TORCH_CHECK(false, "kunlun mse_loss only supports float16/float32");
}

at::Tensor addmm(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
                 const at::Scalar& beta, const at::Scalar& alpha) {
    check_tensor(mat1, "addmm mat1");
    const at::Tensor mat2c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    check_tensor(mat2c, "addmm mat2");
    at::AutoDispatchBelowAutograd guard;
    at::Tensor result = at::mm(mat1, mat2c);
    if (alpha.toDouble() != 1.0) {
        if (result.scalar_type() == at::kHalf) {
            result = scale_impl<float16>(result, alpha.toDouble(), 0.0);
        } else {
            result = scale_impl<float>(result, alpha.toDouble(), 0.0);
        }
    }
    if (beta.toDouble() != 0.0) {
        at::Tensor bias = self;
        if (beta.toDouble() != 1.0) {
            if (bias.scalar_type() == at::kHalf) {
                bias = scale_impl<float16>(bias, beta.toDouble(), 0.0);
            } else {
                bias = scale_impl<float>(bias, beta.toDouble(), 0.0);
            }
        }
        result = at::add(result, bias);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Autograd functions
// ---------------------------------------------------------------------------

class AddFunction : public Function<AddFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor other,
                              at::Scalar alpha) {
        TORCH_CHECK(alpha.toDouble() == 1.0, "kunlun autograd add only supports alpha=1");
        at::AutoDispatchBelowAutograd guard;
        auto out = at::add(self, other, alpha);
        ctx->save_for_backward({self, other, out});
        ctx->saved_data["self_shape"] = self.sizes().vec();
        ctx->saved_data["other_shape"] = other.sizes().vec();
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& self = saved[0];
        const at::Tensor& other = saved[1];
        const at::Tensor& out = saved[2];
        at::Tensor grad = grad_outputs[0].contiguous();
        if (self.sizes().equals(other.sizes())) {
            return {grad, grad, at::Tensor()};
        }
        auto self_shape = ctx->saved_data["self_shape"].toIntVector();
        auto other_shape = ctx->saved_data["other_shape"].toIntVector();
        auto dx = at::empty(self.sizes(), self.options());
        auto dy = at::empty(other.sizes(), other.options());
        ContextGuard guard(self.device().index());
        if (self.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::broadcast_add_grad<float16>(
                guard.dc.ctx, cptr<float16>(self), cptr<float16>(other), cptr<float16>(out),
                cptr<float16>(grad), mptr<float16>(dy), mptr<float16>(dx),
                self_shape, other_shape);
            TORCH_CHECK(ret == 0, "broadcast_add_grad failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::broadcast_add_grad<float>(
                guard.dc.ctx, cptr<float>(self), cptr<float>(other), cptr<float>(out),
                cptr<float>(grad), mptr<float>(dy), mptr<float>(dx),
                self_shape, other_shape);
            TORCH_CHECK(ret == 0, "broadcast_add_grad failed: ret=", ret);
        }
        return {dx, dy, at::Tensor()};
    }
};

class SubFunction : public Function<SubFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor other,
                              at::Scalar alpha) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::sub(self, other, alpha);
        ctx->save_for_backward({});
        ctx->saved_data["dtype_half"] = self.scalar_type() == at::kHalf;
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        at::Tensor grad = grad_outputs[0].contiguous();
        if (ctx->saved_data["dtype_half"].toBool()) {
            return {grad, scale_impl<float16>(grad, -1.0, 0.0), at::Tensor()};
        }
        return {grad, scale_impl<float>(grad, -1.0, 0.0), at::Tensor()};
    }
};

class MulFunction : public Function<MulFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor other) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::mul(self, other);
        ctx->save_for_backward({self, other});
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        at::Tensor grad = grad_outputs[0].contiguous();
        if (saved[0].scalar_type() == at::kHalf) {
            return {mul_impl<float16>(grad, saved[1]), mul_impl<float16>(grad, saved[0]),
                    at::Tensor()};
        }
        return {mul_impl<float>(grad, saved[1]), mul_impl<float>(grad, saved[0]),
                at::Tensor()};
    }
};

class ReluFunction : public Function<ReluFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::relu(self);
        ctx->save_for_backward({self, out});
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        at::Tensor grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(saved[0]);
        ContextGuard guard(saved[0].device().index());
        if (saved[0].scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::relu_grad<float16>(
                guard.dc.ctx, cptr<float16>(saved[0]), cptr<float16>(saved[1]),
                cptr<float16>(grad), mptr<float16>(dx), saved[0].numel());
            TORCH_CHECK(ret == 0, "relu_grad failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::relu_grad<float>(
                guard.dc.ctx, cptr<float>(saved[0]), cptr<float>(saved[1]),
                cptr<float>(grad), mptr<float>(dx), saved[0].numel());
            TORCH_CHECK(ret == 0, "relu_grad failed: ret=", ret);
        }
        return {dx};
    }
};

class MmFunction : public Function<MmFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor mat2) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::mm(self, mat2);
        ctx->save_for_backward({self, mat2});
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& a = saved[0];
        const at::Tensor& b = saved[1];
        at::Tensor grad = grad_outputs[0].contiguous();
        const int64_t m = a.size(0), k = a.size(1), n = b.size(1);
        auto ga = at::empty({m, k}, a.options());
        auto gb = at::empty({k, n}, b.options());
        ContextGuard guard(a.device().index());
        if (a.scalar_type() == at::kHalf) {
            int ret = baidu::xpu::api::fc<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(grad), cptr<float16>(b), mptr<float16>(ga),
                m, k, n, false, true, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "mm backward grad_a failed: ret=", ret);
            ret = baidu::xpu::api::fc<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(a), cptr<float16>(grad), mptr<float16>(gb),
                k, n, m, true, false, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "mm backward grad_b failed: ret=", ret);
        } else {
            int ret = baidu::xpu::api::fc<float, float, float, float>(
                guard.dc.ctx, cptr<float>(grad), cptr<float>(b), mptr<float>(ga),
                m, k, n, false, true, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "mm backward grad_a failed: ret=", ret);
            ret = baidu::xpu::api::fc<float, float, float, float>(
                guard.dc.ctx, cptr<float>(a), cptr<float>(grad), mptr<float>(gb),
                k, n, m, true, false, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "mm backward grad_b failed: ret=", ret);
        }
        return {ga, gb};
    }
};

class LinearFunction : public Function<LinearFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input, at::Tensor weight,
                              std::optional<at::Tensor> bias) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::linear(input, weight, bias);
        ctx->save_for_backward({input, weight, bias.has_value() ? *bias : at::Tensor()});
        ctx->saved_data["has_bias"] = bias.has_value();
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& input = saved[0];
        const at::Tensor& weight = saved[1];
        const at::Tensor& bias = saved[2];
        at::Tensor grad = grad_outputs[0].contiguous();
        const int64_t m = input.size(0), k = input.size(1), n = weight.size(0);
        auto gi = at::empty({m, k}, input.options());
        auto gw = at::empty({n, k}, weight.options());
        at::Tensor gb;
        ContextGuard guard(input.device().index());
        const bool half = input.scalar_type() == at::kHalf;
        if (half) {
            int ret = baidu::xpu::api::fc<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(grad), cptr<float16>(weight), mptr<float16>(gi),
                m, k, n, false, true, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "linear grad_input failed: ret=", ret);
            ret = baidu::xpu::api::fc<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(grad), mptr<float16>(gw),
                k, n, m, true, false, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "linear grad_weight failed: ret=", ret);
            if (ctx->saved_data["has_bias"].toBool()) {
                gb = at::empty({n}, weight.options());
                ret = baidu::xpu::api::reduce_sum<float16>(
                    guard.dc.ctx, cptr<float16>(grad), mptr<float16>(gb),
                    grad.sizes().vec(), {0});
                TORCH_CHECK(ret == 0, "linear grad_bias failed: ret=", ret);
            }
        } else {
            int ret = baidu::xpu::api::fc<float, float, float, float>(
                guard.dc.ctx, cptr<float>(grad), cptr<float>(weight), mptr<float>(gi),
                m, k, n, false, true, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "linear grad_input failed: ret=", ret);
            ret = baidu::xpu::api::fc<float, float, float, float>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(grad), mptr<float>(gw),
                k, n, m, true, false, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "linear grad_weight failed: ret=", ret);
            if (ctx->saved_data["has_bias"].toBool()) {
                gb = at::empty({n}, weight.options());
                ret = baidu::xpu::api::reduce_sum<float>(
                    guard.dc.ctx, cptr<float>(grad), mptr<float>(gb),
                    grad.sizes().vec(), {0});
                TORCH_CHECK(ret == 0, "linear grad_bias failed: ret=", ret);
            }
        }
        return {gi, gw, gb};
    }
};

class AddmmFunction : public Function<AddmmFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor mat1,
                              at::Tensor mat2, at::Scalar beta, at::Scalar alpha) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::addmm(self, mat1, mat2, beta, alpha);
        ctx->save_for_backward({self, mat1, mat2});
        ctx->saved_data["beta"] = beta.toDouble();
        ctx->saved_data["alpha"] = alpha.toDouble();
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& self = saved[0];
        const at::Tensor& mat1 = saved[1];
        const at::Tensor& mat2 = saved[2];
        at::Tensor grad = grad_outputs[0].contiguous();
        const double beta = ctx->saved_data["beta"].toDouble();
        const double alpha = ctx->saved_data["alpha"].toDouble();
        const at::Tensor mat2c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
        const int64_t m = mat1.size(0), k = mat1.size(1), n = mat2c.size(1);
        auto ga = at::empty({m, k}, mat1.options());
        auto gb = at::empty({k, n}, mat2c.options());
        at::Tensor gs;
        ContextGuard guard(mat1.device().index());
        if (mat1.scalar_type() == at::kHalf) {
            int ret = baidu::xpu::api::fc<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(grad), cptr<float16>(mat2c), mptr<float16>(ga),
                m, k, n, false, true, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "addmm grad_mat1 failed: ret=", ret);
            ret = baidu::xpu::api::fc<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(mat1), cptr<float16>(grad), mptr<float16>(gb),
                k, n, m, true, false, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "addmm grad_mat2 failed: ret=", ret);
            if (alpha != 1.0) {
                ga = scale_impl<float16>(ga, alpha, 0.0);
                gb = scale_impl<float16>(gb, alpha, 0.0);
            }
            if (self.dim() == 1 && self.size(0) == n) {
                gs = at::empty({n}, self.options());
                ret = baidu::xpu::api::reduce_sum<float16>(
                    guard.dc.ctx, cptr<float16>(grad), mptr<float16>(gs),
                    grad.sizes().vec(), {0});
                TORCH_CHECK(ret == 0, "addmm grad_bias failed: ret=", ret);
                if (beta != 1.0) gs = scale_impl<float16>(gs, beta, 0.0);
            } else if (self.sizes().equals(grad.sizes())) {
                gs = beta == 1.0 ? grad : scale_impl<float16>(grad, beta, 0.0);
            }
        } else {
            int ret = baidu::xpu::api::fc<float, float, float, float>(
                guard.dc.ctx, cptr<float>(grad), cptr<float>(mat2c), mptr<float>(ga),
                m, k, n, false, true, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "addmm grad_mat1 failed: ret=", ret);
            ret = baidu::xpu::api::fc<float, float, float, float>(
                guard.dc.ctx, cptr<float>(mat1), cptr<float>(grad), mptr<float>(gb),
                k, n, m, true, false, nullptr, nullptr, nullptr);
            TORCH_CHECK(ret == 0, "addmm grad_mat2 failed: ret=", ret);
            if (alpha != 1.0) {
                ga = scale_impl<float>(ga, alpha, 0.0);
                gb = scale_impl<float>(gb, alpha, 0.0);
            }
            if (self.dim() == 1 && self.size(0) == n) {
                gs = at::empty({n}, self.options());
                ret = baidu::xpu::api::reduce_sum<float>(
                    guard.dc.ctx, cptr<float>(grad), mptr<float>(gs),
                    grad.sizes().vec(), {0});
                TORCH_CHECK(ret == 0, "addmm grad_bias failed: ret=", ret);
                if (beta != 1.0) gs = scale_impl<float>(gs, beta, 0.0);
            } else if (self.sizes().equals(grad.sizes())) {
                gs = beta == 1.0 ? grad : scale_impl<float>(grad, beta, 0.0);
            }
        }
        return {gs, ga, gb, at::Tensor(), at::Tensor()};
    }
};

class MseLossFunction : public Function<MseLossFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor target,
                              int64_t reduction) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::mse_loss(self, target, reduction);
        ctx->save_for_backward({self, target});
        ctx->saved_data["reduction"] = reduction;
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& self = saved[0];
        const at::Tensor& target = saved[1];
        at::Tensor grad = grad_outputs[0].contiguous();
        const int64_t reduction = ctx->saved_data["reduction"].toInt();
        const double grad_value = grad.numel() == 1
            ? grad.detach().cpu().item<double>()
            : 1.0;
        double factor = 2.0 * grad_value;
        if (reduction == 1) {
            factor /= static_cast<double>(self.numel());
        }
        auto dx = at::empty_like(self);
        auto dt = at::empty_like(self);
        ContextGuard guard(self.device().index());
        if (self.scalar_type() == at::kHalf) {
            auto diff = at::empty_like(self);
            int ret = baidu::xpu::api::sub<float16>(guard.dc.ctx, cptr<float16>(self),
                                                    cptr<float16>(target), mptr<float16>(diff),
                                                    self.numel());
            TORCH_CHECK(ret == 0, "mse backward sub failed: ret=", ret);
            ret = baidu::xpu::api::scale<float16>(guard.dc.ctx, cptr<float16>(diff),
                                                  mptr<float16>(dx), self.numel(), false,
                                                  static_cast<float>(factor), 0.0f);
            TORCH_CHECK(ret == 0, "mse backward scale failed: ret=", ret);
            ret = baidu::xpu::api::scale<float16>(guard.dc.ctx, cptr<float16>(dx),
                                                  mptr<float16>(dt), self.numel(), false,
                                                  -1.0f, 0.0f);
            TORCH_CHECK(ret == 0, "mse backward neg failed: ret=", ret);
        } else {
            auto diff = at::empty_like(self);
            int ret = baidu::xpu::api::sub<float>(guard.dc.ctx, cptr<float>(self),
                                                  cptr<float>(target), mptr<float>(diff),
                                                  self.numel());
            TORCH_CHECK(ret == 0, "mse backward sub failed: ret=", ret);
            ret = baidu::xpu::api::scale<float>(guard.dc.ctx, cptr<float>(diff),
                                                mptr<float>(dx), self.numel(), false,
                                                static_cast<float>(factor), 0.0f);
            TORCH_CHECK(ret == 0, "mse backward scale failed: ret=", ret);
            ret = baidu::xpu::api::scale<float>(guard.dc.ctx, cptr<float>(dx),
                                                mptr<float>(dt), self.numel(), false,
                                                -1.0f, 0.0f);
            TORCH_CHECK(ret == 0, "mse backward neg failed: ret=", ret);
        }
        return {dx, dt, at::Tensor()};
    }
};


// ---------------------------------------------------------------------------
// CNN / Transformer backward (rough but explicit)
// ---------------------------------------------------------------------------

std::vector<int64_t> ivec(at::IntArrayRef value) {
    return std::vector<int64_t>(value.begin(), value.end());
}

template <typename T>
at::Tensor mul_scalar_impl(const at::Tensor& self, const at::Scalar& other) {
    check_tensor(self, "mul.Scalar");
    auto out = at::empty_like(self);
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::scale<T>(
        guard.dc.ctx, cptr<T>(self), mptr<T>(out), self.numel(),
        /*bias_after_scale=*/false, static_cast<float>(other.toDouble()), 0.0f);
    TORCH_CHECK(ret == 0, "xdnn::scale (mul.Scalar) failed: ret=", ret);
    return out;
}

at::Tensor mul_scalar(const at::Tensor& self, const at::Scalar& other) {
    if (self.scalar_type() == at::kHalf) return mul_scalar_impl<float16>(self, other);
    if (self.scalar_type() == at::kFloat) return mul_scalar_impl<float>(self, other);
    TORCH_CHECK(false, "kunlun mul.Scalar only supports float16/float32");
}

class MulScalarFunction : public Function<MulScalarFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Scalar other) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::mul(self, other);
        ctx->saved_data["factor"] = other.toDouble();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto grad = grad_outputs[0].contiguous();
        const double factor = ctx->saved_data["factor"].toDouble();
        if (grad.scalar_type() == at::kHalf) {
            return {scale_impl<float16>(grad, factor, 0.0), at::Tensor()};
        }
        return {scale_impl<float>(grad, factor, 0.0), at::Tensor()};
    }
};

class Conv2dFunction : public Function<Conv2dFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input, at::Tensor weight,
                              std::optional<at::Tensor> bias, std::vector<int64_t> stride,
                              std::vector<int64_t> padding, std::vector<int64_t> dilation,
                              int64_t groups) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::conv2d(input, weight, bias, stride, padding, dilation, groups);
        ctx->save_for_backward({input, weight, bias.has_value() ? *bias : at::Tensor()});
        ctx->saved_data["has_bias"] = bias.has_value();
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["groups"] = groups;
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& input = saved[0];
        const at::Tensor& weight = saved[1];
        at::Tensor grad = grad_outputs[0].contiguous();
        auto stride = ctx->saved_data["stride"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        auto dilation = ctx->saved_data["dilation"].toIntVector();
        const int64_t groups = ctx->saved_data["groups"].toInt();
        const int64_t n = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
        const int64_t f = weight.size(0);
        auto dx = at::empty_like(input);
        auto dw = at::empty_like(weight);
        const std::vector<int64_t> ksize{weight.size(2), weight.size(3)};
        ContextGuard guard(input.device().index());
        if (input.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::conv2d_grad<float16, float16, float16, short>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(weight), cptr<float16>(grad),
                mptr<float16>(dx), mptr<float16>(dw), n, c, h, w, f, ksize, stride, padding,
                dilation, groups, nullptr, nullptr, nullptr, nullptr, nullptr, true);
            TORCH_CHECK(ret == 0, "xdnn::conv2d_grad failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::conv2d_grad<float, float, float, float>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(weight), cptr<float>(grad),
                mptr<float>(dx), mptr<float>(dw), n, c, h, w, f, ksize, stride, padding,
                dilation, groups, nullptr, nullptr, nullptr, nullptr, nullptr, true);
            TORCH_CHECK(ret == 0, "xdnn::conv2d_grad failed: ret=", ret);
        }
        at::Tensor gb;
        if (ctx->saved_data["has_bias"].toBool()) {
            gb = at::empty({f}, weight.options());
            const std::vector<int64_t> dims{0, 2, 3};
            if (input.scalar_type() == at::kHalf) {
                const int ret = baidu::xpu::api::reduce_sum<float16>(
                    guard.dc.ctx, cptr<float16>(grad), mptr<float16>(gb), grad.sizes().vec(), dims);
                TORCH_CHECK(ret == 0, "conv2d bias grad failed: ret=", ret);
            } else {
                const int ret = baidu::xpu::api::reduce_sum<float>(
                    guard.dc.ctx, cptr<float>(grad), mptr<float>(gb), grad.sizes().vec(), dims);
                TORCH_CHECK(ret == 0, "conv2d bias grad failed: ret=", ret);
            }
        }
        return {dx, dw, gb, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class MaxPool2dFunction : public Function<MaxPool2dFunction> {
public:
    static variable_list forward(
        AutogradContext* ctx, at::Tensor input, std::vector<int64_t> kernel,
        std::vector<int64_t> stride, std::vector<int64_t> padding,
        std::vector<int64_t> dilation, bool ceil_mode) {
        at::AutoDispatchBelowAutograd guard;
        auto [out, indices] = at::max_pool2d_with_indices(
            input, kernel, stride, padding, dilation, ceil_mode);
        ctx->save_for_backward({input, out, indices});
        ctx->saved_data["kernel"] = kernel;
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        return {out, indices};
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& input = saved[0];
        const at::Tensor& out = saved[1];
        const at::Tensor& indices = saved[2];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(input);
        auto kernel = ctx->saved_data["kernel"].toIntVector();
        auto stride = ctx->saved_data["stride"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        ContextGuard guard(input.device().index());
        if (input.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::max_pool2d_grad<float16, int>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(out), cptr<int>(indices),
                cptr<float16>(grad), mptr<float16>(dx), input.size(0), input.size(1),
                input.size(2), input.size(3), kernel, stride, padding, true);
            TORCH_CHECK(ret == 0, "max_pool2d_grad failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::max_pool2d_grad<float, int>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(out), cptr<int>(indices),
                cptr<float>(grad), mptr<float>(dx), input.size(0), input.size(1),
                input.size(2), input.size(3), kernel, stride, padding, true);
            TORCH_CHECK(ret == 0, "max_pool2d_grad failed: ret=", ret);
        }
        return {dx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class AdaptiveAvgPool2dFunction : public Function<AdaptiveAvgPool2dFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input,
                              std::vector<int64_t> output_size) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::adaptive_avg_pool2d(input, output_size);
        ctx->save_for_backward({input});
        ctx->saved_data["out_h"] = out.size(2);
        ctx->saved_data["out_w"] = out.size(3);
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& input = saved[0];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(input);
        const int64_t oh = ctx->saved_data["out_h"].toInt();
        const int64_t ow = ctx->saved_data["out_w"].toInt();
        ContextGuard guard(input.device().index());
        if (input.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::adaptive_avg_pool2d_grad<float16>(
                guard.dc.ctx, cptr<float16>(grad), mptr<float16>(dx), input.size(0),
                input.size(1), input.size(2), input.size(3), oh, ow, true);
            TORCH_CHECK(ret == 0, "adaptive_avg_pool2d_grad failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::adaptive_avg_pool2d_grad<float>(
                guard.dc.ctx, cptr<float>(grad), mptr<float>(dx), input.size(0),
                input.size(1), input.size(2), input.size(3), oh, ow, true);
            TORCH_CHECK(ret == 0, "adaptive_avg_pool2d_grad failed: ret=", ret);
        }
        return {dx, at::Tensor()};
    }
};

class SoftmaxFunction : public Function<SoftmaxFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, int64_t dim,
                              std::optional<at::ScalarType> dtype) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::softmax(self, dim, dtype);
        ctx->save_for_backward({self, out});
        ctx->saved_data["dim"] = dim;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& self = saved[0];
        const at::Tensor& out = saved[1];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(self);
        const int64_t dim = ctx->saved_data["dim"].toInt();
        const int64_t axis = dim < 0 ? dim + self.dim() : dim;
        ContextGuard guard(self.device().index());
        if (self.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::softmax_grad<float16>(
                guard.dc.ctx, cptr<float16>(out), cptr<float16>(grad), mptr<float16>(dx),
                self.sizes().vec(), axis);
            TORCH_CHECK(ret == 0, "softmax_grad failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::softmax_grad<float>(
                guard.dc.ctx, cptr<float>(out), cptr<float>(grad), mptr<float>(dx),
                self.sizes().vec(), axis);
            TORCH_CHECK(ret == 0, "softmax_grad failed: ret=", ret);
        }
        return {dx, at::Tensor(), at::Tensor()};
    }
};

class GeluFunction : public Function<GeluFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, std::string approximate) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::gelu(self, approximate);
        ctx->save_for_backward({self, out});
        ctx->saved_data["approximate"] = approximate;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& self = saved[0];
        const at::Tensor& out = saved[1];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::empty_like(self);
        ContextGuard guard(self.device().index());
        const bool tanh = ctx->saved_data["approximate"].toStringRef() == "tanh";
        if (self.scalar_type() == at::kHalf) {
            const int ret = tanh
                ? baidu::xpu::api::approximate_gelu_grad<float16>(
                      guard.dc.ctx, cptr<float16>(self), cptr<float16>(out),
                      cptr<float16>(grad), mptr<float16>(dx), self.numel())
                : baidu::xpu::api::gelu_grad<float16>(
                      guard.dc.ctx, cptr<float16>(self), cptr<float16>(out),
                      cptr<float16>(grad), mptr<float16>(dx), self.numel());
            TORCH_CHECK(ret == 0, "gelu_grad failed: ret=", ret);
        } else {
            const int ret = tanh
                ? baidu::xpu::api::approximate_gelu_grad<float>(
                      guard.dc.ctx, cptr<float>(self), cptr<float>(out),
                      cptr<float>(grad), mptr<float>(dx), self.numel())
                : baidu::xpu::api::gelu_grad<float>(
                      guard.dc.ctx, cptr<float>(self), cptr<float>(out),
                      cptr<float>(grad), mptr<float>(dx), self.numel());
            TORCH_CHECK(ret == 0, "gelu_grad failed: ret=", ret);
        }
        return {dx, at::Tensor()};
    }
};

class EmbeddingFunction : public Function<EmbeddingFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor weight, at::Tensor indices,
                              int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
        ctx->save_for_backward({weight, indices});
        ctx->saved_data["padding_idx"] = padding_idx;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& weight = saved[0];
        const at::Tensor& indices = saved[1];
        auto grad = grad_outputs[0].contiguous();
        auto dx = at::zeros(weight.sizes(), weight.options());
        const int64_t padding_idx = ctx->saved_data["padding_idx"].toInt();
        ContextGuard guard(weight.device().index());
        if (indices.scalar_type() == at::kInt) {
            if (weight.scalar_type() == at::kHalf) {
                const int ret = baidu::xpu::api::embedding_grad<float16, int>(
                    guard.dc.ctx, cptr<float16>(grad), cptr<int>(indices), mptr<float16>(dx),
                    weight.size(0), indices.numel(), weight.size(1), padding_idx);
                TORCH_CHECK(ret == 0, "embedding_grad failed: ret=", ret);
            } else {
                const int ret = baidu::xpu::api::embedding_grad<float, int>(
                    guard.dc.ctx, cptr<float>(grad), cptr<int>(indices), mptr<float>(dx),
                    weight.size(0), indices.numel(), weight.size(1), padding_idx);
                TORCH_CHECK(ret == 0, "embedding_grad failed: ret=", ret);
            }
        } else {
            if (weight.scalar_type() == at::kHalf) {
                const int ret = baidu::xpu::api::embedding_grad<float16, int64_t>(
                    guard.dc.ctx, cptr<float16>(grad), cptr<int64_t>(indices), mptr<float16>(dx),
                    weight.size(0), indices.numel(), weight.size(1), padding_idx);
                TORCH_CHECK(ret == 0, "embedding_grad failed: ret=", ret);
            } else {
                const int ret = baidu::xpu::api::embedding_grad<float, int64_t>(
                    guard.dc.ctx, cptr<float>(grad), cptr<int64_t>(indices), mptr<float>(dx),
                    weight.size(0), indices.numel(), weight.size(1), padding_idx);
                TORCH_CHECK(ret == 0, "embedding_grad failed: ret=", ret);
            }
        }
        return {dx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class CatFunction : public Function<CatFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, std::vector<at::Tensor> tensors, int64_t dim) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::cat(tensors, dim);
        std::vector<int64_t> sizes;
        for (const auto& t : tensors) sizes.push_back(t.size(dim));
        ctx->saved_data["sizes"] = sizes;
        ctx->saved_data["dim"] = dim;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto grad = grad_outputs[0];
        auto sizes = ctx->saved_data["sizes"].toIntVector();
        const int64_t dim = ctx->saved_data["dim"].toInt();
        variable_list grads;
        int64_t offset = 0;
        for (int64_t size : sizes) {
            grads.push_back(grad.slice(dim, offset, offset + size));
            offset += size;
        }
        return grads;
    }
};

class BmmFunction : public Function<BmmFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor mat2) {
        at::AutoDispatchBelowAutograd guard;
        auto out = at::bmm(self, mat2);
        ctx->save_for_backward({self, mat2});
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& a = saved[0];
        const at::Tensor& b = saved[1];
        auto grad = grad_outputs[0].contiguous();
        auto ga = at::empty_like(a);
        auto gb = at::empty_like(b);
        const int64_t batch = a.size(0), m = a.size(1), k = a.size(2), n = b.size(2);
        ContextGuard guard(a.device().index());
        for (int64_t i = 0; i < batch; ++i) {
            if (a.scalar_type() == at::kHalf) {
                int ret = baidu::xpu::api::fc<float16, float16, float16, short>(
                    guard.dc.ctx, cptr<float16>(grad) + i * m * n, cptr<float16>(b) + i * k * n,
                    mptr<float16>(ga) + i * m * k, m, k, n, false, true, nullptr, nullptr, nullptr);
                TORCH_CHECK(ret == 0, "bmm grad_a failed: ret=", ret);
                ret = baidu::xpu::api::fc<float16, float16, float16, short>(
                    guard.dc.ctx, cptr<float16>(a) + i * m * k, cptr<float16>(grad) + i * m * n,
                    mptr<float16>(gb) + i * k * n, k, n, m, true, false, nullptr, nullptr, nullptr);
                TORCH_CHECK(ret == 0, "bmm grad_b failed: ret=", ret);
            } else {
                int ret = baidu::xpu::api::fc<float, float, float, float>(
                    guard.dc.ctx, cptr<float>(grad) + i * m * n, cptr<float>(b) + i * k * n,
                    mptr<float>(ga) + i * m * k, m, k, n, false, true, nullptr, nullptr, nullptr);
                TORCH_CHECK(ret == 0, "bmm grad_a failed: ret=", ret);
                ret = baidu::xpu::api::fc<float, float, float, float>(
                    guard.dc.ctx, cptr<float>(a) + i * m * k, cptr<float>(grad) + i * m * n,
                    mptr<float>(gb) + i * k * n, k, n, m, true, false, nullptr, nullptr, nullptr);
                TORCH_CHECK(ret == 0, "bmm grad_b failed: ret=", ret);
            }
        }
        return {ga, gb};
    }
};

at::Tensor mul_scalar_autograd(const at::Tensor& self, const at::Scalar& other) {
    return MulScalarFunction::apply(self, other);
}

at::Tensor conv2d_autograd(const at::Tensor& input, const at::Tensor& weight,
                           const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    return Conv2dFunction::apply(input, weight, bias, ivec(stride), ivec(padding),
                                 ivec(dilation), groups);
}

std::tuple<at::Tensor, at::Tensor> max_pool2d_autograd(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    auto out = MaxPool2dFunction::apply(self, ivec(kernel_size), ivec(stride), ivec(padding),
                                        ivec(dilation), ceil_mode);
    return {out[0], out[1]};
}

at::Tensor adaptive_avg_pool2d_autograd(const at::Tensor& self, at::IntArrayRef output_size) {
    return AdaptiveAvgPool2dFunction::apply(self, ivec(output_size));
}

at::Tensor softmax_autograd(const at::Tensor& self, int64_t dim,
                            std::optional<at::ScalarType> dtype) {
    return SoftmaxFunction::apply(self, dim, dtype);
}

at::Tensor _softmax_autograd(const at::Tensor& self, int64_t dim, bool half_to_float) {
    TORCH_CHECK(!half_to_float, "kunlun _softmax half_to_float=True is unsupported");
    return SoftmaxFunction::apply(self, dim, std::nullopt);
}

at::Tensor gelu_autograd(const at::Tensor& self, c10::string_view approximate) {
    return GeluFunction::apply(self, std::string(approximate));
}

at::Tensor embedding_autograd(const at::Tensor& weight, const at::Tensor& indices,
                              int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    return EmbeddingFunction::apply(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

at::Tensor cat_autograd(at::ITensorListRef tensors, int64_t dim) {
    std::vector<at::Tensor> list;
    for (const auto& t : tensors) list.push_back(t);
    return CatFunction::apply(list, dim);
}

at::Tensor bmm_autograd(const at::Tensor& self, const at::Tensor& mat2) {
    return BmmFunction::apply(self, mat2);
}


template <typename T, typename TW>
at::Tensor layer_norm_forward_impl(AutogradContext* ctx, const at::Tensor& input,
                                   int64_t m, int64_t n, double eps,
                                   const std::optional<at::Tensor>& weight,
                                   const std::optional<at::Tensor>& bias,
                                   at::Tensor gamma, at::Tensor beta) {
    auto out = at::empty_like(input);
    auto mean = at::empty({m}, input.options().dtype(at::kFloat));
    auto var = at::empty({m}, input.options().dtype(at::kFloat));
    ContextGuard guard(input.device().index());
    const int ret = baidu::xpu::api::layer_norm<T, TW>(
        guard.dc.ctx, cptr<T>(input), mptr<T>(out), m, n, static_cast<float>(eps),
        cptr<TW>(gamma), cptr<TW>(beta), mptr<float>(mean), mptr<float>(var),
        /*is_rstd=*/false);
    TORCH_CHECK(ret == 0, "xdnn::layer_norm failed: ret=", ret);
    ctx->save_for_backward({input, gamma, beta, mean, var});
    ctx->saved_data["has_weight"] = weight.has_value();
    ctx->saved_data["has_bias"] = bias.has_value();
    ctx->saved_data["m"] = m;
    ctx->saved_data["n"] = n;
    ctx->saved_data["eps"] = eps;
    return out;
}

class LayerNormFunction : public Function<LayerNormFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input,
                              std::vector<int64_t> normalized_shape,
                              std::optional<at::Tensor> weight,
                              std::optional<at::Tensor> bias, double eps, bool) {
        at::AutoDispatchBelowAutograd guard;
        int64_t n = 1;
        for (int64_t s : normalized_shape) n *= s;
        TORCH_CHECK(n > 0 && input.numel() % n == 0, "layer_norm shape mismatch");
        const int64_t m = input.numel() / n;
        const bool weight_half = weight.has_value() && weight->scalar_type() == at::kHalf;
        at::Tensor gamma;
        at::Tensor beta;
        if (weight.has_value()) {
            gamma = weight->contiguous();
            TORCH_CHECK(gamma.numel() == n, "layer_norm weight size mismatch");
        } else {
            gamma = at::ones({n}, input.options().dtype(at::kFloat));
        }
        if (bias.has_value()) {
            beta = bias->contiguous();
            TORCH_CHECK(beta.numel() == n, "layer_norm bias size mismatch");
        } else {
            beta = at::zeros({n}, input.options().dtype(at::kFloat));
        }
        if (input.scalar_type() == at::kHalf) {
            if (weight_half) {
                return layer_norm_forward_impl<float16, float16>(
                    ctx, input, m, n, eps, weight, bias, gamma, beta);
            }
            return layer_norm_forward_impl<float16, float>(
                ctx, input, m, n, eps, weight, bias, gamma, beta);
        }
        return layer_norm_forward_impl<float, float>(
            ctx, input, m, n, eps, weight, bias, gamma, beta);
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& input = saved[0];
        const at::Tensor& gamma = saved[1];
        const at::Tensor& beta = saved[2];
        const at::Tensor& mean = saved[3];
        const at::Tensor& var = saved[4];
        auto grad = grad_outputs[0].contiguous();
        const int64_t m = ctx->saved_data["m"].toInt();
        const int64_t n = ctx->saved_data["n"].toInt();
        const double eps = ctx->saved_data["eps"].toDouble();
        auto dx = at::empty_like(input);
        auto dgamma = at::empty_like(gamma);
        auto dbeta = at::empty_like(beta);
        ContextGuard guard(input.device().index());
        if (input.scalar_type() == at::kHalf && gamma.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::layer_norm_grad<float16, float16>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(grad), mptr<float16>(dx),
                m, n, static_cast<float>(eps), cptr<float16>(gamma), cptr<float>(mean),
                cptr<float>(var), mptr<float16>(dgamma), mptr<float16>(dbeta));
            TORCH_CHECK(ret == 0, "layer_norm_grad failed: ret=", ret);
        } else if (input.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::layer_norm_grad<float16, float>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(grad), mptr<float16>(dx),
                m, n, static_cast<float>(eps), cptr<float>(gamma), cptr<float>(mean),
                cptr<float>(var), mptr<float>(dgamma), mptr<float>(dbeta));
            TORCH_CHECK(ret == 0, "layer_norm_grad failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::layer_norm_grad<float, float>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(grad), mptr<float>(dx),
                m, n, static_cast<float>(eps), cptr<float>(gamma), cptr<float>(mean),
                cptr<float>(var), mptr<float>(dgamma), mptr<float>(dbeta));
            TORCH_CHECK(ret == 0, "layer_norm_grad failed: ret=", ret);
        }
        return {dx,
                at::Tensor(),
                ctx->saved_data["has_weight"].toBool() ? dgamma : at::Tensor(),
                ctx->saved_data["has_bias"].toBool() ? dbeta : at::Tensor(),
                at::Tensor(),
                at::Tensor()};
    }
};

at::Tensor layer_norm_autograd(const at::Tensor& input, at::IntArrayRef normalized_shape,
                               const std::optional<at::Tensor>& weight,
                               const std::optional<at::Tensor>& bias, double eps, bool cudnn) {
    return LayerNormFunction::apply(input, ivec(normalized_shape), weight, bias, eps, cudnn);
}


at::Tensor to_float_tensor(const at::Tensor& t, const char* what) {
    check_tensor(t, what);
    if (t.scalar_type() == at::kFloat) return t;
    TORCH_CHECK(t.scalar_type() == at::kHalf, what, " must be float16/float32");
    auto out = at::empty(t.sizes(), t.options().dtype(at::kFloat));
    ContextGuard guard(t.device().index());
    const int ret = baidu::xpu::api::cast<float16, float>(
        guard.dc.ctx, cptr<float16>(t), mptr<float>(out), t.numel());
    TORCH_CHECK(ret == 0, what, " cast failed: ret=", ret);
    return out;
}

class BatchNormFunction : public Function<BatchNormFunction> {
public:
    static variable_list forward(AutogradContext* ctx, at::Tensor input,
                                 std::optional<at::Tensor> weight,
                                 std::optional<at::Tensor> bias,
                                 std::optional<at::Tensor> running_mean,
                                 std::optional<at::Tensor> running_var,
                                 bool training, double momentum, double eps) {
        if (!training) {
            at::AutoDispatchBelowAutograd guard;
            auto values = at::native_batch_norm(input, weight, bias, running_mean, running_var,
                                                false, momentum, eps);
            return {std::get<0>(values), std::get<1>(values), std::get<2>(values)};
        }
        check_tensor(input, "native_batch_norm");
        TORCH_CHECK(input.dim() == 4, "native_batch_norm training requires 4-D input");
        const int64_t n = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
        at::Tensor scale = weight.has_value()
            ? to_float_tensor(*weight, "native_batch_norm weight")
            : at::ones({c}, input.options().dtype(at::kFloat));
        at::Tensor offset = bias.has_value()
            ? to_float_tensor(*bias, "native_batch_norm bias")
            : at::zeros({c}, input.options().dtype(at::kFloat));
        at::Tensor mean = running_mean.has_value()
            ? to_float_tensor(*running_mean, "native_batch_norm running_mean")
            : at::zeros({c}, input.options().dtype(at::kFloat));
        at::Tensor var = running_var.has_value()
            ? to_float_tensor(*running_var, "native_batch_norm running_var")
            : at::ones({c}, input.options().dtype(at::kFloat));
        auto out = at::empty_like(input);
        auto batch_mean = at::empty({c}, input.options().dtype(at::kFloat));
        auto batch_invstd = at::empty({c}, input.options().dtype(at::kFloat));
        ContextGuard guard(input.device().index());
        int ret = 0;
        if (input.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::batch_norm<float16>(
                guard.dc.ctx, cptr<float16>(input), mptr<float16>(out), n, c, h, w,
                static_cast<float>(eps), static_cast<float>(momentum),
                cptr<float>(scale), cptr<float>(offset), mptr<float>(batch_mean),
                mptr<float>(batch_invstd), mptr<float>(mean), mptr<float>(var),
                /*is_nchw=*/true, /*unbiased_global_var=*/false);
        } else {
            ret = baidu::xpu::api::batch_norm<float>(
                guard.dc.ctx, cptr<float>(input), mptr<float>(out), n, c, h, w,
                static_cast<float>(eps), static_cast<float>(momentum),
                cptr<float>(scale), cptr<float>(offset), mptr<float>(batch_mean),
                mptr<float>(batch_invstd), mptr<float>(mean), mptr<float>(var),
                /*is_nchw=*/true, /*unbiased_global_var=*/false);
        }
        TORCH_CHECK(ret == 0, "xdnn::batch_norm failed: ret=", ret);
        ctx->save_for_backward({input, scale, offset, batch_mean, batch_invstd});
        ctx->saved_data["n"] = n;
        ctx->saved_data["c"] = c;
        ctx->saved_data["h"] = h;
        ctx->saved_data["w"] = w;
        ctx->saved_data["has_weight"] = weight.has_value();
        ctx->saved_data["has_bias"] = bias.has_value();
        ctx->saved_data["weight_half"] = weight.has_value() && weight->scalar_type() == at::kHalf;
        return {out, batch_mean, batch_invstd};
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& input = saved[0];
        const at::Tensor& scale = saved[1];
        const at::Tensor& mean = saved[3];
        const at::Tensor& invstd = saved[4];
        auto grad = grad_outputs[0].contiguous();
        const int64_t n = ctx->saved_data["n"].toInt();
        const int64_t c = ctx->saved_data["c"].toInt();
        const int64_t h = ctx->saved_data["h"].toInt();
        const int64_t w = ctx->saved_data["w"].toInt();
        auto dx = at::empty_like(input);
        auto dscale = at::empty({c}, input.options().dtype(at::kFloat));
        auto dbias = at::empty({c}, input.options().dtype(at::kFloat));
        ContextGuard guard(input.device().index());
        if (input.scalar_type() == at::kHalf) {
            const int ret = baidu::xpu::api::batch_norm_grad<float16>(
                guard.dc.ctx, cptr<float16>(input), cptr<float16>(grad), mptr<float16>(dx),
                n, c, h, w, cptr<float>(scale), cptr<float>(mean), cptr<float>(invstd),
                mptr<float>(dscale), mptr<float>(dbias), true);
            TORCH_CHECK(ret == 0, "batch_norm_grad failed: ret=", ret);
        } else {
            const int ret = baidu::xpu::api::batch_norm_grad<float>(
                guard.dc.ctx, cptr<float>(input), cptr<float>(grad), mptr<float>(dx),
                n, c, h, w, cptr<float>(scale), cptr<float>(mean), cptr<float>(invstd),
                mptr<float>(dscale), mptr<float>(dbias), true);
            TORCH_CHECK(ret == 0, "batch_norm_grad failed: ret=", ret);
        }
        at::Tensor dweight;
        at::Tensor dbias_out;
        if (ctx->saved_data["has_weight"].toBool()) {
            if (ctx->saved_data["weight_half"].toBool()) {
                dweight = at::empty({c}, input.options());
                int ret = baidu::xpu::api::cast<float, float16>(
                    guard.dc.ctx, cptr<float>(dscale), mptr<float16>(dweight), c);
                TORCH_CHECK(ret == 0, "batch_norm dweight cast failed: ret=", ret);
            } else {
                dweight = dscale;
            }
        }
        if (ctx->saved_data["has_bias"].toBool()) {
            if (ctx->saved_data["weight_half"].toBool()) {
                dbias_out = at::empty({c}, input.options());
                int ret = baidu::xpu::api::cast<float, float16>(
                    guard.dc.ctx, cptr<float>(dbias), mptr<float16>(dbias_out), c);
                TORCH_CHECK(ret == 0, "batch_norm dbias cast failed: ret=", ret);
            } else {
                dbias_out = dbias;
            }
        }
        return {dx, dweight, dbias_out, at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor()};
    }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_autograd(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var, bool training, double momentum, double eps) {
    auto out = BatchNormFunction::apply(input, weight, bias, running_mean, running_var,
                                        training, momentum, eps);
    return {out[0], out[1], out[2]};
}


// CrossEntropyLoss fallback: XDNN's CE kernels are not yet characterized on
// KL1, so compute the loss on the CPU via the existing host-staging copy.
// This is intentionally slow and is only a correctness bridge for training.
class CrossEntropyFunction : public Function<CrossEntropyFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor target,
                              std::optional<at::Tensor> weight, int64_t reduction,
                              c10::SymInt ignore_index, double label_smoothing) {
        TORCH_CHECK(self.device().type() == at::DeviceType::PrivateUse1,
                    "cross_entropy_loss requires a kunlun tensor");
        TORCH_CHECK(self.dim() == 2, "kunlun cross_entropy_loss supports 2-D logits");
        note_cpu_fallback("cross_entropy_loss");
        auto cpu_self = at::empty(self.sizes(), self.options().device(at::kCPU));
        auto cpu_target = at::empty(target.sizes(), target.options().device(at::kCPU));
        cpu_self.copy_(self);
        cpu_target.copy_(target);
        at::Tensor cpu_weight;
        if (weight.has_value()) {
            cpu_weight = at::empty(weight->sizes(), weight->options().device(at::kCPU));
            cpu_weight.copy_(*weight);
        }
        const int64_t ignore = ignore_index.expect_int();
        at::Tensor cpu_loss;
        {
            at::AutoDispatchBelowAutograd guard;
            cpu_loss = at::cross_entropy_loss(
                cpu_self, cpu_target,
                weight.has_value() ? std::optional<at::Tensor>(cpu_weight)
                                   : std::optional<at::Tensor>(),
                reduction, ignore, label_smoothing);
        }
        auto out = at::empty(cpu_loss.sizes(), self.options());
        out.copy_(cpu_loss);
        ctx->save_for_backward({cpu_self, cpu_target,
                                weight.has_value() ? cpu_weight : at::Tensor()});
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["ignore"] = ignore;
        ctx->saved_data["label_smoothing"] = label_smoothing;
        ctx->saved_data["has_weight"] = weight.has_value();
        ctx->saved_data["device"] = self.device().index();
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& cpu_self = saved[0];
        const at::Tensor& cpu_target = saved[1];
        const at::Tensor& cpu_weight = saved[2];
        const int64_t reduction = ctx->saved_data["reduction"].toInt();
        const int64_t ignore = ctx->saved_data["ignore"].toInt();
        const double smoothing = ctx->saved_data["label_smoothing"].toDouble();
        auto grad = grad_outputs[0];
        auto cpu_grad = at::empty(grad.sizes(), grad.options().device(at::kCPU));
        cpu_grad.copy_(grad);

        auto logits = cpu_self.detach().clone();
        logits.requires_grad_(true);
        auto loss = at::cross_entropy_loss(
            logits, cpu_target,
            ctx->saved_data["has_weight"].toBool()
                ? std::optional<at::Tensor>(cpu_weight)
                : std::optional<at::Tensor>(),
            reduction, ignore, smoothing);
        loss.backward(cpu_grad);
        auto cpu_dx = logits.grad();
        const int device = ctx->saved_data["device"].toInt();
        auto dx = at::empty(cpu_dx.sizes(), cpu_dx.options().device(
            c10::Device(c10::DeviceType::PrivateUse1, device)));
        dx.copy_(cpu_dx);
        return {dx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor cross_entropy_loss_autograd(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    c10::SymInt ignore_index, double label_smoothing) {
    return CrossEntropyFunction::apply(self, target, weight, reduction,
                                       ignore_index, label_smoothing);
}

at::Tensor add_autograd(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return AddFunction::apply(self, other, alpha);
}

at::Tensor sub_autograd(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return SubFunction::apply(self, other, alpha);
}

at::Tensor mul_autograd(const at::Tensor& self, const at::Tensor& other) {
    return MulFunction::apply(self, other);
}

at::Tensor relu_autograd(const at::Tensor& self) {
    return ReluFunction::apply(self);
}

at::Tensor mm_autograd(const at::Tensor& self, const at::Tensor& mat2) {
    return MmFunction::apply(self, mat2);
}

at::Tensor linear_autograd(const at::Tensor& input, const at::Tensor& weight,
                           const std::optional<at::Tensor>& bias) {
    return LinearFunction::apply(input, weight, bias);
}

at::Tensor mse_loss_autograd(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
    return MseLossFunction::apply(self, target, reduction);
}

at::Tensor addmm_autograd(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
                          const at::Scalar& beta, const at::Scalar& alpha) {
    return AddmmFunction::apply(self, mat1, mat2, beta, alpha);
}

}  // namespace
}  // namespace torch_kunlun

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("sub.Tensor", TORCH_FN(torch_kunlun::sub));
    m.impl("mul.Tensor", TORCH_FN(torch_kunlun::mul));
    m.impl("add_.Tensor", TORCH_FN(torch_kunlun::add_));
    m.impl("mul_.Tensor", TORCH_FN(torch_kunlun::mul_));
    m.impl("mse_loss", TORCH_FN(torch_kunlun::mse_loss));
    m.impl("addmm", TORCH_FN(torch_kunlun::addmm));
    m.impl("mul.Scalar", TORCH_FN(torch_kunlun::mul_scalar));
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(torch_kunlun::add_autograd));
    m.impl("sub.Tensor", TORCH_FN(torch_kunlun::sub_autograd));
    m.impl("mul.Tensor", TORCH_FN(torch_kunlun::mul_autograd));
    m.impl("relu", TORCH_FN(torch_kunlun::relu_autograd));
    m.impl("mm", TORCH_FN(torch_kunlun::mm_autograd));
    m.impl("linear", TORCH_FN(torch_kunlun::linear_autograd));
    m.impl("mse_loss", TORCH_FN(torch_kunlun::mse_loss_autograd));
    m.impl("addmm", TORCH_FN(torch_kunlun::addmm_autograd));
    m.impl("mul.Scalar", TORCH_FN(torch_kunlun::mul_scalar_autograd));
    m.impl("conv2d", TORCH_FN(torch_kunlun::conv2d_autograd));
    m.impl("max_pool2d_with_indices", TORCH_FN(torch_kunlun::max_pool2d_autograd));
    m.impl("adaptive_avg_pool2d", TORCH_FN(torch_kunlun::adaptive_avg_pool2d_autograd));
    m.impl("softmax.int", TORCH_FN(torch_kunlun::softmax_autograd));
    m.impl("_softmax", TORCH_FN(torch_kunlun::_softmax_autograd));
    m.impl("gelu", TORCH_FN(torch_kunlun::gelu_autograd));
    m.impl("embedding", TORCH_FN(torch_kunlun::embedding_autograd));
    m.impl("cat", TORCH_FN(torch_kunlun::cat_autograd));
    m.impl("bmm", TORCH_FN(torch_kunlun::bmm_autograd));
    m.impl("layer_norm", TORCH_FN(torch_kunlun::layer_norm_autograd));
    m.impl("native_batch_norm", TORCH_FN(torch_kunlun::native_batch_norm_autograd));
    m.impl("cross_entropy_loss", TORCH_FN(torch_kunlun::cross_entropy_loss_autograd));
}
