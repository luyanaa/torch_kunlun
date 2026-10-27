#include "kunlun.h"

#include <ATen/ATen.h>
#include "xpu/refactor/nn_loss.h"
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include <optional>

namespace torch_kunlun {
namespace {

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;

enum LossOp : int64_t {
    NLL = 0,
    BCE = 1,
    BCE_LOGITS = 2,
    L1 = 3,
    SMOOTH_L1 = 4,
    KL_DIV = 5,
    HUBER = 6,
};

at::Tensor to_cpu(const at::Tensor& t) {
    if (!t.defined()) return at::Tensor();
    auto out = at::empty(t.sizes(), t.options().device(at::kCPU));
    out.copy_(t);
    return out;
}

at::Tensor to_device(const at::Tensor& t, const at::Tensor& reference) {
    if (!t.defined()) return at::Tensor();
    auto out = at::empty(t.sizes(), reference.options());
    out.copy_(t);
    return out;
}

at::Tensor run_cpu(const LossOp op, const at::Tensor& self, const at::Tensor& target,
                   const std::optional<at::Tensor>& weight,
                   const std::optional<at::Tensor>& pos_weight,
                   int64_t reduction, int64_t ignore, double extra) {
    switch (op) {
        case NLL:
            return at::nll_loss(self, target, weight, reduction, ignore);
        case BCE:
            return at::binary_cross_entropy(self, target, weight, reduction);
        case BCE_LOGITS:
            return at::binary_cross_entropy_with_logits(self, target, weight,
                                                        pos_weight, reduction);
        case L1:
            return at::l1_loss(self, target, reduction);
        case SMOOTH_L1:
            return at::smooth_l1_loss(self, target, reduction, extra);
        case KL_DIV:
            return at::kl_div(self, target, reduction, extra != 0.0);
        case HUBER:
            return at::huber_loss(self, target, reduction, extra);
    }
    TORCH_CHECK(false, "unknown kunlun loss op");
}


struct NllGuard {
    DeviceContext& dc;
    std::unique_lock<std::mutex> lock;
    explicit NllGuard(int device) : dc(get_context(device)), lock(dc.mu) {
        xpu_set_device(dc.device);
        dc.ctx->set_stream(current_stream_native(dc.device));
    }
};

at::Tensor to_dev_nll(const at::Tensor& t, int device) {
    auto out = at::empty(t.sizes(), t.options().device(
        c10::Device(c10::DeviceType::PrivateUse1, device)));
    out.copy_(t);
    return out;
}

class NativeNllFunction : public Function<NativeNllFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor target,
                              std::optional<at::Tensor> weight, int64_t reduction,
                              c10::SymInt ignore_index) {
        const int64_t ignore = ignore_index.expect_int();
        const int64_t n = self.size(0);
        const int64_t c = self.size(1);
        auto out = reduction == 0 ? at::empty({n}, self.options())
                                  : at::empty({}, self.options());
        auto total_weight = at::empty({1}, self.options());
        at::Tensor native_weight;
        if (weight.has_value()) native_weight = weight->contiguous();
        bool native = false;
        try {
            NllGuard guard(self.device().index());
            int ret = 0;
            if (self.scalar_type() == at::kHalf) {
                ret = baidu::xpu::api::nll_loss<float16, int64_t>(
                    guard.dc.ctx, reinterpret_cast<const float16*>(self.data_ptr()),
                    reinterpret_cast<float16*>(out.data_ptr()),
                    reinterpret_cast<float16*>(total_weight.data_ptr()), self.sizes().vec(),
                    reinterpret_cast<const int64_t*>(target.data_ptr()),
                    native_weight.defined() ? reinterpret_cast<const float16*>(native_weight.data_ptr()) : nullptr,
                    reduction, ignore);
            } else {
                ret = baidu::xpu::api::nll_loss<float, int64_t>(
                    guard.dc.ctx, reinterpret_cast<const float*>(self.data_ptr()),
                    reinterpret_cast<float*>(out.data_ptr()),
                    reinterpret_cast<float*>(total_weight.data_ptr()), self.sizes().vec(),
                    reinterpret_cast<const int64_t*>(target.data_ptr()),
                    native_weight.defined() ? reinterpret_cast<const float*>(native_weight.data_ptr()) : nullptr,
                    reduction, ignore);
            }
            native = (ret == 0);
        } catch (...) {
            native = false;
        }
        (void)c;
        if (native) {
            ctx->save_for_backward({self, target, native_weight, total_weight});
            ctx->saved_data["native"] = true;
            ctx->saved_data["reduction"] = reduction;
            ctx->saved_data["ignore"] = ignore;
            return out;
        }
        // If the native reduction ABI is unavailable, leave the stage to the
        // established CPU-staged loss path.  This class is intentionally only
        // installed where the native call can be verified later on K200.
        auto cpu_self = to_cpu(self);
        auto cpu_target = to_cpu(target);
        auto cpu_weight = to_cpu(weight.value_or(at::Tensor()));
        auto cpu_out = at::nll_loss(cpu_self, cpu_target,
            weight.has_value() ? std::optional<at::Tensor>(cpu_weight) : std::nullopt,
            reduction, ignore);
        out = to_dev_nll(cpu_out, self.device().index());
        ctx->save_for_backward({cpu_self, cpu_target, cpu_weight});
        ctx->saved_data["native"] = false;
        ctx->saved_data["has_weight"] = weight.has_value();
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["ignore"] = ignore;
        ctx->saved_data["device"] = self.device().index();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        if (ctx->saved_data["native"].toBool()) {
            auto self = saved[0]; auto target = saved[1]; auto weight = saved[2];
            auto total_weight = saved[3];
            auto grad = grad_outputs[0].contiguous();
            auto dx = at::empty_like(self);
            NllGuard guard(self.device().index());
            int ret = 0;
            if (self.scalar_type() == at::kHalf) {
                ret = baidu::xpu::api::nll_loss_grad<float16, int64_t>(
                    guard.dc.ctx, reinterpret_cast<const float16*>(grad.data_ptr()),
                    reinterpret_cast<float16*>(dx.data_ptr()), self.sizes().vec(),
                    reinterpret_cast<const int64_t*>(target.data_ptr()),
                    weight.defined() ? reinterpret_cast<const float16*>(weight.data_ptr()) : nullptr,
                    ctx->saved_data["reduction"].toInt(), ctx->saved_data["ignore"].toInt(),
                    reinterpret_cast<const float16*>(total_weight.data_ptr()));
            } else {
                ret = baidu::xpu::api::nll_loss_grad<float, int64_t>(
                    guard.dc.ctx, reinterpret_cast<const float*>(grad.data_ptr()),
                    reinterpret_cast<float*>(dx.data_ptr()), self.sizes().vec(),
                    reinterpret_cast<const int64_t*>(target.data_ptr()),
                    weight.defined() ? reinterpret_cast<const float*>(weight.data_ptr()) : nullptr,
                    ctx->saved_data["reduction"].toInt(), ctx->saved_data["ignore"].toInt(),
                    reinterpret_cast<const float*>(total_weight.data_ptr()));
            }
            TORCH_CHECK(ret == 0, "native nll_loss backward failed: ret=", ret);
            return {dx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }
        auto cpu_self = saved[0].detach().clone().requires_grad_(true);
        auto cpu_target = saved[1]; auto cpu_weight = saved[2];
        auto cpu_grad = to_cpu(grad_outputs[0]);
        auto cpu_loss = at::nll_loss(cpu_self, cpu_target,
            ctx->saved_data["has_weight"].toBool() ? std::optional<at::Tensor>(cpu_weight) : std::nullopt,
            ctx->saved_data["reduction"].toInt(), ctx->saved_data["ignore"].toInt());
        cpu_loss.backward(cpu_grad);
        auto dx = to_dev_nll(cpu_self.grad(), ctx->saved_data["device"].toInt());
        return {dx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor native_nll_wrap(const at::Tensor& self, const at::Tensor& target,
                           const std::optional<at::Tensor>& weight, int64_t reduction,
                           c10::SymInt ignore) {
    return NativeNllFunction::apply(self, target, weight, reduction, ignore);
}

class LossFunction : public Function<LossFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, int64_t op, at::Tensor self,
                              at::Tensor target, std::optional<at::Tensor> weight,
                              std::optional<at::Tensor> pos_weight, int64_t reduction,
                              c10::SymInt ignore_index, double extra) {
        auto cpu_self = to_cpu(self);
        auto cpu_target = to_cpu(target);
        auto cpu_weight = to_cpu(weight.value_or(at::Tensor()));
        auto cpu_pos = to_cpu(pos_weight.value_or(at::Tensor()));
        at::Tensor cpu_out;
        {
            at::AutoDispatchBelowAutograd guard;
            cpu_out = run_cpu(static_cast<LossOp>(op), cpu_self, cpu_target,
                              weight.has_value() ? std::optional<at::Tensor>(cpu_weight)
                                                 : std::optional<at::Tensor>(),
                              pos_weight.has_value() ? std::optional<at::Tensor>(cpu_pos)
                                                     : std::optional<at::Tensor>(),
                              reduction, ignore_index.expect_int(), extra);
        }
        auto out = at::empty(cpu_out.sizes(), self.options());
        out.copy_(cpu_out);
        ctx->save_for_backward({cpu_self, cpu_target, cpu_weight, cpu_pos});
        ctx->saved_data["op"] = op;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["ignore"] = ignore_index.expect_int();
        ctx->saved_data["extra"] = extra;
        ctx->saved_data["has_weight"] = weight.has_value();
        ctx->saved_data["has_pos"] = pos_weight.has_value();
        ctx->saved_data["device"] = self.device().index();
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto cpu_self = saved[0].detach().clone();
        auto cpu_target = saved[1];
        auto cpu_weight = saved[2];
        auto cpu_pos = saved[3];
        cpu_self.requires_grad_(true);
        const auto op = static_cast<LossOp>(ctx->saved_data["op"].toInt());
        const int64_t reduction = ctx->saved_data["reduction"].toInt();
        const int64_t ignore = ctx->saved_data["ignore"].toInt();
        const double extra = ctx->saved_data["extra"].toDouble();
        auto grad = grad_outputs[0];
        auto cpu_grad = to_cpu(grad);
        auto cpu_loss = run_cpu(
            op, cpu_self, cpu_target,
            ctx->saved_data["has_weight"].toBool()
                ? std::optional<at::Tensor>(cpu_weight) : std::optional<at::Tensor>(),
            ctx->saved_data["has_pos"].toBool()
                ? std::optional<at::Tensor>(cpu_pos) : std::optional<at::Tensor>(),
            reduction, ignore, extra);
        cpu_loss.backward(cpu_grad);
        auto cpu_dx = cpu_self.grad();
        const int device = ctx->saved_data["device"].toInt();
        auto dx = at::empty(cpu_dx.sizes(), cpu_dx.options().device(
            c10::Device(c10::DeviceType::PrivateUse1, device)));
        dx.copy_(cpu_dx);
        return {dx, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor nll_loss_wrap(const at::Tensor& self, const at::Tensor& target,
                         const std::optional<at::Tensor>& weight, int64_t reduction,
                         c10::SymInt ignore) {
    return LossFunction::apply(static_cast<int64_t>(NLL), self, target, weight,
                               std::nullopt, reduction, ignore, 0.0);
}
at::Tensor bce_wrap(const at::Tensor& self, const at::Tensor& target,
                    const std::optional<at::Tensor>& weight, int64_t reduction) {
    return LossFunction::apply(static_cast<int64_t>(BCE), self, target, weight,
                               std::nullopt, reduction, c10::SymInt(-100), 0.0);
}
at::Tensor bce_logits_wrap(const at::Tensor& self, const at::Tensor& target,
                           const std::optional<at::Tensor>& weight,
                           const std::optional<at::Tensor>& pos_weight, int64_t reduction) {
    return LossFunction::apply(static_cast<int64_t>(BCE_LOGITS), self, target, weight,
                               pos_weight, reduction, -100, 0.0);
}
at::Tensor l1_wrap(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
    return LossFunction::apply(static_cast<int64_t>(L1), self, target, std::nullopt,
                               std::nullopt, reduction, c10::SymInt(-100), 0.0);
}
at::Tensor smooth_l1_wrap(const at::Tensor& self, const at::Tensor& target,
                          int64_t reduction, double beta) {
    return LossFunction::apply(static_cast<int64_t>(SMOOTH_L1), self, target,
                               std::nullopt, std::nullopt, reduction, c10::SymInt(-100), beta);
}
at::Tensor kl_div_wrap(const at::Tensor& self, const at::Tensor& target,
                       int64_t reduction, bool log_target) {
    return LossFunction::apply(static_cast<int64_t>(KL_DIV), self, target, std::nullopt,
                               std::nullopt, reduction, c10::SymInt(-100), log_target ? 1.0 : 0.0);
}
at::Tensor huber_wrap(const at::Tensor& self, const at::Tensor& target,
                      int64_t reduction, double delta) {
    return LossFunction::apply(static_cast<int64_t>(HUBER), self, target, std::nullopt,
                               std::nullopt, reduction, c10::SymInt(-100), delta);
}

}  // namespace
}  // namespace torch_kunlun

#define REG_LOSS(name, fn) \
    m.impl(name, TORCH_FN(fn))

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    REG_LOSS("nll_loss", torch_kunlun::native_nll_wrap);
    REG_LOSS("binary_cross_entropy", torch_kunlun::bce_wrap);
    REG_LOSS("binary_cross_entropy_with_logits", torch_kunlun::bce_logits_wrap);
    REG_LOSS("l1_loss", torch_kunlun::l1_wrap);
    REG_LOSS("smooth_l1_loss", torch_kunlun::smooth_l1_wrap);
    REG_LOSS("kl_div", torch_kunlun::kl_div_wrap);
    REG_LOSS("huber_loss", torch_kunlun::huber_wrap);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    REG_LOSS("nll_loss", torch_kunlun::native_nll_wrap);
    REG_LOSS("binary_cross_entropy", torch_kunlun::bce_wrap);
    REG_LOSS("binary_cross_entropy_with_logits", torch_kunlun::bce_logits_wrap);
    REG_LOSS("l1_loss", torch_kunlun::l1_wrap);
    REG_LOSS("smooth_l1_loss", torch_kunlun::smooth_l1_wrap);
    REG_LOSS("kl_div", torch_kunlun::kl_div_wrap);
    REG_LOSS("huber_loss", torch_kunlun::huber_wrap);
}
