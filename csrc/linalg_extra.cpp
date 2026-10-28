#include "kunlun.h"

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include <optional>
#include <vector>

namespace torch_kunlun {
namespace {

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;

at::Tensor to_cpu(const at::Tensor& t) {
    auto out = at::empty(t.sizes(), t.options().device(at::kCPU));
    out.copy_(t);
    return out;
}
at::Tensor to_dev(const at::Tensor& t, const at::Tensor& reference) {
    auto out = at::empty(t.sizes(), reference.options());
    out.copy_(t);
    return out;
}

at::Tensor to_dev_index(const at::Tensor& t, int device) {
    auto out = at::empty(t.sizes(), t.options().device(
        c10::Device(c10::DeviceType::PrivateUse1, device)));
    out.copy_(t);
    return out;
}

at::Tensor bmm_contig(const at::Tensor& a, const at::Tensor& b) {
    auto ac = a.is_contiguous() ? a : a.contiguous();
    auto bc = b.is_contiguous() ? b : b.contiguous();
    return at::bmm(ac, bc);
}

at::Tensor scale_tensor(const at::Tensor& x, double factor) {
    return at::mul(x, factor);
}

class BaddbmmFunction : public Function<BaddbmmFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor batch1,
                              at::Tensor batch2, at::Scalar beta, at::Scalar alpha) {
        auto mm = bmm_contig(batch1, batch2);
        auto out = scale_tensor(mm, alpha.toDouble());
        if (beta.toDouble() != 0.0) {
            out = at::add(scale_tensor(self, beta.toDouble()), out);
        }
        ctx->save_for_backward({self, batch1, batch2});
        ctx->saved_data["beta"] = beta.toDouble();
        ctx->saved_data["alpha"] = alpha.toDouble();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto self = saved[0]; auto b1 = saved[1]; auto b2 = saved[2];
        auto grad = grad_outputs[0].contiguous();
        const double alpha = ctx->saved_data["alpha"].toDouble();
        const double beta = ctx->saved_data["beta"].toDouble();
        auto b2t = b2.transpose(1, 2);
        auto b1t = b1.transpose(1, 2);
        auto g1 = scale_tensor(bmm_contig(grad, b2t), alpha);
        auto g2 = scale_tensor(bmm_contig(b1t, grad), alpha);
        at::Tensor gs;
        if (self.dim() == grad.dim()) {
            gs = scale_tensor(grad, beta);
        } else if (self.dim() == 2 && grad.dim() == 3) {
            gs = scale_tensor(at::sum(grad, {0}, false), beta);
        }
        return {gs, g1, g2, at::Tensor(), at::Tensor()};
    }
};

at::Tensor baddbmm_autograd(const at::Tensor& self, const at::Tensor& batch1,
                            const at::Tensor& batch2, const at::Scalar& beta,
                            const at::Scalar& alpha) {
    return BaddbmmFunction::apply(self, batch1, batch2, beta, alpha);
}

at::Tensor baddbmm_forward(const at::Tensor& self, const at::Tensor& batch1,
                           const at::Tensor& batch2, const at::Scalar& beta,
                           const at::Scalar& alpha) {
    at::AutoDispatchBelowAutograd guard;
    auto mm = bmm_contig(batch1, batch2);
    auto out = scale_tensor(mm, alpha.toDouble());
    if (beta.toDouble() != 0.0) out = at::add(scale_tensor(self, beta.toDouble()), out);
    return out;
}


class AddbmmFunction : public Function<AddbmmFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor batch1,
                              at::Tensor batch2, at::Scalar beta, at::Scalar alpha) {
        at::AutoDispatchBelowAutograd guard;
        auto mm = at::sum(bmm_contig(batch1, batch2), {0}, false);
        auto out = scale_tensor(mm, alpha.toDouble());
        if (beta.toDouble() != 0.0) out = at::add(scale_tensor(self, beta.toDouble()), out);
        ctx->save_for_backward({self, batch1, batch2});
        ctx->saved_data["beta"] = beta.toDouble();
        ctx->saved_data["alpha"] = alpha.toDouble();
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto self = saved[0]; auto b1 = saved[1]; auto b2 = saved[2];
        auto grad = grad_outputs[0].contiguous();
        const double alpha = ctx->saved_data["alpha"].toDouble();
        const double beta = ctx->saved_data["beta"].toDouble();
        const int64_t batch = b1.size(0);
        auto g = grad.unsqueeze(0).expand({batch, grad.size(0), grad.size(1)}).contiguous();
        g = scale_tensor(g, alpha);
        auto g1 = bmm_contig(g, b2.transpose(1, 2));
        auto g2 = bmm_contig(b1.transpose(1, 2), g);
        auto gs = beta == 0.0 ? at::Tensor() : scale_tensor(grad, beta);
        return {gs, g1, g2, at::Tensor(), at::Tensor()};
    }
};

at::Tensor addbmm_autograd(const at::Tensor& self, const at::Tensor& batch1,
                           const at::Tensor& batch2, const at::Scalar& beta,
                           const at::Scalar& alpha) {
    return AddbmmFunction::apply(self, batch1, batch2, beta, alpha);
}

at::Tensor addbmm_forward(const at::Tensor& self, const at::Tensor& batch1,
                          const at::Tensor& batch2, const at::Scalar& beta,
                          const at::Scalar& alpha) {
    at::AutoDispatchBelowAutograd guard;
    auto mm = at::sum(bmm_contig(batch1, batch2), {0}, false);
    auto out = scale_tensor(mm, alpha.toDouble());
    if (beta.toDouble() != 0.0) out = at::add(scale_tensor(self, beta.toDouble()), out);
    return out;
}

class MatmulFunction : public Function<MatmulFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::Tensor other) {
        const bool native = (self.dim() == 2 && other.dim() == 2) ||
                            (self.dim() == 3 && other.dim() == 3) ||
                            (self.dim() == 4 && other.dim() == 4);
        if (!native) {
            note_cpu_fallback("matmul");
            auto cpu_a = to_cpu(self);
            auto cpu_b = to_cpu(other);
            auto cpu_out = at::matmul(cpu_a, cpu_b);
            auto out = to_dev(cpu_out, self);
            ctx->save_for_backward({cpu_a, cpu_b});
            ctx->saved_data["cpu_fallback"] = true;
            ctx->saved_data["device"] = self.device().index();
            return out;
        }
        ctx->saved_data["cpu_fallback"] = false;
        at::Tensor a = self, b = other;
        std::vector<int64_t> a_shape = self.sizes().vec();
        std::vector<int64_t> b_shape = other.sizes().vec();
        if (self.dim() == 4 && other.dim() == 4) {
            const int64_t ab = self.size(0) * self.size(1);
            const int64_t bb = other.size(0) * other.size(1);
            TORCH_CHECK(ab == bb, "matmul batch mismatch");
            a = self.reshape({ab, self.size(2), self.size(3)});
            b = other.reshape({bb, other.size(2), other.size(3)});
        }
        at::Tensor out;
        if (a.dim() == 2 && b.dim() == 2) {
            out = at::mm(a, b);
        } else if (a.dim() == 3 && b.dim() == 3) {
            out = bmm_contig(a, b);
        } else {
            TORCH_CHECK(false, "kunlun matmul supports 2D/3D/4D (matching batch)");
        }
        if (self.dim() == 4) {
            out = out.reshape({self.size(0), self.size(1), out.size(1), out.size(2)});
        }
        ctx->save_for_backward({self, other});
        ctx->saved_data["a_shape"] = a_shape;
        ctx->saved_data["b_shape"] = b_shape;
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        if (ctx->saved_data["cpu_fallback"].toBool()) {
            auto cpu_a = saved[0].detach().clone().requires_grad_(true);
            auto cpu_b = saved[1].detach().clone().requires_grad_(true);
            auto cpu_grad = to_cpu(grad_outputs[0]);
            auto cpu_out = at::matmul(cpu_a, cpu_b);
            cpu_out.backward(cpu_grad);
            const int device = ctx->saved_data["device"].toInt();
            auto ga = to_dev_index(cpu_a.grad(), device);
            auto gb = to_dev_index(cpu_b.grad(), device);
            return {ga, gb};
        }
        auto self = saved[0]; auto other = saved[1];
        auto grad = grad_outputs[0].contiguous();
        auto a_shape = ctx->saved_data["a_shape"].toIntVector();
        auto b_shape = ctx->saved_data["b_shape"].toIntVector();
        at::Tensor a = self, b = other, g = grad;
        if (self.dim() == 4) {
            a = self.reshape({self.size(0) * self.size(1), self.size(2), self.size(3)});
            b = other.reshape({other.size(0) * other.size(1), other.size(2), other.size(3)});
            g = grad.reshape({self.size(0) * self.size(1), grad.size(2), grad.size(3)});
        }
        at::Tensor ga, gb;
        if (a.dim() == 2) {
            ga = at::mm(g, b.transpose(0, 1).contiguous());
            gb = at::mm(a.transpose(0, 1).contiguous(), g);
        } else {
            ga = bmm_contig(g, b.transpose(1, 2));
            gb = bmm_contig(a.transpose(1, 2), g);
        }
        if (self.dim() == 4) {
            ga = ga.reshape(a_shape);
            gb = gb.reshape(b_shape);
        }
        return {ga, gb};
    }
};

at::Tensor matmul_autograd(const at::Tensor& self, const at::Tensor& other) {
    return MatmulFunction::apply(self, other);
}

at::Tensor matmul_forward(const at::Tensor& self, const at::Tensor& other) {
    const bool native = (self.dim() == 2 && other.dim() == 2) ||
                        (self.dim() == 3 && other.dim() == 3) ||
                        (self.dim() == 4 && other.dim() == 4);
    if (!native) {
        note_cpu_fallback("matmul");
        auto cpu_out = at::matmul(to_cpu(self), to_cpu(other));
        return to_dev(cpu_out, self);
    }
    at::AutoDispatchBelowAutograd guard;
    at::Tensor a = self, b = other;
    if (self.dim() == 4 && other.dim() == 4) {
        const int64_t ab = self.size(0) * self.size(1);
        const int64_t bb = other.size(0) * other.size(1);
        TORCH_CHECK(ab == bb, "matmul batch mismatch");
        a = self.reshape({ab, self.size(2), self.size(3)});
        b = other.reshape({bb, other.size(2), other.size(3)});
    }
    at::Tensor out;
    if (a.dim() == 2 && b.dim() == 2) out = at::mm(a, b);
    else if (a.dim() == 3 && b.dim() == 3) out = bmm_contig(a, b);
    else TORCH_CHECK(false, "kunlun matmul supports 2D/3D/4D (matching batch)");
    if (self.dim() == 4) out = out.reshape({self.size(0), self.size(1), out.size(1), out.size(2)});
    return out;
}

}  // namespace
}  // namespace torch_kunlun

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("baddbmm", TORCH_FN(torch_kunlun::baddbmm_forward));
    m.impl("addbmm", TORCH_FN(torch_kunlun::addbmm_forward));
    m.impl("matmul", TORCH_FN(torch_kunlun::matmul_forward));
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("baddbmm", TORCH_FN(torch_kunlun::baddbmm_autograd));
    m.impl("addbmm", TORCH_FN(torch_kunlun::addbmm_autograd));
    m.impl("matmul", TORCH_FN(torch_kunlun::matmul_autograd));
}
