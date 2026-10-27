#include "kunlun.h"

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include <optional>
#include <string>
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
at::Tensor to_dev(const at::Tensor& t, int device) {
    auto out = at::empty(t.sizes(), t.options().device(
        c10::Device(c10::DeviceType::PrivateUse1, device)));
    out.copy_(t);
    return out;
}

class EinsumFunction : public Function<EinsumFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, std::string equation,
                              std::vector<at::Tensor> tensors) {
        std::vector<at::Tensor> cpu_tensors;
        cpu_tensors.reserve(tensors.size());
        for (const auto& t : tensors) cpu_tensors.push_back(to_cpu(t));
        auto cpu_out = at::einsum(equation, cpu_tensors, std::nullopt);
        auto out = at::empty(cpu_out.sizes(), tensors[0].options());
        out.copy_(cpu_out);
        ctx->saved_data["equation"] = equation;
        ctx->saved_data["device"] = tensors[0].device().index();
        ctx->save_for_backward(cpu_tensors);
        return out;
    }
    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        std::vector<at::Tensor> cpu_tensors;
        for (const auto& t : saved) cpu_tensors.push_back(t.detach().clone().requires_grad_(true));
        auto cpu_grad = to_cpu(grad_outputs[0]);
        auto cpu_out = at::einsum(ctx->saved_data["equation"].toStringRef(), cpu_tensors,
                                  std::nullopt);
        cpu_out.backward(cpu_grad);
        const int device = ctx->saved_data["device"].toInt();
        variable_list grads;
        for (auto& t : cpu_tensors) {
            auto g = t.grad();
            grads.push_back(g.defined() ? to_dev(g, device) : at::Tensor());
        }
        return grads;
    }
};

at::Tensor einsum_wrap(c10::string_view equation, at::TensorList tensors,
                       at::OptionalIntArrayRef path) {
    std::vector<at::Tensor> list;
    for (const auto& t : tensors) list.push_back(t);
    return EinsumFunction::apply(std::string(equation), list);
}

}  // namespace
}  // namespace torch_kunlun

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("einsum", TORCH_FN(torch_kunlun::einsum_wrap));
}
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("einsum", TORCH_FN(torch_kunlun::einsum_wrap));
}
