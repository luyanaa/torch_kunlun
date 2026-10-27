#include "kunlun.h"

#include <ATen/ATen.h>

#include "xpu/refactor/nn.h"

#include <mutex>

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

void check(const at::Tensor& t, const char* what) {
    TORCH_CHECK(t.device().type() == at::DeviceType::PrivateUse1,
                what, " requires a kunlun tensor");
    TORCH_CHECK(t.is_contiguous(), what, " requires contiguous");
}

template <typename T>
void adam_impl(const at::Tensor& param, const at::Tensor& grad,
               at::Tensor& mom1, at::Tensor& mom2, double beta1_pow,
               double beta2_pow, double lr, double beta1, double beta2,
               double eps, double weight_decay, bool decoupled) {
    auto param_out = at::empty_like(param);
    auto mom1_out = at::empty_like(mom1);
    auto mom2_out = at::empty_like(mom2);
    auto scalar = [&](double value) {
        auto t = at::empty({1}, param.options().dtype(at::kFloat));
        t.fill_(value);
        return t;
    };
    auto b1 = scalar(beta1_pow);
    auto b2 = scalar(beta2_pow);
    auto lr_t = scalar(lr);
    ContextGuard guard(param.device().index());
    int ret = 0;
    if (decoupled) {
        ret = baidu::xpu::api::adamw<T>(
            guard.dc.ctx, reinterpret_cast<const T*>(grad.data_ptr()),
            reinterpret_cast<const float*>(mom1.data_ptr()),
            reinterpret_cast<const float*>(mom2.data_ptr()),
            reinterpret_cast<const T*>(param.data_ptr()),
            reinterpret_cast<const float*>(b1.data_ptr()),
            reinterpret_cast<const float*>(b2.data_ptr()),
            reinterpret_cast<const float*>(lr_t.data_ptr()),
            reinterpret_cast<float*>(mom1_out.data_ptr()),
            reinterpret_cast<float*>(mom2_out.data_ptr()),
            reinterpret_cast<T*>(param_out.data_ptr()),
            static_cast<float>(beta1), static_cast<float>(beta2),
            static_cast<float>(eps), static_cast<float>(weight_decay),
            param.numel(), 1.0f, nullptr, nullptr);
    } else {
        ret = baidu::xpu::api::adam<T>(
            guard.dc.ctx, reinterpret_cast<const T*>(grad.data_ptr()),
            reinterpret_cast<const float*>(mom1.data_ptr()),
            reinterpret_cast<const float*>(mom2.data_ptr()),
            reinterpret_cast<const T*>(param.data_ptr()),
            reinterpret_cast<const float*>(b1.data_ptr()),
            reinterpret_cast<const float*>(b2.data_ptr()),
            reinterpret_cast<const float*>(lr_t.data_ptr()),
            reinterpret_cast<float*>(mom1_out.data_ptr()),
            reinterpret_cast<float*>(mom2_out.data_ptr()),
            reinterpret_cast<T*>(param_out.data_ptr()),
            static_cast<float>(beta1), static_cast<float>(beta2),
            static_cast<float>(eps), param.numel());
    }
    TORCH_CHECK(ret == 0, "xdnn adam/adamw failed: ret=", ret);
    param.copy_(param_out);
    mom1.copy_(mom1_out);
    mom2.copy_(mom2_out);
}

}  // namespace

void adam_step(const at::Tensor& param, const at::Tensor& grad,
               at::Tensor& mom1, at::Tensor& mom2, double beta1_pow,
               double beta2_pow, double lr, double beta1, double beta2,
               double eps) {
    check(param, "adam param"); check(grad, "adam grad");
    check(mom1, "adam mom1"); check(mom2, "adam mom2");
    TORCH_CHECK(mom1.scalar_type() == at::kFloat && mom2.scalar_type() == at::kFloat,
                "adam moments must be float32");
    if (param.scalar_type() == at::kHalf) adam_impl<float16>(param, grad, mom1, mom2, beta1_pow, beta2_pow, lr, beta1, beta2, eps, 0.0, false);
    else adam_impl<float>(param, grad, mom1, mom2, beta1_pow, beta2_pow, lr, beta1, beta2, eps, 0.0, false);
}

void adamw_step(const at::Tensor& param, const at::Tensor& grad,
                at::Tensor& mom1, at::Tensor& mom2, double beta1_pow,
                double beta2_pow, double lr, double beta1, double beta2,
                double eps, double weight_decay) {
    check(param, "adamw param"); check(grad, "adamw grad");
    check(mom1, "adamw mom1"); check(mom2, "adamw mom2");
    TORCH_CHECK(mom1.scalar_type() == at::kFloat && mom2.scalar_type() == at::kFloat,
                "adamw moments must be float32");
    if (param.scalar_type() == at::kHalf) adam_impl<float16>(param, grad, mom1, mom2, beta1_pow, beta2_pow, lr, beta1, beta2, eps, weight_decay, true);
    else adam_impl<float>(param, grad, mom1, mom2, beta1_pow, beta2_pow, lr, beta1, beta2, eps, weight_decay, true);
}

}  // namespace torch_kunlun
