#include "kunlun.h"

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
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
    HardTanh, Softsign, SoftShrink, Threshold
};

template <typename T>
at::Tensor unary_impl(const at::Tensor& self, UnaryCode code,
                      float a, float b, float c) {
    check_tensor(self, "unary");
    auto out = at::empty_like(self);
    const int64_t len = self.numel();
    ContextGuard guard(self.device().index());
    auto* ctx = guard.dc.ctx;
    int ret = 0;
    switch (code) {
        case UnaryCode::Abs: ret = baidu::xpu::api::abs<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Neg: ret = baidu::xpu::api::neg<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Exp: ret = baidu::xpu::api::exp<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Log: ret = baidu::xpu::api::log<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Sqrt: ret = baidu::xpu::api::sqrt<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Rsqrt: ret = baidu::xpu::api::rsqrt<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Sigmoid: ret = baidu::xpu::api::sigmoid<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Tanh: ret = baidu::xpu::api::tanh<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Silu: ret = baidu::xpu::api::swish<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::LeakyRelu: ret = baidu::xpu::api::leaky_relu<T>(ctx, cptr<T>(self), mptr<T>(out), len, a); break;
        case UnaryCode::Elu: ret = baidu::xpu::api::elu<T>(ctx, cptr<T>(self), mptr<T>(out), len, a); break;
        case UnaryCode::Softplus: ret = baidu::xpu::api::softplus<T>(ctx, cptr<T>(self), mptr<T>(out), len, a, b); break;
        case UnaryCode::Clamp: {
            const T lo = static_cast<T>(a);
            const T hi = static_cast<T>(b);
            ret = baidu::xpu::api::clip<T>(ctx, cptr<T>(self), mptr<T>(out), len, lo, hi);
            break;
        }
        case UnaryCode::Square: ret = baidu::xpu::api::square<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Reciprocal: ret = baidu::xpu::api::reciprocal<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Sign: ret = baidu::xpu::api::sign<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Ceil: ret = baidu::xpu::api::ceil<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Floor: ret = baidu::xpu::api::floor<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Round: ret = baidu::xpu::api::round<T>(ctx, cptr<T>(self), mptr<T>(out), len, 0); break;
        case UnaryCode::Log1p: ret = baidu::xpu::api::log1p<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Exp2: ret = baidu::xpu::api::exp2<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Erf: ret = baidu::xpu::api::erf<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Erfinv: ret = baidu::xpu::api::erfinv<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::Selu: ret = baidu::xpu::api::selu<T>(ctx, cptr<T>(self), mptr<T>(out), len, a, b); break;
        case UnaryCode::Celu: ret = baidu::xpu::api::celu<T>(ctx, cptr<T>(self), mptr<T>(out), len, a); break;
        case UnaryCode::Mish: ret = baidu::xpu::api::mish<T>(ctx, cptr<T>(self), mptr<T>(out), len, a); break;
        case UnaryCode::HardSwish: ret = baidu::xpu::api::hard_swish<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::HardSigmoid: ret = baidu::xpu::api::hard_sigmoid<T>(ctx, cptr<T>(self), mptr<T>(out), len, a); break;
        case UnaryCode::HardTanh: ret = baidu::xpu::api::hard_tanh<T>(ctx, cptr<T>(self), mptr<T>(out), len, a, b); break;
        case UnaryCode::Softsign: ret = baidu::xpu::api::softsign<T>(ctx, cptr<T>(self), mptr<T>(out), len); break;
        case UnaryCode::SoftShrink: ret = baidu::xpu::api::soft_shrink<T>(ctx, cptr<T>(self), mptr<T>(out), len, a); break;
        case UnaryCode::Threshold: ret = baidu::xpu::api::threshold<T>(ctx, cptr<T>(self), mptr<T>(out), len, a, b); break;
    }
    TORCH_CHECK(ret == 0, "xdnn unary op failed: ret=", ret);
    return out;
}


std::vector<int64_t> normalize_dims(at::IntArrayRef dims, int64_t rank) {
    std::vector<int64_t> out;
    if (dims.empty()) {
        for (int64_t i = 0; i < rank; ++i) out.push_back(i);
        return out;
    }
    for (int64_t d : dims) {
        if (d < 0) d += rank;
        TORCH_CHECK(d >= 0 && d < rank, "dimension out of range: ", d);
        out.push_back(d);
    }
    return out;
}

std::vector<int64_t> reduced_shape(at::IntArrayRef shape,
                                   const std::vector<int64_t>& dims,
                                   bool keepdim) {
    std::vector<int64_t> out;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
        const bool reduce = std::find(dims.begin(), dims.end(), i) != dims.end();
        if (reduce) {
            if (keepdim) out.push_back(1);
        } else {
            out.push_back(shape[i]);
        }
    }
    return out;
}

template <typename T>
at::Tensor reduce_impl(const at::Tensor& self, const std::vector<int64_t>& dims,
                       bool keepdim, bool mean) {
    check_tensor(self, "reduce");
    auto out_shape = reduced_shape(self.sizes(), dims, keepdim);
    auto out = at::empty(out_shape, self.options());
    ContextGuard guard(self.device().index());
    int ret = mean
        ? baidu::xpu::api::reduce_mean<T>(guard.dc.ctx, cptr<T>(self), mptr<T>(out),
                                          self.sizes().vec(), dims)
        : baidu::xpu::api::reduce_sum<T>(guard.dc.ctx, cptr<T>(self), mptr<T>(out),
                                         self.sizes().vec(), dims);
    TORCH_CHECK(ret == 0, "xdnn reduce failed: ret=", ret);
    return out;
}

at::Tensor sum_dim(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                   std::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype.has_value() || *dtype == self.scalar_type(),
                "kunlun sum dtype must be omitted/same");
    auto dims = normalize_dims(dim, self.dim());
    if (self.scalar_type() == at::kHalf) return reduce_impl<float16>(self, dims, keepdim, false);
    if (self.scalar_type() == at::kFloat) return reduce_impl<float>(self, dims, keepdim, false);
    TORCH_CHECK(false, "kunlun sum only supports float16/float32");
}

at::Tensor sum_all(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
    return sum_dim(self, at::IntArrayRef{}, false, dtype);
}

at::Tensor mean_dim(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                    std::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype.has_value() || *dtype == self.scalar_type(),
                "kunlun mean dtype must be omitted/same");
    auto dims = normalize_dims(dim, self.dim());
    if (self.scalar_type() == at::kHalf) return reduce_impl<float16>(self, dims, keepdim, true);
    if (self.scalar_type() == at::kFloat) return reduce_impl<float>(self, dims, keepdim, true);
    TORCH_CHECK(false, "kunlun mean only supports float16/float32");
}

at::Tensor mean_all(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
    return mean_dim(self, at::IntArrayRef{}, false, dtype);
}

template <typename T>
at::Tensor amax_impl(const at::Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    check_tensor(self, "amax");
    auto out_shape = reduced_shape(self.sizes(), dims, keepdim);
    auto out = at::empty(out_shape, self.options());
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::reduce_max<T>(
        guard.dc.ctx, cptr<T>(self), mptr<T>(out), self.sizes().vec(), dims);
    TORCH_CHECK(ret == 0, "xdnn reduce_max failed: ret=", ret);
    return out;
}

at::Tensor amax(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    auto dims = normalize_dims(dim, self.dim());
    if (self.scalar_type() == at::kHalf) return amax_impl<float16>(self, dims, keepdim);
    if (self.scalar_type() == at::kFloat) return amax_impl<float>(self, dims, keepdim);
    TORCH_CHECK(false, "kunlun amax only supports float16/float32");
}

template <typename T>
at::Tensor amin_impl(const at::Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    check_tensor(self, "amin");
    auto out_shape = reduced_shape(self.sizes(), dims, keepdim);
    auto out = at::empty(out_shape, self.options());
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::reduce_min<T>(
        guard.dc.ctx, cptr<T>(self), mptr<T>(out), self.sizes().vec(), dims);
    TORCH_CHECK(ret == 0, "xdnn reduce_min failed: ret=", ret);
    return out;
}

at::Tensor amin(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    auto dims = normalize_dims(dim, self.dim());
    if (self.scalar_type() == at::kHalf) return amin_impl<float16>(self, dims, keepdim);
    if (self.scalar_type() == at::kFloat) return amin_impl<float>(self, dims, keepdim);
    TORCH_CHECK(false, "kunlun amin only supports float16/float32");
}

at::Tensor argmax(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    check_tensor(self, "argmax");
    at::Tensor out;
    if (!dim.has_value()) {
        out = at::empty({}, self.options().dtype(at::kLong));
    } else {
        auto d = *dim;
        if (d < 0) d += self.dim();
        out = at::empty(reduced_shape(self.sizes(), {d}, keepdim),
                        self.options().dtype(at::kLong));
    }
    ContextGuard guard(self.device().index());
    int ret = 0;
    if (!dim.has_value()) {
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::argmax<float16>(
                guard.dc.ctx, cptr<float16>(self), mptr<int64_t>(out),
                {self.numel()}, 0);
        } else {
            ret = baidu::xpu::api::argmax<float>(
                guard.dc.ctx, cptr<float>(self), mptr<int64_t>(out),
                {self.numel()}, 0);
        }
    } else {
        int64_t axis = *dim;
        if (axis < 0) axis += self.dim();
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::argmax<float16>(
                guard.dc.ctx, cptr<float16>(self), mptr<int64_t>(out),
                self.sizes().vec(), axis);
        } else {
            ret = baidu::xpu::api::argmax<float>(
                guard.dc.ctx, cptr<float>(self), mptr<int64_t>(out),
                self.sizes().vec(), axis);
        }
    }
    TORCH_CHECK(ret == 0, "xdnn argmax failed: ret=", ret);
    return out;
}

at::Tensor argmin(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    check_tensor(self, "argmin");
    at::Tensor out;
    if (!dim.has_value()) {
        out = at::empty({}, self.options().dtype(at::kLong));
    } else {
        auto d = *dim;
        if (d < 0) d += self.dim();
        out = at::empty(reduced_shape(self.sizes(), {d}, keepdim),
                        self.options().dtype(at::kLong));
    }
    ContextGuard guard(self.device().index());
    int ret = 0;
    if (!dim.has_value()) {
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::argmin<float16>(
                guard.dc.ctx, cptr<float16>(self), mptr<int64_t>(out),
                {self.numel()}, 0);
        } else {
            ret = baidu::xpu::api::argmin<float>(
                guard.dc.ctx, cptr<float>(self), mptr<int64_t>(out),
                {self.numel()}, 0);
        }
    } else {
        int64_t axis = *dim;
        if (axis < 0) axis += self.dim();
        if (self.scalar_type() == at::kHalf) {
            ret = baidu::xpu::api::argmin<float16>(
                guard.dc.ctx, cptr<float16>(self), mptr<int64_t>(out),
                self.sizes().vec(), axis);
        } else {
            ret = baidu::xpu::api::argmin<float>(
                guard.dc.ctx, cptr<float>(self), mptr<int64_t>(out),
                self.sizes().vec(), axis);
        }
    }
    TORCH_CHECK(ret == 0, "xdnn argmin failed: ret=", ret);
    return out;
}

at::Tensor logsumexp(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    check_tensor(self, "logsumexp");
    auto dims = normalize_dims(dim, self.dim());
    auto out_shape = reduced_shape(self.sizes(), dims, keepdim);
    auto out = at::empty(out_shape, self.options());
    ContextGuard guard(self.device().index());
    int ret = 0;
    if (self.scalar_type() == at::kHalf) {
        ret = baidu::xpu::api::logsumexp<float16>(
            guard.dc.ctx, cptr<float16>(self), mptr<float16>(out),
            self.sizes().vec(), dims);
    } else {
        ret = baidu::xpu::api::logsumexp<float>(
            guard.dc.ctx, cptr<float>(self), mptr<float>(out),
            self.sizes().vec(), dims);
    }
    TORCH_CHECK(ret == 0, "xdnn logsumexp failed: ret=", ret);
    return out;
}

template <typename T>
at::Tensor where_impl(const at::Tensor& condition, const at::Tensor& x, const at::Tensor& y) {
    check_tensor(condition, "where condition");
    check_tensor(x, "where x");
    check_tensor(y, "where y");
    TORCH_CHECK(condition.scalar_type() == at::kBool, "where condition must be bool");
    auto out_shape = at::infer_size(x.sizes(), y.sizes());
    auto out = at::empty(out_shape, x.options());
    ContextGuard guard(x.device().index());
    const int ret = baidu::xpu::api::select<T>(
        guard.dc.ctx, cptr<bool>(condition), cptr<T>(x), cptr<T>(y), mptr<T>(out),
        condition.sizes().vec(), x.sizes().vec());
    TORCH_CHECK(ret == 0, "xdnn select(where) failed: ret=", ret);
    return out;
}

at::Tensor where(const at::Tensor& condition, const at::Tensor& x, const at::Tensor& y) {
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "where dtype mismatch");
    if (x.scalar_type() == at::kHalf) return where_impl<float16>(condition, x, y);
    if (x.scalar_type() == at::kFloat) return where_impl<float>(condition, x, y);
    TORCH_CHECK(false, "kunlun where only supports float16/float32");
}

template <typename T>
at::Tensor masked_fill_impl(const at::Tensor& self, const at::Tensor& mask,
                            const at::Scalar& value) {
    check_tensor(self, "masked_fill");
    check_tensor(mask, "masked_fill mask");
    TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_fill mask must be bool");
    auto out = at::empty_like(self);
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::masked_fill_scalar<T>(
        guard.dc.ctx, cptr<bool>(mask), cptr<T>(self), static_cast<T>(value.toDouble()),
        mptr<T>(out), mask.sizes().vec(), self.sizes().vec());
    TORCH_CHECK(ret == 0, "xdnn masked_fill failed: ret=", ret);
    return out;
}

at::Tensor masked_fill(const at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {
    if (self.scalar_type() == at::kHalf) return masked_fill_impl<float16>(self, mask, value);
    if (self.scalar_type() == at::kFloat) return masked_fill_impl<float>(self, mask, value);
    TORCH_CHECK(false, "kunlun masked_fill only supports float16/float32");
}

template <typename T, bool Upper>
at::Tensor triangle_impl(const at::Tensor& self, int64_t diagonal) {
    check_tensor(self, "triu/tril");
    auto out = at::empty_like(self);
    ContextGuard guard(self.device().index());
    const int ret = Upper
        ? baidu::xpu::api::triu<T>(guard.dc.ctx, cptr<T>(self), mptr<T>(out),
                                   self.sizes().vec(), diagonal)
        : baidu::xpu::api::tril<T>(guard.dc.ctx, cptr<T>(self), mptr<T>(out),
                                   self.sizes().vec(), diagonal);
    TORCH_CHECK(ret == 0, "xdnn triu/tril failed: ret=", ret);
    return out;
}

at::Tensor triu(const at::Tensor& self, int64_t diagonal) {
    if (self.scalar_type() == at::kHalf) return triangle_impl<float16, true>(self, diagonal);
    if (self.scalar_type() == at::kFloat) return triangle_impl<float, true>(self, diagonal);
    TORCH_CHECK(false, "kunlun triu only supports float16/float32");
}

at::Tensor tril(const at::Tensor& self, int64_t diagonal) {
    if (self.scalar_type() == at::kHalf) return triangle_impl<float16, false>(self, diagonal);
    if (self.scalar_type() == at::kFloat) return triangle_impl<float, false>(self, diagonal);
    TORCH_CHECK(false, "kunlun tril only supports float16/float32");
}


template <typename T, bool Max>
at::Tensor extremum_binary_impl(const at::Tensor& self, const at::Tensor& other) {
    check_tensor(self, "maximum/minimum");
    check_tensor(other, "maximum/minimum");
    const auto shape = at::infer_size(self.sizes(), other.sizes());
    auto out = at::empty(shape, self.options());
    ContextGuard guard(self.device().index());
    int ret = 0;
    if (self.sizes().equals(other.sizes())) {
        ret = Max ? baidu::xpu::api::max<T>(guard.dc.ctx, cptr<T>(self), cptr<T>(other),
                                            mptr<T>(out), self.numel())
                  : baidu::xpu::api::min<T>(guard.dc.ctx, cptr<T>(self), cptr<T>(other),
                                            mptr<T>(out), self.numel());
    } else {
        ret = Max ? baidu::xpu::api::broadcast_max<T>(
                        guard.dc.ctx, cptr<T>(self), cptr<T>(other), mptr<T>(out),
                        self.sizes().vec(), other.sizes().vec())
                  : baidu::xpu::api::broadcast_min<T>(
                        guard.dc.ctx, cptr<T>(self), cptr<T>(other), mptr<T>(out),
                        self.sizes().vec(), other.sizes().vec());
    }
    TORCH_CHECK(ret == 0, "xdnn maximum/minimum failed: ret=", ret);
    return out;
}

at::Tensor maximum(const at::Tensor& self, const at::Tensor& other) {
    if (self.scalar_type() == at::kHalf) return extremum_binary_impl<float16, true>(self, other);
    if (self.scalar_type() == at::kFloat) return extremum_binary_impl<float, true>(self, other);
    TORCH_CHECK(false, "kunlun maximum only supports float16/float32");
}

at::Tensor minimum(const at::Tensor& self, const at::Tensor& other) {
    if (self.scalar_type() == at::kHalf) return extremum_binary_impl<float16, false>(self, other);
    if (self.scalar_type() == at::kFloat) return extremum_binary_impl<float, false>(self, other);
    TORCH_CHECK(false, "kunlun minimum only supports float16/float32");
}

template <typename T, bool Cdiv>
at::Tensor add_c_mul_div_impl(const at::Tensor& self, const at::Tensor& a,
                              const at::Tensor& b, double value) {
    check_tensor(self, "addcmul/addcdiv");
    check_tensor(a, "addcmul/addcdiv");
    check_tensor(b, "addcmul/addcdiv");
    TORCH_CHECK(self.sizes().equals(a.sizes()) && self.sizes().equals(b.sizes()),
                "kunlun addcmul/addcdiv requires equal shapes");
    auto out = at::empty_like(self);
    ContextGuard guard(self.device().index());
    const int ret = Cdiv
        ? baidu::xpu::api::addcdiv<T>(guard.dc.ctx, cptr<T>(self), cptr<T>(a), cptr<T>(b),
                                      mptr<T>(out), static_cast<float>(value), self.numel())
        : baidu::xpu::api::addcmul<T>(guard.dc.ctx, cptr<T>(self), cptr<T>(a), cptr<T>(b),
                                      mptr<T>(out), static_cast<float>(value), self.numel());
    TORCH_CHECK(ret == 0, "xdnn addcmul/addcdiv failed: ret=", ret);
    return out;
}

at::Tensor addcmul(const at::Tensor& self, const at::Tensor& a, const at::Tensor& b,
                   const at::Scalar& value) {
    if (self.scalar_type() == at::kHalf) return add_c_mul_div_impl<float16, false>(self, a, b, value.toDouble());
    if (self.scalar_type() == at::kFloat) return add_c_mul_div_impl<float, false>(self, a, b, value.toDouble());
    TORCH_CHECK(false, "kunlun addcmul only supports float16/float32");
}

at::Tensor addcdiv(const at::Tensor& self, const at::Tensor& a, const at::Tensor& b,
                   const at::Scalar& value) {
    if (self.scalar_type() == at::kHalf) return add_c_mul_div_impl<float16, true>(self, a, b, value.toDouble());
    if (self.scalar_type() == at::kFloat) return add_c_mul_div_impl<float, true>(self, a, b, value.toDouble());
    TORCH_CHECK(false, "kunlun addcdiv only supports float16/float32");
}
at::Tensor unary(const at::Tensor& self, UnaryCode code,
                 float a = 0.0f, float b = 20.0f, float c = 1.0f) {
    if (self.scalar_type() == at::kHalf) return unary_impl<float16>(self, code, a, b, c);
    if (self.scalar_type() == at::kFloat) return unary_impl<float>(self, code, a, b, c);
    TORCH_CHECK(false, "kunlun unary op only supports float16/float32");
}

at::Tensor abs(const at::Tensor& self) { return unary(self, UnaryCode::Abs); }
at::Tensor neg(const at::Tensor& self) { return unary(self, UnaryCode::Neg); }
at::Tensor exp(const at::Tensor& self) { return unary(self, UnaryCode::Exp); }
at::Tensor log(const at::Tensor& self) { return unary(self, UnaryCode::Log); }
at::Tensor sqrt(const at::Tensor& self) { return unary(self, UnaryCode::Sqrt); }
at::Tensor rsqrt(const at::Tensor& self) { return unary(self, UnaryCode::Rsqrt); }
at::Tensor sigmoid(const at::Tensor& self) { return unary(self, UnaryCode::Sigmoid); }
at::Tensor tanh(const at::Tensor& self) { return unary(self, UnaryCode::Tanh); }
at::Tensor silu(const at::Tensor& self) { return unary(self, UnaryCode::Silu); }
at::Tensor square(const at::Tensor& self) { return unary(self, UnaryCode::Square); }
at::Tensor reciprocal(const at::Tensor& self) { return unary(self, UnaryCode::Reciprocal); }
at::Tensor sign(const at::Tensor& self) { return unary(self, UnaryCode::Sign); }
at::Tensor ceil(const at::Tensor& self) { return unary(self, UnaryCode::Ceil); }
at::Tensor floor(const at::Tensor& self) { return unary(self, UnaryCode::Floor); }
at::Tensor round(const at::Tensor& self) { return unary(self, UnaryCode::Round); }
at::Tensor log1p(const at::Tensor& self) { return unary(self, UnaryCode::Log1p); }
at::Tensor exp2(const at::Tensor& self) { return unary(self, UnaryCode::Exp2); }
at::Tensor erf(const at::Tensor& self) { return unary(self, UnaryCode::Erf); }
at::Tensor erfinv(const at::Tensor& self) { return unary(self, UnaryCode::Erfinv); }
at::Tensor selu(const at::Tensor& self) { return unary(self, UnaryCode::Selu, 1.050701f, 1.6732632f); }
at::Tensor celu(const at::Tensor& self, const at::Scalar& alpha) { return unary(self, UnaryCode::Celu, static_cast<float>(alpha.toDouble())); }
at::Tensor mish(const at::Tensor& self) { return unary(self, UnaryCode::Mish, 20.0f); }
at::Tensor hardswish(const at::Tensor& self) { return unary(self, UnaryCode::HardSwish); }
at::Tensor hardsigmoid(const at::Tensor& self) { return unary(self, UnaryCode::HardSigmoid, 1.0f / 6.0f); }
at::Tensor hardtanh(const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val) {
    return unary(self, UnaryCode::HardTanh, static_cast<float>(min_val.toDouble()),
                 static_cast<float>(max_val.toDouble()));
}
at::Tensor softshrink(const at::Tensor& self, const at::Scalar& lambd) {
    return unary(self, UnaryCode::SoftShrink, static_cast<float>(lambd.toDouble()));
}
at::Tensor threshold(const at::Tensor& self, const at::Scalar& thre, const at::Scalar& value) {
    return unary(self, UnaryCode::Threshold, static_cast<float>(thre.toDouble()),
                 static_cast<float>(value.toDouble()));
}

at::Tensor leaky_relu(const at::Tensor& self, const at::Scalar& negative_slope) {
    return unary(self, UnaryCode::LeakyRelu, static_cast<float>(negative_slope.toDouble()));
}

at::Tensor elu(const at::Tensor& self, const at::Scalar& alpha,
               const at::Scalar& scale, const at::Scalar& input_scale) {
    TORCH_CHECK(scale.toDouble() == 1.0 && input_scale.toDouble() == 1.0,
                "kunlun elu only supports scale=1,input_scale=1");
    return unary(self, UnaryCode::Elu, static_cast<float>(alpha.toDouble()));
}

at::Tensor softplus(const at::Tensor& self, const at::Scalar& beta,
                    const at::Scalar& threshold) {
    return unary(self, UnaryCode::Softplus, static_cast<float>(beta.toDouble()),
                 static_cast<float>(threshold.toDouble()));
}

at::Tensor clamp_scalar(const at::Tensor& self, const std::optional<at::Scalar>& min,
                        const std::optional<at::Scalar>& max) {
    const float lo = min.has_value() ? static_cast<float>(min->toDouble())
                                     : -std::numeric_limits<float>::infinity();
    const float hi = max.has_value() ? static_cast<float>(max->toDouble())
                                     : std::numeric_limits<float>::infinity();
    return unary(self, UnaryCode::Clamp, lo, hi);
}

template <typename T>
at::Tensor div_impl(const at::Tensor& self, const at::Tensor& other) {
    check_tensor(self, "div");
    check_tensor(other, "div");
    const auto out_shape = at::infer_size(self.sizes(), other.sizes());
    auto out = at::empty(out_shape, self.options());
    ContextGuard guard(self.device().index());
    int ret = 0;
    if (self.sizes().equals(other.sizes())) {
        ret = baidu::xpu::api::div<T>(guard.dc.ctx, cptr<T>(self), cptr<T>(other),
                                      mptr<T>(out), self.numel());
    } else {
        ret = baidu::xpu::api::broadcast_div<T>(
            guard.dc.ctx, cptr<T>(self), cptr<T>(other), mptr<T>(out),
            self.sizes().vec(), other.sizes().vec());
    }
    TORCH_CHECK(ret == 0, "xdnn::div failed: ret=", ret);
    return out;
}

at::Tensor div_tensor(const at::Tensor& self, const at::Tensor& other) {
    TORCH_CHECK(self.scalar_type() == other.scalar_type(), "div dtype mismatch");
    if (self.scalar_type() == at::kHalf) return div_impl<float16>(self, other);
    if (self.scalar_type() == at::kFloat) return div_impl<float>(self, other);
    TORCH_CHECK(false, "kunlun div only supports float16/float32");
}

at::Tensor div_scalar(const at::Tensor& self, const at::Scalar& other) {
    const double value = other.toDouble();
    TORCH_CHECK(value != 0.0, "division by zero");
    check_tensor(self, "div.Scalar");
    auto out = at::empty_like(self);
    ContextGuard guard(self.device().index());
    int ret = 0;
    if (self.scalar_type() == at::kHalf) {
        ret = baidu::xpu::api::scale<float16>(
            guard.dc.ctx, cptr<float16>(self), mptr<float16>(out), self.numel(),
            false, static_cast<float>(1.0 / value), 0.0f);
    } else {
        ret = baidu::xpu::api::scale<float>(
            guard.dc.ctx, cptr<float>(self), mptr<float>(out), self.numel(),
            false, static_cast<float>(1.0 / value), 0.0f);
    }
    TORCH_CHECK(ret == 0, "xdnn::scale (div.Scalar) failed: ret=", ret);
    return out;
}

at::Tensor log_softmax_op(const at::Tensor& self, int64_t dim,
                       std::optional<at::ScalarType> dtype) {
    TORCH_CHECK(!dtype.has_value() || *dtype == self.scalar_type(),
                "kunlun log_softmax dtype must be omitted/same");
    check_tensor(self, "log_softmax");
    const int64_t axis = dim < 0 ? dim + self.dim() : dim;
    auto out = at::empty_like(self);
    ContextGuard guard(self.device().index());
    int ret = 0;
    if (self.scalar_type() == at::kHalf) {
        ret = baidu::xpu::api::log_softmax<float16>(
            guard.dc.ctx, cptr<float16>(self), mptr<float16>(out),
            self.sizes().vec(), axis);
    } else {
        ret = baidu::xpu::api::log_softmax<float>(
            guard.dc.ctx, cptr<float>(self), mptr<float>(out),
            self.sizes().vec(), axis);
    }
    TORCH_CHECK(ret == 0, "xdnn::log_softmax failed: ret=", ret);
    return out;
}

at::Tensor _log_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
    TORCH_CHECK(!half_to_float, "kunlun _log_softmax half_to_float=True unsupported");
    return log_softmax_op(self, dim, std::nullopt);
}

template <typename T>
at::Tensor avg_pool2d_impl(const at::Tensor& self, at::IntArrayRef kernel,
                           at::IntArrayRef stride, at::IntArrayRef padding,
                           bool ceil_mode, bool count_include_pad,
                           std::optional<int64_t> divisor_override) {
    check_tensor(self, "avg_pool2d");
    TORCH_CHECK(self.dim() == 4, "avg_pool2d requires 4-D input");
    TORCH_CHECK(!ceil_mode, "kunlun avg_pool2d does not support ceil_mode=True");
    TORCH_CHECK(!divisor_override.has_value(), "kunlun avg_pool2d divisor_override unsupported");
    const auto k = pair(kernel);
    const auto s = stride.empty() ? k : pair(stride);
    const auto p = pair(padding);
    const int64_t n = self.size(0), c = self.size(1), h = self.size(2), w = self.size(3);
    const int64_t oh = (h + 2 * p[0] - k[0]) / s[0] + 1;
    const int64_t ow = (w + 2 * p[1] - k[1]) / s[1] + 1;
    TORCH_CHECK(oh > 0 && ow > 0, "avg_pool2d output shape non-positive");
    auto out = at::empty({n, c, oh, ow}, self.options());
    ContextGuard guard(self.device().index());
    const int ret = baidu::xpu::api::avg_pool2d<T>(
        guard.dc.ctx, cptr<T>(self), mptr<T>(out), n, c, h, w, k, s, p,
        count_include_pad, /*is_nchw=*/true);
    TORCH_CHECK(ret == 0, "xdnn::avg_pool2d failed: ret=", ret);
    return out;
}


at::Tensor empty_private(c10::SymIntArrayRef size,
                         const std::optional<at::ScalarType>& dtype,
                         const std::optional<at::Layout>& layout,
                         const std::optional<at::Device>& device,
                         const std::optional<bool>& pin_memory) {
    TORCH_CHECK(!layout.has_value() || *layout == at::Layout::Strided,
                "kunlun creation only supports strided layout");
    TORCH_CHECK(!pin_memory.value_or(false), "kunlun creation does not support pin_memory");
    auto opts = at::TensorOptions()
                    .dtype(dtype.value_or(at::kFloat))
                    .layout(at::Layout::Strided)
                    .device(device.value_or(at::Device(at::DeviceType::PrivateUse1,
                                                       current_device())))
                    .pinned_memory(false);
    return at::empty_symint(size, opts);
}

at::Tensor zeros(c10::SymIntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<at::Device> device,
                 std::optional<bool> pin_memory) {
    auto out = empty_private(size, dtype, layout, device, pin_memory);
    return out.fill_(0);
}

at::Tensor ones(c10::SymIntArrayRef size, std::optional<at::ScalarType> dtype,
                std::optional<at::Layout> layout, std::optional<at::Device> device,
                std::optional<bool> pin_memory) {
    auto out = empty_private(size, dtype, layout, device, pin_memory);
    return out.fill_(1);
}

at::Tensor full(c10::SymIntArrayRef size, const at::Scalar& fill_value,
                std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                std::optional<at::Device> device, std::optional<bool> pin_memory) {
    auto out = empty_private(size, dtype, layout, device, pin_memory);
    return out.fill_(fill_value);
}

at::Tensor zeros_like(const at::Tensor& self, std::optional<at::ScalarType> dtype,
                      std::optional<at::Layout> layout, std::optional<at::Device> device,
                      std::optional<bool> pin_memory,
                      std::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(!layout.has_value() || *layout == at::Layout::Strided,
                "kunlun zeros_like only supports strided");
    TORCH_CHECK(!pin_memory.value_or(false), "kunlun zeros_like does not support pin_memory");
    auto opts = self.options().dtype(dtype.value_or(self.scalar_type()))
                    .device(device.value_or(self.device()));
    auto out = at::empty(self.sizes(), opts);
    return out.fill_(0);
}

at::Tensor ones_like(const at::Tensor& self, std::optional<at::ScalarType> dtype,
                     std::optional<at::Layout> layout, std::optional<at::Device> device,
                     std::optional<bool> pin_memory,
                     std::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(!layout.has_value() || *layout == at::Layout::Strided,
                "kunlun ones_like only supports strided");
    TORCH_CHECK(!pin_memory.value_or(false), "kunlun ones_like does not support pin_memory");
    auto opts = self.options().dtype(dtype.value_or(self.scalar_type()))
                    .device(device.value_or(self.device()));
    auto out = at::empty(self.sizes(), opts);
    return out.fill_(1);
}

at::Tensor full_like(const at::Tensor& self, const at::Scalar& fill_value,
                     std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                     std::optional<at::Device> device, std::optional<bool> pin_memory,
                     std::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(!layout.has_value() || *layout == at::Layout::Strided,
                "kunlun full_like only supports strided");
    TORCH_CHECK(!pin_memory.value_or(false), "kunlun full_like does not support pin_memory");
    auto opts = self.options().dtype(dtype.value_or(self.scalar_type()))
                    .device(device.value_or(self.device()));
    auto out = at::empty(self.sizes(), opts);
    return out.fill_(fill_value);
}
at::Tensor stack(const at::TensorList& tensors, int64_t dim) {
    TORCH_CHECK(!tensors.empty(), "stack got an empty tensor list");
    const auto& first = tensors[0];
    check_tensor(first, "stack");
    const int64_t rank = first.dim() + 1;
    if (dim < 0) dim += rank;
    TORCH_CHECK(dim >= 0 && dim < rank, "stack dim out of range");
    std::vector<at::Tensor> expanded;
    expanded.reserve(tensors.size());
    for (const auto& t : tensors) {
        check_tensor(t, "stack");
        TORCH_CHECK(t.sizes().equals(first.sizes()), "stack requires equal shapes");
        expanded.push_back(t.unsqueeze(dim));
    }
    return at::cat(expanded, dim);
}

at::Tensor avg_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                      at::IntArrayRef stride, at::IntArrayRef padding,
                      bool ceil_mode, bool count_include_pad,
                      std::optional<int64_t> divisor_override) {
    if (self.scalar_type() == at::kHalf) {
        return avg_pool2d_impl<float16>(self, kernel_size, stride, padding,
                                        ceil_mode, count_include_pad, divisor_override);
    }
    if (self.scalar_type() == at::kFloat) {
        return avg_pool2d_impl<float>(self, kernel_size, stride, padding,
                                      ceil_mode, count_include_pad, divisor_override);
    }
    TORCH_CHECK(false, "kunlun avg_pool2d only supports float16/float32");
}

}  // namespace
}  // namespace torch_kunlun

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("abs", TORCH_FN(torch_kunlun::abs));
    m.impl("neg", TORCH_FN(torch_kunlun::neg));
    m.impl("exp", TORCH_FN(torch_kunlun::exp));
    m.impl("log", TORCH_FN(torch_kunlun::log));
    m.impl("sqrt", TORCH_FN(torch_kunlun::sqrt));
    m.impl("rsqrt", TORCH_FN(torch_kunlun::rsqrt));
    m.impl("sigmoid", TORCH_FN(torch_kunlun::sigmoid));
    m.impl("tanh", TORCH_FN(torch_kunlun::tanh));
    m.impl("silu", TORCH_FN(torch_kunlun::silu));
    m.impl("leaky_relu", TORCH_FN(torch_kunlun::leaky_relu));
    m.impl("elu", TORCH_FN(torch_kunlun::elu));
    m.impl("softplus", TORCH_FN(torch_kunlun::softplus));
    m.impl("clamp", TORCH_FN(torch_kunlun::clamp_scalar));
    m.impl("square", TORCH_FN(torch_kunlun::square));
    m.impl("reciprocal", TORCH_FN(torch_kunlun::reciprocal));
    m.impl("sign", TORCH_FN(torch_kunlun::sign));
    m.impl("ceil", TORCH_FN(torch_kunlun::ceil));
    m.impl("floor", TORCH_FN(torch_kunlun::floor));
    m.impl("round", TORCH_FN(torch_kunlun::round));
    m.impl("log1p", TORCH_FN(torch_kunlun::log1p));
    m.impl("exp2", TORCH_FN(torch_kunlun::exp2));
    m.impl("erf", TORCH_FN(torch_kunlun::erf));
    m.impl("erfinv", TORCH_FN(torch_kunlun::erfinv));
    m.impl("selu", TORCH_FN(torch_kunlun::selu));
    m.impl("celu", TORCH_FN(torch_kunlun::celu));
    m.impl("mish", TORCH_FN(torch_kunlun::mish));
    m.impl("hardswish", TORCH_FN(torch_kunlun::hardswish));
    m.impl("hardsigmoid", TORCH_FN(torch_kunlun::hardsigmoid));
    m.impl("hardtanh", TORCH_FN(torch_kunlun::hardtanh));
    m.impl("softshrink", TORCH_FN(torch_kunlun::softshrink));
    m.impl("threshold", TORCH_FN(torch_kunlun::threshold));
    m.impl("div.Tensor", TORCH_FN(torch_kunlun::div_tensor));
    m.impl("div.Scalar", TORCH_FN(torch_kunlun::div_scalar));
    m.impl("log_softmax.int", TORCH_FN(torch_kunlun::log_softmax_op));
    m.impl("_log_softmax", TORCH_FN(torch_kunlun::_log_softmax));
    m.impl("avg_pool2d", TORCH_FN(torch_kunlun::avg_pool2d));
    m.impl("sum.dim_IntList", TORCH_FN(torch_kunlun::sum_dim));
    m.impl("sum", TORCH_FN(torch_kunlun::sum_all));
    m.impl("mean.dim", TORCH_FN(torch_kunlun::mean_dim));
    m.impl("mean", TORCH_FN(torch_kunlun::mean_all));
    m.impl("amax", TORCH_FN(torch_kunlun::amax));
    m.impl("amin", TORCH_FN(torch_kunlun::amin));
    m.impl("argmax", TORCH_FN(torch_kunlun::argmax));
    m.impl("argmin", TORCH_FN(torch_kunlun::argmin));
    m.impl("logsumexp", TORCH_FN(torch_kunlun::logsumexp));
    m.impl("where.self", TORCH_FN(torch_kunlun::where));
    m.impl("masked_fill.Scalar", TORCH_FN(torch_kunlun::masked_fill));
    m.impl("triu", TORCH_FN(torch_kunlun::triu));
    m.impl("tril", TORCH_FN(torch_kunlun::tril));
    m.impl("stack", TORCH_FN(torch_kunlun::stack));
    m.impl("maximum", TORCH_FN(torch_kunlun::maximum));
    m.impl("minimum", TORCH_FN(torch_kunlun::minimum));
    m.impl("addcmul", TORCH_FN(torch_kunlun::addcmul));
    m.impl("addcdiv", TORCH_FN(torch_kunlun::addcdiv));
    m.impl("zeros", TORCH_FN(torch_kunlun::zeros));
    m.impl("ones", TORCH_FN(torch_kunlun::ones));
    m.impl("full", TORCH_FN(torch_kunlun::full));
    m.impl("zeros_like", TORCH_FN(torch_kunlun::zeros_like));
    m.impl("ones_like", TORCH_FN(torch_kunlun::ones_like));
    m.impl("full_like", TORCH_FN(torch_kunlun::full_like));
}
