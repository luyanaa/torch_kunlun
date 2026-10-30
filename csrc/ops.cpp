#include "kunlun.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

#include "xpu/refactor/math.h"
#include "xpu/refactor/nn.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace torch_kunlun {
namespace {

using baidu::xpu::api::Context;

void check_same_dtype(const at::Tensor& a, const at::Tensor& b, const char* what) {
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), what,
                " requires matching dtypes, got ", a.scalar_type(), " and ", b.scalar_type());
}

void check_contiguous(const at::Tensor& t, const char* what) {
    TORCH_CHECK(t.is_contiguous(), what, " requires contiguous tensors");
}

template <typename T>
const T* cptr(const at::Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}

template <typename T>
T* mptr(at::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

// ---- device/host transfer helpers -----------------------------------------
// xpu_memcpy has no stream argument: it executes on XPURT's default stream.
// Every transfer therefore drains the caller's stream first (so the copy sees
// earlier launches) and waits for the copy before returning (so later launches
// on the caller's stream see the result).

void drain_stream(int device) {
    xpu_set_device(device);
    check_xpu(xpu_wait(current_stream_native(device)), "xpu_wait(stream drain)");
}

void memcpy_d2h(int device, void* host, const void* dev, size_t bytes) {
    drain_stream(device);
    check_xpu(xpu_memcpy(host, dev, bytes, XPU_DEVICE_TO_HOST), "xpu_memcpy D2H");
    check_xpu(xpu_wait(nullptr), "xpu_wait D2H");
}

void memcpy_h2d(int device, void* dev, const void* host, size_t bytes) {
    drain_stream(device);
    check_xpu(xpu_memcpy(dev, host, bytes, XPU_HOST_TO_DEVICE), "xpu_memcpy H2D");
    check_xpu(xpu_wait(nullptr), "xpu_wait H2D");
}

void memcpy_d2d(int device, void* dst, const void* src, size_t bytes) {
    drain_stream(device);
    check_xpu(xpu_memcpy(dst, src, bytes, XPU_DEVICE_TO_DEVICE), "xpu_memcpy D2D");
    check_xpu(xpu_wait(nullptr), "xpu_wait D2D");
}

void* host_staging_alloc(size_t bytes, bool* pinned) {
    void* host = nullptr;
    *pinned = xpu_host_alloc(&host, bytes, 0) == 0 && host != nullptr;
    if (!*pinned) {
        host = std::malloc(bytes);
    }
    return host;
}

void host_staging_free(void* host, bool pinned) {
    if (pinned) {
        xpu_host_free(host);
    } else {
        std::free(host);
    }
}

template <typename T>
T scalar_value(const at::Scalar& value) {
    if constexpr (std::is_same_v<T, bool>) {
        return value.toBool();
    } else if constexpr (std::is_integral_v<T>) {
        TORCH_CHECK(!value.isFloatingPoint() || std::trunc(value.toDouble()) == value.toDouble(),
                    "kunlun fill_ value ", value.toDouble(), " is not an integer");
        const int64_t v = value.toLong();
        if constexpr (sizeof(T) < sizeof(int64_t)) {
            TORCH_CHECK(v >= static_cast<int64_t>(std::numeric_limits<T>::min()) &&
                            v <= static_cast<int64_t>(std::numeric_limits<T>::max()),
                        "kunlun fill_ value ", v, " overflows ", sizeof(T) * 8, "-bit dtype");
        }
        return static_cast<T>(v);
    } else {
        return static_cast<T>(value.toDouble());
    }
}

template <>
float16 scalar_value<float16>(const at::Scalar& value) {
    return float16(static_cast<float>(value.toDouble()));
}

template <typename T>
at::Tensor add_impl(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    TORCH_CHECK(alpha.toDouble() == 1.0, "kunlun add only supports alpha=1");
    check_same_dtype(self, other, "add");
    check_contiguous(self, "add");
    check_contiguous(other, "add");

    const auto out_shape = at::infer_size(self.sizes(), other.sizes());
    auto out = at::empty(out_shape, self.options());
    ContextGuard guard(self.device().index(), out.numel());
    if (self.sizes().equals(other.sizes())) {
        TORCH_CHECK(baidu::xpu::api::add<T>(
                        guard.dc.ctx, cptr<T>(self), cptr<T>(other), mptr<T>(out),
                        self.numel()) == 0,
                    "xdnn::add failed");
    } else if (other.sizes().equals(out_shape)) {
        // XDNN broadcasts x onto y: x is the narrow operand, y holds the output
        // shape (see xtcl gemm_impl.h: broadcast_add(ctx, bias, C, C, {N}, {M,N})).
        TORCH_CHECK(baidu::xpu::api::broadcast_add<T>(
                        guard.dc.ctx, cptr<T>(self), cptr<T>(other), mptr<T>(out),
                        self.sizes().vec(), out_shape) == 0,
                    "xdnn::broadcast_add failed");
    } else if (self.sizes().equals(out_shape)) {
        TORCH_CHECK(baidu::xpu::api::broadcast_add<T>(
                        guard.dc.ctx, cptr<T>(other), cptr<T>(self), mptr<T>(out),
                        other.sizes().vec(), out_shape) == 0,
                    "xdnn::broadcast_add failed");
    } else {
        TORCH_CHECK(false, "kunlun add does not support two-way broadcasting: ",
                    self.sizes(), " + ", other.sizes());
    }
    return out;
}

template <typename T>
at::Tensor relu_impl(const at::Tensor& self, bool inplace) {
    check_contiguous(self, "relu");
    at::Tensor out = inplace ? self : at::empty(self.sizes(), self.options());
    ContextGuard guard(self.device().index());
    TORCH_CHECK(baidu::xpu::api::relu<T>(
                    guard.dc.ctx, cptr<T>(self), mptr<T>(out), self.numel(), nullptr, nullptr) == 0,
                "xdnn::relu failed");
    return out;
}

template <typename T>
at::Tensor& fill_impl(at::Tensor& self, const at::Scalar& value) {
    check_contiguous(self, "fill_");
    const T v = scalar_value<T>(value);
    ContextGuard guard(self.device().index(), self.numel());
    TORCH_CHECK(baidu::xpu::api::constant<T>(guard.dc.ctx, mptr<T>(self), self.numel(), v) == 0,
                "xdnn::constant failed");
    return self;
}

// XPURT has no memset, so non-floating dtypes are filled by broadcasting a
// host pattern block.  The DMA length is kept well below the 32-bit limit.
template <typename T>
at::Tensor& fill_pattern(at::Tensor& self, T v) {
    const int64_t numel = self.numel();
    if (numel == 0) {
        return self;
    }
    // 1 MiB per DMA keeps the transfer count low while staying far below the
    // runtime's 32-bit length limit.
    constexpr size_t kChunkBytes = 1ULL << 20;
    const int64_t max_chunk = static_cast<int64_t>(kChunkBytes / sizeof(T));
    const int64_t chunk = std::min<int64_t>(numel, std::max<int64_t>(1, max_chunk));
    const size_t chunk_bytes = static_cast<size_t>(chunk) * sizeof(T);
    bool pinned = false;
    void* host = host_staging_alloc(chunk_bytes, &pinned);
    TORCH_CHECK(host != nullptr, "kunlun fill_ staging allocation failed");
    auto* pattern = static_cast<unsigned char*>(host);
    for (int64_t i = 0; i < chunk; ++i) {
        std::memcpy(pattern + static_cast<size_t>(i) * sizeof(T), &v, sizeof(T));
    }
    const size_t total = static_cast<size_t>(numel) * sizeof(T);
    auto* dst = static_cast<unsigned char*>(self.data_ptr());
    drain_stream(self.device().index());
    size_t done = 0;
    int ret = 0;
    while (done < total && ret == 0) {
        const size_t n = std::min(chunk_bytes, total - done);
        ret = xpu_memcpy(dst + done, pattern, n, XPU_HOST_TO_DEVICE);
        done += n;
    }
    const int wait_ret = xpu_wait(nullptr);
    host_staging_free(host, pinned);
    check_xpu(ret, "xpu_memcpy H2D fill_");
    check_xpu(wait_ret, "xpu_wait fill_");
    return self;
}

template <typename T, typename TGEMM>
at::Tensor fc_impl(const at::Tensor& x, const at::Tensor& w, bool w_trans) {
    check_same_dtype(x, w, "fc");
    check_contiguous(x, "fc");
    check_contiguous(w, "fc");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "kunlun fc requires 2-D tensors");

    const int64_t m = x.size(0);
    const int64_t k = w_trans ? w.size(1) : x.size(1);
    const int64_t n = w_trans ? w.size(0) : w.size(1);
    TORCH_CHECK((w_trans ? x.size(1) : w.size(0)) == k, "kunlun fc inner dimensions mismatch");

    auto out = at::empty({m, n}, x.options());
    ContextGuard guard(x.device().index(), m * n * k);
    const int fc_ret = baidu::xpu::api::fc<T, T, T, TGEMM>(
            guard.dc.ctx, cptr<T>(x), cptr<T>(w), mptr<T>(out),
            m, n, k, /*x_trans=*/false, w_trans,
            /*x_maxptr=*/nullptr, /*w_maxptr=*/nullptr, /*y_maxptr=*/nullptr);
    TORCH_CHECK(fc_ret == 0, "xdnn::fc failed: ret=", fc_ret);
    return out;
}

std::vector<int64_t> pair_from_int_array(at::IntArrayRef value) {
    TORCH_CHECK(value.size() == 1 || value.size() == 2, "expected 1 or 2 element array");
    return {value[0], value.size() == 1 ? value[0] : value[1]};
}

template <typename T, typename TGEMM>
at::Tensor conv2d_impl(const at::Tensor& input, const at::Tensor& weight,
                       const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                       at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    check_same_dtype(input, weight, "conv2d");
    check_contiguous(input, "conv2d");
    check_contiguous(weight, "conv2d");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4,
                "kunlun conv2d requires 4-D NCHW input and OIHW weight");

    const int64_t n = input.size(0);
    const int64_t c = input.size(1);
    const int64_t h = input.size(2);
    const int64_t w = input.size(3);
    const int64_t f = weight.size(0);
    const int64_t kh = weight.size(2);
    const int64_t kw = weight.size(3);

    const auto stride_v = pair_from_int_array(stride);
    const auto pad_v = pair_from_int_array(padding);
    const auto dilation_v = pair_from_int_array(dilation);
    const int64_t oh = (h + 2 * pad_v[0] - dilation_v[0] * (kh - 1) - 1) / stride_v[0] + 1;
    const int64_t ow = (w + 2 * pad_v[1] - dilation_v[1] * (kw - 1) - 1) / stride_v[1] + 1;
    TORCH_CHECK(oh > 0 && ow > 0, "kunlun conv2d produced non-positive output shape");

    auto out = at::empty({n, f, oh, ow}, input.options());
    ContextGuard guard(input.device().index(), n * f * oh * ow * c * kh * kw);
    const int conv_ret = baidu::xpu::api::conv2d<T, T, T, TGEMM>(
            guard.dc.ctx, cptr<T>(input), cptr<T>(weight), mptr<T>(out),
            n, c, h, w, f, {kh, kw}, stride_v, pad_v, dilation_v, groups,
            /*x_maxptr=*/nullptr, /*weight_maxptr=*/nullptr, /*y_maxptr=*/nullptr,
            /*is_nchw=*/true);
    TORCH_CHECK(conv_ret == 0, "xdnn::conv2d failed: ret=", conv_ret);

    if (bias.has_value()) {
        check_same_dtype(input, *bias, "conv2d bias");
        check_contiguous(*bias, "conv2d bias");
        TORCH_CHECK(bias->numel() == f, "kunlun conv2d bias size mismatch");
        // XDNN broadcasts with numpy right-alignment, so a {f} shape would be
        // matched against the W axis; {1,f,1,1} matches the channel axis.
        auto tmp = at::empty_like(out);
        TORCH_CHECK(baidu::xpu::api::broadcast_add<T>(
                        guard.dc.ctx, cptr<T>(*bias), cptr<T>(out), mptr<T>(tmp),
                        {1, f, 1, 1}, {n, f, oh, ow}) == 0,
                    "xdnn::broadcast_add (conv2d bias) failed");
        out = tmp;
    }
    return out;
}

}  // namespace

at::Tensor materialize_private_contiguous(const at::Tensor& src) {
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
                "materialize_private_contiguous requires a kunlun tensor");
    if (src.is_contiguous()) {
        return src;
    }
    if (src.numel() == 0) {
        return at::empty(src.sizes(), src.options());
    }

    const int64_t elem_size = src.element_size();
    const int64_t storage_elems =
        static_cast<int64_t>(src.storage().nbytes()) / elem_size;
    // Stage only the address range the view touches: the storage may be much
    // larger than the view (one row of a big matrix, a slice of a buffer), and
    // copying the whole storage is both slower and an unnecessary DMA.
    int64_t min_elem = 0;
    int64_t max_elem = storage_elems - 1;
    bool negative_stride = false;
    for (int64_t d = 0; d < src.dim(); ++d) {
        negative_stride = negative_stride || src.stride(d) < 0;
    }
    if (!negative_stride) {
        min_elem = src.storage_offset();
        max_elem = min_elem;
        for (int64_t d = 0; d < src.dim(); ++d) {
            if (src.size(d) > 1) {
                max_elem += (src.size(d) - 1) * src.stride(d);
            }
        }
        TORCH_CHECK(min_elem >= 0 && max_elem >= min_elem && max_elem < storage_elems,
                    "kunlun contiguous materialization found an out-of-range view");
    }
    const size_t offset_bytes = static_cast<size_t>(min_elem) * elem_size;
    const size_t bytes = static_cast<size_t>(max_elem - min_elem + 1) * elem_size;

    bool pinned = false;
    void* host = host_staging_alloc(bytes, &pinned);
    TORCH_CHECK(host != nullptr, "kunlun contiguous staging allocation failed");
    const int device = src.device().index();
    const auto* base = static_cast<const unsigned char*>(src.storage().data_ptr().get());
    memcpy_d2h(device, host, base + offset_bytes, bytes);

    auto cpu_bytes = at::from_blob(
        host, {static_cast<int64_t>(bytes / elem_size)},
        [pinned](void* ptr) {
            if (pinned) {
                xpu_host_free(ptr);
            } else {
                std::free(ptr);
            }
        },
        src.options().device(at::kCPU));
    auto cpu_view = cpu_bytes.as_strided(src.sizes(), src.strides(),
                                         src.storage_offset() - min_elem);
    auto cpu_contig = cpu_view.contiguous();

    auto dst = at::empty(src.sizes(), src.options());
    memcpy_h2d(device, dst.data_ptr(), cpu_contig.data_ptr(),
               cpu_contig.numel() * cpu_contig.element_size());
    return dst;
}

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    check_same_dtype(self, src, "copy_");
    check_contiguous(self, "copy_ destination");
    // Safe subset until broadcast/dtype conversion is implemented: never read
    // more bytes from src than its allocation owns.
    TORCH_CHECK(self.sizes().equals(src.sizes()),
                "kunlun copy_ requires equal shapes; broadcast copy is not implemented");
    if (self.data_ptr() == src.data_ptr() && self.device() == src.device() &&
        src.is_contiguous()) {
        // Same contiguous storage and byte range: nothing to move.  A
        // non-contiguous src with the same base pointer (x.copy_(x.t()) on a
        // square tensor) is a real transpose and must still be copied.
        return self;
    }
    at::Tensor source;
    if (src.is_contiguous()) {
        source = src;
    } else if (src.device().type() == c10::DeviceType::PrivateUse1) {
        source = materialize_private_contiguous(src);
    } else {
        source = src.contiguous();
    }

    const auto self_type = self.device().type();
    const auto src_type = source.device().type();
    const size_t bytes = self.numel() * self.element_size();
    if (bytes == 0) {
        return self;
    }

    if (self_type == c10::DeviceType::PrivateUse1 &&
        src_type == c10::DeviceType::CPU) {
        memcpy_h2d(self.device().index(), self.data_ptr(), source.data_ptr(), bytes);
    } else if (self_type == c10::DeviceType::CPU &&
               src_type == c10::DeviceType::PrivateUse1) {
        memcpy_d2h(source.device().index(), self.data_ptr(), source.data_ptr(), bytes);
    } else if (self_type == c10::DeviceType::PrivateUse1 &&
               src_type == c10::DeviceType::PrivateUse1) {
        const int dst_device = self.device().index();
        const int src_device = source.device().index();
        const auto dst_addr = reinterpret_cast<uintptr_t>(self.data_ptr());
        const auto src_addr = reinterpret_cast<uintptr_t>(source.data_ptr());
        // Partially overlapping ranges inside one storage (x[1:].copy_(x[:-1]))
        // cannot be copied in place; stage through the host instead.
        const bool overlap = dst_addr < src_addr + bytes && src_addr < dst_addr + bytes;
        if (dst_device == src_device && !overlap) {
            memcpy_d2d(dst_device, self.data_ptr(), source.data_ptr(), bytes);
        } else {
            // Cross-PD staging also covers the overlap case.  This avoids
            // xpu_memcpy_peer, which only works between same-card PDs.
            bool pinned = false;
            void* host = host_staging_alloc(bytes, &pinned);
            TORCH_CHECK(host != nullptr, "kunlun copy_ host staging allocation failed");
            memcpy_d2h(src_device, host, source.data_ptr(), bytes);
            memcpy_h2d(dst_device, self.data_ptr(), host, bytes);
            host_staging_free(host, pinned);
        }
    } else {
        TORCH_CHECK(false, "kunlun copy_ unsupported device combination");
    }

    // xpu_memcpy is synchronous: every branch above completed the transfer
    // before returning, so non_blocking cannot defer anything.  XPURT has no
    // stream-aware memcpy and the host staging buffer is released here.
    (void)non_blocking;
    return self;
}


at::Tensor clone(const at::Tensor& self, std::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(!memory_format.has_value() ||
                    *memory_format == at::MemoryFormat::Contiguous ||
                    *memory_format == at::MemoryFormat::Preserve,
                "kunlun clone only supports contiguous/preserve");
    auto out = at::empty(self.sizes(), self.options());
    return ::torch_kunlun::copy_(out, self, /*non_blocking=*/false);
}

at::Tensor empty_memory_format(at::SymIntArrayRef size,
                               std::optional<at::ScalarType> dtype_opt,
                               std::optional<at::Layout> layout_opt,
                               std::optional<at::Device> device_opt,
                               std::optional<bool> pin_memory_opt,
                               std::optional<at::MemoryFormat> memory_format_opt) {
    TORCH_CHECK(!layout_opt.has_value() || *layout_opt == at::Layout::Strided,
                "kunlun empty only supports strided layout");
    TORCH_CHECK(!pin_memory_opt.value_or(false),
                "kunlun empty does not support pinned memory");
    const auto device = device_opt.value_or(at::Device(at::DeviceType::PrivateUse1,
                                                       current_device()));
    TORCH_CHECK(device.type() == at::DeviceType::PrivateUse1,
                "kunlun empty received a non-kunlun device");
    const int device_index = device.index() < 0 ? current_device() : device.index();
    const int old_device = current_device();
    set_device(device_index);
    auto base = at::detail::empty_generic_symint(
        size, get_allocator(), at::DispatchKeySet(at::DispatchKey::PrivateUse1),
        at::dtype_or_default(dtype_opt), memory_format_opt);
    set_device(old_device);
    return at::Tensor(std::move(base));
}

at::Tensor empty_strided(at::SymIntArrayRef size,
                         at::SymIntArrayRef stride,
                         std::optional<at::ScalarType> dtype_opt,
                         std::optional<at::Layout> layout_opt,
                         std::optional<at::Device> device_opt,
                         std::optional<bool> pin_memory_opt) {
    TORCH_CHECK(!layout_opt.has_value() || *layout_opt == at::Layout::Strided,
                "kunlun empty_strided only supports strided layout");
    TORCH_CHECK(!pin_memory_opt.value_or(false),
                "kunlun empty_strided does not support pinned memory");
    const auto device = device_opt.value_or(at::Device(at::DeviceType::PrivateUse1,
                                                       current_device()));
    TORCH_CHECK(device.type() == at::DeviceType::PrivateUse1,
                "kunlun empty_strided received a non-kunlun device");
    const int device_index = device.index() < 0 ? current_device() : device.index();
    const int old_device = current_device();
    set_device(device_index);
    auto base = at::detail::empty_strided_symint_generic(
        size, stride, get_allocator(), at::DispatchKeySet(at::DispatchKey::PrivateUse1),
        at::dtype_or_default(dtype_opt));
    set_device(old_device);
    return at::Tensor(std::move(base));
}

at::Tensor add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);

at::Tensor linear(const at::Tensor& input, const at::Tensor& weight,
                  const std::optional<at::Tensor>& bias) {
    TORCH_CHECK(input.dim() == 2, "kunlun linear only supports 2-D input");
    at::Tensor out;
    if (input.scalar_type() == at::kHalf) {
        out = fc_impl<float16, short>(input, weight, true);
    } else if (input.scalar_type() == at::kFloat) {
        out = fc_impl<float, float>(input, weight, true);
    } else {
        TORCH_CHECK(false, "kunlun linear only supports float16/float32");
    }
    if (bias.has_value()) {
        out = ::torch_kunlun::add(out, *bias, 1);
    }
    return out;
}

at::Tensor& fill_(at::Tensor& self, const at::Scalar& value) {
    if (!self.is_contiguous()) {
        // XDNN constant and the host-pattern path both write a contiguous
        // range.  Walk the outer dimension instead of staging the whole
        // tensor; the recursion stops at contiguous sub-tensors.  Small
        // tensors only: a transposed large view would otherwise turn into one
        // tiny DMA per element.
        constexpr int64_t kMaxStridedFill = 1 << 12;
        TORCH_CHECK(self.numel() <= kMaxStridedFill,
                    "kunlun fill_ on a non-contiguous tensor is limited to ",
                    kMaxStridedFill, " elements; materialize it with .contiguous() first");
        TORCH_CHECK(self.dim() > 0, "kunlun fill_ requires a contiguous tensor");
        for (int64_t i = 0; i < self.size(0); ++i) {
            auto row = self.select(0, i);
            fill_(row, value);
        }
        return self;
    }
    switch (self.scalar_type()) {
        case at::kHalf:
            return fill_impl<float16>(self, value);
        case at::kFloat:
            return fill_impl<float>(self, value);
        case at::kDouble:
            return fill_pattern<double>(self, scalar_value<double>(value));
        case at::kBool:
            return fill_pattern<bool>(self, scalar_value<bool>(value));
        case at::kByte:
            return fill_pattern<uint8_t>(self, scalar_value<uint8_t>(value));
        case at::kChar:
            return fill_pattern<int8_t>(self, scalar_value<int8_t>(value));
        case at::kShort:
            return fill_pattern<int16_t>(self, scalar_value<int16_t>(value));
        case at::kInt:
            return fill_pattern<int32_t>(self, scalar_value<int32_t>(value));
        case at::kLong:
            return fill_pattern<int64_t>(self, scalar_value<int64_t>(value));
        default:
            TORCH_CHECK(false, "kunlun fill_ does not support dtype ", self.scalar_type());
    }
    return self;
}

at::Tensor& zero_(at::Tensor& self) {
    return ::torch_kunlun::fill_(self, 0);
}

at::Tensor add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    if (self.scalar_type() == at::kHalf) {
        return add_impl<float16>(self, other, alpha);
    }
    if (self.scalar_type() == at::kFloat) {
        return add_impl<float>(self, other, alpha);
    }
    TORCH_CHECK(false, "kunlun add only supports float16/float32");
    return {};
}

at::Tensor relu(const at::Tensor& self) {
    if (self.scalar_type() == at::kHalf) {
        return relu_impl<float16>(self, false);
    }
    if (self.scalar_type() == at::kFloat) {
        return relu_impl<float>(self, false);
    }
    TORCH_CHECK(false, "kunlun relu only supports float16/float32");
    return {};
}

at::Tensor& relu_(at::Tensor& self) {
    if (self.scalar_type() == at::kHalf) {
        relu_impl<float16>(self, true);
        return self;
    }
    if (self.scalar_type() == at::kFloat) {
        relu_impl<float>(self, true);
        return self;
    }
    TORCH_CHECK(false, "kunlun relu_ only supports float16/float32");
    return self;
}

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
    if (self.scalar_type() == at::kHalf) {
        return fc_impl<float16, short>(self, mat2, false);
    }
    if (self.scalar_type() == at::kFloat) {
        return fc_impl<float, float>(self, mat2, false);
    }
    TORCH_CHECK(false, "kunlun mm only supports float16/float32");
    return {};
}

at::Tensor conv2d(const at::Tensor& input, const at::Tensor& weight,
                  const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                  at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    if (input.scalar_type() == at::kHalf) {
        return conv2d_impl<float16, short>(input, weight, bias, stride, padding, dilation, groups);
    }
    if (input.scalar_type() == at::kFloat) {
        return conv2d_impl<float, float>(input, weight, bias, stride, padding, dilation, groups);
    }
    TORCH_CHECK(false, "kunlun conv2d only supports float16/float32");
    return {};
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    const std::string name = op.schema().name();
    // No PrivateUse1 kernel exists for this op: deny unless the user opts in.
    check_generic_fallback(name.c_str());
    at::native::cpu_fallback(op, stack);
}

}  // namespace torch_kunlun

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", TORCH_FN(torch_kunlun::empty_memory_format));
    m.impl("empty_strided", TORCH_FN(torch_kunlun::empty_strided));
    m.impl("copy_.Tensor", TORCH_FN(torch_kunlun::copy_));
    m.impl("clone", TORCH_FN(torch_kunlun::clone));
    m.impl("fill_.Scalar", TORCH_FN(torch_kunlun::fill_));
    m.impl("zero_", TORCH_FN(torch_kunlun::zero_));
    m.impl("add.Tensor", TORCH_FN(torch_kunlun::add));
    m.impl("relu", TORCH_FN(torch_kunlun::relu));
    m.impl("relu_", TORCH_FN(torch_kunlun::relu_));
    m.impl("mm", TORCH_FN(torch_kunlun::mm));
    m.impl("linear", TORCH_FN(torch_kunlun::linear));
    m.impl("conv2d", TORCH_FN(torch_kunlun::conv2d));
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&torch_kunlun::cpu_fallback>());
}
