# torch_kunlun operator TODO

This file records the remaining operator work.  "Native" means a PrivateUse1
kernel backed by XDNN.  "Composition" means it can be assembled from already
registered operators, but it still needs correctness/autograd tests.

## Implemented in this pass

Native forward + autograd:

- `abs`, `neg`, `exp`, `log`, `sqrt`, `rsqrt`
- `sigmoid`, `tanh`, `silu`
- `leaky_relu`, `elu`, `softplus`, `clamp`
- `square`, `reciprocal`, `sign`, `ceil`, `floor`, `round`
- `div.Tensor`, `div.Scalar`
- `log_softmax.int`, `_log_softmax`
- `avg_pool2d`
- `stack` (composition via `unsqueeze` + `cat`)

Native forward + autograd:

- `sum`, `sum.dim_IntList`, `mean`, `mean.dim`
- `amax`, `amin`, `logsumexp`
- `where.self`, `masked_fill.Scalar`, `triu`, `tril`

Non-differentiable forward:

- `argmax`, `argmin` (keepdim/flattened coverage still needs tests)

## Implemented extra elementwise / linalg

- `maximum`, `minimum`, `addcmul`, `addcdiv`
- `log1p`, `exp2`, `erf`, `erfinv`
- `selu`, `celu`, `mish`, `hardswish`, `hardsigmoid`, `hardtanh`,
  `softshrink`, `threshold`
- Broadcast/1-D `matmul` via CPU fallback; native 2D/3D/4D matching batch
- `einsum` via CPU host-staged fwd/bwd

## Implemented NN / linalg additions

- `nll_loss`: native XDNN forward/backward, CPU host-staged fallback if the ABI rejects
- `binary_cross_entropy`, `binary_cross_entropy_with_logits`, `l1_loss`,
  `smooth_l1_loss`, `kl_div`, `huber_loss`: CPU host-staged fwd/bwd
- Native XDNN `Adam`/`AdamW` step wrappers and `torch_kunlun.optim.Adam/AdamW`
- Native `group_norm`, `instance_norm`, `rms_norm`, `adaptive_max_pool2d`,
  `avg_pool3d`, `conv1d`, `conv3d`, `conv_transpose2d` with fwd+bwd
- Composition `matmul` (2D/3D/4D matching batch), `baddbmm`, `addbmm`
  with fwd+bwd

## Immediate autograd gaps

- None for the currently registered reduction/where/masking forward set;
  `argmax`/`argmin` are non-differentiable by design.
- `pow.Tensor_Tensor`, `pow.Tensor_Scalar`, `pow.Scalar`
- `addcmul`, `addcdiv`, `lerp`
- `floor_divide`, `remainder`, `fmod`
- `clamp.Tensor` (tensor min/max)

## Native operators still missing

### Elementwise / activation

- `silu_`, `sigmoid_`, `tanh_` and other in-place variants
- `softsign` (not a standalone ATen op in this build; compose `x/(1+abs(x))`)
- `addcmul`/`addcdiv` broadcast and in-place variants
- `pow` tensor/scalar combinations
- `fmod`, `floor_divide`, `remainder`
- `lerp`
- `addcmul_`, `addcdiv_`, `lerp_` in-place variants

### Reduction / sorting

- `var`, `std`, `norm`
- `cumsum`, `cumprod`, `logcumsumexp`
- `any`, `all`
- `topk`, `sort`, `unique`
- `argmax`/`argmin` keepdim and flattened-full fixes beyond current subset

### Shape / indexing

- `stack` has a composition implementation; still needs K200 test
- `split`, `chunk`, `repeat`, `expand`, `flatten`
- `index_select`, `gather`, `scatter`, `scatter_add`, `index_add`
- `masked_select`, `nonzero`
- `embedding_bag`
- `flip`, `roll`, `diagonal`, `trace`

### Creation / RNG

- `zeros`, `ones`, `full`, `zeros_like`, `ones_like`, `full_like` are native now
- `zeros.names`, `ones.names`, `full.names`
- `arange`, `linspace`, `eye`
- `rand`, `randn`, `randint`, `bernoulli`, `randperm`
- dropout training / XPU RNG

### NN (remaining)

- `embedding_bag`
- `interpolate`, `upsample`, `grid_sample`
- fused `scaled_dot_product_attention`
- native (non-CPU-staged) XDNN losses

### Linear algebra / optimizers

- broadcast `matmul` and 1-D cases; current matmul is 2D/3D/4D matching batch
- `einsum` currently CPU host-staged, not XDNN native
- `Adam`/`AdamW` are implemented; missing remaining optimizer in-place ops are still TODO

## 2026-10-28 safety review fixes

- `copy_`: equal-shape guard; host copies synchronize even with non_blocking.
- `addbmm` is now separate from `baddbmm` (batch sum vs batch output).
- XCCL `Work`: conservative synchronous completion; no early `is_completed`.
- `create_stream` selects the requested device; `getDeviceFromPtr` uses
  allocator ownership; global stream pool is per-device.
- `KUNLUN_CPU_FALLBACK` (default unset) rejects the generic boxed fallback
  while allowing deliberate host-staged kernels with a one-time notice;
  `error` rejects both, `warn`/`silent` allow both.

## 2026-10-30 second safety review fixes

Memory / transfer:

- `copy_`: transfers run on XPURT's default stream, so each one drains the
  caller's stream first and waits for the copy before returning.  The global
  `synchronize()` after every copy is gone; `non_blocking` is accepted but the
  transfer still completes synchronously.
- Overlapping source/destination ranges (`x[1:].copy_(x[:-1])`) are staged
  through the host; cross-PD copies keep the host-staging path.
- `materialize_private_contiguous` copies only the byte range the view touches
  (storage offset + span) instead of the whole storage, and drains the
  producing stream before the D2H transfer.  Negative strides fall back to the
  whole storage, which is the only safe superset.
- `fill_` covers float16/float32 through XDNN `constant<T>` and
  bool/uint8/int8/int16/int32/int64/double through a chunked host-pattern
  broadcast (XPURT has no memset).  Non-contiguous tensors are filled by
  walking the outer dimension.
- `copy_` still requires a contiguous destination, so `x.t().to("kunlun")`
  fails; a strided-destination DMA loop is not implemented.  A non-contiguous
  *source* is handled.
- `scalar_value<T>` range-checks integer scalars instead of silently
  truncating.

Allocator:

- A single live block must stay below 2 GiB (`int32` copy lengths and 32-bit
  device pointers); `KUNLUN_MAX_ALLOC_MB` defaults to 2048 and can only lower
  that bound.
- Sizes round to 512 B; blocks split when the remainder is at least 1 MiB;
  best-fit reuses the smallest free block; address-adjacent free blocks are
  coalesced.  `memory_allocated()` reports block capacity, `memory_reserved()`
  reports device bytes actually held.
- `release()` records events on both the allocating stream and the freeing
  stream (C2).  `empty_cache()` drains all streams if an event belonged to a
  torn-down stream, then frees.
- Releasing a pointer this allocator did not allocate warns and leaks instead
  of calling `xpu_free` on foreign memory.
- `device_of_ptr()` resolves interior pointers by interval lookup over live
  blocks, not exact base-address equality.

XDNN / scheduling / diagnostics:

- One XDNN context per device protected by a mutex; a launch on a different
  stream drains the previous one (`ContextGuard`).  Ops on one stream still
  pipeline.
- Cluster scheduling is per launch: `torch.kunlun.set_cluster_budget(n, nsdnn)`
  caps XPU clusters/SDNN engines (1..4), `KUNLUN_NCLUSTER`/`KUNLUN_NSDNN` set the
  initial value, and `KUNLUN_CLUSTER_POLICY=auto` uses one cluster for small
  launches.  Lower the budget when co-running a Triton kernel on the same PD.
  `auto` only reacts to ops that pass a work estimate (`fill_`, `add`, `relu`,
  `fc`, `conv2d`); other launches keep the fixed budget.  There is no
  reservation: XDNN exposes no non-blocking completion query, so the next
  launch cannot learn when a co-tenant's clusters are free.
- `torch.kunlun.xdnn_version()` reports the `XDNN_VERSION_*` tag of the loaded
  `libxpuapi`; import warns on a major/minor mismatch with the 2.0 headers,
  `KUNLUN_XDNN_REQUIRE` turns that into an error.
- `torch.kunlun.is_pinned_ptr` / `is_pinned_tensor` expose the XPURT pinned
  host allocator used by `DataLoader(pin_memory=True)`.

BKCL:

- `xccl_barrier` drains this rank's stream before `bkcl_barrier`, which is a
  host-side rendezvous with no stream argument.
- Collectives reject dtypes outside float32/float16/int32 on KL1 (BKCL's enum
  lists BF16/FLOAT64/INT64/UINT8, but KL1 coverage is unverified and UINT8
  would silently accept bool/uint8 masks).

Numerics (D1, confirmed):

- `fc<float, float, float, float>` and `fc<float16, float16, float16, short>`
  are **int16-quantized** GEMMs, not IEEE FP32/FP16 FMA.  XDNN scales each
  operand by its per-tensor max into int16 (`mac_int16_helper`,
  `mm_cfg_basic_int16`), accumulates in int32 and dequantizes with a single
  scale (`XFP16` uses 1/32767²).  FP32 GEMM therefore has roughly 16-bit
  mantissa accuracy per element; do not expect bit-exact FP32 results or
  FP32-accurate accumulation for long K reductions.  Verify with
  `tests/test_numeric_semantics.py` on hardware.

## Rule

The generic `cpu_fallback` is disabled unless `KUNLUN_CPU_FALLBACK=warn|silent`,
because it silently moves tensors to CPU without autograd or performance
parity.  Do not build training paths on either the generic fallback or the
host-staged kernels; they exist for correctness coverage, not for throughput.
