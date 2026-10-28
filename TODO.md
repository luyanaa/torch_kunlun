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
- `KUNLUN_CPU_FALLBACK=error|warn|silent` for explicit and generic fallbacks.

## Rule

Do not rely on the generic `cpu_fallback` for training paths.  It can silently
move tensors to CPU, and it does not guarantee autograd or performance parity.
