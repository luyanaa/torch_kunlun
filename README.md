# torch_kunlun (basic runtime)

Minimal PyTorch 2.6 `PrivateUse1` runtime for Kunlun K200 (XPU1), backed by XDNN.
This is the first slice: device/allocator/guard plus a few FP16/FP32 ops. It is
**not** a complete backend and has not been run on hardware yet.

## Supported

- `torch.empty` / `empty_like` / `empty_strided` (via the PrivateUse1 allocator)
- `fill_`, `zero_`
- `copy_` CPU↔kunlun and kunlun↔kunlun (same dtype, contiguous)
- `add` (broadcast, `alpha=1`), `relu` / `relu_`
- `mm` / `linear` / `bmm` (FP16/FP32)
- `conv2d` (NCHW/OIHW, FP16/FP32, bias, groups)
- `batch_norm` inference/training forward, `max_pool2d`, `adaptive_avg_pool2d`, `softmax`, `layer_norm`, `gelu`, `embedding`, `cat`
- XPU streams/events that are real `torch.Stream`/`torch.Event`-compatible objects (`torch.kunlun.Stream`, `Event`, `current_stream`, `stream()`); `Tensor.record_stream()` works with `Stream`
- multi-device `copy_` via host staging; same-device D2D
- Single-node multi-card collectives: raw `torch_kunlun.xccl.*` plus a `torch.distributed` ProcessGroup backend named `bkcl` (DDP-compatible subset: broadcast/allreduce/barrier/all_gather_base/reduce_scatter)
- CNN/Transformer ops: `batch_norm` (eval), `max_pool2d`, `adaptive_avg_pool2d`, `softmax`, `layer_norm`, `gelu`, `embedding`, `cat`, `bmm`
- native extra ops: `abs`/`neg`/`exp`/`log`/`sqrt`/`rsqrt`/`sigmoid`/`tanh`/`silu`/`leaky_relu`/`elu`/`softplus`/`clamp`/`square`/`reciprocal`/`sign`/`ceil`/`floor`/`round`, `div`, `log_softmax`, `avg_pool2d`, `sum`/`mean`/`amax`/`amin`/`argmax`/`argmin`/`logsumexp`, `where`, `masked_fill`, `triu`/`tril`, `stack`, `zeros`/`ones`/`full` and `*_like`, `maximum`/`minimum`/`addcmul`/`addcdiv`, `selu`/`celu`/`mish`/`hardswish`/`hardsigmoid`/`hardtanh`/`softshrink`/`threshold`, `log1p`/`exp2`/`erf`/`erfinv`, `einsum` (CPU fallback), native `Adam`/`AdamW` via `torch_kunlun.optim`, `nll_loss` (native with CPU fallback), `nll_loss`/BCE/L1/SmoothL1/KL/Huber (CPU host-staged), `group_norm`/`instance_norm`/`rms_norm`/`adaptive_max_pool2d`/`avg_pool3d`/`conv1d`/`conv3d`/`conv_transpose2d`, `matmul` (2D/3D/4D)/`baddbmm`/`addbmm`
- remaining operator work is tracked in `TODO.md`
- `torch.kunlun.synchronize()` waits all known streams
- attention: `torch_kunlun.kunlun.scaled_dot_product_attention` (bmm/softmax composition, not fused XDNN yet)
- autograd: `add`/`sub`/`mul`/`relu`/`mm`/`linear`/`addmm`/`mse_loss`, `conv2d`, `max_pool2d`, `adaptive_avg_pool2d`, `softmax`, `layer_norm`, `gelu`, `embedding`, `cat`, `bmm`, training `native_batch_norm`, and `cross_entropy_loss` (CPU-host-staged correctness fallback)
- basic training: `torch.optim.SGD` via in-place `add_`/`mul_`; `MSELoss`

## Not supported yet

- RNG
- non-contiguous arithmetic inputs (`.contiguous()` now uses a host-staging fallback)
- dropout training / RNG
- other optimizers beyond Adam/AdamW
- `torch.compile`
- multi-node distributed; only the single-node `bkcl` ProcessGroup is provided
- INT8 (generic path; attention INT8 is pending Gate 0-A)
- single allocations ≥ 2 GiB (K200 device pointers and DMA lengths are 32-bit)

## Linux x86_64 build record

- **2026-10-28**: `python setup.py build_ext --inplace` completed with `EXIT:0` on an x86_64 Linux host using torch `2.6.0+cpu` and g++ 14.
- Produced `torch_kunlun/_C.cpython-313-x86_64-linux-gnu.so`.
- `tools/kunlun/attention_int8_probe.cpp` also compiled and linked on that host; `ldd` resolved only the expected XRE/XDNN libraries.
- The build was made before the final `native_batch_norm` registry-return fix. Re-run the same build on the K200 host before importing.
- Importing the produced extension on a host **without** the K200 driver wedged the machine; do not use a CPU-only host for runtime import.

## Safety / fallback policy

- `copy_` currently requires equal shapes and matching dtype. Broadcasting and
  dtype conversion are intentionally rejected rather than reading past the
  source allocation.
- `xpu_memcpy` takes no stream and runs on XPURT's default stream, and
  `xpu_wait(nullptr)` waits only that stream.  Every host↔device transfer
  therefore drains the caller's stream, copies, then waits for the copy, so
  `non_blocking=True` cannot defer anything.
- Non-contiguous sources are materialized through host staging that copies only
  the address range the view touches (not the whole storage).
- `KUNLUN_CPU_FALLBACK` controls CPU execution:
  - unset (default): the generic boxed fallback for ops without a device kernel
    is **rejected**; operators with a deliberate host-staged implementation
    (einsum, broadcast matmul, host-staged losses) run and print a one-time notice.
  - `error`: rejects both.
  - `warn`: allows both and prints every call.
  - `silent`: allows both without messages.
- XCCL Work uses a conservative completed model: ProcessGroup calls synchronize
  before reporting completion; `is_completed()` never returns true early.
  `barrier()` drains the current stream before joining the host rendezvous, and
  only float32/float16/int32 collectives are accepted.
- `getDeviceFromPtr` resolves pointers through an address-interval index; unknown
  pointers are rejected instead of being attributed to the current device.
  Releasing an unknown pointer is reported and leaked, never passed to
  `xpu_free`.
- FP32 `mm`/`conv` are **not** IEEE FP32: XDNN runs int16 MAC with per-tensor
  max dequantization for both FP16 and FP32, so only the FP16-scale error bound
  holds.

## Allocation guard

K200 `xpu_malloc` can hang instead of returning OOM for oversized requests.
The allocator therefore keeps a per-device reservation budget and checks it
before calling `xpu_malloc`:

| env | default | meaning |
|---|---:|---|
| `KUNLUN_MEMORY_FRACTION` | `0.85` | usable fraction of reported device capacity |
| `KUNLUN_MEMORY_RESERVE_MB` | `256` | headroom reserved for XDNN/driver workspaces |
| `KUNLUN_MEMORY_LIMIT_MB` | `0` | optional hard capacity override; `0` disables |
| `KUNLUN_MAX_ALLOC_MB` | `2048` | max single allocation; capped at 2 GiB, `0` disables the check |

Cached free blocks still count as reserved until `empty_cache()` actually
frees them. `torch.kunlun.memory_reserved()` reports the reserved bytes.
The allocator rounds to 512 B, reuses free blocks best-fit, splits remainders
≥ 1 MiB, coalesces address-adjacent blocks, and records an event on the
allocating and freeing streams so a block is only reused after both are
complete.  These checks reduce, but cannot eliminate, driver-side hangs caused
by allocations made outside this allocator.

## Pinned host memory

`torch_kunlun` binds XPURT's page-locked host pool to the PrivateUse1 hooks,
so `DataLoader(pin_memory=True)` and `tensor.pin_memory()` produce buffers that
`xpu_memcpy` can DMA directly.  Use `torch.kunlun.is_pinned_tensor(t)` /
`is_pinned_ptr(addr)` to test them; `torch.is_pinned` only knows CUDA's pool.

## Scheduling and version checks

| env | default | meaning |
|---|---:|---|
| `KUNLUN_NCLUSTER` | `4` | clusters per XDNN launch (1..4) |
| `KUNLUN_NSDNN` | `4` | SDNN engines per XDNN launch (1..4) |
| `KUNLUN_CLUSTER_POLICY` | `fixed` | `auto` drops launches below 4 Mi elements to 1 cluster |
| `KUNLUN_XDNN_REQUIRE` | unset | fail import unless the loaded XDNN version contains this tag |
| `KUNLUN_XDNN_ALLOW_MISMATCH` | unset | silence the "compiled against 2.0.x" warning |
| `KUNLUN_SKIP_XDNN_VERSION_CHECK` | unset | skip the version check entirely |
| `KUNLUN_XDNN_VERSION` | unset | override the detected version tag |

The budget is applied per launch (`torch.kunlun.set_cluster_budget()` at
runtime), which lets a Triton kernel started with `ncluster=2` share the PD.
`auto` only reacts to ops that supply a work estimate (`fill_`, `add`, `relu`,
`fc`, `conv2d`); everything else uses the fixed budget.  It is a cap, not a
reservation: XDNN has no non-blocking completion query, so the next launch
cannot know when the other engine's clusters are free.

## Host safety

Build only on Linux x86_64.  Do **not** import `torch_kunlun` or run the
vendor probes on a host that has no K200 driver: loading `libxpuapi`/`libxpurt`
there can wedge the machine.  The `.so` build itself is the only step that is
safe without a card; runtime/import validation belongs on the K200 host.

## Build (on the K200 Linux host)

```bash
# torch 2.6 must already be installed in this environment.
# Defaults assume the sibling SDK directories; override when needed.
export KUNLUN_XRE=/path/to/xre-ubuntu_x86_64
export KUNLUN_XDNN=/path/to/xdnn-ubuntu_x86_64
export KUNLUN_XCCL=/path/to/xccl-ubuntu_x86_64

cd torch_kunlun
python -m pip install -e . --no-build-isolation
```

`setup.py` links `libxpuapi.so`, `libxpurt.so`, and `libbkcl.so` and embeds rpaths for the
default `xre/so` and `xdnn/so` paths.

## Smoke test

```bash
python - <<'PY'
import torch, torch_kunlun
print(torch_kunlun.kunlun.is_available(), torch_kunlun.kunlun.device_count())
x = torch.empty(1024, dtype=torch.float16, device="kunlun")
x.fill_(1)
print(x.cpu()[:4])
PY
```

```bash
pytest -q torch_kunlun/tests
```

## Basic training

```python
model = torch.nn.Sequential(
    torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
).half().to("kunlun")
opt = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = torch.nn.MSELoss()

x = torch.randn(32, 8, dtype=torch.float16).to("kunlun")
y = torch.randn(32, 1, dtype=torch.float16).to("kunlun")
for _ in range(20):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
print(loss.detach().cpu().item())
```

The first training slice covers MLP/regression with `nn.Linear` + `ReLU` + `MSELoss`
+ `SGD`. CNN/Transformer backward is not wired yet.

## Single-node multi-card collectives

```python
import torch, torch_kunlun

uid = torch_kunlun.xccl.get_unique_id() if rank == 0 else recv_uid
torch_kunlun.xccl.init_rank(rank, world_size, uid, device=rank)
x = torch.ones(1024, dtype=torch.float16, device=f"kunlun:{rank}")
torch_kunlun.xccl.all_reduce(x, op="sum")
torch_kunlun.xccl.barrier(rank)
```

**Scope:** one host only; one BKCL context per process and one process per rank.
No multi-node launcher, no IB/RDMA, and no `torch.distributed` ProcessGroup backend;
the unique ID must be exchanged within the host (e.g. a multiprocessing queue or file).
Cross-device `copy_` uses host staging (same-card `xpu_memcpy_peer` optimization is not used yet).

## Notes

- The device name is `kunlun`, not `xpu`.
- XDNN work is serialized per device context; the current `torch.kunlun` stream is passed to XDNN. Freed blocks are reused after their per-stream XPU events complete (`record_stream` is honored).
- `add`/`relu`/`mm`/`conv2d` require contiguous tensors; `.contiguous()` is available but uses host staging for non-contiguous views.
- The C++ extension registers `PrivateUse1` and calls `c10::register_privateuse1_backend("kunlun")`;
  the Python package then calls `torch.utils.rename_privateuse1_backend("kunlun")`.
- Do not load `libLLVM-10`/`libtvm` into the PyTorch process.
