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

- pinned host memory, RNG
- non-contiguous arithmetic inputs (`.contiguous()` now uses a host-staging fallback)
- dropout training / RNG
- other optimizers beyond Adam/AdamW
- `torch.compile`
- multi-node distributed; only the single-node `bkcl` ProcessGroup is provided
- INT8 (generic path; attention INT8 is pending Gate 0-A)
- >4 GB single allocation (device pointers are 32-bit)

## Linux x86_64 build record

- **2026-10-28**: `python setup.py build_ext --inplace` completed with `EXIT:0` on an x86_64 Linux host using torch `2.6.0+cpu` and g++ 14.
- Produced `torch_kunlun/_C.cpython-313-x86_64-linux-gnu.so`.
- `tools/kunlun/attention_int8_probe.cpp` also compiled and linked on that host; `ldd` resolved only the expected XRE/XDNN libraries.
- The build was made before the final `native_batch_norm` registry-return fix. Re-run the same build on the K200 host before importing.
- Importing the produced extension on a host **without** the K200 driver wedged the machine; do not use a CPU-only host for runtime import.

## Allocation guard

K200 `xpu_malloc` can hang instead of returning OOM for oversized requests.
The allocator therefore keeps a per-device reservation budget and checks it
before calling `xpu_malloc`:

| env | default | meaning |
|---|---:|---|
| `KUNLUN_MEMORY_FRACTION` | `0.85` | usable fraction of reported device capacity |
| `KUNLUN_MEMORY_RESERVE_MB` | `256` | headroom reserved for XDNN/driver workspaces |
| `KUNLUN_MEMORY_LIMIT_MB` | `0` | optional hard capacity override; `0` disables |
| `KUNLUN_MAX_ALLOC_MB` | `1024` | max single allocation; `0` disables the check |

Cached free blocks still count as reserved until `empty_cache()` actually
frees them. `torch.kunlun.memory_reserved()` reports the reserved bytes.
These checks reduce, but cannot eliminate, driver-side hangs caused by
allocations made outside this allocator.

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
