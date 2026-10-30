"""On-demand probes for XDNN numeric semantics and runtime invariants.

These run on a real K200 only.  Their job is to confirm (or refute) the
documented contract, not to check PyTorch coverage:

* ``fc``/``mm`` FP32 and FP16 are int16-quantized GEMMs, so FP32 is *not*
  IEEE FP32 arithmetic (see TODO.md, D1).
* ``fill_`` covers non-floating dtypes through a host pattern.
* ``copy_`` handles overlapping views and non-contiguous sources.
* The allocator enforces a hard per-allocation cap below 2 GiB.
"""

import os

import pytest
import torch

torch_kunlun = pytest.importorskip("torch_kunlun")
kunlun = torch_kunlun.kunlun

pytestmark = pytest.mark.skipif(
    not kunlun.is_available(),
    reason="Kunlun K200 device is not available",
)


def test_fp32_mm_is_int16_quantized():
    """FP32 matmul loses small terms that a per-tensor int16 scale cannot hold."""
    # Row [1e-5, 1.0] times column [1.0, 1e5]: the exact result is 100000.00001.
    # With a per-tensor scale of 1e5 the 1.0 operand becomes ~0.33 int16 steps
    # and the product 1e-5 * 1.0 rounds to zero, so the result differs from an
    # IEEE FP32 dot product by ~1.0.
    x = torch.tensor([[1e-5, 1.0]], dtype=torch.float32, device="kunlun")
    w = torch.tensor([[1.0], [1e5]], dtype=torch.float32, device="kunlun")
    out = (x @ w).cpu().item()
    exact = 1e-5 * 1.0 + 1.0 * 1e5

    fp32_limit = abs(exact) * 1e-5  # what an IEEE FP32 dot product allows
    error = abs(out - exact)
    print(f"fp32 mm: got {out!r}, exact {exact!r}, error {error!r}")
    assert error > fp32_limit, (
        "FP32 mm matched IEEE FP32 arithmetic; the int16-quantized contract in "
        "TODO.md (D1) is wrong and should be revised"
    )
    # The error must still be bounded: int16 has ~5 decimal digits.
    assert error < max(1e-2, abs(exact) * 1e-3)


def test_fp16_mm_accuracy():
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float16)
    w = torch.randn(128, 64, dtype=torch.float16)
    exact = x.float() @ w.float()
    out = (x.to("kunlun") @ w.to("kunlun")).cpu().float()
    rel = (out - exact).norm() / exact.norm()
    print(f"fp16 mm relative L2 error: {rel.item():.4e}")
    assert rel < 5e-3


@pytest.mark.parametrize(
    "dtype,value",
    [
        (torch.bool, True),
        (torch.uint8, 200),
        (torch.int8, -7),
        (torch.int16, -3000),
        (torch.int32, 123456),
        (torch.int64, -1234567890123),
        (torch.float64, 1.5),
    ],
)
def test_fill_non_float_dtypes(dtype, value):
    x = torch.empty(1000, dtype=dtype, device="kunlun")
    x.fill_(value)
    kunlun.synchronize()
    out = x.cpu()
    assert torch.equal(out, torch.full((1000,), value, dtype=dtype))


def test_fill_rejects_overflowing_scalar():
    x = torch.empty(4, dtype=torch.int8, device="kunlun")
    with pytest.raises(RuntimeError):
        x.fill_(1000)


def test_zero_int_tensor():
    x = torch.full((128,), 9, dtype=torch.int32, device="kunlun")
    x.zero_()
    kunlun.synchronize()
    assert torch.equal(x.cpu(), torch.zeros(128, dtype=torch.int32))


def test_copy_overlapping_views():
    x = torch.arange(16, dtype=torch.float16).to("kunlun")
    x[1:].copy_(x[:-1])
    kunlun.synchronize()
    expected = torch.arange(16, dtype=torch.float16)
    expected[1:] = expected[:-1].clone()
    assert torch.equal(x.cpu(), expected)


def test_copy_transposed_source():
    src = torch.arange(12, dtype=torch.float16).reshape(3, 4).to("kunlun")
    dst = torch.empty(4, 3, dtype=torch.float16, device="kunlun")
    dst.copy_(src.t())
    kunlun.synchronize()
    assert torch.equal(dst.cpu(), src.cpu().t())


def test_allocation_cap_below_2gib():
    with pytest.raises(RuntimeError):
        torch.empty(3 * (1 << 30), dtype=torch.uint8, device="kunlun")


def test_cluster_budget_roundtrip():
    original = kunlun.cluster_budget()
    try:
        assert kunlun.set_cluster_budget(2) == 2
        assert kunlun.cluster_budget() == 2
        assert kunlun.sdnn_budget() == 2
        # A launch still works with the reduced budget.
        x = torch.ones(256, dtype=torch.float16, device="kunlun")
        assert torch.equal((x + x).cpu(), torch.full((256,), 2.0, dtype=torch.float16))
    finally:
        kunlun.set_cluster_budget(original)


@pytest.mark.skipif(
    os.environ.get("KUNLUN_CPU_FALLBACK") is not None,
    reason="KUNLUN_CPU_FALLBACK explicitly set",
)
def test_cpu_fallback_denied_by_default():
    x = torch.ones(8, dtype=torch.float32, device="kunlun")
    with pytest.raises(RuntimeError, match="generic CPU fallback is disabled"):
        torch.cumsum(x, 0)
