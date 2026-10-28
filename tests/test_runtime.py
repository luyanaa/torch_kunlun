import pytest
import torch

torch_kunlun = pytest.importorskip("torch_kunlun")
kunlun = torch_kunlun.kunlun

pytestmark = pytest.mark.skipif(
    not kunlun.is_available(),
    reason="Kunlun K200 device is not available",
)


def test_device_count_and_properties():
    assert kunlun.device_count() > 0
    props = kunlun.get_device_properties(0)
    assert props.multi_processor_count == 64
    assert props.warp_size == 8
    assert props.total_memory > 0


def test_empty_fill_copy():
    x = torch.empty(1024, dtype=torch.float16, device="kunlun")
    assert x.device.type == "kunlun"
    x.fill_(1.0)
    y = torch.empty_like(x)
    y.copy_(x)
    kunlun.synchronize()
    out = y.cpu()
    assert torch.all(out == 1.0)


def test_add_relu():
    a = torch.randn(128, dtype=torch.float16).to("kunlun")
    b = torch.randn(128, dtype=torch.float16).to("kunlun")
    c = torch.add(a, b)
    r = torch.relu(c)
    kunlun.synchronize()
    assert torch.allclose(c.cpu(), a.cpu() + b.cpu(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(r.cpu(), torch.relu(a.cpu() + b.cpu()), atol=1e-2, rtol=1e-2)


def test_mm_fp16():
    a = torch.randn(16, 32, dtype=torch.float16)
    b = torch.randn(32, 24, dtype=torch.float16)
    c = torch.mm(a.to("kunlun"), b.to("kunlun"))
    kunlun.synchronize()
    ref = torch.mm(a, b)
    assert torch.allclose(c.cpu(), ref, atol=2e-2, rtol=2e-2)


def test_conv2d_fp16():
    x = torch.randn(1, 4, 8, 8, dtype=torch.float16)
    w = torch.randn(8, 4, 3, 3, dtype=torch.float16)
    b = torch.randn(8, dtype=torch.float16)
    y = torch.conv2d(x.to("kunlun"), w.to("kunlun"), b.to("kunlun"), padding=1)
    kunlun.synchronize()
    ref = torch.conv2d(x, w, b, padding=1)
    assert torch.allclose(y.cpu(), ref, atol=5e-2, rtol=5e-2)


def test_linear_fp16():
    x = torch.randn(4, 8, dtype=torch.float16)
    w = torch.randn(6, 8, dtype=torch.float16)
    b = torch.randn(6, dtype=torch.float16)
    y = torch.nn.functional.linear(x.to("kunlun"), w.to("kunlun"), b.to("kunlun"))
    kunlun.synchronize()
    ref = torch.nn.functional.linear(x, w, b)
    assert torch.allclose(y.cpu(), ref, atol=3e-2, rtol=3e-2)


def test_empty_strided():
    x = torch.empty_strided((4, 8), (8, 1), dtype=torch.float16, device="kunlun")
    assert x.device.type == "kunlun"
    assert x.stride() == (8, 1)
    x.zero_()
    kunlun.synchronize()
    assert torch.all(x.cpu() == 0)


def test_copy_shape_mismatch_is_rejected():
    dst = torch.empty(1024, dtype=torch.float16, device="kunlun")
    src = torch.ones(1, dtype=torch.float16, device="kunlun")
    with pytest.raises(RuntimeError, match="equal shapes"):
        dst.copy_(src)
