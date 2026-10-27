import pytest
import torch

torch_kunlun = pytest.importorskip("torch_kunlun")
kunlun = torch_kunlun.kunlun

pytestmark = pytest.mark.skipif(
    not kunlun.is_available(),
    reason="Kunlun K200 device is not available",
)


def test_softmax_layer_norm_gelu_cat():
    x = torch.randn(8, 32, dtype=torch.float16)
    xk = x.to("kunlun")
    y = torch.softmax(xk, dim=-1)
    assert torch.allclose(y.cpu(), torch.softmax(x, dim=-1), atol=2e-2, rtol=2e-2)

    ln = torch.nn.LayerNorm(32).half()
    z = torch.nn.functional.layer_norm(xk, (32,), ln.weight.to("kunlun"), ln.bias.to("kunlun"), 1e-5)
    ref = torch.nn.functional.layer_norm(x, (32,), ln.weight, ln.bias, 1e-5)
    assert torch.allclose(z.cpu(), ref, atol=3e-2, rtol=3e-2)

    g = torch.nn.functional.gelu(xk)
    assert torch.allclose(g.cpu(), torch.nn.functional.gelu(x), atol=3e-2, rtol=3e-2)

    c = torch.cat([xk, xk], dim=-1)
    assert c.shape == (8, 64)


def test_batch_norm_eval_and_pool():
    x = torch.randn(2, 4, 8, 8, dtype=torch.float16)
    bn = torch.nn.BatchNorm2d(4).half().eval()
    y = bn(x.to("kunlun"))
    ref = bn(x)
    assert torch.allclose(y.cpu(), ref, atol=5e-2, rtol=5e-2)

    p = torch.nn.functional.max_pool2d(x.to("kunlun"), 2, 2)
    pref = torch.nn.functional.max_pool2d(x, 2, 2)
    assert torch.allclose(p.cpu(), pref, atol=1e-2, rtol=1e-2)

    a = torch.nn.functional.adaptive_avg_pool2d(x.to("kunlun"), (1, 1))
    aref = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    assert torch.allclose(a.cpu(), aref, atol=3e-2, rtol=3e-2)


def test_noncontiguous_contiguous():
    x = torch.randn(4, 8, dtype=torch.float16, device="kunlun")
    xt = x.transpose(0, 1)
    y = xt.contiguous()
    kunlun.synchronize()
    ref = x.cpu().transpose(0, 1).contiguous()
    assert y.is_contiguous()
    assert torch.allclose(y.cpu(), ref, atol=1e-2, rtol=1e-2)
