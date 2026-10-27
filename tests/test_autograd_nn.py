import pytest
import torch
import torch.nn.functional as F

torch_kunlun = pytest.importorskip("torch_kunlun")
kunlun = torch_kunlun.kunlun

pytestmark = pytest.mark.skipif(
    not kunlun.is_available(),
    reason="Kunlun K200 device is not available",
)


def _ones_like(x):
    out = torch.empty_like(x)
    out.fill_(1.0)
    return out


def test_conv2d_backward():
    x = torch.randn(1, 2, 5, 5, dtype=torch.float16)
    w = torch.randn(3, 2, 3, 3, dtype=torch.float16)
    b = torch.randn(3, dtype=torch.float16)
    xr = x.clone().requires_grad_()
    wr = w.clone().requires_grad_()
    br = b.clone().requires_grad_()
    ref = F.conv2d(xr, wr, br, padding=1)
    ref.backward(_ones_like(ref))

    xk = x.clone().to("kunlun").requires_grad_()
    wk = w.clone().to("kunlun").requires_grad_()
    bk = b.clone().to("kunlun").requires_grad_()
    out = F.conv2d(xk, wk, bk, padding=1)
    out.backward(_ones_like(out))
    kunlun.synchronize()
    assert torch.allclose(out.cpu(), ref.detach(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(wk.grad.cpu(), wr.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(bk.grad.cpu(), br.grad, atol=5e-2, rtol=5e-2)


def test_pool_softmax_gelu_backward():
    x = torch.randn(2, 3, 6, 6, dtype=torch.float16)
    xr = x.clone().requires_grad_()
    yr = F.max_pool2d(xr, 2, 2)
    yr.backward(_ones_like(yr))
    xk = x.clone().to("kunlun").requires_grad_()
    yk = F.max_pool2d(xk, 2, 2)
    yk.backward(_ones_like(yk))
    kunlun.synchronize()
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=1e-2, rtol=1e-2)

    x = torch.randn(4, 16, dtype=torch.float16)
    xr = x.clone().requires_grad_()
    yr = torch.softmax(xr, -1)
    yr.backward(_ones_like(yr))
    xk = x.clone().to("kunlun").requires_grad_()
    yk = torch.softmax(xk, -1)
    yk.backward(_ones_like(yk))
    kunlun.synchronize()
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=2e-2, rtol=2e-2)

    xr = x.clone().requires_grad_()
    yr = F.gelu(xr)
    yr.backward(_ones_like(yr))
    xk = x.clone().to("kunlun").requires_grad_()
    yk = F.gelu(xk)
    yk.backward(_ones_like(yk))
    kunlun.synchronize()
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=3e-2, rtol=3e-2)


def test_embedding_cat_bmm_backward():
    w = torch.randn(8, 4, dtype=torch.float16)
    idx = torch.tensor([0, 2, 2, 7], dtype=torch.int64)
    wr = w.clone().requires_grad_()
    ref = F.embedding(idx, wr)
    ref.backward(_ones_like(ref))
    wk = w.clone().to("kunlun").requires_grad_()
    out = F.embedding(idx.to("kunlun"), wk)
    out.backward(_ones_like(out))
    kunlun.synchronize()
    assert torch.allclose(wk.grad.cpu(), wr.grad, atol=1e-3, rtol=1e-3)

    a = torch.randn(2, 3, 4, dtype=torch.float16)
    b = torch.randn(2, 3, 4, dtype=torch.float16)
    ar = a.clone().requires_grad_()
    br = b.clone().requires_grad_()
    cr = torch.cat([ar, br], dim=-1)
    cr.backward(_ones_like(cr))
    ak = a.clone().to("kunlun").requires_grad_()
    bk = b.clone().to("kunlun").requires_grad_()
    ck = torch.cat([ak, bk], dim=-1)
    ck.backward(_ones_like(ck))
    kunlun.synchronize()
    assert torch.allclose(ak.grad.cpu(), ar.grad, atol=1e-3, rtol=1e-3)

    a = torch.randn(2, 4, 5, dtype=torch.float16)
    b = torch.randn(2, 5, 3, dtype=torch.float16)
    ar = a.clone().requires_grad_()
    br = b.clone().requires_grad_()
    cr = torch.bmm(ar, br)
    cr.backward(_ones_like(cr))
    ak = a.clone().to("kunlun").requires_grad_()
    bk = b.clone().to("kunlun").requires_grad_()
    ck = torch.bmm(ak, bk)
    ck.backward(_ones_like(ck))
    kunlun.synchronize()
    assert torch.allclose(ak.grad.cpu(), ar.grad, atol=3e-2, rtol=3e-2)
    assert torch.allclose(bk.grad.cpu(), br.grad, atol=3e-2, rtol=3e-2)


def test_layer_norm_batch_norm_backward():
    x = torch.randn(4, 8, dtype=torch.float16)
    g = torch.randn(8, dtype=torch.float16)
    b = torch.randn(8, dtype=torch.float16)
    xr = x.clone().requires_grad_()
    gr = g.clone().requires_grad_()
    br = b.clone().requires_grad_()
    yr = F.layer_norm(xr, (8,), gr, br, 1e-5)
    yr.backward(_ones_like(yr))
    xk = x.clone().to("kunlun").requires_grad_()
    gk = g.clone().to("kunlun").requires_grad_()
    bk = b.clone().to("kunlun").requires_grad_()
    yk = F.layer_norm(xk, (8,), gk, bk, 1e-5)
    yk.backward(_ones_like(yk))
    kunlun.synchronize()
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=3e-2, rtol=3e-2)
    assert torch.allclose(gk.grad.cpu(), gr.grad, atol=3e-2, rtol=3e-2)
    assert torch.allclose(bk.grad.cpu(), br.grad, atol=3e-2, rtol=3e-2)

    x = torch.randn(2, 4, 5, 5, dtype=torch.float16)
    bn = torch.nn.BatchNorm2d(4).half().train()
    xr = x.clone().requires_grad_()
    yr = bn(xr)
    yr.backward(_ones_like(yr))
    bnk = torch.nn.BatchNorm2d(4).half().train()
    bnk.load_state_dict(bn.state_dict())
    bnk = bnk.to("kunlun")
    xk = x.clone().to("kunlun").requires_grad_()
    yk = bnk(xk)
    yk.backward(_ones_like(yk))
    kunlun.synchronize()
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=5e-2, rtol=5e-2)


def test_cross_entropy_backward():
    x = torch.randn(4, 5, dtype=torch.float16)
    target = torch.tensor([0, 2, 4, 1], dtype=torch.int64)
    xr = x.clone().requires_grad_()
    ref = F.cross_entropy(xr, target, reduction="mean")
    ref.backward()
    xk = x.clone().to("kunlun").requires_grad_()
    out = F.cross_entropy(xk, target.to("kunlun"), reduction="mean")
    out.backward()
    kunlun.synchronize()
    assert torch.allclose(out.cpu(), ref.detach(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=5e-2, rtol=5e-2)
