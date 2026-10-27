import pytest
import torch
import torch.nn.functional as F

torch_kunlun = pytest.importorskip("torch_kunlun")
kunlun = torch_kunlun.kunlun

pytestmark = pytest.mark.skipif(
    not kunlun.is_available(),
    reason="Kunlun K200 device is not available",
)


def _ones(x):
    out = torch.empty_like(x)
    out.fill_(1.0)
    return out


@pytest.mark.parametrize(
    "name,fn,lo,hi",
    [
        ("abs", torch.abs, -2.0, 2.0),
        ("neg", torch.neg, -2.0, 2.0),
        ("exp", torch.exp, -2.0, 2.0),
        ("log", torch.log, 0.1, 2.0),
        ("sqrt", torch.sqrt, 0.1, 2.0),
        ("rsqrt", torch.rsqrt, 0.1, 2.0),
        ("sigmoid", torch.sigmoid, -2.0, 2.0),
        ("tanh", torch.tanh, -2.0, 2.0),
        ("silu", F.silu, -2.0, 2.0),
        ("leaky_relu", lambda x: F.leaky_relu(x, 0.2), -2.0, 2.0),
        ("elu", lambda x: F.elu(x, 0.5), -2.0, 2.0),
        ("softplus", F.softplus, -2.0, 2.0),
        ("clamp", lambda x: torch.clamp(x, -0.5, 0.7), -2.0, 2.0),
        ("square", torch.square, -2.0, 2.0),
        ("reciprocal", torch.reciprocal, 0.1, 2.0),
        ("sign", torch.sign, -2.0, 2.0),
        ("ceil", torch.ceil, -2.0, 2.0),
        ("floor", torch.floor, -2.0, 2.0),
        ("round", torch.round, -2.0, 2.0),
        ("log1p", torch.log1p, 0.1, 2.0),
        ("exp2", torch.exp2, -2.0, 2.0),
        ("erf", torch.erf, -2.0, 2.0),
        ("erfinv", torch.erfinv, -0.8, 0.8),
        ("selu", F.selu, -2.0, 2.0),
        ("celu", lambda x: F.celu(x, 0.5), -2.0, 2.0),
        ("mish", F.mish, -2.0, 2.0),
        ("hardswish", F.hardswish, -2.0, 2.0),
        ("hardsigmoid", F.hardsigmoid, -2.0, 2.0),
        ("hardtanh", lambda x: F.hardtanh(x, -0.5, 0.7), -2.0, 2.0),
        ("softshrink", lambda x: F.softshrink(x, 0.2), -2.0, 2.0),
        ("threshold", lambda x: F.threshold(x, 0.2, 0.0), -2.0, 2.0),
    ],
)
def test_unary_forward_backward(name, fn, lo, hi):
    x = torch.empty(32).uniform_(lo, hi).half()
    xr = x.clone().requires_grad_()
    yr = fn(xr)
    yr.backward(_ones(yr))
    xk = x.clone().to("kunlun").requires_grad_()
    yk = fn(xk)
    yk.backward(_ones(yk))
    kunlun.synchronize()
    assert torch.allclose(yk.cpu(), yr.detach(), atol=5e-2, rtol=5e-2), name
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=5e-2, rtol=5e-2), name


def test_div_forward_backward():
    a = torch.randn(8, 16, dtype=torch.float16)
    b = torch.rand(8, 16, dtype=torch.float16) + 0.5
    ar, br = a.clone().requires_grad_(), b.clone().requires_grad_()
    cr = ar / br
    cr.backward(_ones(cr))
    ak, bk = a.to("kunlun").requires_grad_(), b.to("kunlun").requires_grad_()
    ck = ak / bk
    ck.backward(_ones(ck))
    kunlun.synchronize()
    assert torch.allclose(ck.cpu(), cr.detach(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(ak.grad.cpu(), ar.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(bk.grad.cpu(), br.grad, atol=5e-2, rtol=5e-2)

    ar = a.clone().requires_grad_()
    cr = ar / 2.0
    cr.backward(_ones(cr))
    ak = a.to("kunlun").requires_grad_()
    ck = ak / 2.0
    ck.backward(_ones(ck))
    kunlun.synchronize()
    assert torch.allclose(ak.grad.cpu(), ar.grad, atol=5e-2, rtol=5e-2)


def test_log_softmax_avg_pool_autograd():
    x = torch.randn(4, 8, dtype=torch.float16)
    xr = x.clone().requires_grad_()
    yr = F.log_softmax(xr, -1)
    yr.backward(_ones(yr))
    xk = x.to("kunlun").requires_grad_()
    yk = F.log_softmax(xk, -1)
    yk.backward(_ones(yk))
    kunlun.synchronize()
    assert torch.allclose(yk.cpu(), yr.detach(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=3e-2, rtol=3e-2)

    x = torch.randn(2, 3, 6, 6, dtype=torch.float16)
    xr = x.clone().requires_grad_()
    yr = F.avg_pool2d(xr, 2, 2)
    yr.backward(_ones(yr))
    xk = x.to("kunlun").requires_grad_()
    yk = F.avg_pool2d(xk, 2, 2)
    yk.backward(_ones(yk))
    kunlun.synchronize()
    assert torch.allclose(yk.cpu(), yr.detach(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=3e-2, rtol=3e-2)


def test_reductions_and_masking_forward():
    x = torch.randn(2, 4, 5, dtype=torch.float16)
    xk = x.to("kunlun")
    assert torch.allclose(torch.sum(xk).cpu(), x.sum(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(torch.sum(xk, dim=1).cpu(), x.sum(dim=1), atol=5e-2, rtol=5e-2)
    assert torch.allclose(torch.mean(xk, dim=-1).cpu(), x.mean(dim=-1), atol=5e-2, rtol=5e-2)
    assert torch.allclose(torch.amax(xk, dim=1).cpu(), x.amax(dim=1), atol=5e-2, rtol=5e-2)
    assert torch.allclose(torch.amin(xk, dim=1).cpu(), x.amin(dim=1), atol=5e-2, rtol=5e-2)
    assert torch.equal(torch.argmax(xk, dim=1).cpu(), x.argmax(dim=1))
    assert torch.allclose(torch.logsumexp(xk, dim=-1).cpu(), torch.logsumexp(x, dim=-1), atol=5e-2, rtol=5e-2)

    c_cpu = x > 0
    c = c_cpu.to("kunlun")
    a = torch.randn_like(x).to("kunlun")
    b = torch.randn_like(x).to("kunlun")
    assert torch.allclose(torch.where(c, a, b).cpu(), torch.where(c_cpu, a.cpu(), b.cpu()), atol=5e-2, rtol=5e-2)
    assert torch.allclose(torch.masked_fill(xk, c, 0.0).cpu(), torch.masked_fill(x, c_cpu, 0.0), atol=5e-2, rtol=5e-2)
    assert torch.allclose(torch.triu(xk).cpu(), torch.triu(x), atol=5e-2, rtol=5e-2)
    assert torch.allclose(torch.tril(xk).cpu(), torch.tril(x), atol=5e-2, rtol=5e-2)


def test_stack_composition():
    a = torch.randn(3, 4, dtype=torch.float16)
    b = torch.randn(3, 4, dtype=torch.float16)
    ref = torch.stack([a, b], dim=1)
    got = torch.stack([a.to("kunlun"), b.to("kunlun")], dim=1)
    kunlun.synchronize()
    assert torch.allclose(got.cpu(), ref, atol=3e-2, rtol=3e-2)


def test_sum_mean_where_mask_triangle_autograd():
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    xr = x.clone().requires_grad_()
    yr = xr.sum(dim=1).mean()
    yr.backward()
    xk = x.to("kunlun").requires_grad_()
    yk = xk.sum(dim=1).mean()
    yk.backward()
    kunlun.synchronize()
    assert torch.allclose(yk.cpu(), yr.detach(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=3e-2, rtol=3e-2)

    x = torch.randn(2, 4, dtype=torch.float16)
    a = torch.randn_like(x)
    b = torch.randn_like(x)
    c_cpu = x > 0
    xr, ar, br = x.clone().requires_grad_(), a.clone().requires_grad_(), b.clone().requires_grad_()
    yr = torch.where(c_cpu, xr, ar) + torch.masked_fill(br, c_cpu, 0.0)
    yr.backward(_ones(yr))
    xk = x.to("kunlun").requires_grad_()
    ak = a.to("kunlun").requires_grad_()
    bk = b.to("kunlun").requires_grad_()
    yk = torch.where(c_cpu.to("kunlun"), xk, ak) + torch.masked_fill(bk, c_cpu.to("kunlun"), 0.0)
    yk.backward(_ones(yk))
    kunlun.synchronize()
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=3e-2, rtol=3e-2)
    assert torch.allclose(ak.grad.cpu(), ar.grad, atol=3e-2, rtol=3e-2)
    assert torch.allclose(bk.grad.cpu(), br.grad, atol=3e-2, rtol=3e-2)

    x = torch.randn(4, 4, dtype=torch.float16)
    xr = x.clone().requires_grad_()
    yr = torch.triu(xr) + torch.tril(xr)
    yr.backward(_ones(yr))
    xk = x.to("kunlun").requires_grad_()
    yk = torch.triu(xk) + torch.tril(xk)
    yk.backward(_ones(yk))
    kunlun.synchronize()
    assert torch.allclose(xk.grad.cpu(), xr.grad, atol=3e-2, rtol=3e-2)


def test_native_creation_ops():
    z = torch.zeros((2, 3), dtype=torch.float16, device="kunlun")
    o = torch.ones((2, 3), dtype=torch.float16, device="kunlun")
    f = torch.full((2, 3), 0.5, dtype=torch.float16, device="kunlun")
    kunlun.synchronize()
    assert torch.all(z.cpu() == 0)
    assert torch.all(o.cpu() == 1)
    assert torch.all(f.cpu() == 0.5)
    zl = torch.zeros_like(o)
    ol = torch.ones_like(z)
    assert torch.all(zl.cpu() == 0)
    assert torch.all(ol.cpu() == 1)


def test_extremum_logsumexp_autograd():
    x = torch.randn(3, 5, dtype=torch.float16)
    for fn in (lambda t: torch.amax(t, dim=1), lambda t: torch.amin(t, dim=1),
               lambda t: torch.logsumexp(t, dim=1)):
        xr = x.clone().requires_grad_()
        yr = fn(xr)
        yr.backward(_ones(yr))
        xk = x.to("kunlun").requires_grad_()
        yk = fn(xk)
        yk.backward(_ones(yk))
        kunlun.synchronize()
        assert torch.allclose(yk.cpu(), yr.detach(), atol=5e-2, rtol=5e-2)
        assert torch.allclose(xk.grad.cpu(), xr.grad, atol=5e-2, rtol=5e-2)


def test_matmul_1d_broadcast_and_einsum():
    for a_shape, b_shape in [((4,), (4,)), ((4,), (4, 3)), ((2, 3, 4), (4,)),
                             ((2, 1, 3, 4), (5, 4, 2))]:
        a = torch.randn(*a_shape, dtype=torch.float16)
        b = torch.randn(*b_shape, dtype=torch.float16)
        ar, br = a.clone().requires_grad_(), b.clone().requires_grad_()
        cr = torch.matmul(ar, br)
        cr.backward(_ones(cr))
        ak = a.to("kunlun").requires_grad_()
        bk = b.to("kunlun").requires_grad_()
        ck = torch.matmul(ak, bk)
        ck.backward(_ones(ck))
        kunlun.synchronize()
        assert torch.allclose(ck.cpu(), cr.detach(), atol=8e-2, rtol=8e-2)
        assert torch.allclose(ak.grad.cpu(), ar.grad, atol=8e-2, rtol=8e-2)
        assert torch.allclose(bk.grad.cpu(), br.grad, atol=8e-2, rtol=8e-2)

    a = torch.randn(2, 3, dtype=torch.float16)
    b = torch.randn(3, 4, dtype=torch.float16)
    ar, br = a.clone().requires_grad_(), b.clone().requires_grad_()
    cr = torch.einsum("ij,jk->ik", ar, br)
    cr.backward(_ones(cr))
    ak = a.to("kunlun").requires_grad_()
    bk = b.to("kunlun").requires_grad_()
    ck = torch.einsum("ij,jk->ik", ak, bk)
    ck.backward(_ones(ck))
    kunlun.synchronize()
    assert torch.allclose(ck.cpu(), cr.detach(), atol=8e-2, rtol=8e-2)
    assert torch.allclose(ak.grad.cpu(), ar.grad, atol=8e-2, rtol=8e-2)
    assert torch.allclose(bk.grad.cpu(), br.grad, atol=8e-2, rtol=8e-2)
