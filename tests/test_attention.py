import math
import pytest
import torch

torch_kunlun = pytest.importorskip("torch_kunlun")
kunlun = torch_kunlun.kunlun

pytestmark = pytest.mark.skipif(
    not kunlun.is_available(),
    reason="Kunlun K200 device is not available",
)


def _ref(q, k, v, causal=False):
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(torch.ones(s.shape[-2], s.shape[-1], dtype=s.dtype), diagonal=1) * -1e4
        s = s + mask
    return torch.matmul(torch.softmax(s, -1), v)


@pytest.mark.parametrize("shape", [(2, 3, 5, 4), (2, 4, 5, 8)])
def test_attention_forward_backward(shape):
    q = torch.randn(*shape, dtype=torch.float16)
    k = torch.randn(*shape, dtype=torch.float16)
    v = torch.randn(*shape, dtype=torch.float16)
    qr, kr, vr = q.clone().requires_grad_(), k.clone().requires_grad_(), v.clone().requires_grad_()
    ref = _ref(qr, kr, vr)
    ref.backward(torch.ones_like(ref))

    qk = q.clone().to("kunlun").requires_grad_()
    kk = k.clone().to("kunlun").requires_grad_()
    vk = v.clone().to("kunlun").requires_grad_()
    out = kunlun.scaled_dot_product_attention(qk, kk, vk)
    gout = torch.empty_like(out)
    gout.fill_(1.0)
    out.backward(gout)
    kunlun.synchronize()
    assert torch.allclose(out.cpu(), ref.detach(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(qk.grad.cpu(), qr.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(kk.grad.cpu(), kr.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(vk.grad.cpu(), vr.grad, atol=5e-2, rtol=5e-2)
