import pytest
import torch

torch_kunlun = pytest.importorskip("torch_kunlun")
kunlun = torch_kunlun.kunlun

pytestmark = pytest.mark.skipif(
    not kunlun.is_available(),
    reason="Kunlun K200 device is not available",
)


def _run(opt_cls):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2).half().to("kunlun")
    opt = opt_cls(model.parameters(), lr=1e-2)
    x = torch.randn(8, 4, dtype=torch.float16).to("kunlun")
    y = torch.randn(8, 2, dtype=torch.float16).to("kunlun")
    initial = torch.nn.functional.mse_loss(model(x), y)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    kunlun.synchronize()
    final = torch.nn.functional.mse_loss(model(x), y)
    assert float(final.cpu()) < float(initial.cpu())


def test_adam_and_adamw_step():
    _run(torch_kunlun.optim.Adam)
    _run(torch_kunlun.optim.AdamW)
