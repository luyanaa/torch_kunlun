import pytest
import torch

torch_kunlun = pytest.importorskip("torch_kunlun")
kunlun = torch_kunlun.kunlun

pytestmark = pytest.mark.skipif(
    not kunlun.is_available(),
    reason="Kunlun K200 device is not available",
)


def _loss_value(loss):
    return float(loss.detach().cpu().item())


def test_mlp_mse_sgd_training():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    ).half().to("kunlun")
    opt = torch.optim.SGD(model.parameters(), lr=0.05, foreach=False)
    loss_fn = torch.nn.MSELoss()

    x = torch.randn(32, 8, dtype=torch.float16).to("kunlun")
    target = torch.randn(32, 1, dtype=torch.float16).to("kunlun")

    losses = []
    for _ in range(20):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        losses.append(_loss_value(loss))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
