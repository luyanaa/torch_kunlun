"""Native XDNN Adam/AdamW for torch_kunlun.

The optimizer keeps FP32 moments and calls XDNN's `adam`/`adamw` kernels.
It is intentionally a small Optimizer subclass; advanced PyTorch features
(foreach, fused, capturable, amsgrad) are not implemented.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

from . import _C


class _KunlunAdamBase(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
        amsgrad: bool = False,
    ):
        if amsgrad:
            raise NotImplementedError("kunlun Adam/AdamW does not implement amsgrad")
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("invalid Adam/AdamW hyperparameter")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._decoupled = decoupled_weight_decay

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise NotImplementedError("kunlun Adam/AdamW does not support sparse gradients")
                if param.device.type != "kunlun":
                    raise ValueError("kunlun optimizer requires kunlun parameters")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = param.grad.contiguous()

                if not self._decoupled and group["weight_decay"] != 0:
                    # Classic Adam L2: add wd * param to the gradient.
                    grad = grad.add(param.detach(), alpha=group["weight_decay"]).contiguous()

                beta1_pow = beta1 ** step
                beta2_pow = beta2 ** step
                if self._decoupled:
                    _C.adamw_step(
                        param, grad, exp_avg, exp_avg_sq,
                        beta1_pow, beta2_pow, float(group["lr"]),
                        float(beta1), float(beta2), float(group["eps"]),
                        float(group["weight_decay"]),
                    )
                else:
                    _C.adam_step(
                        param, grad, exp_avg, exp_avg_sq,
                        beta1_pow, beta2_pow, float(group["lr"]),
                        float(beta1), float(beta2), float(group["eps"]),
                    )
        return loss


class Adam(_KunlunAdamBase):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        super().__init__(params, lr, betas, eps, weight_decay,
                         decoupled_weight_decay=False, amsgrad=amsgrad)


class AdamW(_KunlunAdamBase):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        super().__init__(params, lr, betas, eps, weight_decay,
                         decoupled_weight_decay=True, amsgrad=amsgrad)


__all__ = ["Adam", "AdamW"]
