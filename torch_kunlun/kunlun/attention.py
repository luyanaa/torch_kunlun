"""Rough fused-free attention built from Kunlun bmm/softmax primitives.

This is intentionally a composition, not an XDNN fused kernel: it gives a
correctness/autograd reference while `qk_attention`/`qkv_attention` INT8
support is still gated.  Shapes follow PyTorch SDPA:

    query: (..., L, E)
    key:   (..., S, E)
    value: (..., S, Ev)

The leading dimensions are flattened to a single batch for the two bmm calls.
No dropout is implemented (XPU RNG is not available yet).
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def _mask_to_scores(mask: torch.Tensor, batch: int, heads: int, q_len: int, kv_len: int):
    if mask.dim() == 2:
        return mask.reshape(1, q_len, kv_len)
    if mask.dim() == 3:
        # (B, L, S) -> (B*1, L, S) or (B*H, L, S) if already head-major.
        if mask.size(0) == batch:
            return mask.reshape(batch, q_len, kv_len)
        return mask.reshape(batch * heads, q_len, kv_len)
    if mask.dim() == 4:
        return mask.reshape(batch * heads, q_len, kv_len)
    raise ValueError(f"unsupported attention mask rank: {mask.dim()}")


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if dropout_p != 0.0:
        raise NotImplementedError("Kunlun attention dropout is not implemented (no XPU RNG)")
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError("attention expects tensors with at least 2 dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key head dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key/value sequence lengths must match")

    if query.dim() == 2:
        batch, heads = 1, 1
        q_len, head_dim = query.shape
        kv_len = key.shape[0]
        value_dim = value.shape[1]
        q3 = query.reshape(1, q_len, head_dim)
        k3 = key.reshape(1, kv_len, head_dim).transpose(1, 2)
        v3 = value.reshape(1, kv_len, value_dim)
        out_shape = (q_len, value_dim)
    elif query.dim() == 3:
        batch, q_len, head_dim = query.shape
        heads = 1
        kv_len = key.shape[1]
        value_dim = value.shape[2]
        q3 = query.reshape(batch, q_len, head_dim)
        k3 = key.reshape(batch, kv_len, head_dim).transpose(1, 2)
        v3 = value.reshape(batch, kv_len, value_dim)
        out_shape = (batch, q_len, value_dim)
    else:
        batch, heads, q_len, head_dim = query.shape
        kv_len = key.shape[-2]
        value_dim = value.shape[-1]
        q3 = query.reshape(batch * heads, q_len, head_dim)
        k3 = key.reshape(batch * heads, kv_len, head_dim).transpose(1, 2)
        v3 = value.reshape(batch * heads, kv_len, value_dim)
        out_shape = (batch, heads, q_len, value_dim)

    q3 = q3.contiguous()
    k3 = k3.contiguous()
    v3 = v3.contiguous()
    scores = torch.bmm(q3, k3)
    factor = (1.0 / math.sqrt(head_dim)) if scale is None else float(scale)
    if factor != 1.0:
        scores = scores * factor

    if is_causal:
        # Build the additive mask on CPU, then copy once.  This keeps the
        # Kunlun path free of an untested triu kernel.
        causal = torch.triu(
            torch.ones(q_len, kv_len, dtype=scores.dtype), diagonal=1
        ) * -1.0e4
        scores = scores + causal.to(scores.device).reshape(1, q_len, kv_len)

    if attn_mask is not None:
        mask = _mask_to_scores(attn_mask, batch, heads, q_len, kv_len)
        mask = mask.to(scores.device, dtype=scores.dtype)
        if attn_mask.dtype == torch.bool:
            # Boolean True means "mask out"; convert to an additive bias.
            mask = mask * -1.0e4
        scores = scores + mask

    probs = torch.softmax(scores, dim=-1)
    out = torch.bmm(probs, v3)
    return out.reshape(out_shape)


__all__ = ["scaled_dot_product_attention"]
