"""Per-step InfoNCE against ALL frozen instruction keys (pseudocode section 4)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def false_negative_mask(U: torch.Tensor, cand_group: torch.Tensor, own_key: torch.Tensor,
                        own_group: torch.Tensor, target: torch.Tensor, theta_fn: float) -> torch.Tensor:
    """[B, N] True where a candidate must NOT count as a negative: same instruction group, or a
    near-duplicate intent (cos > theta_fn). The session's own column is never masked."""
    m = (own_key @ U.T > theta_fn) | (own_group[:, None] == cand_group[None, :])
    m[torch.arange(len(target), device=U.device), target] = False
    return m


def step_logits(d, k, U, pos_keys, target, fn_mask):
    """d: [B,T,D] k: [B,T,1] U: [N,D] pos_keys: [B,D] target: [B] -> logits [B,T,N] (fp32)."""
    d, k = d.float(), k.float()
    logits = k * torch.einsum("btd,nd->btn", d, U)
    pos = k.squeeze(-1) * torch.einsum("btd,bd->bt", d, pos_keys)        # paraphrased positive
    logits = logits.masked_fill(fn_mask[:, None, :], float("-inf"))
    B, T, _ = logits.shape
    logits = logits.scatter(2, target.view(B, 1, 1).expand(B, T, 1), pos[..., None])
    return logits


def info_nce(d, k, valid, U, pos_keys, target, fn_mask, return_steps=False):
    logits = step_logits(d, k, U, pos_keys, target, fn_mask)
    B, T, N = logits.shape
    ce = F.cross_entropy(logits.reshape(B * T, N), target.repeat_interleave(T), reduction="none").view(B, T)
    ce = ce * valid
    per_session = ce.sum(1) / valid.sum(1)                               # each session weighs the same
    if return_steps:
        return per_session.mean(), ce, logits
    return per_session.mean()
