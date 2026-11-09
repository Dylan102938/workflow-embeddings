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


def step_logits(d, k, U, pos_keys, target, fn_mask, logpi=None):
    """d: [B,T,D] k: [B,T,1] U: [N,D] pos_keys: [B,D] target: [B] -> logits [B,T,N] (fp32).
    Mixture (logpi [B,T,K], d [B,T,K,D], k [B,T,K,1]): returns log p = log sum_k pi_k softmax(kappa_k d_k . U),
    already normalized, so downstream log_softmax / cross_entropy / ranking are unchanged."""
    if logpi is not None:
        d, k = d.float(), k.float()
        B, T, K, _ = d.shape
        comp = k * torch.einsum("btkd,nd->btkn", d, U)
        pos = k.squeeze(-1) * torch.einsum("btkd,bd->btk", d, pos_keys)
        # Finite fill inside the mixture: logsumexp over all -inf entries has a NaN gradient. Masked columns are
        # set to -inf only after combining (masked_fill passes no gradient there).
        comp = comp.masked_fill(fn_mask[:, None, None, :], -1e4)
        comp = comp.scatter(3, target.view(B, 1, 1, 1).expand(B, T, K, 1), pos[..., None])
        logp = torch.logsumexp(logpi.float()[..., None] + comp.log_softmax(-1), dim=2)
        return logp.masked_fill(fn_mask[:, None, :], float("-inf"))
    d, k = d.float(), k.float()
    logits = k * torch.einsum("btd,nd->btn", d, U)
    pos = k.squeeze(-1) * torch.einsum("btd,bd->bt", d, pos_keys)        # paraphrased positive
    logits = logits.masked_fill(fn_mask[:, None, :], float("-inf"))
    B, T, _ = logits.shape
    logits = logits.scatter(2, target.view(B, 1, 1).expand(B, T, 1), pos[..., None])
    return logits


def smooth_targets(U, own_key, target, fn_mask, eps, tau):
    """(1 - eps) * one-hot(target) + eps * softmax(cos(own_key, U) / tau) over the unmasked candidates
    other than the target itself (its self-similarity would otherwise absorb all of the smoothing mass)."""
    rows = torch.arange(len(target), device=U.device)
    sims = (own_key @ U.T / tau).masked_fill(fn_mask, float("-inf"))
    sims[rows, target] = float("-inf")
    q = eps * sims.softmax(-1)
    q[rows, target] += 1.0 - eps
    return q


def hard_negative_bias(U, own_key, target, fn_mask, beta):
    """[B, N] additive logit bias that weights each negative j by w_j = exp(beta * cos(own_key, U_j)), normalized
    so the weights of a row's negatives average 1 (Robinson et al. 2021, with hardness measured in the label space).
    The target column gets 0; masked columns are -inf in the logits anyway."""
    rows = torch.arange(len(target), device=U.device)
    s = beta * (own_key @ U.T)
    neg = ~fn_mask
    neg[rows, target] = False
    lme = torch.logsumexp(s.masked_fill(~neg, float("-inf")), -1, keepdim=True) - neg.sum(-1, keepdim=True).log()
    bias = (s - lme).masked_fill(~neg, 0.0)
    return bias


def info_nce(d, k, valid, U, pos_keys, target, fn_mask, return_steps=False, soft_targets=None, logpi=None,
             logit_bias=None):
    """soft_targets [B, N] (training-time label smoothing) replaces the one-hot target when given.
    logit_bias [B, N] (training only, e.g. hard-negative weights) is added to every step's logits."""
    logits = step_logits(d, k, U, pos_keys, target, fn_mask, logpi)
    if logit_bias is not None:
        logits = logits + logit_bias[:, None, :]
    B, T, N = logits.shape
    if soft_targets is None:
        ce = F.cross_entropy(logits.reshape(B * T, N), target.repeat_interleave(T), reduction="none").view(B, T)
    else:
        logp = logits.log_softmax(-1).masked_fill(fn_mask[:, None, :], 0.0)   # masked columns have q = 0
        ce = -(soft_targets[:, None, :] * logp).sum(-1)
    ce = ce * valid
    per_session = ce.sum(1) / valid.sum(1)                               # each session weighs the same
    if return_steps:
        return per_session.mean(), ce, logits
    return per_session.mean()
