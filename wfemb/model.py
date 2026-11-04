"""SessionStateMachine: a recurrent state machine over (screenshot, input events) steps.

At each step it outputs a direction d_t (unit vector in the frozen text-key space: the intent it
currently believes) and a confidence kappa_t (inverse temperature).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import EVENT_TYPES


@dataclass
class ModelConfig:
    d_key: int = 1024        # frozen text-key dimension (D in the pseudocode)
    d_vis: int = 1152        # frozen vision token dimension
    n_tok: int = 64          # frame tokens per step (global grid, + crop tokens if enabled)
    d: int = 512             # internal width
    h: int = 1024            # recurrent state size (H)
    n_queries: int = 4       # Q queries for the event-conditioned frame pool
    n_heads: int = 8
    n_keys: int = 512        # key-token vocab size
    d_text: int = 1024       # typed-text embedding dim
    n_freq: int = 16         # Fourier frequencies for coordinates
    k_min: float = 1.0
    k_max: float = 100.0
    k_init: float = 5.0      # initial kappa (~1/0.2)
    dropout: float = 0.0
    tok_drop: float = 0.0    # training-time fraction of frame tokens hidden from the attention pool


class Fourier(nn.Module):
    def __init__(self, n_freq: int):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)

    def forward(self, x):                                   # [..., k] -> [..., k * 2F]
        a = x[..., None] * self.freqs
        return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class EventEncoder(nn.Module):
    """type embed + Fourier(x, y, x2, y2) + scroll + key-token embed + proj(text_vec); summed over a step's events."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d
        self.n_freq = cfg.n_freq
        self.type_emb = nn.Embedding(len(EVENT_TYPES), d)
        self.fourier = Fourier(cfg.n_freq)
        self.xy_proj = nn.Linear(4 * 2 * cfg.n_freq + 4, d)
        self.scroll_proj = nn.Linear(2, d)
        self.key_emb = nn.Embedding(cfg.n_keys, d, padding_idx=0)
        self.text_proj = nn.Linear(cfg.d_text, d)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, ev_type, ev_xy, ev_scroll, ev_keys, ev_text, text_table):
        present = ev_type > 0                                            # [..., E]
        xy_ok = ~torch.isnan(ev_xy)                                      # [..., E, 4]
        fx = self.fourier(torch.nan_to_num(ev_xy, 0.0))                  # [..., E, 4*2F]
        fx = fx * xy_ok.repeat_interleave(2 * self.n_freq, -1)
        h = self.type_emb(ev_type) + self.xy_proj(torch.cat([fx, xy_ok.float()], -1))
        h = h + self.scroll_proj(torch.stack([ev_scroll.sign(), torch.log1p(ev_scroll.abs())], -1))
        h = h + self.key_emb(ev_keys).sum(-2)
        has_text = ev_text >= 0
        tv = text_table[ev_text.clamp(min=0)].float() * has_text[..., None]
        h = h + self.text_proj(tv)
        h = h + self.mlp(h)
        h = h * present[..., None]
        return h.sum(-2), present.any(-1)                                # [..., d], [...]


class CrossAttnPool(nn.Module):
    """Event-conditioned attention over frame tokens: the action decides where to look."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, Q = cfg.d, cfg.n_queries
        self.Q, self.heads, self.tok_drop = Q, cfg.n_heads, cfg.tok_drop
        self.kv_norm = nn.LayerNorm(cfg.d_vis)
        self.k_proj = nn.Linear(cfg.d_vis, d)
        self.v_proj = nn.Linear(cfg.d_vis, d)
        self.pos = nn.Parameter(torch.randn(cfg.n_tok, d) * 0.02)
        self.q_base = nn.Parameter(torch.randn(Q, d) * 0.02)
        self.q_from_a = nn.Linear(d, Q * d)
        self.out = nn.Linear(Q * d, d)

    def forward(self, a, tokens):                                        # a [N, d], tokens [N, n_tok, d_vis]
        N, n_tok, _ = tokens.shape
        x = self.kv_norm(tokens)
        k = self.k_proj(x) + self.pos[:n_tok]
        v = self.v_proj(x)
        q = self.q_base + self.q_from_a(a).view(N, self.Q, -1)
        sh = lambda t: t.view(N, t.shape[1], self.heads, -1).transpose(1, 2)
        mask = None
        if self.training and self.tok_drop > 0:
            keep = torch.rand(N, n_tok, device=tokens.device) >= self.tok_drop
            keep[torch.arange(N, device=tokens.device), torch.randint(0, n_tok, (N,), device=tokens.device)] = True
            mask = keep[:, None, None, :]                                # True = attend
        o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=mask)   # [N, heads, Q, d/heads]
        return self.out(o.transpose(1, 2).reshape(N, -1))


class SessionStateMachine(nn.Module):
    def __init__(self, cfg: ModelConfig, text_table: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.d, cfg.h
        self.register_buffer("text_table", text_table.to(torch.float16), persistent=False)
        self.event_enc = EventEncoder(cfg)
        self.no_event = nn.Parameter(torch.randn(d) * 0.02)              # used for the final screen (no action)
        self.frame_pool = CrossAttnPool(cfg)
        self.fuse = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(cfg.dropout)
        self.cell = nn.GRUCell(d, H)                                     # gated recurrent state
        self.norm = nn.LayerNorm(H)
        self.h0 = nn.Parameter(torch.zeros(H))
        self.dir_head = nn.Linear(H, cfg.d_key)
        self.conf_head = nn.Linear(H, 1)
        nn.init.zeros_(self.conf_head.weight)
        nn.init.constant_(self.conf_head.bias, math.log(math.expm1(cfg.k_init - cfg.k_min)))

    def inputs(self, b):
        """z_t for every step. It does not depend on h, so all steps are computed in parallel."""
        B, T = b.valid.shape
        a, has_ev = self.event_enc(b.ev_type, b.ev_xy, b.ev_scroll, b.ev_keys, b.ev_text, self.text_table)
        a = torch.where(has_ev[..., None], a, self.no_event.to(a.dtype))
        f = self.frame_pool(a.reshape(B * T, -1), b.frames.reshape(B * T, *b.frames.shape[2:])).view(B, T, -1)
        return self.drop(self.fuse(torch.cat([f, a], -1)))

    def heads(self, h):
        d = F.normalize(self.dir_head(h).float(), dim=-1)                               # intent direction
        k = (F.softplus(self.conf_head(h).float()) + self.cfg.k_min).clamp(max=self.cfg.k_max)  # confidence
        return d, k

    def step(self, h, z):
        return self.norm(self.cell(z, h))

    def forward(self, b):
        B, T = b.valid.shape
        z = self.inputs(b)
        h = self.h0.expand(B, -1).to(z.dtype)
        hs = []
        for t in range(T):                                               # full BPTT; T is small
            h_new = self.step(h, z[:, t])
            h = torch.where(b.valid[:, t, None], h_new, h)               # freeze state after a session ends
            hs.append(h)
        hs = self.drop(torch.stack(hs, 1))
        return self.heads(hs)                                            # [B, T, D], [B, T, 1]
