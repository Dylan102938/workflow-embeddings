"""Input-event tokens: one d_model vector per input event of a step."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..prep.actions import EVENT_TYPES


class Fourier(nn.Module):
    def __init__(self, n_freq: int):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)

    def forward(self, x):                                   # [..., k] -> [..., k * 2F]
        a = x[..., None] * self.freqs
        return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class EventEncoder(nn.Module):
    """type embedding + Fourier(x, y, x2, y2) + scroll + key-token embeddings + projected typed-text embedding,
    then a residual MLP. Returns per-event vectors [..., E, d] and the presence mask [..., E]."""

    def __init__(self, d: int, n_key_tokens: int, d_text: int, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq
        self.type_emb = nn.Embedding(len(EVENT_TYPES), d)
        self.fourier = Fourier(n_freq)
        self.xy_proj = nn.Linear(4 * 2 * n_freq + 4, d)
        self.scroll_proj = nn.Linear(2, d)
        self.key_emb = nn.Embedding(n_key_tokens, d, padding_idx=0)
        self.text_proj = nn.Linear(d_text, d)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, ev_type, ev_xy, ev_scroll, ev_keys, ev_text, text_table):
        present = ev_type > 0                                            # [..., E]
        xy_ok = ~torch.isnan(ev_xy)                                      # [..., E, 4]
        fx = self.fourier(torch.nan_to_num(ev_xy, 0.0)) * xy_ok.repeat_interleave(2 * self.n_freq, -1)
        h = self.type_emb(ev_type) + self.xy_proj(torch.cat([fx, xy_ok.float()], -1))
        h = h + self.scroll_proj(torch.stack([ev_scroll.sign(), torch.log1p(ev_scroll.abs())], -1))
        h = h + self.key_emb(ev_keys).sum(-2)
        has_text = ev_text >= 0
        h = h + self.text_proj(text_table[ev_text.clamp(min=0)].float() * has_text[..., None])
        h = h + self.mlp(h)
        return h * present[..., None], present
