"""The final block of arch `gru`: a GRU cell integrates the transformer's history-position output into the state."""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUHead(nn.Module):
    """h_t = LayerNorm(GRUCell(LayerNorm(x_t), h_{t-1})), x_t = the last transformer block's output at the history
    position."""

    def __init__(self, d: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(d)
        self.cell = nn.GRUCell(d, d)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.norm_out(self.cell(self.norm_in(x), h.to(x.dtype))).float()
