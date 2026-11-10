"""The final block of architecture `gru`: a GRU cell folds the transformer's output into the state."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.cell = nn.GRUCell(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, block_output: torch.Tensor, previous_state: torch.Tensor) -> torch.Tensor:
        new_state = self.cell(self.input_norm(block_output), previous_state.to(block_output.dtype))
        return self.output_norm(new_state).float()
