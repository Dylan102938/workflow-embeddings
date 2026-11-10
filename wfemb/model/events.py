"""Event tokens: one d_model vector per input event of a step."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..prep.actions import EVENT_TYPES


class FourierFeatures(nn.Module):
    def __init__(self, n_frequencies: int):
        super().__init__()
        self.register_buffer("frequencies", (2.0 ** torch.arange(n_frequencies)) * math.pi, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """[..., k] -> [..., k * 2 * n_frequencies]"""
        angles = values[..., None] * self.frequencies
        return torch.cat([angles.sin(), angles.cos()], -1).flatten(-2)


class EventEncoder(nn.Module):
    """Sums an event's type, screen points, scroll amount, key tokens and typed text, then a residual MLP."""

    def __init__(self, d_model: int, n_key_tokens: int, d_text: int, n_frequencies: int = 16):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.type_embedding = nn.Embedding(len(EVENT_TYPES), d_model)
        self.fourier = FourierFeatures(n_frequencies)
        self.point_projection = nn.Linear(4 * 2 * n_frequencies + 4, d_model)
        self.scroll_projection = nn.Linear(2, d_model)
        self.key_embedding = nn.Embedding(n_key_tokens, d_model, padding_idx=0)
        self.text_projection = nn.Linear(d_text, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )

    def forward(self, batch, text_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch -> event tokens [B, T, E, d_model] (zero for empty slots) and the slot mask [B, T, E]."""
        is_present = batch.event_types > 0
        has_point = ~torch.isnan(batch.event_points)
        point_features = self.fourier(torch.nan_to_num(batch.event_points, 0.0))
        point_features = point_features * has_point.repeat_interleave(2 * self.n_frequencies, -1)
        scroll_features = torch.stack([batch.event_scrolls.sign(), torch.log1p(batch.event_scrolls.abs())], -1)
        has_text = batch.event_texts >= 0
        text_features = text_embeddings[batch.event_texts.clamp(min=0)].float() * has_text[..., None]

        tokens = (
            self.type_embedding(batch.event_types)
            + self.point_projection(torch.cat([point_features, has_point.float()], -1))
            + self.scroll_projection(scroll_features)
            + self.key_embedding(batch.event_keys).sum(-2)
            + self.text_projection(text_features)
        )
        tokens = tokens + self.mlp(tokens)
        return tokens * is_present[..., None], is_present
