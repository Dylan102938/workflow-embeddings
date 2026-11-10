"""The workflow encoder: a per-step transformer over [screen tokens | event tokens | history token].

Design and the three architectures: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .events import EventEncoder
from .gru import GRUHead

SCREEN, EVENT, HISTORY = 0, 1, 2  # rows of the token-type embedding


@dataclass
class ModelConfig:
    architecture: str
    d_model: int
    d_head: int
    n_layers: int
    n_heads: int
    dropout: float
    d_label: int  # dimension of the frozen instruction keys
    d_screen: int  # dimension of the frozen screen features
    n_patches: int
    n_key_tokens: int
    d_text: int  # dimension of the frozen typed-text embeddings
    max_events: int  # event slots per step
    screen_token_dropout: float = 0.0
    mlp_ratio: int = 4
    confidence_min: float = 1.0
    confidence_max: float = 50.0
    confidence_init: float = 5.0
    activation_checkpointing: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class TransformerBlock(nn.Module):
    """Pre-norm block: multi-head self-attention, then an MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model, d_attention = config.d_model, config.n_heads * config.d_head
        self.n_heads, self.d_head = config.n_heads, config.d_head
        self.query = nn.Linear(d_model, d_attention)
        self.key = nn.Linear(d_model, d_attention)
        self.value = nn.Linear(d_model, d_attention)
        self.output = nn.Linear(d_attention, d_model)
        self.attention_norm = nn.LayerNorm(d_model)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, config.mlp_ratio * d_model), nn.GELU(), nn.Linear(config.mlp_ratio * d_model, d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self, tokens: torch.Tensor, attend_mask: torch.Tensor | None = None, last_token_only: bool = False
    ) -> torch.Tensor:
        """tokens [N, L, d_model]; attend_mask [N, L] (True = may be attended to).

        Returns [N, L, d_model], or [N, 1, d_model] when only the last token's output is needed.
        """
        normed = self.attention_norm(tokens)
        keys, values = self._split_heads(self.key(normed)), self._split_heads(self.value(normed))
        if last_token_only:
            normed, tokens = normed[:, -1:], tokens[:, -1:]
        queries = self._split_heads(self.query(normed))
        mask = None if attend_mask is None else attend_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[1], -1)
        tokens = tokens + self.dropout(self.output(attended))
        return tokens + self.dropout(self.mlp(self.mlp_norm(tokens)))


class WorkflowTransformer(nn.Module):
    def __init__(self, config: ModelConfig, text_embeddings: torch.Tensor):
        super().__init__()
        self.config = config
        d_model = config.d_model
        n_blocks = config.n_layers - 1 if config.architecture == "gru" else config.n_layers
        # Blocks before this index see only screen and event tokens, so they run for all steps at once.
        self.first_block_with_history = n_blocks - 1 if config.architecture == "late-history-transf" else 0

        self.register_buffer("text_embeddings", text_embeddings.to(torch.float16), persistent=False)
        self.event_encoder = EventEncoder(d_model, config.n_key_tokens, config.d_text)
        self.no_event_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.event_positions = nn.Parameter(torch.randn(config.max_events, d_model) * 0.02)
        self.screen_norm = nn.LayerNorm(config.d_screen)
        self.screen_projection = nn.Linear(config.d_screen, d_model)
        self.screen_positions = nn.Parameter(torch.randn(config.n_patches, d_model) * 0.02)
        self.token_type_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(n_blocks)])
        self.initial_state = nn.Parameter(torch.zeros(d_model))
        if config.architecture == "gru":
            self.gru = GRUHead(d_model)
        else:
            self.state_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.direction_head = nn.Linear(d_model, config.d_label)
        self.confidence_head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        residual_std = 0.02 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)
            nn.init.normal_(block.output.weight, std=residual_std)
            nn.init.normal_(block.mlp[-1].weight, std=residual_std)
        config = self.config
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.constant_(
            self.confidence_head.bias, math.log(math.expm1(config.confidence_init - config.confidence_min))
        )

    def _run_block(self, block, tokens, attend_mask, last_token_only=False):
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, tokens, attend_mask, last_token_only, use_reentrant=False)
        return block(tokens, attend_mask, last_token_only)

    def screen_tokens(self, batch) -> torch.Tensor:
        """[B * T, n_screen_tokens, d_model]; in training a random subset of each frame's patches is kept."""
        features = batch.frames.flatten(0, 1)
        n_frames, n_patches, _ = features.shape
        positions = self.screen_positions[:n_patches]
        if self.training and self.config.screen_token_dropout > 0:
            n_kept = max(1, int(round(n_patches * (1 - self.config.screen_token_dropout))))
            kept = torch.rand(n_frames, n_patches, device=features.device).argsort(1)[:, :n_kept]
            features = features.gather(1, kept[..., None].expand(-1, -1, features.shape[-1]))
            positions = positions[kept]
        return self.screen_projection(self.screen_norm(features)) + positions + self.token_type_embedding[SCREEN]

    def event_tokens(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """[B * T, E, d_model] and the mask [B * T, E] of slots that hold a token.

        A step without events gets the no-event token in slot 0.
        """
        tokens, is_present = self.event_encoder(batch, self.text_embeddings)
        first_slot = torch.zeros_like(is_present)
        first_slot[..., 0] = True
        is_placeholder = ~is_present.any(-1, keepdim=True) & first_slot
        tokens = torch.where(is_placeholder[..., None], self.no_event_token.to(tokens.dtype), tokens)
        tokens = tokens + self.event_positions[: tokens.shape[2]] + self.token_type_embedding[EVENT]
        return tokens.flatten(0, 1), (is_present | is_placeholder).flatten(0, 1)

    def step_tokens(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Every step's screen and event tokens [B * T, L, d_model] and their attend mask [B * T, L]."""
        screen = self.screen_tokens(batch)
        events, event_mask = self.event_tokens(batch)
        screen_mask = torch.ones(screen.shape[:2], dtype=torch.bool, device=screen.device)
        return torch.cat([screen, events.to(screen.dtype)], 1), torch.cat([screen_mask, event_mask], 1)

    def heads(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        directions = F.normalize(self.direction_head(states).float(), dim=-1)
        confidences = F.softplus(self.confidence_head(states).float()) + self.config.confidence_min
        return directions, confidences.clamp(max=self.config.confidence_max)

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch -> unit directions [B, T, d_label] and confidences [B, T, 1], both float32.

        After a session's last step its state is frozen, so padded steps repeat its final output.
        """
        batch_size, n_steps = batch.valid.shape
        tokens, attend_mask = self.step_tokens(batch)
        for block in self.blocks[: self.first_block_with_history]:
            tokens = self._run_block(block, tokens, attend_mask)

        # unbind (rather than indexing [:, t] in the loop) keeps one stacked gradient instead of one per step
        tokens_by_step = tokens.view(batch_size, n_steps, *tokens.shape[1:]).unbind(1)
        masks_by_step = attend_mask.view(batch_size, n_steps, -1).unbind(1)
        valid_by_step = batch.valid.unbind(1)
        history_blocks = self.blocks[self.first_block_with_history :]
        history_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
        history_type = self.token_type_embedding[HISTORY]

        state = self.initial_state.expand(batch_size, -1).float()
        states = []
        for step in range(n_steps):
            history_token = (state + history_type).to(tokens.dtype)[:, None]
            sequence = torch.cat([tokens_by_step[step], history_token], 1)
            sequence_mask = torch.cat([masks_by_step[step], history_mask], 1)
            for index, block in enumerate(history_blocks):
                is_last = index == len(history_blocks) - 1
                sequence = self._run_block(block, sequence, sequence_mask, last_token_only=is_last)
            block_output = sequence[:, -1]
            if self.config.architecture == "gru":
                new_state = self.gru(block_output, state)
            else:
                new_state = self.state_norm(block_output).float()
            state = torch.where(valid_by_step[step][:, None], new_state, state)
            states.append(state)
        return self.heads(self.dropout(torch.stack(states, 1)))
