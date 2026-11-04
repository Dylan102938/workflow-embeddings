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
    prev_query: str = ""     # "" | "sum" (prev action added to the same queries) | "sep" (its own query group)
    gru_layers: int = 1      # stacked recurrent cells; the state is their concatenation, heads read the top one
    keep_bias: float = 0.0   # added to the GRU update-gate bias at init (> 0 = biased toward keeping the state)
    n_mix: int = 1           # readout components (pseudocode upgrade 1: K directions with mixture weights)
    compress_latents: int = 0  # Perceiver-style learned compression: L latent queries over dense tokens (0 = off)
    compress_blocks: int = 1   # (cross-attention + latent self-attention) blocks in the compressor
    hist_queries: int = 0      # readout queries derived from the recurrent state h[t-1] (pooling becomes sequential)
    static_queries: int = 0    # learned state- and action-independent readout queries (whole-screen inspection)

    def __post_init__(self):
        # checkpoints from before prev_query became a mode string stored a bool; True meant "sum"
        if isinstance(self.prev_query, bool):
            self.prev_query = "sum" if self.prev_query else ""


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


def _mlp(d):
    return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


def _token_drop_mask(N, n, p, device):
    """[N, 1, 1, n] True = attend; drops a fraction p of tokens, always keeping at least one."""
    keep = torch.rand(N, n, device=device) >= p
    keep[torch.arange(N, device=device), torch.randint(0, n, (N,), device=device)] = True
    return keep[:, None, None, :]


class PerceiverBlock(nn.Module):
    """Latents cross-attend to (shared) token keys/values, then self-attend; pre-norm residual MLPs."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln_q, self.q, self.o = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, d)
        self.ln_ff1, self.ff1 = nn.LayerNorm(d), _mlp(d)
        self.ln_sa, self.qkv, self.o_sa = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ln_ff2, self.ff2 = nn.LayerNorm(d), _mlp(d)

    def forward(self, lat, k, v, mask=None):                            # lat [N, L, d]; k, v [N, heads, n, dh]
        N, L, d = lat.shape
        sh = lambda t: t.view(N, L, self.heads, -1).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(self.q(self.ln_q(lat))), k, v, attn_mask=mask)
        lat = lat + self.o(o.transpose(1, 2).reshape(N, L, d))
        lat = lat + self.ff1(self.ln_ff1(lat))
        q, kk, vv = self.qkv(self.ln_sa(lat)).chunk(3, -1)
        o = F.scaled_dot_product_attention(sh(q), sh(kk), sh(vv))
        lat = lat + self.o_sa(o.transpose(1, 2).reshape(N, L, d))
        return lat + self.ff2(self.ln_ff2(lat))


class Compressor(nn.Module):
    """Learned compression of dense screen features (Perceiver pattern): L latent queries attend over the
    frame's dense tokens and return L compact tokens. State-independent, so all frames run in parallel."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, self.heads, self.tok_drop = cfg.d, cfg.n_heads, cfg.tok_drop
        self.kv_norm = nn.LayerNorm(cfg.d_vis)
        self.k_proj, self.v_proj = nn.Linear(cfg.d_vis, d), nn.Linear(cfg.d_vis, d)
        self.pos = nn.Parameter(torch.randn(cfg.n_tok, d) * 0.02)
        self.latents = nn.Parameter(torch.randn(cfg.compress_latents, d) * 0.02)
        self.blocks = nn.ModuleList([PerceiverBlock(d, cfg.n_heads) for _ in range(cfg.compress_blocks)])

    def forward(self, tokens):                                           # [N, n_tok, d_vis] -> [N, L, d]
        N, n, _ = tokens.shape
        x = self.kv_norm(tokens)
        sh = lambda t: t.view(N, n, self.heads, -1).transpose(1, 2)
        k, v = sh(self.k_proj(x) + self.pos[:n]), sh(self.v_proj(x))
        mask = _token_drop_mask(N, n, self.tok_drop, tokens.device) if self.training and self.tok_drop > 0 else None
        lat = self.latents.expand(N, -1, -1).to(k.dtype)
        for blk in self.blocks:
            lat = blk(lat, k, v, mask)
        return lat


class CrossAttnPool(nn.Module):
    """Readout over a frame's tokens (raw frame tokens, or the compressor's latents).
    Query groups: action-conditioned (current action, optionally + previous action), learned static queries
    (whole-screen), and history queries derived from the recurrent state h[t-1]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, Q = cfg.d, cfg.n_queries
        self.Q, self.heads = Q, cfg.n_heads
        self.compressed = cfg.compress_latents > 0
        self.tok_drop = 0.0 if self.compressed else cfg.tok_drop        # token dropout lives in the compressor then
        src_dim = d if self.compressed else cfg.d_vis
        n_src = cfg.compress_latents if self.compressed else cfg.n_tok
        self.kv_norm = nn.LayerNorm(src_dim)
        self.k_proj = nn.Linear(src_dim, d)
        self.v_proj = nn.Linear(src_dim, d)
        self.pos = nn.Parameter(torch.randn(n_src, d) * 0.02)
        self.q_base = nn.Parameter(torch.randn(Q, d) * 0.02)
        self.q_from_a = nn.Linear(d, Q * d)
        self.prev_mode = cfg.prev_query
        self.q_from_prev = nn.Linear(d, Q * d) if cfg.prev_query else None
        if cfg.prev_query == "sep":
            self.q_base_prev = nn.Parameter(torch.randn(Q, d) * 0.02)
        self.Qs, self.Qh = cfg.static_queries, cfg.hist_queries
        if self.Qs:
            self.q_static = nn.Parameter(torch.randn(self.Qs, d) * 0.02)
        if self.Qh:
            self.q_from_h = nn.Linear(cfg.h, self.Qh * d)
        n_q = (2 if cfg.prev_query == "sep" else 1) * Q + self.Qs + self.Qh
        self.out = nn.Linear(n_q * d, d)

    def prepare(self, a, src, a_prev=None):
        """State-independent part (parallel over N frames): queries [N, Q', d], keys/values [N, heads, S, dh]."""
        N, n, _ = src.shape
        x = self.kv_norm(src)
        sh = lambda t: t.view(N, t.shape[1], self.heads, -1).transpose(1, 2)
        k, v = sh(self.k_proj(x) + self.pos[:n]), sh(self.v_proj(x))
        q = self.q_base + self.q_from_a(a).view(N, self.Q, -1)
        if self.prev_mode == "sum":
            q = q + self.q_from_prev(a_prev).view(N, self.Q, -1)
        elif self.prev_mode == "sep":
            q = torch.cat([q, self.q_base_prev + self.q_from_prev(a_prev).view(N, self.Q, -1)], 1)
        if self.Qs:
            q = torch.cat([q, self.q_static.to(q.dtype).expand(N, -1, -1)], 1)
        mask = _token_drop_mask(N, n, self.tok_drop, src.device) if self.training and self.tok_drop > 0 else None
        return q, k, v, mask

    def attend(self, q, k, v, mask=None, h=None):
        """q [N, Q', d] (+ history queries from h [N, H]) -> pooled frame vector [N, d]."""
        N = q.shape[0]
        if self.Qh:
            q = torch.cat([q, self.q_from_h(h).view(N, self.Qh, -1).to(q.dtype)], 1)
        qh = q.view(N, q.shape[1], self.heads, -1).transpose(1, 2)
        o = F.scaled_dot_product_attention(qh, k, v, attn_mask=mask)    # [N, heads, Q, d/heads]
        return self.out(o.transpose(1, 2).reshape(N, -1))

    def forward(self, a, src, a_prev=None):                              # parallel path (no history queries)
        return self.attend(*self.prepare(a, src, a_prev))


class SessionStateMachine(nn.Module):
    def __init__(self, cfg: ModelConfig, text_table: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.d, cfg.h
        self.register_buffer("text_table", text_table.to(torch.float16), persistent=False)
        self.event_enc = EventEncoder(cfg)
        self.no_event = nn.Parameter(torch.randn(d) * 0.02)              # used for the final screen (no action)
        self.compressor = Compressor(cfg) if cfg.compress_latents else None
        self.frame_pool = CrossAttnPool(cfg)
        self.fuse = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(cfg.dropout)
        self.cell = nn.GRUCell(d, H)                                     # gated recurrent state
        self.norm = nn.LayerNorm(H)
        L = cfg.gru_layers
        self.cells_extra = nn.ModuleList([nn.GRUCell(H, H) for _ in range(L - 1)])
        self.norms_extra = nn.ModuleList([nn.LayerNorm(H) for _ in range(L - 1)])
        self.h0 = nn.Parameter(torch.zeros(L * H))
        if cfg.keep_bias:
            with torch.no_grad():                                        # gate order (r, z, n); h' = (1-z) n + z h
                for c in [self.cell, *self.cells_extra]:
                    c.bias_ih[H:2 * H] += cfg.keep_bias
        K = cfg.n_mix
        self.dir_head = nn.Linear(H, cfg.d_key * K)
        self.conf_head = nn.Linear(H, K)
        nn.init.zeros_(self.conf_head.weight)
        nn.init.constant_(self.conf_head.bias, math.log(math.expm1(cfg.k_init - cfg.k_min)))
        self.mix_head = None
        if K > 1:                                                        # mixture weights, uniform at init
            self.mix_head = nn.Linear(H, K)
            nn.init.zeros_(self.mix_head.weight)
            nn.init.zeros_(self.mix_head.bias)

    def actions(self, b):
        """Action embeddings a [B, T, d] and the previous step's action a_prev [B*T, d] (or None)."""
        B, T = b.valid.shape
        a, has_ev = self.event_enc(b.ev_type, b.ev_xy, b.ev_scroll, b.ev_keys, b.ev_text, self.text_table)
        a = torch.where(has_ev[..., None], a, self.no_event.to(a.dtype))
        a_prev = None
        if self.cfg.prev_query:   # the action that produced this frame; the no-event embedding at t = 0
            start = self.no_event.to(a.dtype).expand(B, 1, -1)
            a_prev = torch.cat([start, a[:, :-1]], 1).reshape(B * T, -1)
        return a, a_prev

    def frame_source(self, b):
        """Per-frame tokens the readout attends over: raw frame tokens, or learned latents (parallel)."""
        B, T = b.valid.shape
        src = b.frames.reshape(B * T, *b.frames.shape[2:])
        return self.compressor(src) if self.compressor is not None else src

    def inputs(self, b):
        """z_t for every step when pooling does not depend on h (no history queries): all steps in parallel."""
        B, T = b.valid.shape
        a, a_prev = self.actions(b)
        f = self.frame_pool(a.reshape(B * T, -1), self.frame_source(b), a_prev).view(B, T, -1)
        return self.drop(self.fuse(torch.cat([f, a], -1)))

    def heads(self, h):
        """K = 1: d [.., D], k [.., 1], None.   K > 1: d [.., K, D], k [.., K, 1], log pi [.., K]."""
        h = h[..., -self.cfg.h:]                                                         # top recurrent layer
        k = (F.softplus(self.conf_head(h).float()) + self.cfg.k_min).clamp(max=self.cfg.k_max)  # confidence
        if self.mix_head is None:
            return F.normalize(self.dir_head(h).float(), dim=-1), k, None              # intent direction
        K = self.cfg.n_mix
        d = F.normalize(self.dir_head(h).float().view(*h.shape[:-1], K, -1), dim=-1)
        return d, k.unsqueeze(-1), self.mix_head(h).float().log_softmax(-1)

    def step(self, h, z):
        H = self.cfg.h
        x = self.norm(self.cell(z, h[:, :H]))
        outs = [x]
        for l, (cell, norm) in enumerate(zip(self.cells_extra, self.norms_extra), start=1):
            x = norm(cell(x, h[:, l * H:(l + 1) * H]))
            outs.append(x)
        return outs[0] if len(outs) == 1 else torch.cat(outs, -1)

    def forward(self, b):
        B, T = b.valid.shape
        if not self.cfg.hist_queries:
            z = self.inputs(b)
            h = self.h0.expand(B, -1).to(z.dtype)
            hs = []
            for t in range(T):                                           # full BPTT; T is small
                h_new = self.step(h, z[:, t])
                h = torch.where(b.valid[:, t, None], h_new, h)           # freeze state after a session ends
                hs.append(h)
        else:
            # History-conditioned pooling: queries depend on h[t-1], so the readout runs inside the recurrence.
            # Keys/values (and the compressor) are state-independent and precomputed for all steps in parallel.
            a, a_prev = self.actions(b)
            q, k, v, mask = self.frame_pool.prepare(a.reshape(B * T, -1), self.frame_source(b), a_prev)
            q = q.view(B, T, *q.shape[1:])
            k, v = k.view(B, T, *k.shape[1:]), v.view(B, T, *v.shape[1:])
            mask = mask.view(B, T, *mask.shape[1:]) if mask is not None else None
            h = self.h0.expand(B, -1).to(a.dtype)
            hs = []
            for t in range(T):
                f = self.frame_pool.attend(q[:, t], k[:, t], v[:, t], None if mask is None else mask[:, t],
                                           h=h[:, -self.cfg.h:])
                z = self.drop(self.fuse(torch.cat([f, a[:, t]], -1)))
                h_new = self.step(h, z)
                h = torch.where(b.valid[:, t, None], h_new, h)
                hs.append(h)
        hs = self.drop(torch.stack(hs, 1))
        return self.heads(hs)                                            # d, kappa, log pi (None if K = 1)
