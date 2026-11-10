"""The workflow encoder: a per-step transformer over [screen patch tokens | event tokens | history token].

At step t the sequence holds every screen patch token of the step's screenshot (frozen SigLIP2 features projected to
d_model, + patch position and type embeddings), every input event of the step as its own token (+ its order within the
step), and one history token carrying the state h_{t-1}. No causal mask: a step only ever sees its own screen, its own
events and the state from earlier steps (the target instruction is never an input). Masks are padding masks only (empty
event slots); screen-token dropout in training removes tokens.

The state h_t is read at the history position; heads map it to a unit direction d_t in the frozen text-key space and a
confidence kappa_t (inverse temperature of the InfoNCE softmax). arch (what the final block does):
  transf               the history token takes part in every block
  late-history-transf  blocks 0..nl-2 see only screen + event tokens (they do not depend on the state, so they run for
                       all frames of the batch in parallel); the history token joins in the last block
  gru                  nl-1 transformer blocks with the history token in every block, then a GRU cell (model/gru.py)
The last transformer block computes only the history row (no other output is used).
qkv (projections per token type: screen, event, history): shared | sep-q (W_Q) | sep-qkv (W_Q, W_K, W_V).
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

SCREEN, EVENT, HIST = 0, 1, 2


@dataclass
class ModelConfig:
    arch: str
    d: int                   # d_model
    d_head: int
    n_layers: int            # blocks including the final block (for gru: nl - 1 transformer blocks + the GRU)
    n_heads: int
    dropout: float
    qkv: str
    d_key: int               # frozen label (text-key) dimension
    d_vis: int               # frozen screen-feature dimension
    n_tok: int               # screen patch tokens per frame
    n_key_tokens: int        # key-token vocabulary size
    d_text: int              # typed-text embedding dimension
    e_max: int               # event slots per step
    tok_drop: float = 0.0    # training-time fraction of screen tokens dropped
    mlp_ratio: int = 4
    k_min: float = 1.0
    k_max: float = 50.0
    k_init: float = 5.0
    checkpointing: bool = True

    def to_dict(self):
        return asdict(self)


class XBlock(nn.Module):
    """Pre-LN transformer block (multi-head attention + MLP) with optional per-token-type Q/K/V projections."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, dh, nh = cfg.d, cfg.d_head, cfg.n_heads
        self.nh, self.dh = nh, dh
        n_q = 3 if cfg.qkv in ("sep-q", "sep-qkv") else 1
        n_kv = 3 if cfg.qkv == "sep-qkv" else 1
        self.wq = nn.ModuleList([nn.Linear(d, nh * dh) for _ in range(n_q)])
        self.wk = nn.ModuleList([nn.Linear(d, nh * dh) for _ in range(n_kv)])
        self.wv = nn.ModuleList([nn.Linear(d, nh * dh) for _ in range(n_kv)])
        self.wo = nn.Linear(nh * dh, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_ratio * d), nn.GELU(), nn.Linear(cfg.mlp_ratio * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    @staticmethod
    def _proj(ws, x, segs):
        """x [N, L, d]; segs = [(type, length), ...] covering L in order; one weight per token type if len(ws) > 1."""
        if len(ws) == 1:
            return ws[0](x)
        out, s = [], 0
        for typ, n in segs:
            out.append(ws[typ](x[:, s:s + n]))
            s += n
        return torch.cat(out, 1)

    def forward(self, x, segs, key_mask=None, last_only=False):
        """x [N, L, d] -> [N, L, d], or [N, 1, d] (the last token only). key_mask [N, L] bool, True = attend."""
        N, L, _ = x.shape
        y = self.ln1(x)
        k = self._proj(self.wk, y, segs).view(N, L, self.nh, self.dh).transpose(1, 2)
        v = self._proj(self.wv, y, segs).view(N, L, self.nh, self.dh).transpose(1, 2)
        if last_only:
            q = self.wq[segs[-1][0] if len(self.wq) > 1 else 0](y[:, -1:])
            x = x[:, -1:]
        else:
            q = self._proj(self.wq, y, segs)
        Lq = q.shape[1]
        q = q.view(N, Lq, self.nh, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if key_mask is None else key_mask[:, None, None, :])
        x = x + self.drop(self.wo(o.transpose(1, 2).reshape(N, Lq, -1)))
        return x + self.drop(self.mlp(self.ln2(x)))


class WorkflowTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, text_table: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        n_tf = cfg.n_layers - 1 if cfg.arch == "gru" else cfg.n_layers
        self.h_join = n_tf - 1 if cfg.arch == "late-history-transf" else 0   # first block with the history token
        self.register_buffer("text_table", text_table.to(torch.float16), persistent=False)
        self.event_enc = EventEncoder(d, cfg.n_key_tokens, cfg.d_text)
        self.no_event = nn.Parameter(torch.randn(d) * 0.02)           # token of a step without events
        self.ev_pos = nn.Parameter(torch.randn(cfg.e_max, d) * 0.02)  # order of an event within its step
        self.scr_norm = nn.LayerNorm(cfg.d_vis)
        self.scr_proj = nn.Linear(cfg.d_vis, d)
        self.scr_pos = nn.Parameter(torch.randn(cfg.n_tok, d) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList([XBlock(cfg) for _ in range(n_tf)])
        self.h0 = nn.Parameter(torch.zeros(d))
        if cfg.arch == "gru":
            self.gru = GRUHead(d)
        else:
            self.h_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        self.dir_head = nn.Linear(d, cfg.d_key)
        self.conf_head = nn.Linear(d, 1)
        self._init_weights()

    def _init_weights(self):
        resid = 0.02 / math.sqrt(2 * len(self.blocks))                   # GPT-2 style residual scaling
        for blk in self.blocks:
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
            nn.init.normal_(blk.wo.weight, std=resid)
            nn.init.normal_(blk.mlp[-1].weight, std=resid)
        nn.init.zeros_(self.conf_head.weight)
        nn.init.constant_(self.conf_head.bias, math.log(math.expm1(self.cfg.k_init - self.cfg.k_min)))

    def _block(self, blk, x, segs, mask, last_only=False):
        if self.cfg.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(blk, x, segs, mask, last_only, use_reentrant=False)
        return blk(x, segs, mask, last_only)

    def tokens(self, b):
        """State-independent tokens of every frame: x [B*T, n_s + E, d], key mask, segments [(type, length)]."""
        cfg = self.cfg
        B, T = b.valid.shape
        N = B * T
        xs = b.frames.reshape(N, *b.frames.shape[2:])
        n_s = xs.shape[1]
        pos = self.scr_pos[:n_s]
        if self.training and cfg.tok_drop > 0:                           # screen-token dropout = fewer tokens
            keep = max(1, int(round(n_s * (1 - cfg.tok_drop))))
            idx = torch.rand(N, n_s, device=xs.device).argsort(1)[:, :keep]
            xs = xs.gather(1, idx[..., None].expand(-1, -1, xs.shape[-1]))
            pos = pos[idx]
            n_s = keep
        xs = self.scr_proj(self.scr_norm(xs)) + pos + self.type_emb[SCREEN]
        ev, present = self.event_enc(b.ev_type, b.ev_xy, b.ev_scroll, b.ev_keys, b.ev_text, self.text_table)
        E = ev.shape[2]                                                  # [B, T, E, d], [B, T, E]
        slot0 = torch.zeros_like(present)
        slot0[..., 0] = True
        empty = (~present.any(-1, keepdim=True)) & slot0                # no events: slot 0 holds the no-event token
        ev = torch.where(empty[..., None], self.no_event.to(ev.dtype), ev)
        present = present | empty
        xe = (ev + self.ev_pos[:E] + self.type_emb[EVENT]).reshape(N, E, -1)
        x = torch.cat([xs, xe.to(xs.dtype)], 1)
        mask = torch.cat([torch.ones(N, n_s, dtype=torch.bool, device=x.device), present.reshape(N, E)], 1)
        return x, mask, [(SCREEN, n_s), (EVENT, E)]

    def heads(self, h):
        k = (F.softplus(self.conf_head(h).float()) + self.cfg.k_min).clamp(max=self.cfg.k_max)
        return F.normalize(self.dir_head(h).float(), dim=-1), k

    def forward(self, b):
        """Batch -> d [B, T, D_key] unit directions (fp32), kappa [B, T, 1]. The state is frozen after a session
        ends (padded steps repeat its last output)."""
        B, T = b.valid.shape
        x, mask, segs = self.tokens(b)
        j = self.h_join
        for blk in self.blocks[:j]:                                     # parallel over all B*T frames
            x = self._block(blk, x, segs, mask)
        L0, dt = x.shape[1], x.dtype
        xt = x.view(B, T, L0, -1).unbind(1)          # unbind, not [:, t]: one stacked gradient instead of T copies
        mt = mask.view(B, T, L0).unbind(1)
        val = b.valid.unbind(1)
        seq = self.blocks[j:]
        segs_h = segs + [(HIST, 1)]
        one = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        h = self.h0.expand(B, -1).float()
        hs = []
        for t in range(T):
            X = torch.cat([xt[t], (h + self.type_emb[HIST]).to(dt)[:, None]], 1)
            m = torch.cat([mt[t], one], 1)
            for i, blk in enumerate(seq):
                X = self._block(blk, X, segs_h, m, last_only=i == len(seq) - 1)
            out = X[:, -1]
            h_new = self.gru(out, h) if self.cfg.arch == "gru" else self.h_norm(out).float()
            h = torch.where(val[t][:, None], h_new, h)
            hs.append(h)
        return self.heads(self.drop(torch.stack(hs, 1)))
