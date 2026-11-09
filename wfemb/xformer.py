"""TransformerStateMachine: a per-step transformer over [screen tokens | event tokens | history token].

At step t the sequence holds every screen patch token (frozen SigLIP2 dense grid, projected to d_model), every
input event of the step as its own token (the same event encoder as the pooling model, not summed, plus a learned
position embedding for its order within the step) and one history token h[t-1]. There is no causal mask: the
target (the instruction) is not among the inputs, and a step only ever sees its own screen, its own events and
the state carried from earlier steps. The only masks are padding masks for empty event slots and, in training,
screen-token dropout (implemented by dropping tokens).

  arch xf      h_t = LayerNorm(output of the last block at the history position)
  arch xf_gru  h_t = LayerNorm(GRU(that output, h[t-1]))

h_join = j: blocks 0..j-1 see only screen + event tokens. They do not depend on the state, so they run for every
frame of the batch in parallel; the history token joins at block j and blocks j.. run inside the recurrence.
j = 0 is the full model (history in every block); j = n_layers - 1 reads the history out in the last block only.
The last block only computes the history row (no other output is used).

qkv_mode: shared (standard transformer) | sep_q (W_Q per token type, shared W_K, W_V) | sep_qkv (all three per
token type). Token types: screen, event, history. Grouped-query attention: n_kv_heads key/value heads shared by
n_heads query heads of width d_head (n_kv_heads = n_heads is multi-head attention).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import EventEncoder, ModelConfig

SCREEN, EVENT, HIST = 0, 1, 2


class XBlock(nn.Module):
    """Pre-LN transformer block with token-type-specific projections."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, dh = cfg.d, cfg.d_head
        self.nq, self.nkv, self.dh = cfg.n_heads, cfg.n_kv_heads or cfg.n_heads, dh
        assert self.nq % self.nkv == 0, "n_heads must be a multiple of n_kv_heads"
        n_q = 3 if cfg.qkv_mode in ("sep_q", "sep_qkv") else 1
        n_kv = 3 if cfg.qkv_mode == "sep_qkv" else 1
        self.wq = nn.ModuleList([nn.Linear(d, self.nq * dh) for _ in range(n_q)])
        self.wk = nn.ModuleList([nn.Linear(d, self.nkv * dh) for _ in range(n_kv)])
        self.wv = nn.ModuleList([nn.Linear(d, self.nkv * dh) for _ in range(n_kv)])
        self.wo = nn.Linear(self.nq * dh, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_ratio * d), nn.GELU(), nn.Linear(cfg.mlp_ratio * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    @staticmethod
    def _proj(ws, x, segs):
        """x [N, L, d]; segs = [(type, length), ...] covers L in order. One weight per type if len(ws) > 1."""
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
        k = self._proj(self.wk, y, segs).view(N, L, self.nkv, self.dh).transpose(1, 2)
        v = self._proj(self.wv, y, segs).view(N, L, self.nkv, self.dh).transpose(1, 2)
        if last_only:
            q = self.wq[segs[-1][0] if len(self.wq) > 1 else 0](y[:, -1:])
            x = x[:, -1:]
        else:
            q = self._proj(self.wq, y, segs)
        Lq = q.shape[1]
        q = q.view(N, Lq, self.nq, self.dh).transpose(1, 2)
        if self.nkv != self.nq:
            r = self.nq // self.nkv
            k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if key_mask is None else key_mask[:, None, None, :])
        x = x + self.drop(self.wo(o.transpose(1, 2).reshape(N, Lq, -1)))
        return x + self.drop(self.mlp(self.ln2(x)))


class TransformerStateMachine(nn.Module):
    def __init__(self, cfg: ModelConfig, text_table: torch.Tensor):
        super().__init__()
        assert cfg.arch in ("xf", "xf_gru"), cfg.arch
        assert cfg.h == cfg.d, "the history token lives in the model width: h must equal d"
        assert 0 <= cfg.h_join < cfg.n_layers, "the history token must join at some block"
        self.cfg = cfg
        d = cfg.d
        self.register_buffer("text_table", text_table.to(torch.float16), persistent=False)
        self.event_enc = EventEncoder(cfg)
        self.no_event = nn.Parameter(torch.randn(d) * 0.02)           # token for a step without events
        self.ev_pos = nn.Parameter(torch.randn(cfg.e_max, d) * 0.02) if cfg.ev_pos else None
        self.scr_norm = nn.LayerNorm(cfg.d_vis)
        self.scr_proj = nn.Linear(cfg.d_vis, d)
        self.scr_pos = nn.Parameter(torch.randn(cfg.n_tok, d) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList([XBlock(cfg) for _ in range(cfg.n_layers)])
        self.h0 = nn.Parameter(torch.zeros(d))
        self.h_norm = nn.LayerNorm(d)
        self.gru = nn.GRUCell(d, d) if cfg.arch == "xf_gru" else None
        self.gru_in = nn.LayerNorm(d) if cfg.arch == "xf_gru" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.dir_head = nn.Linear(d, cfg.d_key)
        self.conf_head = nn.Linear(d, 1)
        self._init_weights()

    def _init_weights(self):
        resid = 0.02 / math.sqrt(2 * self.cfg.n_layers)                    # GPT-2 style residual scaling
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
        if self.cfg.xf_ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(blk, x, segs, mask, last_only, use_reentrant=False)
        return blk(x, segs, mask, last_only)

    def tokens(self, b):
        """State-independent tokens of every frame: x [B*T, n_s + E, d], key mask, segments."""
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
        ev, present = self.event_enc(b.ev_type, b.ev_xy, b.ev_scroll, b.ev_keys, b.ev_text, self.text_table,
                                     per_event=True)                   # [B, T, E, d], [B, T, E]
        E = ev.shape[2]
        slot0 = torch.zeros_like(present)
        slot0[..., 0] = True
        empty = (~present.any(-1, keepdim=True)) & slot0                # no events: slot 0 holds the no-event token
        ev = torch.where(empty[..., None], self.no_event.to(ev.dtype), ev)
        present = present | empty
        if self.ev_pos is not None:
            ev = ev + self.ev_pos[:E]
        xe = (ev + self.type_emb[EVENT]).reshape(N, E, -1)
        x = torch.cat([xs, xe.to(xs.dtype)], 1)
        mask = torch.cat([torch.ones(N, n_s, dtype=torch.bool, device=x.device), present.reshape(N, E)], 1)
        return x, mask, [(SCREEN, n_s), (EVENT, E)]

    def heads(self, h):
        cfg = self.cfg
        k = (F.softplus(self.conf_head(h).float()) + cfg.k_min).clamp(max=cfg.k_max)
        return F.normalize(self.dir_head(h).float(), dim=-1), k, None

    def forward(self, b):
        B, T = b.valid.shape
        x, mask, segs = self.tokens(b)
        j = self.cfg.h_join
        for blk in self.blocks[:j]:                                     # parallel over all B*T frames
            x = self._block(blk, x, segs, mask)
        L0, dt = x.shape[1], x.dtype
        xt = x.view(B, T, L0, -1).unbind(1)
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
            if self.gru is None:
                h_new = self.h_norm(out).float()
            else:
                h_new = self.h_norm(self.gru(self.gru_in(out), h.to(out.dtype))).float()
            h = torch.where(val[t][:, None], h_new, h)
            hs.append(h)
        hs = self.drop(torch.stack(hs, 1))
        return self.heads(hs)
