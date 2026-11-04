"""Evaluation on held-out val/test (pseudocode section 6).

U_ref = keys of the training instructions + the evaluated split's instructions. The split's sessions
never share an instruction group with train, so each one's own key is a genuinely new candidate.
"""
from __future__ import annotations

import collections
import json

import numpy as np
import torch
import torch.nn.functional as F

from . import paths
from .data import SessionData, length_bucketed_batches
from .loss import false_negative_mask, info_nce


@torch.no_grad()
def embed_split(model, data: SessionData, sidx: np.ndarray, device, batch: int = 64, max_frames: int = 4096):
    """Returns d [S, T_max, D] fp16, k [S, T_max] fp32, lengths [S] (CPU), in the order of sidx."""
    model.eval()
    T_max = int(data.length[sidx].max())
    D = model.cfg.d_key
    d_all = torch.zeros(len(sidx), T_max, D, dtype=torch.float16)
    k_all = torch.zeros(len(sidx), T_max)
    pos = {int(s): i for i, s in enumerate(sidx)}
    rng = np.random.default_rng(0)
    for bidx in length_bucketed_batches(data.length, sidx, batch, rng, drop_last=False, max_frames=max_frames):
        b = data.finalize(data.make_batch(bidx).to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d, k = model(b)
        rows = [pos[int(s)] for s in bidx]
        T = d.shape[1]
        d_all[rows, :T] = d.half().cpu()
        k_all[rows, :T] = k.squeeze(-1).float().cpu()
    model.train()
    return d_all, k_all, torch.from_numpy(data.length[sidx].astype(np.int64))


def _retrieval(E: torch.Tensor, groups: np.ndarray, apps: np.ndarray, within_app: bool):
    """Session-to-session retrieval with relevance = same duplicate-instruction group."""
    E = F.normalize(E.float(), dim=-1)
    S = E @ E.T
    S.fill_diagonal_(-float("inf"))
    cnt = collections.Counter(groups.tolist())
    q = [i for i in range(len(groups)) if cnt[groups[i]] > 1]
    if not q:
        return {}
    r1, r5, mrr = [], [], []
    for i in q:
        s = S[i].clone()
        if within_app:
            s[torch.from_numpy(apps != apps[i])] = -float("inf")
        rel = torch.from_numpy(groups == groups[i])
        rel[i] = False
        order = s.argsort(descending=True)
        ranks = torch.nonzero(rel[order]).flatten()
        first = int(ranks[0]) + 1
        r1.append(first <= 1); r5.append(first <= 5); mrr.append(1.0 / first)
    tag = "within_app" if within_app else "cross_app"
    return {f"session_R@1_{tag}": float(np.mean(r1)), f"session_R@5_{tag}": float(np.mean(r5)),
            f"session_MRR_{tag}": float(np.mean(mrr)), "n_session_queries": len(q)}


def load_judged(split: str, sidx: np.ndarray, groups: np.ndarray):
    """Graded relevance between sessions of a split: trace-judged labels (0/1/2) from
    KEYS/judged_<split>.jsonl, plus duplicate-instruction groups (gain 2). Keys are positions in sidx."""
    path = paths.KEYS / f"judged_{split}.jsonl"
    if not path.exists():
        return None
    pos = {int(s): i for i, s in enumerate(sidx)}
    rel = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        a, b = pos.get(r["a"]), pos.get(r["b"])
        if a is None or b is None:
            continue
        g = int(r["label"])
        rel[a][b] = max(rel[a].get(b, 0), g)
        rel[b][a] = max(rel[b].get(a, 0), g)
    members = collections.defaultdict(list)
    for i, g in enumerate(groups.tolist()):
        members[g].append(i)
    for ms in members.values():
        for a in ms:
            for b in ms:
                if a != b:
                    rel[a][b] = 2
    return rel


def _judged_retrieval(E: torch.Tensor, rel, apps: np.ndarray, within_app: bool, k: int = 10):
    """Pooled relevance (unjudged = 0). R@k / MRR over queries with a 'same workflow' (gain 2) partner;
    nDCG@k with gains 2^g - 1 over queries with any positive gain."""
    E = F.normalize(E.float(), dim=-1)
    S = E @ E.T
    S.fill_diagonal_(-float("inf"))
    r1, r5, mrr, ndcg = [], [], [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for q, cands in rel.items():
        if not any(g > 0 for g in cands.values()):
            continue
        s = S[q].clone()
        if within_app:
            s[torch.from_numpy(apps != apps[q])] = -float("inf")
            cands = {j: g for j, g in cands.items() if apps[j] == apps[q]}
            if not any(g > 0 for g in cands.values()):
                continue
        order = s.argsort(descending=True).tolist()
        gains = np.array([cands.get(j, 0) for j in order[:k]], float)
        ideal = np.sort(np.array(list(cands.values()), float))[::-1][:k]
        idcg = ((2 ** ideal - 1) * disc[:len(ideal)]).sum()
        ndcg.append(((2 ** gains - 1) * disc[:len(gains)]).sum() / idcg)
        if any(g == 2 for g in cands.values()):
            rank = next(i for i, j in enumerate(order) if cands.get(j, 0) == 2) + 1
            r1.append(rank <= 1); r5.append(rank <= 5); mrr.append(1.0 / rank)
    tag = "within_app" if within_app else "cross_app"
    out = {f"jsession_nDCG@{k}_{tag}": float(np.mean(ndcg)) if ndcg else float("nan"),
           f"n_jsession_queries_{tag}": len(r1)}
    if r1:
        out.update({f"jsession_R@1_{tag}": float(np.mean(r1)), f"jsession_R@5_{tag}": float(np.mean(r5)),
                    f"jsession_MRR_{tag}": float(np.mean(mrr))})
    return out


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def _probe(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if len(set(y.tolist())) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    return float(cross_val_score(clf, X, y, cv=5).mean())


@torch.no_grad()
def evaluate(model, data: SessionData, split: str, device, theta_fn: float, U_train: torch.Tensor,
             train_idx: np.ndarray, full: bool = False, max_sessions: int | None = None):
    sidx = data.indices(split)
    if max_sessions and len(sidx) > max_sessions:
        sidx = np.sort(np.random.default_rng(0).choice(sidx, max_sessions, replace=False))
    d_all, k_all, lens = embed_split(model, data, sidx, device)
    U_split = torch.from_numpy(data.keys[sidx]).to(device)
    U_ref = torch.cat([U_train, U_split])
    g_ref = torch.from_numpy(np.concatenate([data.group[train_idx], data.group[sidx]])).to(device)
    n_tr = len(train_idx)
    S = len(sidx)

    tot_loss, tot_last, ranks_last, conf_last, correct_last, p_true_last = 0.0, 0.0, [], [], [], []
    T_bins = 10
    ent_bins = np.zeros(T_bins); rank_bins = np.zeros(T_bins); r5_bins = np.zeros(T_bins); n_bins = np.zeros(T_bins)
    ent_abs = collections.defaultdict(list)
    for s0 in range(0, S, 64):
        sl = slice(s0, min(S, s0 + 64))
        L = lens[sl].to(device)
        T = int(L.max())
        d = d_all[sl, :T].to(device).float()
        k = k_all[sl, :T, None].to(device)
        valid = torch.arange(T, device=device)[None, :] < L[:, None]
        target = torch.arange(sl.start, sl.stop, device=device) + n_tr
        own = U_ref[target]
        fn = false_negative_mask(U_ref, g_ref, own, g_ref[target], target, theta_fn)
        loss, ce, logits = info_nce(d, k, valid.float(), U_ref, own, target, fn, return_steps=True)
        B = d.shape[0]
        tot_loss += loss.item() * B
        last = L - 1
        tot_last += ce[torch.arange(B), last].sum().item()
        logp = logits.log_softmax(-1)                                    # [B, T, N]
        p = logp.exp()
        ent = -(p * logp.nan_to_num(0.0, neginf=0.0)).sum(-1)            # [B, T]
        true_lp = logp.gather(2, target.view(B, 1, 1).expand(B, T, 1)).squeeze(-1)
        rank = (logp > true_lp[..., None]).sum(-1) + 1                   # [B, T]
        relb = (torch.arange(T, device=device)[None, :].float() / (L[:, None] - 1).clamp(min=1).float()
                * (T_bins - 1e-6)).long().clamp(max=T_bins - 1)                # [B, T]
        vm = valid
        rb = relb[vm].cpu().numpy()
        np.add.at(ent_bins, rb, ent[vm].cpu().numpy())
        np.add.at(rank_bins, rb, rank[vm].float().cpu().numpy())
        np.add.at(r5_bins, rb, (rank[vm] <= 5).float().cpu().numpy())
        np.add.at(n_bins, rb, 1)
        for t in range(min(T, 12)):
            m = valid[:, t]
            ent_abs[t].extend(ent[m, t].cpu().tolist())
        rl = rank[torch.arange(B), last]
        ranks_last.append(rl.cpu())
        pl = p[torch.arange(B), last]                                    # [B, N]
        conf_last.append(pl.max(-1).values.cpu())
        correct_last.append((pl.argmax(-1) == target).cpu())
        p_true_last.append(pl.gather(1, target[:, None]).squeeze(1).cpu())

    ranks_last = torch.cat(ranks_last).float()
    out = {
        "loss": tot_loss / S, "loss_last": tot_last / S,
        "key_R@1": float((ranks_last <= 1).float().mean()), "key_R@5": float((ranks_last <= 5).float().mean()),
        "key_R@10": float((ranks_last <= 10).float().mean()), "key_MRR": float((1.0 / ranks_last).mean()),
        "key_median_rank": float(ranks_last.median()),
        "kappa_first": float(k_all[:, 0].mean()), "kappa_last": float(k_all[torch.arange(S), lens - 1].mean()),
        "entropy_vs_relpos": (ent_bins / np.maximum(n_bins, 1)).round(3).tolist(),
        "true_key_R@5_vs_relpos": (r5_bins / np.maximum(n_bins, 1)).round(3).tolist(),
        "entropy_vs_t": [round(float(np.mean(ent_abs[t])), 3) for t in sorted(ent_abs)],
        "calibration_ECE": _ece(torch.cat(conf_last).numpy(), torch.cat(correct_last).numpy().astype(float)),
        "n_sessions": S, "n_candidates": int(U_ref.shape[0]),
    }
    E = d_all[torch.arange(S), lens - 1]                                 # terminal direction
    groups = data.group[sidx]
    apps = np.array([data.sessions[i]["app"] for i in sidx])
    out.update(_retrieval(E, groups, apps, within_app=False))
    out.update(_retrieval(E, groups, apps, within_app=True))
    rel = load_judged(split, sidx, groups)
    if rel is not None:
        out.update(_judged_retrieval(E, rel, apps, within_app=False))
        out.update(_judged_retrieval(E, rel, apps, within_app=True))
    if full:
        os_lab = np.array([data.sessions[i]["os"] for i in sidx])
        mean_frame = np.stack([mean_frame_feature(data, i) for i in sidx])
        out["nuisance_probe_os_E"] = _probe(E.float().numpy(), os_lab)
        out["nuisance_probe_os_meanframe"] = _probe(mean_frame, os_lab)
        out["nuisance_probe"] = out["nuisance_probe_os_E"] - out["nuisance_probe_os_meanframe"]
    return out


def mean_frame_feature(data: SessionData, sidx: int, last: bool = False) -> np.ndarray:
    o, L = int(data.offset[sidx]), int(data.length[sidx])
    g = np.arange(o, o + L)[-1:] if last else np.arange(o, o + L)
    x = data.frames.gather(g)                                            # [L, n_tok, D] bf16 bits
    x = torch.from_numpy(x.view(np.int16)).view(torch.bfloat16).float()
    return x.mean((0, 1)).numpy()
