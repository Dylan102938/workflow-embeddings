"""Baselines (pseudocode section 6): action n-gram TF-IDF, untrained mean/last-frame features,
the text-key oracle, plus a non-recurrent ridge map from bag features into key space (scored with the
same instruction-retrieval metrics as the model)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import paths
from .actions import EVENT_TYPES
from .data import SessionData
from .evaluate import _retrieval
from .loss import false_negative_mask


def action_docs(data: SessionData) -> list[str]:
    ev = data.ev
    docs = []
    for si in range(len(data.sessions)):
        o, L = int(data.offset[si]), int(data.length[si])
        toks = []
        for g in range(o, o + L):
            for j in range(ev["ev_type"].shape[1]):
                t = int(ev["ev_type"][g, j])
                if t == 0:
                    continue
                tok = EVENT_TYPES[t]
                xy = ev["ev_xy"][g, j]
                if not np.isnan(xy[0]):
                    tok += f"@{int(min(xy[0], 0.999) * 4)}{int(min(xy[1], 0.999) * 4)}"
                ks = [k for k in ev["ev_keys"][g, j] if k > 0]
                if ks:
                    tok += "+" + "_".join(map(str, ks))
                toks.append(tok)
        docs.append(" ".join(toks) if toks else "empty")
    return docs


def frame_feats(data: SessionData, chunk: int = 256):
    """mean-over-session and last-frame features from the global tokens (mean over tokens)."""
    S = len(data.sessions)
    D = data.frames.dim
    mean_f, last_f = np.zeros((S, D), np.float32), np.zeros((S, D), np.float32)
    for s0 in range(0, S, chunk):
        ss = np.arange(s0, min(S, s0 + chunk))
        g = np.concatenate([np.arange(data.offset[s], data.offset[s] + data.length[s]) for s in ss])
        x = data.frames.gather(g)[:, :64]
        x = torch.from_numpy(x.view(np.int16)).view(torch.bfloat16).float().mean(1).numpy()
        pos = 0
        for i, s in enumerate(ss):
            L = int(data.length[s])
            mean_f[s] = x[pos:pos + L].mean(0)
            last_f[s] = x[pos + L - 1]
            pos += L
    return mean_f, last_f


def key_metrics(pred: np.ndarray, data: SessionData, split: str, theta_fn: float, device="cuda"):
    tr, sp = data.indices("train"), data.indices(split)
    U = torch.from_numpy(np.concatenate([data.keys[tr], data.keys[sp]])).to(device)
    g = torch.from_numpy(np.concatenate([data.group[tr], data.group[sp]])).to(device)
    E = F.normalize(torch.from_numpy(pred[sp]).float().to(device), dim=-1)
    target = torch.arange(len(sp), device=device) + len(tr)
    fn = false_negative_mask(U, g, U[target], g[target], target, theta_fn)
    s = (E @ U.T).masked_fill(fn, -float("inf"))
    rank = (s > s.gather(1, target[:, None])).sum(1) + 1
    r = rank.float()
    return {"key_R@1": float((r <= 1).float().mean()), "key_R@5": float((r <= 5).float().mean()),
            "key_R@10": float((r <= 10).float().mean()), "key_MRR": float((1 / r).mean()),
            "key_median_rank": float(r.median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default=str(paths.FRAMES))
    ap.add_argument("--theta_fn", type=float, default=0.9)
    ap.add_argument("--out", default=str(paths.RUNS / "baselines.json"))
    args = ap.parse_args()
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    data = SessionData(Path(args.frames_dir))
    docs = action_docs(data)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"\S+", min_df=2, sublinear_tf=True)
    tr = data.indices("train")
    tfidf.fit([docs[i] for i in tr])
    A = tfidf.transform(docs)
    svd = TruncatedSVD(256, random_state=0).fit(A[tr])
    A_svd = svd.transform(A).astype(np.float32)
    mean_f, last_f = frame_feats(data)

    feats = {"action_tfidf": A_svd, "mean_frame": mean_f, "last_frame": last_f,
             "text_key_oracle": data.keys}
    res = {}
    for split in ("val", "test"):
        sp = data.indices(split)
        groups = data.group[sp]
        apps = np.array([data.sessions[i]["app"] for i in sp])
        for name, X in feats.items():
            E = torch.from_numpy(X[sp])
            r = {}
            r.update(_retrieval(E, groups, apps, within_app=False))
            r.update(_retrieval(E, groups, apps, within_app=True))
            res[f"{split}/{name}"] = r
        # non-recurrent map into key space: ridge on [mean frame, last frame, action tf-idf svd]
        Xb = np.concatenate([mean_f, last_f, A_svd], 1)
        mu, sd = Xb[tr].mean(0), Xb[tr].std(0) + 1e-6
        Xb = (Xb - mu) / sd
        for alpha in (10.0, 100.0, 1000.0):
            reg = Ridge(alpha=alpha).fit(Xb[tr], data.keys[tr])
            pred = reg.predict(Xb).astype(np.float32)
            res[f"{split}/ridge_bag_alpha{alpha:g}"] = key_metrics(pred, data, split, args.theta_fn)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(k, {a: round(b, 3) for a, b in v.items()})


if __name__ == "__main__":
    main()
