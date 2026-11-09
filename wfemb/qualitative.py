"""Worked examples for the report: for selected test sessions, the model's top instructions at every step (with
probabilities, entropy, kappa, rank of the true instruction), the action taken, and downscaled screenshots.

  python -m wfemb.qualitative RUN --out DIR [--n 12] [--ckpt best.pt]

Writes DIR/examples.json and DIR/<session id>/<t>.jpg. Scoring matches evaluate(): candidates are the training
instructions plus the test instructions, with the same false-negative mask.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import paths
from .data import SessionData
from .evaluate import embed_split, load_judged
from .loss import false_negative_mask, info_nce
from .model import ModelConfig, build_model
from .splitzip import SplitZip


def raw_codes(ids: set[str]) -> dict[str, list[str]]:
    """task id -> the pyautogui code of each step, in step order (read from the raw AgentNet jsonl)."""
    out = {}
    for jp in paths.JSONLS.values():
        with open(jp) as f:
            for line in f:
                if not any(i in line for i in ids):
                    continue
                d = json.loads(line)
                if d["task_id"] in ids:
                    out[d["task_id"]] = [(s["value"].get("code") or "")
                                         for s in sorted(d["traj"], key=lambda s: s["index"])]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    state = torch.load(paths.RUNS / args.run / args.ckpt, map_location=device, weights_only=False)
    a = state["args"]
    paths.KEYS = Path(a["keys_dir"]) if a.get("keys_dir") else paths.SHARED
    data = SessionData(Path(a.get("frames_dir") or paths.FRAMES), use_crop=bool(a.get("use_crop")),
                       frames_mode="gpu_fp8", device=device, avg_grid=a.get("avg_grid", 0))
    model = build_model(ModelConfig(**state["cfg"]), torch.from_numpy(data.text_vecs)).to(device)
    model.load_state_dict(state["model"])
    theta = a.get("eval_theta_fn", 0.9)

    train_idx, sidx = data.indices("train"), data.indices("test")
    U_train = torch.from_numpy(data.keys[train_idx]).to(device)
    U_ref = torch.cat([U_train, torch.from_numpy(data.keys[sidx]).to(device)])
    g_ref = torch.from_numpy(np.concatenate([data.group[train_idx], data.group[sidx]])).to(device)
    texts = [data.sessions[i]["instruction"] for i in train_idx] + [data.sessions[i]["instruction"] for i in sidx]
    n_tr = len(train_idx)

    d_all, k_all, lp_all, lens = embed_split(model, data, sidx, device)
    assert lp_all is None, "mixture readouts not supported here"

    # per-step entropy / true rank for every test session, to choose illustrative ones
    S = len(sidx)
    stats = []
    for s0 in range(0, S, 64):
        sl = slice(s0, min(S, s0 + 64))
        L = lens[sl].to(device)
        T = int(L.max())
        d = d_all[sl, :T].to(device).float()
        k = k_all[sl, :T, None].to(device)
        valid = torch.arange(T, device=device)[None, :] < L[:, None]
        target = torch.arange(sl.start, sl.stop, device=device) + n_tr
        fn = false_negative_mask(U_ref, g_ref, U_ref[target], g_ref[target], target, theta)
        _, _, logits = info_nce(d, k, valid.float(), U_ref, U_ref[target], target, fn, return_steps=True)
        logp = logits.log_softmax(-1)
        ent = -(logp.exp() * logp.nan_to_num(0.0, neginf=0.0)).sum(-1)
        true_lp = logp.gather(2, target.view(-1, 1, 1).expand(-1, T, 1)).squeeze(-1)
        rank = (logp > true_lp[..., None]).sum(-1) + 1
        for j in range(d.shape[0]):
            n = int(L[j])
            stats.append({"pos": s0 + j, "len": n, "ent": ent[j, :n].tolist(), "rank": rank[j, :n].tolist()})

    def score(st):
        e, r = st["ent"], st["rank"]
        return (e[0] - e[-1]) + (2.0 if r[-1] == 1 else 0.0) + (1.0 if r[0] > 20 else 0.0)

    cands = [st for st in stats if args.min_len <= st["len"] <= args.max_len and st["rank"][-1] <= 3
             and len(data.sessions[sidx[st["pos"]]]["instruction"]) <= 140]
    cands.sort(key=score, reverse=True)
    # one per app, for variety
    chosen, apps = [], set()
    for st in cands:
        app = data.sessions[sidx[st["pos"]]]["app"]
        if app in apps:
            continue
        apps.add(app)
        chosen.append(st)
        if len(chosen) >= args.n:
            break
    # aggregate distributions over all test sessions (for context in the report)
    summary = {"n_test": S,
               "frac_rank1_last": float(np.mean([st["rank"][-1] == 1 for st in stats])),
               "frac_rank1_first": float(np.mean([st["rank"][0] == 1 for st in stats])),
               "median_rank_first": float(np.median([st["rank"][0] for st in stats])),
               "median_rank_last": float(np.median([st["rank"][-1] for st in stats]))}

    ids = {data.sessions[sidx[st["pos"]]]["id"] for st in chosen}
    codes = raw_codes(ids)
    zips = {src: SplitZip(zp) for src, zp in paths.ZIPS.items()}
    index = {src: z.basename_index() for src, z in zips.items()}
    focus = data.ev["focus"]
    groups = data.group[sidx]
    rel = load_judged("test", sidx, groups) or {}
    E = torch.nn.functional.normalize(d_all[torch.arange(S), lens - 1].float(), dim=-1)

    examples = []
    for st in chosen:
        p = st["pos"]
        gi = int(sidx[p])
        s = data.sessions[gi]
        n = st["len"]
        d = d_all[p:p + 1, :n].to(device).float()
        k = k_all[p:p + 1, :n, None].to(device)
        target = torch.tensor([p + n_tr], device=device)
        fn = false_negative_mask(U_ref, g_ref, U_ref[target], g_ref[target], target, theta)
        _, _, logits = info_nce(d, k, torch.ones(1, n, device=device), U_ref, U_ref[target], target, fn,
                                return_steps=True)
        prob = logits.softmax(-1)[0]                                       # [n, N]
        sdir = out / s["id"]
        sdir.mkdir(exist_ok=True)
        code = codes.get(s["id"], [])
        steps = []
        for t in range(n):
            pv, ix = prob[t].topk(args.topk)
            img_rel = None
            member = index[s["source"]].get(s["images"][t])
            if member is not None:
                im = Image.open(io.BytesIO(zips[s["source"]].read(member))).convert("RGB")
                im.thumbnail((720, 720))
                im.save(sdir / f"{t}.jpg", quality=82)
                img_rel = f"{s['id']}/{t}.jpg"
            g = int(data.offset[gi]) + t
            steps.append({
                "t": t, "image": img_rel, "code": code[t] if t < len(code) else "",
                "focus": [float(focus[g, 0]), float(focus[g, 1])],
                "kappa": float(k_all[p, t]), "entropy": st["ent"][t], "true_rank": st["rank"][t],
                "top": [{"text": texts[int(j)], "p": float(v), "is_true": int(j) == p + n_tr,
                         "split": "train" if int(j) < n_tr else "test"} for v, j in zip(pv.tolist(), ix.tolist())],
            })
        # nearest test sessions by terminal direction (the retrieval task), with judged labels where available
        sims = E @ E[p]
        sims[p] = -2
        nb = sims.topk(5)
        neighbours = [{"instruction": data.sessions[int(sidx[int(j)])]["instruction"],
                       "app": data.sessions[int(sidx[int(j)])]["app"], "cos": float(v),
                       "judged": rel.get(p, {}).get(int(j)) if isinstance(rel, dict) else None,
                       "same_group": bool(groups[int(j)] == groups[p])}
                      for v, j in zip(nb.values.tolist(), nb.indices.tolist())]
        examples.append({"id": s["id"], "app": s["app"], "os": s["os"], "instruction": s["instruction"],
                         "n_steps": n, "steps": steps, "neighbours": neighbours})
        print(f"{s['app']:30s} len={n} ent {st['ent'][0]:.2f}->{st['ent'][-1]:.2f} "
              f"rank {st['rank'][0]}->{st['rank'][-1]}  {s['instruction'][:90]}", flush=True)
    json.dump({"run": args.run, "ckpt": args.ckpt, "n_candidates": int(U_ref.shape[0]), "summary": summary,
               "examples": examples}, open(out / "examples.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
