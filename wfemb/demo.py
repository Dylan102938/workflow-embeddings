"""Scrubbable workflow demo: the model's intent beliefs at every step, and the 2D trajectory of its embedding among all
test workflows. Writes a bundle that demo/index.html plays (screenshots, actions, top intents, PCA map).

  # every test step's direction d_t and confidence kappa_t (cached; needs the test frame features, see `extract`)
  python -m wfemb.demo embed      --run r20_xf4_d512
  # 2D PCA of all test workflows' final embeddings + clusters -> BUNDLE/background.json
  python -m wfemb.demo background --run r20_xf4_d512 --out BUNDLE
  # workflows: illustrative test sessions, specific AgentNet test sessions, or ANY workflow given as raw steps
  python -m wfemb.demo export --run r20_xf4_d512 --out BUNDLE --auto 8
  python -m wfemb.demo export --run r20_xf4_d512 --out BUNDLE --session <task_id> [--session ...]
  python -m wfemb.demo export --run r20_xf4_d512 --out BUNDLE --raw my_workflow.json [--single_file]
  # (on rnn, where the AgentNet archives are) screenshots of the exported AgentNet sessions
  python -m wfemb.demo frames --out BUNDLE
  # (on rnn) the test sessions' frame features as one file, for running the demo on another machine
  python -m wfemb.demo extract

Raw workflow format (JSON). Each step is the screenshot the user saw and the pyautogui-style action code they ran on
it (AgentNet convention: coordinates as fractions of the screen; pixel coordinates are detected and normalized):
  {"instruction": "optional ground truth", "app": "optional", "os": "optional",
   "steps": [{"image": "shot_000.png", "code": "pyautogui.click(x=0.41, y=0.22)"}, ...]}
An AgentNet jsonl record ({"task_id", "instruction", "traj": [{"index", "image", "value": {"code"}}]}) is accepted as
well (--images DIR holds its screenshots). Screenshots are encoded with the frozen SigLIP2 encoder and channel PCA and
typed text with Qwen3-Embedding-0.6B, exactly as for training (cache_frames.py, text_embed.py, prepare.py).

Intent descriptors are the instructions of the training and test sessions (the evaluation's candidate set, unique
texts); their probabilities are softmax(kappa_t * cos(d_t, key)) as in the loss.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import math
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import actions, paths
from .actions import EVENT_TYPES, TYPE_ID, parse_code
from .data import Batch, np_load, read_chunked
from .model import ModelConfig, build_model

E_MAX, K_MAX = 8, 4
SIGLIP = "google/siglip2-so400m-patch16-naflex"
TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEMO = paths.DATA / "demo"


# ----------------------------------------------------------------------------------------------------------- assets
class Assets:
    """Checkpoint, sessions, event arrays, vocabularies and the intent bank."""

    def __init__(self, run: str, ckpt: str = "best.pt", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.run = run
        self.state = torch.load(paths.RUNS / run / ckpt, map_location="cpu", weights_only=False)
        a = self.state["args"]
        kd = Path(a["keys_dir"]) if a.get("keys_dir") else paths.SHARED
        if not kd.exists():
            kd = paths.DATA / kd.name
        self.keys_dir = kd
        self.frames_dir = Path(a.get("frames_dir") or paths.FRAMES)
        self.sessions = pickle.load(read_chunked(paths.PREP / "sessions.pkl"))
        ev = np_load(paths.PREP / "events.npz")
        self.ev = {k: ev[k] for k in ev.files}
        self.vocab = json.load(open(paths.PREP / "vocab.json"))
        self.key_id = {k: i for i, k in enumerate(self.vocab["key_vocab"])}
        self.keys = np_load(kd / "keys.npy").astype(np.float32)
        self.text_vecs = np_load(kd / "text_vecs.npy")
        self.split = np.array([s["split"] for s in self.sessions])
        self.test_idx = np.nonzero(self.split == "test")[0]
        self.id2idx = {s["id"]: i for i, s in enumerate(self.sessions)}
        self.cfg = ModelConfig(**self.state["cfg"])
        self.model = self.build(torch.from_numpy(self.text_vecs))
        # intent bank: unique instruction texts of the training + test sessions (the evaluation's candidates)
        first = {}
        for i in np.nonzero(np.isin(self.split, ["train", "test"]))[0]:
            first.setdefault(self.sessions[i]["instruction"], int(i))
        self.bank_text = list(first)
        self.bank_sess = np.array(list(first.values()))
        self.bank_pos = {t: j for j, t in enumerate(self.bank_text)}
        self.U = F.normalize(torch.from_numpy(self.keys[self.bank_sess]), dim=-1).to(self.device)

    def build(self, text_table: torch.Tensor):
        m = build_model(self.cfg, text_table).to(self.device)
        m.load_state_dict(self.state["model"])
        return m.eval()

    def session_events(self, gi: int) -> dict:
        o, n = int(self.ev["sess_offset"][gi]), int(self.ev["sess_len"][gi])
        return {k: self.ev[k][o:o + n] for k in ("ev_type", "ev_xy", "ev_scroll", "ev_keys", "ev_text", "focus")}


@torch.no_grad()
def run_model(model, device, frames: list[np.ndarray], evs: list[dict]):
    """frames: per session [L, n_tok, D] bf16 bits (uint16); evs: per session event arrays (see session_events).
    Returns per session (d [L, D_key] float32, kappa [L])."""
    B, T = len(frames), max(len(f) for f in frames)

    def pad(x, fill=0):                                   # fill=None repeats the last step (padding is masked)
        out = np.empty((T,) + x.shape[1:], dtype=x.dtype)
        out[:len(x)] = x
        out[len(x):] = x[-1] if fill is None else fill
        return out

    fr = np.stack([pad(f, None) for f in frames]).view(np.int16)
    b = Batch(
        idx=torch.zeros(B, dtype=torch.long),
        frames=torch.from_numpy(fr).to(device).view(torch.bfloat16),
        ev_type=torch.from_numpy(np.stack([pad(e["ev_type"]) for e in evs]).astype(np.int64)).to(device),
        ev_xy=torch.from_numpy(np.stack([pad(e["ev_xy"].astype(np.float32), np.nan) for e in evs])).to(device),
        ev_scroll=torch.from_numpy(np.stack([pad(e["ev_scroll"].astype(np.float32)) for e in evs])).to(device),
        ev_keys=torch.from_numpy(np.stack([pad(e["ev_keys"]) for e in evs]).astype(np.int64)).to(device),
        ev_text=torch.from_numpy(np.stack([pad(e["ev_text"], -1) for e in evs]).astype(np.int64)).to(device),
        valid=torch.arange(T, device=device)[None, :] < torch.tensor([len(f) for f in frames], device=device)[:, None],
        lengths=torch.tensor([len(f) for f in frames], device=device),
        gidx=torch.zeros(B, T, dtype=torch.long, device=device),
    )
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    with ctx:
        d, k, _ = model(b)
    d, k = d.float().cpu().numpy(), k.squeeze(-1).float().cpu().numpy()
    return [(d[i, :len(f)], k[i, :len(f)]) for i, f in enumerate(frames)]


class TestFrames:
    """Frame features of the test sessions: DATA/demo/test_frames.npy (see `extract`), else the full frame cache."""

    def __init__(self, A: Assets):
        sub = DEMO / "test_frames.npy"
        if sub.exists():
            self.mm = np.load(sub, mmap_mode="r")
            ix = np.load(DEMO / "test_index.npz")
            self.pos = {int(s): (int(o), int(n)) for s, o, n in zip(ix["sidx"], ix["offset"], ix["length"])}
            self.store = None
        else:
            from .data import FrameStore
            self.store = FrameStore(A.frames_dir, use_crop=False, pread=True)
            self.A = A

    def get(self, gi: int) -> np.ndarray:
        if self.store is None:
            o, n = self.pos[gi]
            return np.ascontiguousarray(self.mm[o:o + n])
        o, n = int(self.A.ev["sess_offset"][gi]), int(self.A.ev["sess_len"][gi])
        return self.store.gather(np.arange(o, o + n))


def emb_path(run):
    return DEMO / f"{run}_test_embeddings.npz"


def load_embeddings(A: Assets):
    z = np.load(emb_path(A.run))
    out = {}
    for s, o, n in zip(z["sidx"], z["offset"], z["length"]):
        out[int(s)] = (z["d"][o:o + n].astype(np.float32), z["k"][o:o + n])
    return out


def scores(A: Assets, d: np.ndarray, k: np.ndarray, topk: int = 5, true_text: str | None = None):
    """Per step: top-k intents (bank index, prob), entropy, and the true instruction's rank / prob if it is in the bank."""
    D = F.normalize(torch.from_numpy(d).to(A.device), dim=-1)
    logits = torch.from_numpy(k).to(A.device)[:, None] * (D @ A.U.T)          # [L, N]
    logp = logits.log_softmax(-1)
    p = logp.exp()
    ent = -(p * logp).sum(-1)
    tv, ti = p.topk(topk, dim=-1)
    out = {"top_p": tv.cpu().numpy(), "top_i": ti.cpu().numpy(), "entropy": ent.cpu().numpy(),
           "true_rank": None, "true_p": None}
    j = A.bank_pos.get(true_text) if true_text else None
    if j is not None:
        tl = logp[:, j]
        out["true_rank"] = ((logp > tl[:, None]).sum(-1) + 1).cpu().numpy()
        out["true_p"] = tl.exp().cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------------------------- embed
def cmd_embed(args):
    A = Assets(args.run, args.ckpt)
    TF_ = TestFrames(A)
    sidx = A.test_idx
    lens = A.ev["sess_len"][sidx].astype(np.int64)
    order = np.argsort(lens)
    res = {}
    t0 = time.time()
    batch, n_fr = [], 0
    for i in list(order) + [None]:
        if i is not None and (n_fr + lens[i] <= args.max_frames or not batch):
            batch.append(int(sidx[i]))
            n_fr += int(lens[i])
            continue
        outs = run_model(A.model, A.device, [TF_.get(g) for g in batch], [A.session_events(g) for g in batch])
        res.update(zip(batch, outs))
        batch, n_fr = ([int(sidx[i])], int(lens[i])) if i is not None else ([], 0)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    d = np.concatenate([res[int(g)][0] for g in sidx]).astype(np.float16)
    k = np.concatenate([res[int(g)][1] for g in sidx]).astype(np.float32)
    DEMO.mkdir(parents=True, exist_ok=True)
    np.savez(emb_path(args.run), sidx=sidx, offset=off, length=lens, d=d, k=k)
    print(f"{len(sidx)} test sessions, {len(d)} steps embedded in {time.time() - t0:.0f}s -> {emb_path(args.run)}")


# -------------------------------------------------------------------------------------------------------- background
def pca_path(run):
    return DEMO / f"{run}_pca.npz"


def cmd_background(args):
    from sklearn.cluster import KMeans
    A = Assets(args.run, args.ckpt)
    emb = load_embeddings(A)
    sidx = A.test_idx
    E = np.stack([emb[int(g)][0][-1] for g in sidx])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    mu = E.mean(0)
    _, S, Vt = np.linalg.svd(E - mu, full_matrices=False)
    W = Vt[:2].T.copy()
    W *= np.sign(W[np.abs(W).argmax(0), [0, 1]])                       # deterministic orientation
    explained = (S[:2] ** 2 / (S ** 2).sum()).tolist()
    xy = (E - mu) @ W
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(E)
    lab = km.labels_
    sizes = np.bincount(lab, minlength=args.k)
    rank_of = {c: r for r, c in enumerate(np.argsort(-sizes))}           # cluster 0 = largest
    lab = np.array([rank_of[c] for c in lab])
    cent = np.stack([E[lab == c].mean(0) for c in range(args.k)])
    clusters = []
    for c in range(args.k):
        m = np.nonzero(lab == c)[0]
        apps = {}
        for i in m:
            a = A.sessions[sidx[i]]["app"].split(":", 1)[-1]
            apps[a] = apps.get(a, 0) + 1
        top = sorted(apps.items(), key=lambda kv: -kv[1])[:3]
        rep = m[np.argmax(E[m] @ cent[c])]
        clusters.append({"id": c, "n": int(len(m)), "apps": [[a, n] for a, n in top],
                         "label": " / ".join(a for a, _ in top[:2]),
                         "example": A.sessions[sidx[rep]]["instruction"][:160],
                         "xy": ((cent[c] / np.linalg.norm(cent[c]) - mu) @ W).round(4).tolist()})
    # first / last step stats of every test session (to choose illustrative workflows)
    first_rank, last_rank, ent0, entT = [], [], [], []
    for g in sidx:
        d, k = emb[int(g)]
        sc = scores(A, d[[0, -1]], k[[0, -1]], true_text=A.sessions[g]["instruction"])
        first_rank.append(int(sc["true_rank"][0]))
        last_rank.append(int(sc["true_rank"][1]))
        ent0.append(float(sc["entropy"][0]))
        entT.append(float(sc["entropy"][1]))
    np.savez(pca_path(args.run), mu=mu, W=W, explained=np.array(explained), E=E.astype(np.float16), sidx=sidx,
             cluster=lab, first_rank=first_rank, last_rank=last_rank, ent0=ent0, entT=entT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bg = {"run": args.run, "n": int(len(sidx)), "explained": [round(v, 4) for v in explained],
          "clusters": clusters,
          "points": [[round(float(x), 4), round(float(y), 4), int(c)] for (x, y), c in zip(xy, lab)],
          "items": [{"id": A.sessions[g]["id"], "app": A.sessions[g]["app"].split(":", 1)[-1],
                     "os": A.sessions[g]["os"], "instruction": A.sessions[g]["instruction"][:200]} for g in sidx]}
    json.dump(bg, open(out / "background.json", "w"), separators=(",", ":"))
    print(f"background: {len(sidx)} test workflows, PCA explains {explained[0]:.3f} + {explained[1]:.3f} of variance, "
          f"{args.k} clusters; final true-intent rank 1: {np.mean(np.array(last_rank) == 1):.3f}, "
          f"<=5: {np.mean(np.array(last_rank) <= 5):.3f}")


# ------------------------------------------------------------------------------------------------------------ export
def describe(etype: str, keys, text, scroll) -> str:
    if etype in ("click", "doubleClick", "tripleClick", "rightClick", "middleClick"):
        return {"click": "Click", "doubleClick": "Double-click", "tripleClick": "Triple-click",
                "rightClick": "Right-click", "middleClick": "Middle-click"}[etype]
    if etype == "write":
        t = (text or "").replace("\n", "⏎")
        return f"Type “{t[:80]}{'…' if len(t) > 80 else ''}”"
    if etype in ("press", "keyDown", "keyUp"):
        return {"press": "Press", "keyDown": "Hold", "keyUp": "Release"}[etype] + " " + " ".join(
            k.capitalize() if len(k) > 1 else k.upper() for k in keys)
    if etype == "hotkey":
        return "Shortcut " + "+".join(k.capitalize() if len(k) > 1 else k.upper() for k in keys)
    if etype in ("scroll", "hscroll"):
        s = float(scroll or 0)
        arrow = ("↑" if s > 0 else "↓") if etype == "scroll" else ("→" if s > 0 else "←")
        return f"Scroll {arrow} {abs(s):g}"
    return {"moveTo": "Move pointer", "dragTo": "Drag", "mouseDown": "Mouse down", "mouseUp": "Mouse up",
            "wait": "Wait", "other": "Other action"}.get(etype, etype)


def step_actions(A: Assets, ev: dict, t: int, texts: list[str]) -> list[dict]:
    out = []
    for j in range(ev["ev_type"].shape[1]):
        ty = int(ev["ev_type"][t, j])
        if ty == 0:
            continue
        name = EVENT_TYPES[ty]
        xy = [None if not np.isfinite(v) else round(float(v), 4) for v in ev["ev_xy"][t, j]]
        keys = [A.vocab["key_vocab"][int(i)] for i in ev["ev_keys"][t, j] if int(i) > 0]
        ti = int(ev["ev_text"][t, j])
        text = texts[ti] if ti >= 0 else None
        sc = float(ev["ev_scroll"][t, j])
        out.append({"type": name, "x": xy[0], "y": xy[1], "x2": xy[2], "y2": xy[3], "keys": keys, "text": text,
                    "scroll": sc, "label": describe(name, keys, text, sc)})
    return out


def build_workflow(A, pca, meta, d, k, ev, texts, images, topk=5, bg_index=None, partners=(), related=()):
    """Assemble one workflow record for the viewer. images: per step a relative path or data URL (or None)."""
    sc = scores(A, d, k, topk, meta.get("instruction"))
    D = d / np.linalg.norm(d, axis=1, keepdims=True)
    xy = (D - pca["mu"]) @ pca["W"]
    E = pca["E"].astype(np.float32)
    sims = E @ D[-1]
    if bg_index is not None:
        sims[bg_index] = -2
    nb = np.argsort(-sims)[:8]
    steps = []
    for t in range(len(d)):
        acts = step_actions(A, ev, t, texts)
        steps.append({
            "t": t, "image": images[t] if images else None, "actions": acts,
            "focus": [round(float(v), 4) for v in ev["focus"][t]] if "focus" in ev else None,
            "kappa": round(float(k[t]), 3), "entropy": round(float(sc["entropy"][t]), 3),
            "true_rank": None if sc["true_rank"] is None else int(sc["true_rank"][t]),
            "true_p": None if sc["true_p"] is None else round(float(sc["true_p"][t]), 5),
            "top": [[A.bank_text[int(i)], round(float(p), 5),
                     bool(meta.get("instruction") is not None and A.bank_text[int(i)] == meta.get("instruction"))]
                    for p, i in zip(sc["top_p"][t], sc["top_i"][t])],
            "xy": [round(float(v), 4) for v in xy[t]],
        })
    return {"version": 1, "run": A.run, **meta, "n_steps": len(d), "bank_size": len(A.bank_text),
            "true_in_bank": sc["true_rank"] is not None, "bg_index": bg_index,
            "partners": [int(i) for i in partners], "related": [int(i) for i in related],
            "neighbors": [{"bg": int(i), "cos": round(float(sims[i]), 4)} for i in nb], "steps": steps}


def load_pca(run):
    z = np.load(pca_path(run))
    return {k: z[k] for k in z.files}


def judged_partners(A: Assets, pca) -> dict:
    """bg position -> (same-workflow positions, related positions) from the trace-judged test labels."""
    pos = {int(g): i for i, g in enumerate(pca["sidx"])}
    same, rel = {}, {}
    p = paths.SHARED / "judged_test.jsonl"
    for line in open(p):
        r = json.loads(line)
        a, b, lab = pos.get(r["a"]), pos.get(r["b"]), int(r["label"])
        if a is None or b is None or lab == 0:
            continue
        tgt = same if lab == 2 else rel
        tgt.setdefault(a, set()).add(b)
        tgt.setdefault(b, set()).add(a)
    groups = {}
    for i, g in enumerate(pca["sidx"]):
        groups.setdefault(A.sessions[g]["group"], []).append(i)
    for ms in groups.values():
        for a in ms:
            for b in ms:
                if a != b:
                    same.setdefault(a, set()).add(b)
    return {i: (sorted(same.get(i, set())), sorted(rel.get(i, set()) - same.get(i, set())))
            for i in range(len(pca["sidx"]))}


def choose(A: Assets, pca, partners, n: int, n_hard: int, seed: int = 0):
    """Illustrative test workflows: diverse apps / OSes, belief sharpening to the true intent, plus a few hard cases."""
    rng = np.random.default_rng(seed)
    sidx = pca["sidx"]
    fr, lr = np.array(pca["first_rank"]), np.array(pca["last_rank"])
    drop = np.array(pca["ent0"]) - np.array(pca["entT"])
    L = A.ev["sess_len"][sidx]
    ok_len = (L >= 8) & (L <= 22)
    short = np.array([len(A.sessions[g]["instruction"]) <= 150 for g in sidx])
    has_p = np.array([len(partners[i][0]) > 0 for i in range(len(sidx))])
    good = np.nonzero(ok_len & short & (lr <= 3) & (fr > 5))[0]
    score = drop[good] + 1.5 * has_p[good].astype(float) + 0.1 * rng.random(len(good))
    good = good[np.argsort(-score)]
    chosen, apps, os_count = [], set(), {}
    for cap in (max(2, (n - n_hard + 2) // 3), 10 ** 9):          # balance OSes; relax if one runs out
        for i in good:
            g = sidx[i]
            app, osn = A.sessions[g]["app"], A.sessions[g]["os"]
            if len(chosen) >= n - n_hard or app in apps or os_count.get(osn, 0) >= cap:
                continue
            chosen.append(int(i))
            apps.add(app)
            os_count[osn] = os_count.get(osn, 0) + 1
    hard = np.nonzero(ok_len & short & (lr > 30))[0]
    hard = hard[np.argsort(-has_p[hard].astype(float) - 0.1 * rng.random(len(hard)))]
    for i in hard:
        if sum(1 for c in chosen if lr[c] > 30) >= n_hard:
            break
        app = A.sessions[sidx[i]]["app"]
        if app not in apps:
            chosen.append(int(i))
            apps.add(app)
    return chosen


def write_workflow(out: Path, wf: dict, single_file: bool = False):
    (out / "workflows").mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", wf["id"])
    path = out / "workflows" / f"{name}.json"
    json.dump(wf, open(path, "w"), separators=(",", ":"), ensure_ascii=False)
    idx_p = out / "workflows" / "index.json"
    idx = json.load(open(idx_p)) if idx_p.exists() else []
    idx = [e for e in idx if e["id"] != wf["id"]]
    last = wf["steps"][-1]
    idx.append({"id": wf["id"], "file": f"workflows/{name}.json", "title": wf.get("instruction") or wf["id"],
                "app": wf.get("app"), "os": wf.get("os"), "n_steps": wf["n_steps"], "source": wf["source"],
                "final_rank": last["true_rank"], "thumb": wf["steps"][0]["image"] if not single_file else None})
    json.dump(idx, open(idx_p, "w"), indent=1, ensure_ascii=False)
    return path


def jpeg(img, max_w=1280, quality=78) -> bytes:
    from PIL import Image
    im = img.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return b.getvalue()


def cmd_export(args):
    A = Assets(args.run, args.ckpt)
    pca = load_pca(args.run)
    out = Path(args.out)
    bg_pos = {int(g): i for i, g in enumerate(pca["sidx"])}
    part = judged_partners(A, pca)
    targets = []
    if args.auto:
        targets += [int(pca["sidx"][i]) for i in choose(A, pca, part, args.auto, args.hard)]
    for sid in args.session:
        gi = A.id2idx.get(sid)
        if gi is None or A.split[gi] != "test":
            raise SystemExit(f"{sid}: not a test session (use --raw with its screenshots and code for other sessions)")
        targets.append(gi)
    if targets:
        emb = load_embeddings(A)
        texts = A.vocab["texts"]
        for gi in targets:
            s = A.sessions[gi]
            d, k = emb[gi]
            bi = bg_pos[gi]
            meta = {"id": s["id"], "source": "agentnet-test", "instruction": s["instruction"],
                    "app": s["app"].split(":", 1)[-1], "os": s["os"],
                    "agentnet": {"archive": s["source"], "images": s["images"]}}
            images = [f"frames/{s['id']}/{t}.jpg" for t in range(len(d))]
            wf = build_workflow(A, pca, meta, d, k, A.session_events(gi), texts, images, args.topk, bi,
                                part[bi][0], part[bi][1])
            write_workflow(out, wf)
            st = wf["steps"]
            print(f"{wf['app'][:24]:24s} {wf['os']:8s} {len(st):3d} steps  true rank {st[0]['true_rank']:>5} -> "
                  f"{st[-1]['true_rank']:<4} partners {len(wf['partners'])}  {s['instruction'][:70]}", flush=True)
    for rp in args.raw:
        wf = export_raw(A, pca, Path(rp), args)
        p = write_workflow(out, wf, args.single_file)
        st = wf["steps"]
        print(f"raw workflow {wf['id']}: {len(st)} steps -> {p}; final top intent: {st[-1]['top'][0][0][:90]}")


# ------------------------------------------------------------------------------------------------- raw workflows
@contextlib.contextmanager
def _pixel_coords(W: int, H: int):
    """Make actions.parse_code divide x / y arguments by the screen size (code written with pixel coordinates)."""
    orig = actions._get

    def get(args, kwargs, pos, *names):
        v = orig(args, kwargs, pos, *names)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and names and names[0] in ("x", "y"):
            v = v / (W if names[0] == "x" else H)
        return v
    actions._get = get
    try:
        yield
    finally:
        actions._get = orig


def _uses_pixels(code: str) -> bool:
    try:
        calls = actions._calls_from_ast(code)
    except SyntaxError:
        return False
    for _, _, _, a, kw in calls:
        for pos, nm in ((0, "x"), (1, "y")):
            v = actions._get(a, kw, pos, nm)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1.0:
                return True
    return False


def read_raw(path: Path, images_dir: Path | None):
    wf = json.loads(path.read_text()) if path.suffix != ".jsonl" else json.loads(path.read_text().splitlines()[0])
    base = images_dir or path.parent
    if "traj" in wf:                                             # AgentNet record
        steps = [{"image": s["image"], "code": s["value"].get("code") or ""}
                 for s in sorted(wf["traj"], key=lambda s: s["index"])]
        meta = {"id": wf.get("task_id") or path.stem, "instruction": wf.get("instruction"),
                "app": wf.get("app"), "os": wf.get("os")}
    else:
        steps = wf["steps"]
        meta = {"id": wf.get("id") or path.stem, "instruction": wf.get("instruction"), "app": wf.get("app"),
                "os": wf.get("os")}
    return meta, steps, base


def export_raw(A: Assets, pca, path: Path, args):
    from PIL import Image
    meta, steps, base = read_raw(path, Path(args.images) if args.images else None)
    imgs = [Image.open(Path(s["image"]) if Path(s["image"]).is_absolute() else base / s["image"]).convert("RGB")
            for s in steps]
    # events, exactly as prepare.py (terminate dropped, <= E_MAX per step, cursor carried across steps)
    n = len(steps)
    ev = {"ev_type": np.zeros((n, E_MAX), np.int8), "ev_xy": np.full((n, E_MAX, 4), np.nan, np.float32),
          "ev_scroll": np.zeros((n, E_MAX), np.float32), "ev_keys": np.zeros((n, E_MAX, K_MAX), np.int16),
          "ev_text": np.full((n, E_MAX), -1, np.int32), "focus": np.full((n, 2), 0.5, np.float32)}
    new_texts, cursor = {}, None
    base_n = len(A.vocab["texts"])
    for t, (s, im) in enumerate(zip(steps, imgs)):
        code = s.get("code") or ""
        ctx = _pixel_coords(*im.size) if _uses_pixels(code) else contextlib.nullcontext()
        with ctx:
            events, cursor = parse_code(code, cursor)
        events = [e for e in events if e.type != "terminate"][:E_MAX]
        pt = None
        for j, e in enumerate(events):
            ev["ev_type"][t, j] = TYPE_ID[e.type]
            ev["ev_xy"][t, j] = [np.nan if v is None else v for v in (e.x, e.y, e.x2, e.y2)]
            ev["ev_scroll"][t, j] = 0.0 if e.scroll is None else float(np.clip(e.scroll, -1000, 1000))
            ks = [A.key_id.get(kk, 1) for kk in e.keys[:K_MAX]]
            ev["ev_keys"][t, j, :len(ks)] = ks
            if e.text:
                ev["ev_text"][t, j] = base_n + new_texts.setdefault(e.text, len(new_texts))
            if e.point is not None:
                pt = e.point
        f = pt if pt is not None else cursor
        if f is not None:
            ev["focus"][t] = f
    texts = A.vocab["texts"] + list(new_texts)
    model = A.model
    if new_texts:
        tv = encode_texts(list(new_texts), A.device)
        model = A.build(torch.cat([torch.from_numpy(A.text_vecs).float(), torch.from_numpy(tv).float()]))
    frames = encode_frames(imgs, A.device, args.frames_pca)
    (d, k), = run_model(model, A.device, [frames], [ev])
    images = []
    fdir = Path(args.out) / "frames" / re.sub(r"[^A-Za-z0-9_.-]", "_", meta["id"])
    if not args.single_file:
        fdir.mkdir(parents=True, exist_ok=True)
    for t, im in enumerate(imgs):
        b = jpeg(im)
        if args.single_file:
            images.append("data:image/jpeg;base64," + base64.b64encode(b).decode())
        else:
            (fdir / f"{t}.jpg").write_bytes(b)
            images.append(f"frames/{fdir.name}/{t}.jpg")
    meta = {**meta, "source": "raw"}
    return build_workflow(A, pca, meta, d, k, ev, texts, images, args.topk)


_ENC = {}


@torch.no_grad()
def encode_frames(imgs, device, pca_file: str | None = None, grid: int = 16, patches: int = 1024) -> np.ndarray:
    """Screenshots -> [L, grid*grid, 512] bf16 bits, as cache_frames.py --mode dense (SigLIP2, NaFlex <= 1024
    patches, adaptive average pooling to grid x grid, channel PCA fitted on training frames)."""
    from transformers import AutoImageProcessor, Siglip2VisionModel
    from .cache_frames import pool_grid, to_bf16_bits
    if "siglip" not in _ENC:
        proc = AutoImageProcessor.from_pretrained(SIGLIP)
        vis = Siglip2VisionModel.from_pretrained(SIGLIP, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").vision_model.to(device).eval()
        z = np.load(pca_file or paths.BULK / f"dense_pca_g{grid}_d512.npz")
        _ENC["siglip"] = (proc, vis, torch.from_numpy(z["mean"]).to(device), torch.from_numpy(z["comps"]).to(device))
    proc, vis, mean, comps = _ENC["siglip"]
    out = []
    for im in imgs:
        g = proc(images=[im], max_num_patches=patches, return_tensors="pt")
        h = vis(pixel_values=g["pixel_values"].to(device, torch.bfloat16),
                attention_mask=g["pixel_attention_mask"].to(device),
                spatial_shapes=g["spatial_shapes"].to(device)).last_hidden_state
        x = pool_grid(h, g["spatial_shapes"], (grid, grid))
        out.append(to_bf16_bits((x[0] - mean) @ comps))
    return np.stack(out)


@torch.no_grad()
def encode_texts(texts, device) -> np.ndarray:
    """Typed strings -> Qwen3-Embedding-0.6B vectors (as text_embed.py: max length 256, L2-normalized, fp16)."""
    from sentence_transformers import SentenceTransformer
    if "text" not in _ENC:
        m = SentenceTransformer(TEXT_MODEL, device=str(device), model_kwargs={"torch_dtype": torch.bfloat16})
        m.max_seq_length = 256
        _ENC["text"] = m
    v = _ENC["text"].encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return v.astype(np.float16)


# -------------------------------------------------------------------------------------------- rnn-side helpers
def cmd_frames(args):
    """Screenshots of the exported AgentNet sessions, read from the split archives (run where paths.ZIPS exist)."""
    from PIL import Image
    from .splitzip import SplitZip
    out = Path(args.out)
    zips, index = {}, {}
    for f in sorted((out / "workflows").glob("*.json")):
        if f.name == "index.json":
            continue
        wf = json.load(open(f))
        an = wf.get("agentnet")
        if not an:
            continue
        src = an["archive"]
        if src not in zips:
            zips[src] = SplitZip(paths.ZIPS[src])
            index[src] = zips[src].basename_index()
        fdir = out / "frames" / wf["id"]
        fdir.mkdir(parents=True, exist_ok=True)
        for t, name in enumerate(an["images"][:wf["n_steps"]]):
            if (fdir / f"{t}.jpg").exists():
                continue
            member = index[src].get(name)
            if member is None:
                continue
            (fdir / f"{t}.jpg").write_bytes(jpeg(Image.open(io.BytesIO(zips[src].read(member)))))
        print(f"{wf['id']}: {wf['n_steps']} frames", flush=True)


def cmd_extract(args):
    from .data import SessionData
    t0 = time.time()
    A_state = torch.load(paths.RUNS / args.run / args.ckpt, map_location="cpu", weights_only=False)
    data = SessionData(Path(A_state["args"]["frames_dir"]), frames_mode="pread")
    sidx = data.indices("test")
    lens, offs = data.length[sidx].astype(np.int64), data.offset[sidx].astype(np.int64)
    DEMO.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(DEMO / "test_frames.npy", mode="w+", dtype=np.uint16,
                                   shape=(int(lens.sum()), data.frames.n_tok, data.frames.dim))
    pos = 0
    for o, n in zip(offs, lens):
        data.frames.gather_into(np.arange(o, o + n), mm[pos:pos + n])
        pos += n
    mm.flush()
    np.savez(DEMO / "test_index.npz", sidx=sidx, offset=np.r_[0, np.cumsum(lens)[:-1]], length=lens)
    print(f"{len(sidx)} test sessions, {pos} frames in {time.time() - t0:.0f}s -> {DEMO}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["embed", "background", "export", "frames", "extract"])
    ap.add_argument("--run", default="r20_xf4_d512")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", default="demo/bundle")
    ap.add_argument("--max_frames", type=int, default=1024, help="embed: frames per batch")
    ap.add_argument("--k", type=int, default=10, help="background: number of clusters")
    ap.add_argument("--auto", type=int, default=0, help="export: number of illustrative test workflows to choose")
    ap.add_argument("--hard", type=int, default=2, help="export --auto: how many of them are hard cases")
    ap.add_argument("--session", action="append", default=[], help="export: AgentNet test session id")
    ap.add_argument("--raw", action="append", default=[], help="export: raw workflow JSON (see module doc)")
    ap.add_argument("--images", default="", help="export --raw: directory with the screenshots")
    ap.add_argument("--frames_pca", default="", help="export --raw: channel-PCA basis (default BULK)")
    ap.add_argument("--single_file", action="store_true", help="export --raw: embed screenshots in the JSON")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    {"embed": cmd_embed, "background": cmd_background, "export": cmd_export, "frames": cmd_frames,
     "extract": cmd_extract}[args.cmd](args)


if __name__ == "__main__":
    main()
