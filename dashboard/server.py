#!/usr/bin/env python3
"""Local dashboard for the results tree (docs/EXPERIMENTS.md): runs table, curves, final metrics, queue, demo bundles.

    python dashboard/server.py [--results results] [--bundles dashboard/bundles] [--jobs jobs.txt] [--port 8765]

Standard library only. Serves the static frontend in dashboard/static/ and a small JSON API:

    GET /api/runs                      every run folder (a folder holding config.json) with its summary
    GET /api/run?id=<run id>           one run: summary + raw config.json and final.json
    GET /api/series?id=<run id>        steps.jsonl and evals.jsonl as columns, downsampled to ~max_points rows
        [&max_points=2000] [&minmax=1] [&which=steps,evals] [&cols=loss,lr]
    GET /api/queue                     the sections of jobs.txt with each id's status in the results tree
    GET /bundles/<run id>/<path>       files of a demo bundle (wfemb.demo export --out dashboard/bundles/<run id>)

Parsed files are cached by (path, mtime, size); a growing steps.jsonl is parsed incrementally (only the new bytes).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import mimetypes
import os
import re
import sys
import threading
import time
from array import array
from collections import OrderedDict
from email.utils import formatdate, parsedate_to_datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STATIC = HERE / "static"

# The best model so far (read from docs/EXPERIMENTS.md at startup; this is the fallback). Only used to order variables and
# to mark the segments of an id that differ from it; every run's variables come from its own folder path.
REFERENCE_ID = ("arch-late-history-transf_dm-512_dh-64_nl-4_nh-8_drop-0.3_smooth-0.2_patch-576_lr-5e-4_bs-64_ep-45"
                "_beta-0_seed-0")
DOC_REF = re.compile(r"best model so far is\s*`([^`]+)`")


def doc_reference_id() -> str:
    """The best model's id as docs/EXPERIMENTS.md states it ('The best model so far is `...`'), else REFERENCE_ID."""
    try:
        m = DOC_REF.search((REPO / "docs" / "EXPERIMENTS.md").read_text(encoding="utf-8"))
    except OSError:
        m = None
    return m.group(1).strip() if m else REFERENCE_ID


X_COLS = {"step", "epoch", "time"}       # kept at more precision when downsampled / serialized
NAN = float("nan")


# --------------------------------------------------------------------------------------------------- run ids
def parse_id(run_id: str) -> list[tuple[str, str]]:
    """'arch-gru_dm-512' -> [('arch', 'gru'), ('dm', '512')]; key = text before the first '-'. A segment without '-'
    continues the previous value ('qkv-sep_q' -> ('qkv', 'sep_q')), so values spelled with '_' still parse."""
    out: list[tuple[str, str]] = []
    for seg in run_id.split("_"):
        k, sep, v = seg.partition("-")
        if not sep and out:
            pk, pv = out[-1]
            out[-1] = (pk, f"{pv}_{seg}")
        else:
            out.append((k, v) if sep else (seg, ""))
    return out


def canon(v) -> str:
    """Canonical id spelling of a value (docs/EXPERIMENTS.md: ints as ints, floats < 0.01 as 5e-4, others %g)."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        if abs(v) < 0.01:
            m, e = f"{v:e}".split("e")
            m = m.rstrip("0").rstrip(".")
            return f"{m}e{int(e)}"
        return f"{v:g}"
    return "off" if v is None else str(v)


def clean(o):
    """JSON-safe copy: NaN / inf -> None (browsers' JSON.parse rejects them)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def fnum(v, digits: int = 6):
    return None if v != v or v in (math.inf, -math.inf) else float(f"{v:.{digits}g}")


# --------------------------------------------------------------------------------------------------- caches
class LRU:
    def __init__(self, max_items: int | None):
        self.max, self.d, self.lock = max_items, OrderedDict(), threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                return self.d[key]
        return None

    def put(self, key, val):
        with self.lock:
            self.d[key] = val
            self.d.move_to_end(key)
            while self.max and len(self.d) > self.max:
                self.d.popitem(last=False)


_path_locks: dict[str, threading.Lock] = {}
_path_locks_lock = threading.Lock()


def path_lock(p: Path) -> threading.Lock:
    with _path_locks_lock:
        return _path_locks.setdefault(str(p), threading.Lock())


def stat_sig(p: Path):
    try:
        st = p.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


class JsonCache:
    """json files (config.json, final.json) by (mtime, size)."""

    def __init__(self):
        self.c = LRU(None)

    def load(self, p: Path):
        sig = stat_sig(p)
        if sig is None:
            return None
        hit = self.c.get(str(p))
        if hit and hit[0] == sig:
            return hit[1]
        try:
            val = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            val = {"_error": f"{p.name}: {e}"}
        self.c.put(str(p), (sig, val))
        return val


class Table:
    """A parsed jsonl file as numeric columns (array('d'), NaN where a row lacks the key or it is not a number)."""

    __slots__ = ("cols", "n", "offset", "n_committed", "sig", "bad", "anchor")
    ANCHOR = 256                    # bytes before `offset` remembered to recognise an appended (not rewritten) file

    def __init__(self):
        self.cols: dict[str, array] = {}
        self.n = 0                  # rows, including a provisional last row without a trailing newline
        self.n_committed = 0        # rows ending in a newline (an appending writer may still be writing the tail)
        self.offset = 0             # bytes consumed by committed rows
        self.sig = None
        self.bad = 0                # unparseable lines
        self.anchor = b""           # the last ANCHOR committed bytes

    def copy(self) -> "Table":
        t = Table()
        t.cols = {k: array("d", a) for k, a in self.cols.items()}
        t.n, t.n_committed, t.offset, t.sig, t.bad = self.n, self.n_committed, self.offset, self.sig, self.bad
        t.anchor = self.anchor
        return t

    def truncate(self, n: int):
        for a in self.cols.values():
            del a[n:]
        self.n = n

    def add(self, row: dict):
        n = self.n
        for k, v in row.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            a = self.cols.get(k)
            if a is None:
                a = self.cols[k] = array("d", [NAN]) * n
            a.append(float(v))
        self.n = n + 1
        for a in self.cols.values():
            if len(a) < self.n:
                a.append(NAN)

    def feed(self, data: bytes):
        """Parse bytes following self.offset (which ends at a line boundary)."""
        self.truncate(self.n_committed)
        end = data.rfind(b"\n")
        complete, tail = (data[:end], data[end + 1:]) if end >= 0 else (b"", data)
        for line in complete.split(b"\n") if complete else ():
            self._line(line)
        self.n_committed = self.n
        if end >= 0:
            self.offset += end + 1
            self.anchor = (self.anchor + data[:end + 1])[-self.ANCHOR:]
        if tail.strip():
            self._line(tail, provisional=True)

    def _line(self, line: bytes, provisional: bool = False):
        line = line.strip()
        if not line:
            return
        try:
            row = json.loads(line)
        except ValueError:
            if not provisional:
                self.bad += 1
            return
        if isinstance(row, dict):
            self.add(row)


class JsonlCache:
    def __init__(self, max_items: int | None):
        self.c = LRU(max_items)

    def load(self, p: Path) -> Table | None:
        sig = stat_sig(p)
        if sig is None:
            return None
        key = str(p)
        t = self.c.get(key)
        if t is not None and t.sig == sig:
            return t
        with path_lock(p):
            t = self.c.get(key)
            if t is not None and t.sig == sig:
                return t
            try:
                with open(p, "rb") as f:
                    # an appended file: same inode, not shorter, and the bytes before the parsed offset unchanged
                    # (inode numbers get reused when a run restarts and rewrites its files)
                    grow = t is not None and t.sig[2] == sig[2] and sig[1] >= t.offset and sig[1] >= t.sig[1]
                    if grow:
                        f.seek(t.offset - len(t.anchor))
                        grow = f.read(len(t.anchor)) == t.anchor
                    # published tables are never mutated (other threads may be reading them): extend a copy
                    t = t.copy() if grow else Table()
                    f.seek(t.offset)
                    t.feed(f.read())
            except OSError:
                return None
            t.sig = sig
            self.c.put(key, t)
            return t


def tail_row(p: Path, cache: LRU):
    """Last complete-looking json row of a jsonl file without parsing all of it."""
    sig = stat_sig(p)
    if sig is None:
        return None
    hit = cache.get(str(p))
    if hit and hit[0] == sig:
        return hit[1]
    row = None
    try:
        with open(p, "rb") as f:
            size = sig[1]
            for back in (8192, 65536, size):
                f.seek(max(0, size - back))
                lines = f.read().split(b"\n")
                if back < size:
                    lines = lines[1:]          # the first line is probably cut
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict):
                        row = r
                        break
                if row is not None or back >= size:
                    break
    except OSError:
        return None
    cache.put(str(p), (sig, row))
    return row


def downsample(t: Table, max_points: int, minmax: bool, cols: list[str] | None):
    """Bucket means over consecutive rows: at most ~max_points rows per column. NaNs are skipped."""
    keys = [k for k in (cols or t.cols) if k in t.cols]
    n = t.n
    k = max(1, math.ceil(n / max_points)) if max_points > 0 else 1
    out, lo, hi = {}, {}, {}
    for key in keys:
        a = t.cols[key]
        dig = 10 if key in X_COLS else 6
        if k == 1:
            vals = [fnum(v, dig) for v in a]
            if all(v is None for v in vals):
                continue
            out[key] = vals
            continue
        m, mn, mx = [], [], []
        for i in range(0, n, k):
            chunk = [v for v in a[i:i + k] if v == v]
            if chunk:
                m.append(fnum(math.fsum(chunk) / len(chunk), dig))
                if minmax:
                    mn.append(fnum(min(chunk), dig))
                    mx.append(fnum(max(chunk), dig))
            else:
                m.append(None)
                if minmax:
                    mn.append(None)
                    mx.append(None)
        if all(v is None for v in m):
            continue
        out[key] = m
        if minmax and key not in X_COLS:
            lo[key], hi[key] = mn, mx
    res = {"rows": n, "bucket": k, "points": math.ceil(n / k) if n else 0, "columns": out}
    if minmax and k > 1:
        res["min"], res["max"] = lo, hi
    if t.bad:
        res["bad_lines"] = t.bad
    return res


# --------------------------------------------------------------------------------------------------- results
class Results:
    def __init__(self, root: Path, bundles: Path, jobs: Path, stale_minutes: float, reference_id: str):
        self.root, self.bundles, self.jobs = root, bundles, jobs
        self.stale_s = stale_minutes * 60
        self.reference_id = reference_id
        self.json = JsonCache()
        self.evals = JsonlCache(None)          # small; every run's is needed for the run list
        self.steps = JsonlCache(48)            # large; only for runs being plotted
        self.tails = LRU(None)
        self.ds_cache = LRU(256)          # downsampled series
        self.index: dict[str, Path] = {}
        self.scan_lock = threading.Lock()

    # ------------------------------------------------------------------ discovery
    def scan(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        if self.root.is_dir():
            for dirpath, dirnames, filenames in os.walk(self.root):
                if "config.json" in filenames:
                    d = Path(dirpath)
                    rel = d.relative_to(self.root)
                    if rel.parts:
                        found["_".join(rel.parts)] = d
                    dirnames[:] = []                      # run folders are leaves
                else:
                    dirnames[:] = sorted(x for x in dirnames if not x.startswith("."))
        with self.scan_lock:
            self.index = found
        return found

    def run_dir(self, run_id: str) -> Path | None:
        d = self.index.get(run_id)
        if d is not None and (d / "config.json").exists():
            return d
        segs = run_id.split("_")
        if not run_id or any(s in ("", ".", "..") or "/" in s or "\\" in s or "\x00" in s for s in segs):
            return None
        d = self.root.joinpath(*segs)
        if (d / "config.json").is_file():
            with self.scan_lock:
                self.index[run_id] = d
            return d
        return None

    def bundle_ids(self) -> list[str]:
        if not self.bundles.is_dir():
            return []
        return sorted(p.name for p in self.bundles.iterdir()
                      if p.is_dir() and ((p / "background.json").exists() or (p / "workflows" / "index.json").exists()))

    # ------------------------------------------------------------------ one run
    def summary(self, run_id: str, d: Path, now: float, bundles: set[str]) -> dict:
        cfg = self.json.load(d / "config.json") or {}
        final_p, steps_p, evals_p = d / "final.json", d / "steps.jsonl", d / "evals.jsonl"
        final = self.json.load(final_p) if final_p.exists() else None
        meta = cfg.get("meta") or {} if isinstance(cfg, dict) else {}
        exp = cfg.get("experiment") or {} if isinstance(cfg, dict) else {}
        variables = parse_id(run_id)
        warnings = []
        if isinstance(cfg, dict) and cfg.get("_error"):
            warnings.append(cfg["_error"])
        if isinstance(final, dict) and final.get("_error"):
            warnings.append(final["_error"])
            final = None
        if "-" not in run_id.split("_")[0] or any(not v for _, v in variables):
            warnings.append("folder path is not a run id (key-value segments)")
        cid = cfg.get("id") if isinstance(cfg, dict) else None
        if cid and cid != run_id:
            warnings.append(f"config.json id {cid} differs from the folder path")
        if isinstance(exp, dict):
            for k, v in variables:
                if k in exp and canon(exp[k]) != v:
                    warnings.append(f"config.json experiment.{k} = {canon(exp[k])}, folder says {v}")

        mtimes = [s[0] / 1e9 for s in (stat_sig(steps_p), stat_sig(evals_p), stat_sig(d / "config.json")) if s]
        last_update = max(mtimes) if mtimes else None
        if final is not None:
            status = "done"
        elif last_update is not None and now - last_update < self.stale_s:
            status = "running"
        else:
            status = "stale"

        tail = tail_row(steps_p, self.tails) or {}
        ev = self.evals.load(evals_p)
        best_val = best_val_step = None
        last_eval = {}
        if ev is not None and ev.n:
            vl, st = ev.cols.get("val_loss"), ev.cols.get("step")
            if vl is not None:
                finite = [(v, i) for i, v in enumerate(vl) if v == v]
                if finite:
                    best_val, bi = min(finite)
                    best_val_step = st[bi] if st is not None else None
            last_eval = {k: a[ev.n - 1] for k, a in ev.cols.items()}
        total = meta.get("total_steps")
        if total is None and meta.get("steps_per_epoch") and isinstance(exp, dict) and exp.get("ep"):
            try:
                total = int(meta["steps_per_epoch"]) * int(exp["ep"])
            except (TypeError, ValueError):
                total = None
        sv, ev_sig = stat_sig(steps_p), stat_sig(evals_p)
        return clean({
            "id": run_id,
            "path": str(d.relative_to(self.root)) if d.is_relative_to(self.root) else str(d),
            "vars": dict(variables),
            "status": status,
            "last_step": tail.get("step"),
            "last_epoch": tail.get("epoch"),
            "last_time": tail.get("time"),
            "last_loss": tail.get("loss"),
            "total_steps": total,
            "steps_per_epoch": meta.get("steps_per_epoch"),
            "n_evals": ev.n if ev is not None else 0,
            "best_val_loss": best_val,
            "best_val_step": best_val_step,
            "last_eval": last_eval,
            "final": final,
            "legacy": bool(meta.get("legacy")),
            "legacy_name": meta.get("legacy_name"),
            "start_time": meta.get("started_at") or meta.get("start_time"),
            "host": meta.get("host"),
            "launcher": meta.get("launcher"),
            "n_params": meta.get("n_params"),
            "last_update": last_update,
            "has_bundle": run_id in bundles,
            "version": f"{sv[0] if sv else 0}.{sv[1] if sv else 0}.{ev_sig[0] if ev_sig else 0}.{ev_sig[1] if ev_sig else 0}",
            "warnings": warnings,
        })

    def runs(self) -> dict:
        now = time.time()
        bundles = self.bundle_ids()
        bset = set(bundles)
        index = self.scan()
        runs = [self.summary(rid, d, now, bset) for rid, d in sorted(index.items())]
        try:                                   # how to spell the bundles folder in commands run from the repo root
            bundles_out = str(self.bundles.relative_to(REPO))
        except ValueError:
            bundles_out = str(self.bundles)
        return {"now": now, "results_root": str(self.root), "bundles_root": str(self.bundles), "bundles_out": bundles_out,
                "jobs_root": str(self.jobs),
                "stale_minutes": self.stale_s / 60, "reference_id": self.reference_id, "runs": runs,
                "bundles": bundles}

    def run(self, run_id: str) -> dict | None:
        d = self.run_dir(run_id)
        if d is None:
            return None
        bset = set(self.bundle_ids())
        files = {}
        for p in sorted(d.iterdir()):
            if p.is_file():
                st = p.stat()
                files[p.name] = {"size": st.st_size, "mtime": st.st_mtime}
        final_p = d / "final.json"
        return clean({"run": self.summary(run_id, d, time.time(), bset), "config": self.json.load(d / "config.json"),
                      "final": self.json.load(final_p) if final_p.exists() else None, "files": files})

    def series(self, run_id: str, max_points: int, minmax: bool, which: list[str], cols: list[str] | None) -> dict | None:
        d = self.run_dir(run_id)
        if d is None:
            return None
        out = {"id": run_id}
        for name, cache in (("steps", self.steps), ("evals", self.evals)):
            if name not in which:
                continue
            p = d / f"{name}.jsonl"
            sig = stat_sig(p)
            if sig is None:
                out[name] = None
                continue
            key = (str(p), sig, max_points, minmax, tuple(cols) if cols else None)
            hit = self.ds_cache.get(key)
            if hit is None:
                t = cache.load(p)
                hit = downsample(t, max_points, minmax, cols) if t is not None else None
                self.ds_cache.put(key, hit)
            out[name] = hit
        return out

    # ------------------------------------------------------------------ queue
    def queue(self) -> dict:
        """jobs.txt: one group per [section] (comments before its first run = its header), each run with its status."""
        now = time.time()
        index = self.scan()
        bset = set(self.bundle_ids())
        status = {}
        sections = []
        ref = dict(parse_id(self.reference_id))
        seen_ids = set()
        try:
            text = self.jobs.read_text(encoding="utf-8", errors="replace") if self.jobs.is_file() else ""
            mtime = self.jobs.stat().st_mtime if self.jobs.is_file() else None
        except OSError as e:
            return clean({"jobs_root": str(self.jobs), "reference_id": self.reference_id, "error": str(e),
                          "files": [], "unqueued": sorted(index)})
        cur = None
        for ln, raw in enumerate(text.splitlines(), 1):
            body, _, comment = raw.partition("#")
            body, comment = body.strip(), comment.strip()
            m = re.fullmatch(r"\[([A-Za-z0-9_.-]+)\]", body)
            if m:
                cur = {"name": m.group(1), "header": [], "entries": [], "mtime": mtime, "line": ln}
                sections.append(cur)
                continue
            if not body:
                if cur is not None and comment and not cur["entries"]:
                    cur["header"].append(comment)
                continue
            if cur is None:                                   # a run before any [section]
                cur = {"name": "(no section)", "header": [], "entries": [], "mtime": mtime, "line": ln}
                sections.append(cur)
            toks = body.split()
            rid, overrides = toks[0], toks[1:]
            problems = []
            segs = parse_id(rid)
            if any(not v for _, v in segs):
                problems.append("segment without '-'")
            elif [k for k, _ in segs] != [k for k in ref if k in dict(segs)] and ref:
                problems.append("variables out of order")
            unknown = [k for k, _ in segs if k not in ref]
            if unknown and ref:
                problems.append("unknown variable " + ", ".join(unknown))
            bad_ov = [o for o in overrides if "=" not in o]
            if bad_ov:
                problems.append("override without '=': " + " ".join(bad_ov))
            if rid not in status:
                d = index.get(rid) or self.run_dir(rid)
                status[rid] = None if d is None else self.summary(rid, d, now, bset)
            st = status[rid]
            seen_ids.add(rid)
            cur["entries"].append({
                "line": ln, "id": rid, "overrides": overrides, "comment": comment or None,
                "status": st["status"] if st else "not started", "problems": problems,
                "last_step": st["last_step"] if st else None, "total_steps": st["total_steps"] if st else None,
                "best_val_loss": st["best_val_loss"] if st else None,
                "final": {k: v for k, v in (st.get("final") or {}).items() if k.startswith("test_")} if st else None,
            })
        unqueued = sorted(rid for rid in index if rid not in seen_ids)
        # "files" = the sections (the frontend groups by it)
        return clean({"jobs_root": str(self.jobs), "reference_id": self.reference_id, "files": sections,
                      "unqueued": unqueued})


# --------------------------------------------------------------------------------------------------- http
def safe_join(root: Path, rel: str) -> Path | None:
    """root/rel for a URL-decoded relative path, or None if it could leave root (.., absolute, backslash, NUL)."""
    if "\x00" in rel or "\\" in rel:
        return None
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts) or (parts and (":" in parts[0])):
        return None
    return root.joinpath(*parts) if parts else root


class Handler(BaseHTTPRequestHandler):
    server_version = "wfemb-dashboard/1"
    protocol_version = "HTTP/1.1"
    R: Results = None  # set in main()
    verbose = False

    def log_message(self, fmt, *args):
        if self.verbose:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------------ responses
    def send_bytes(self, body: bytes, ctype: str, status: int = 200, headers: dict | None = None, head=False):
        if (len(body) > 32_768 and "gzip" in (self.headers.get("Accept-Encoding") or "")
                and (ctype.startswith("application/json") or ctype.startswith("text/"))):
            body = gzip.compress(body, compresslevel=5)
            headers = {**(headers or {}), "Content-Encoding": "gzip", "Vary": "Accept-Encoding"}
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def send_json(self, obj, status: int = 200, head=False):
        body = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()
        self.send_bytes(body, "application/json; charset=utf-8", status, {"Cache-Control": "no-store"}, head)

    def send_error_json(self, status: int, msg: str, head=False):
        self.send_json({"error": msg, "status": status}, status, head)

    def send_file(self, p: Path, head=False, cache: str = "no-cache"):
        try:
            st = p.stat()
        except OSError:
            return self.send_error_json(404, "not found", head)
        if not p.is_file():
            return self.send_error_json(404, "not found", head)
        ims = self.headers.get("If-Modified-Since")
        if ims:
            try:
                if int(st.st_mtime) <= parsedate_to_datetime(ims).timestamp():
                    self.send_response(304)
                    self.send_header("Cache-Control", cache)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            except (TypeError, ValueError, OverflowError):
                pass
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self.send_bytes(p.read_bytes(), ctype, 200,
                        {"Last-Modified": formatdate(st.st_mtime, usegmt=True), "Cache-Control": cache}, head)

    # ------------------------------------------------------------------ routing
    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        try:
            self.route(head)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001  (report, keep serving)
            import traceback
            traceback.print_exc()
            try:
                self.send_error_json(500, f"{type(e).__name__}: {e}", head)
            except OSError:
                pass

    def route(self, head):
        u = urlsplit(self.path)
        path = unquote(u.path)
        q = {k: v[-1] for k, v in parse_qs(u.query, keep_blank_values=True).items()}
        R = self.R
        if path.startswith("/api/"):
            if path == "/api/runs":
                return self.send_json(R.runs(), head=head)
            if path == "/api/run":
                if not q.get("id"):
                    return self.send_error_json(400, "missing id", head)
                r = R.run(q["id"])
                return self.send_json(r, head=head) if r else self.send_error_json(404, f"no run {q['id']}", head)
            if path == "/api/series":
                if not q.get("id"):
                    return self.send_error_json(400, "missing id", head)
                try:
                    mp = int(q.get("max_points", "2000"))
                except ValueError:
                    return self.send_error_json(400, "max_points must be an integer", head)
                mp = max(0, min(mp, 1_000_000))
                which = [w for w in (q.get("which") or "steps,evals").split(",") if w in ("steps", "evals")]
                cols = [c for c in q["cols"].split(",") if c] if q.get("cols") else None
                r = R.series(q["id"], mp, q.get("minmax") in ("1", "true"), which, cols)
                return self.send_json(r, head=head) if r else self.send_error_json(404, f"no run {q['id']}", head)
            if path == "/api/queue":
                return self.send_json(R.queue(), head=head)
            return self.send_error_json(404, "unknown endpoint", head)
        if path.startswith("/bundles/"):
            p = safe_join(R.bundles, path[len("/bundles/"):])
            if p is None:
                return self.send_error_json(403, "bad path", head)
            return self.send_file(p, head)
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        p = safe_join(STATIC, rel)
        if p is None:
            return self.send_error_json(403, "bad path", head)
        return self.send_file(p, head)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=os.environ.get("WFEMB_RESULTS", str(REPO / "results")),
                    help="results root (default $WFEMB_RESULTS or <repo>/results)")
    ap.add_argument("--bundles", default=str(REPO / "dashboard" / "bundles"), help="demo bundles, one folder per run id")
    ap.add_argument("--jobs", default=str(REPO / "jobs.txt"), help="the run queue file")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--stale-minutes", type=float, default=15,
                    help="a run without final.json is 'running' if its files changed within this many minutes")
    ap.add_argument("--reference-id", default=None,
                    help="id whose values count as the defaults (default: the best model in docs/EXPERIMENTS.md)")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    a = ap.parse_args(argv)

    def rel(p):
        p = Path(p).expanduser()
        return (p if p.is_absolute() else Path.cwd() / p).resolve()

    Handler.R = Results(rel(a.results), rel(a.bundles), rel(a.jobs), a.stale_minutes, a.reference_id or doc_reference_id())
    Handler.verbose = a.verbose
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    try:
        srv = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError as e:
        sys.exit(f"cannot listen on {a.host}:{a.port}: {e} (try --port)")
    srv.daemon_threads = True
    R = Handler.R
    print(f"results {R.root}{'' if R.root.is_dir() else ' (missing)'}\n"
          f"bundles {R.bundles}{'' if R.bundles.is_dir() else ' (missing)'}\n"
          f"jobs    {R.jobs}{'' if R.jobs.is_file() else ' (missing)'}\n"
          f"dashboard on http://{a.host}:{a.port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
