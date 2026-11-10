"""The results tree (docs/EXPERIMENTS.md): raw data of every run, one folder level per run-id segment.

  <root>/arch-<v>/dm-<v>/.../seed-<v>/
      config.json    full configuration + meta (written at start)
      steps.jsonl    one row per training step
      evals.jsonl    one row per validation pass
      final.json     val/test metrics at the chosen checkpoint (written at the end; absent while running)
      best.pt, last.pt

Only raw data goes here; summaries and plots are computed by the dashboard.

  python -m wfemb.storage.results list [--root results]       # every run: id, status, last step
  python -m wfemb.storage.results migrate [--root results]    # re-derive folders from config.json (new variables)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterator

from ..config import Experiment, default_results

FILES = ("config.json", "steps.jsonl", "evals.jsonl", "final.json")


def clean(x):
    """JSON-safe copy: NaN / inf -> None (browsers' JSON.parse rejects NaN), numpy scalars -> Python numbers."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def dumps(x) -> str:
    return json.dumps(clean(x), allow_nan=False)


class RunWriter:
    """Appends a run's rows. fresh=True starts the run over (drops rows of an earlier, unfinished attempt)."""

    def __init__(self, run_dir: Path, fresh: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if fresh:
            for f in ("steps.jsonl", "evals.jsonl", "final.json"):
                (self.dir / f).unlink(missing_ok=True)
        self._steps = open(self.dir / "steps.jsonl", "a")
        self._n = 0

    def config(self, cfg: dict):
        (self.dir / "config.json").write_text(json.dumps(clean(cfg), indent=1, allow_nan=False))

    def step(self, row: dict):
        self._steps.write(dumps(row) + "\n")
        self._n += 1
        if self._n % 50 == 0:
            self._steps.flush()

    def eval(self, row: dict):
        self._steps.flush()
        with open(self.dir / "evals.jsonl", "a") as f:
            f.write(dumps(row) + "\n")

    def final(self, metrics: dict):
        self._steps.flush()
        (self.dir / "final.json").write_text(json.dumps(clean(metrics), indent=1, allow_nan=False))

    def close(self):
        self._steps.close()


def iter_runs(root: str | Path) -> Iterator[tuple[Experiment, Path]]:
    """(experiment, run folder) of every run under root (folders holding a config.json)."""
    for cfg in sorted(Path(root).rglob("config.json")):
        c = json.loads(cfg.read_text())
        yield Experiment(**c["experiment"]), cfg.parent


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:        # a row being written right now
                pass
    return out


def status(run_dir: Path) -> str:
    return "done" if (run_dir / "final.json").exists() else "unfinished"


def migrate(root: Path, dry_run: bool = False):
    """Move every run to the folder its config.json implies (after an experiment variable was added or renamed)."""
    moved = 0
    for exp, d in list(iter_runs(root)):
        want = Path(root) / exp.path
        if want.resolve() == d.resolve():
            continue
        print(f"{d.relative_to(root)} -> {want.relative_to(root)}")
        moved += 1
        if dry_run:
            continue
        if want.exists():
            raise FileExistsError(want)
        want.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(want))
        c = json.loads((want / "config.json").read_text())
        c["id"], c["experiment"] = exp.id, json.loads(json.dumps(clean(vars_of(exp))))
        (want / "config.json").write_text(json.dumps(c, indent=1))
    print(f"{moved} run(s) {'to move' if dry_run else 'moved'}")


def vars_of(exp: Experiment) -> dict:
    return {k: getattr(exp, k) for k in Experiment.keys()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "migrate"])
    ap.add_argument("--root", default=default_results())
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "migrate":
        migrate(Path(a.root), a.dry_run)
        return
    for exp, d in iter_runs(a.root):
        steps = read_jsonl(d / "steps.jsonl")
        print(f"{status(d):10s} step {steps[-1]['step'] if steps else 0:>6}  {exp.id}")


if __name__ == "__main__":
    main()
