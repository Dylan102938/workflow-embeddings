"""Copy run results (raw data) from where jobs ran into the local results tree (default ./results, gitignored).

  python -m wfemb.storage.fetch rnn                    # rsync rnn:$WFEMB_DATA/results (json/jsonl only)
  python -m wfemb.storage.fetch baseten                # every job of the Baseten training project "wfemb"
  python -m wfemb.storage.fetch baseten --job q9gge93  # one job
  ... --ckpt                                           # also the checkpoints (best.pt, last.pt)

Baseten jobs leave their results tree in the job's checkpoint directory (runners/baseten/train.sh), and demo jobs their
bundles (runners/baseten/demo.sh, -> dashboard/bundles/); `truss train get_checkpoint_urls` lists those files with
download URLs.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from ..config import REPO, default_results

RNN_RESULTS = "rnn:/nas/ucb/dfeng/data/workflow-embeddings/results/"
BASETEN_PROJECT = "wfemb"


def fetch_rnn(root: Path, ckpt: bool, src: str = RNN_RESULTS):
    inc = ["--include=*/", "--include=*.json", "--include=*.jsonl"] + (["--include=*.pt"] if ckpt else [])
    cmd = ["rsync", "-a", "--prune-empty-dirs", *inc, "--exclude=*", src, str(root) + "/"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _api_key() -> str:
    if os.environ.get("BASETEN_API_KEY"):
        return os.environ["BASETEN_API_KEY"]
    rc = configparser.ConfigParser()
    rc.read(Path.home() / ".trussrc")
    return rc["baseten"]["api_key"]


def _api(path: str):
    req = urllib.request.Request("https://api.baseten.co/v1" + path, headers={"Authorization": "Api-Key " + _api_key()})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def baseten_jobs(project: str = BASETEN_PROJECT) -> list[dict]:
    projects = _api("/training_projects")
    projects = projects.get("training_projects", projects)
    pid = next(p["id"] for p in projects if p["name"] == project)
    jobs = _api(f"/training_projects/{pid}/jobs")
    return jobs.get("training_jobs", jobs)


def fetch_baseten(root: Path, ckpt: bool, job_ids: list[str] | None = None,
                  bundles: Path = REPO / "dashboard" / "bundles"):
    jobs = job_ids or [j["id"] for j in baseten_jobs() if j["current_status"].endswith(("COMPLETED", "RUNNING"))]
    for jid in jobs:
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run(["truss", "train", "get_checkpoint_urls", "--job-id", jid], cwd=tmp,
                               capture_output=True, text=True)
            files = list(Path(tmp).glob("*.json"))
            if r.returncode or not files:
                print(f"{jid}: no checkpoint listing ({r.stderr.strip()[-200:]})")
                continue
            arts = json.loads(files[0].read_text()).get("checkpoint_artifacts", [])
        n_new = 0
        for a in arts:
            rel = a["relative_file_name"]
            if rel.startswith("results/") and (ckpt or not rel.endswith(".pt")):
                dst = root / rel[len("results/"):]
            elif rel.startswith("bundles/"):
                dst = bundles / rel[len("bundles/"):]
            else:
                continue
            if dst.exists() and dst.stat().st_size == a.get("size_bytes"):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(a["url"], timeout=300) as resp, open(dst, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            n_new += 1
        print(f"{jid}: {n_new} file(s) downloaded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["rnn", "baseten"])
    ap.add_argument("--root", default=default_results())
    ap.add_argument("--job", action="append", help="Baseten job id (repeatable; default: all jobs of the project)")
    ap.add_argument("--ckpt", action="store_true", help="also download checkpoints")
    a = ap.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    if a.source == "rnn":
        fetch_rnn(root, a.ckpt)
    else:
        fetch_baseten(root, a.ckpt, a.job)


if __name__ == "__main__":
    main()
