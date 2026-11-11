"""Copies run results from rnn or Baseten into the local results tree.

  python -m wfemb.storage.fetch rnn
  python -m wfemb.storage.fetch baseten [--job <job id>]
Options: --embeddings (test_embeddings.npz), --checkpoints (*.pt), --screenshots (Baseten demo screenshot cache).
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

from ..config import REPO, Experiment, default_results_root
from . import paths

RNN_RESULTS = "rnn:/nas/ucb/dfeng/data/workflow-embeddings/results/"
BASETEN_PROJECT = "wfemb"
BUNDLES_DIR = REPO / "dashboard" / "bundles"
SCREENSHOTS_ARCHIVE = "demo_screens.tar"


def fetch_rnn(results_root: Path, checkpoints: bool, embeddings: bool):
    patterns = ["*/", "*.json", "*.jsonl"]
    if checkpoints:
        patterns.append("*.pt")
    if embeddings:
        patterns.append("*_embeddings.npz")
    command = [
        "rsync",
        "-a",
        "--prune-empty-dirs",
        *[f"--include={pattern}" for pattern in patterns],
        "--exclude=*",
        RNN_RESULTS,
        f"{results_root}/",
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def baseten_api(path: str):
    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        truss_config = configparser.ConfigParser()
        truss_config.read(Path.home() / ".trussrc")
        api_key = truss_config["baseten"]["api_key"]
    request = urllib.request.Request(
        f"https://api.baseten.co/v1{path}", headers={"Authorization": f"Api-Key {api_key}"}
    )
    return json.loads(urllib.request.urlopen(request, timeout=60).read())


def baseten_jobs() -> list[dict]:
    projects = baseten_api("/training_projects")["training_projects"]
    project_id = next(project["id"] for project in projects if project["name"] == BASETEN_PROJECT)
    return baseten_api(f"/training_projects/{project_id}/jobs")["training_jobs"]


def baseten_job_files(job_id: str) -> list[dict]:
    """Files a job left in its checkpoint directory: {"relative_file_name", "url", "size_bytes"} each."""
    with tempfile.TemporaryDirectory() as workdir:
        listing = subprocess.run(
            ["truss", "train", "get_checkpoint_urls", "--job-id", job_id], cwd=workdir, capture_output=True, text=True
        )
        written = list(Path(workdir).glob("*.json"))
        if listing.returncode or not written:
            print(f"{job_id}: no file listing ({listing.stderr.strip()[-200:]})")
            return []
        return json.loads(written[0].read_text()).get("checkpoint_artifacts", [])


def download(url: str, destination: Path, expected_size: int | None = None, max_attempts: int = 10):
    """Downloads to destination. With expected_size, a connection that closes early is resumed where it stopped."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if expected_size is None:
        with urllib.request.urlopen(url, timeout=300) as response, open(destination, "wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
        return
    destination.unlink(missing_ok=True)
    for _ in range(max_attempts):
        done = destination.stat().st_size if destination.exists() else 0
        if done >= expected_size:
            break
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(request, timeout=300) as response, open(destination, "ab") as file:
                while chunk := response.read(1 << 20):
                    file.write(chunk)
        except OSError as error:
            print(f"{destination.name}: {error}; resuming at byte {destination.stat().st_size}", flush=True)
    if destination.stat().st_size != expected_size:
        raise OSError(
            f"{destination}: {destination.stat().st_size} of {expected_size} bytes after {max_attempts} attempts"
        )


def current_run_path(path: Path) -> Path:
    """A file of a run stored before an experiment variable was added: its place in today's results tree."""
    return Experiment.from_id(str(path.parent)).relative_dir / path.name


def fetch_baseten(
    results_root: Path, job_ids: list[str] | None, checkpoints: bool, embeddings: bool, screenshots: bool
):
    if not job_ids:
        job_ids = [job["id"] for job in baseten_jobs() if job["current_status"].endswith(("COMPLETED", "RUNNING"))]
    demo_dir = paths.DATA / "demo"

    def local_path(name: str) -> Path | None:
        if name.startswith("results/"):
            wanted = (checkpoints or not name.endswith(".pt")) and (embeddings or not name.endswith(".npz"))
            return results_root / current_run_path(Path(name.removeprefix("results/"))) if wanted else None
        if name.startswith("bundles/"):
            return BUNDLES_DIR / name.removeprefix("bundles/")
        if name == SCREENSHOTS_ARCHIVE and screenshots:
            return demo_dir / name
        return None

    for job_id in job_ids:
        n_downloaded = 0
        for file in baseten_job_files(job_id):
            destination = local_path(file["relative_file_name"])
            if destination is None:
                continue
            if destination.exists() and destination.stat().st_size == file.get("size_bytes"):
                continue
            download(file["url"], destination, file.get("size_bytes"))
            n_downloaded += 1
        print(f"{job_id}: {n_downloaded} file(s) downloaded")

    archive = demo_dir / SCREENSHOTS_ARCHIVE
    if archive.exists():
        subprocess.run(["tar", "xf", str(archive), "-C", str(demo_dir)], check=True)
        archive.unlink()
        print(f"screenshots -> {demo_dir / 'screens'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["rnn", "baseten"])
    parser.add_argument("--root", default=default_results_root())
    parser.add_argument("--job", action="append", help="Baseten job id (repeatable; default: every job)")
    parser.add_argument("--checkpoints", action="store_true")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--screenshots", action="store_true")
    arguments = parser.parse_args()
    results_root = Path(arguments.root)
    results_root.mkdir(parents=True, exist_ok=True)
    if arguments.source == "rnn":
        fetch_rnn(results_root, arguments.checkpoints, arguments.embeddings)
    else:
        fetch_baseten(results_root, arguments.job, arguments.checkpoints, arguments.embeddings, arguments.screenshots)


if __name__ == "__main__":
    main()
