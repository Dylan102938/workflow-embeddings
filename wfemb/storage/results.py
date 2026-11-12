"""The results tree: each run's raw data in <root>/<one folder per id segment>/ (layout: docs/EXPERIMENTS.md).

python -m wfemb.storage.results list       # every run: status, last step, id
python -m wfemb.storage.results migrate    # move runs to the folders their config.json implies; delete obsolete ones
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Iterator
from pathlib import Path

import chz

from ..config import Experiment, default_results_root


def json_safe(value):
    """NaN / inf become None (JSON.parse in browsers rejects them); numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def to_json(value, indent: int | None = None) -> str:
    return json.dumps(json_safe(value), indent=indent, allow_nan=False)


class RunWriter:
    """Writes one run's files. Starting a writer discards the rows of an earlier, unfinished attempt."""

    FLUSH_EVERY = 50

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("steps.jsonl", "evals.jsonl", "final.json"):
            (self.run_dir / name).unlink(missing_ok=True)
        self._steps_file = open(self.run_dir / "steps.jsonl", "a")
        self._unflushed = 0

    def write_config(self, config: dict):
        (self.run_dir / "config.json").write_text(to_json(config, indent=1))

    def write_step(self, row: dict):
        self._steps_file.write(to_json(row) + "\n")
        self._unflushed += 1
        if self._unflushed >= self.FLUSH_EVERY:
            self._flush()

    def write_eval(self, row: dict):
        self._flush()
        with open(self.run_dir / "evals.jsonl", "a") as file:
            file.write(to_json(row) + "\n")

    def write_final(self, metrics: dict):
        self._flush()
        (self.run_dir / "final.json").write_text(to_json(metrics, indent=1))

    def close(self):
        self._steps_file.close()

    def _flush(self):
        self._steps_file.flush()
        self._unflushed = 0


def iter_runs(root: str | Path) -> Iterator[tuple[Experiment, Path]]:
    """(experiment, run folder) for every folder under root that holds a config.json of a run that still exists."""
    for config_path in sorted(Path(root).rglob("config.json")):
        experiment = Experiment.from_saved(json.loads(config_path.read_text())["experiment"])
        if experiment is not None:
            yield experiment, config_path.parent


def obsolete_runs(root: str | Path) -> list[Path]:
    """Folders of runs that used a variable that was removed since."""
    config_paths = sorted(Path(root).rglob("config.json"))
    return [
        path.parent
        for path in config_paths
        if Experiment.from_saved(json.loads(path.read_text())["experiment"]) is None
    ]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # blank, or the row being written right now
                pass
    return rows


def migrate(root: Path, dry_run: bool = False):
    """After an experiment variable is added, renamed or removed: move every run to the folder its config.json
    implies, and delete the runs that used a removed variable."""
    for run_dir in obsolete_runs(root):
        print(f"{run_dir.relative_to(root)}: used a removed variable, {'to delete' if dry_run else 'deleted'}")
        if not dry_run:
            shutil.rmtree(run_dir)
    n_moved = 0
    for experiment, run_dir in list(iter_runs(root)):
        target = Path(root) / experiment.relative_dir
        if target.resolve() == run_dir.resolve():
            continue
        print(f"{run_dir.relative_to(root)} -> {target.relative_to(root)}")
        n_moved += 1
        if dry_run:
            continue
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(target))
        config = json.loads((target / "config.json").read_text())
        config["id"], config["experiment"] = experiment.id, chz.asdict(experiment)
        (target / "config.json").write_text(to_json(config, indent=1))
    print(f"{n_moved} run(s) {'to move' if dry_run else 'moved'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["list", "migrate"])
    parser.add_argument("--root", default=default_results_root())
    parser.add_argument("--dry_run", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "migrate":
        migrate(Path(arguments.root), arguments.dry_run)
        return
    for experiment, run_dir in iter_runs(arguments.root):
        steps = read_jsonl(run_dir / "steps.jsonl")
        status = "done" if (run_dir / "final.json").exists() else "unfinished"
        print(f"{status:10s} step {steps[-1]['step'] if steps else 0:>6}  {experiment.id}")


if __name__ == "__main__":
    main()
