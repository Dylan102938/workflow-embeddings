"""The run queue, jobs.txt: run ids grouped into named sections (format: docs/EXPERIMENTS.md).

python -m wfemb.storage.jobs list              # every section's runs and their status
python -m wfemb.storage.jobs runs <section>    # "<run id> [overrides]" lines, read by the launchers
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import REPO, Experiment, default_results_root

JOBS_FILE = Path(os.environ.get("WFEMB_JOBS", REPO / "jobs.txt"))
SECTION_HEADER = re.compile(r"^\[([A-Za-z0-9_.-]+)\]$")


@dataclass
class QueuedRun:
    run_id: str
    overrides: list[str]


@dataclass
class Section:
    name: str
    notes: list[str] = field(default_factory=list)
    runs: list[QueuedRun] = field(default_factory=list)


def parse_jobs(path: Path = JOBS_FILE) -> dict[str, Section]:
    """Sections in file order. Raises ValueError naming the line of a bad id or a run outside a section."""
    sections: dict[str, Section] = {}
    current = None
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        content, _, comment = line.partition("#")
        content, comment = content.strip(), comment.strip()
        where = f"{path}:{line_number}"
        if header := SECTION_HEADER.match(content):
            name = header.group(1)
            if name in sections:
                raise ValueError(f"{where}: section [{name}] defined twice")
            current = sections[name] = Section(name)
        elif not content:
            if current is not None and comment:
                current.notes.append(comment)
        elif current is None:
            raise ValueError(f"{where}: run outside a section")
        else:
            run_id, *overrides = content.split()
            try:
                run_id = Experiment.from_id(run_id).id
            except Exception as error:
                raise ValueError(f"{where}: {error}") from None
            if any("=" not in override for override in overrides):
                raise ValueError(f"{where}: overrides must be key=value")
            current.runs.append(QueuedRun(run_id, overrides))
    return sections


def run_status(run_id: str, results_root: Path) -> str:
    run_dir = results_root / Experiment.from_id(run_id).relative_dir
    if (run_dir / "final.json").exists():
        return "done"
    return "started" if (run_dir / "config.json").exists() else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["list", "runs"])
    parser.add_argument("section", nargs="?")
    parser.add_argument("--file", default=str(JOBS_FILE))
    arguments = parser.parse_args()
    sections = parse_jobs(Path(arguments.file))
    if arguments.command == "runs":
        if arguments.section not in sections:
            sys.exit(f"no section [{arguments.section}] in {arguments.file} (have: {', '.join(sections)})")
        for run in sections[arguments.section].runs:
            print(" ".join([run.run_id, *run.overrides]))
        return
    results_root = Path(default_results_root())
    for section in sections.values():
        print(f"[{section.name}]")
        for run in section.runs:
            print(f"  {run_status(run.run_id, results_root):8s} {run.run_id} {' '.join(run.overrides)}")


if __name__ == "__main__":
    main()
