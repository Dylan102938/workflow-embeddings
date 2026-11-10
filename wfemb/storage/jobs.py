"""The run queue: jobs.txt at the repo root, run ids grouped into named sections (format in its header).

  python -m wfemb.storage.jobs list                # every section: its runs and their status in the results tree
  python -m wfemb.storage.jobs runs <section>      # the section's "<id> [overrides]" lines (used by the launchers)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import REPO, Experiment, default_results

JOBS = Path(os.environ.get("WFEMB_JOBS", REPO / "jobs.txt"))       # WFEMB_JOBS: another queue file (tests)
SECTION = re.compile(r"^\[([A-Za-z0-9_.-]+)\]$")


@dataclass
class Section:
    name: str
    notes: list[str] = field(default_factory=list)          # comment lines
    runs: list[tuple[str, list[str]]] = field(default_factory=list)   # (canonical id, overrides)


def parse(path: Path = JOBS) -> dict[str, Section]:
    """Sections in file order. Raises ValueError (with the line number) on a bad id or a run outside a section."""
    sections: dict[str, Section] = {}
    cur = None
    for ln, raw in enumerate(Path(path).read_text().splitlines(), 1):
        body, _, comment = raw.partition("#")
        body = body.strip()
        if m := SECTION.match(body):
            if m.group(1) in sections:
                raise ValueError(f"{path}:{ln}: section [{m.group(1)}] defined twice")
            cur = sections[m.group(1)] = Section(m.group(1))
            continue
        if not body:
            if cur is not None and comment.strip():
                cur.notes.append(comment.strip())
            continue
        if cur is None:
            raise ValueError(f"{path}:{ln}: run outside a section")
        rid, *overrides = body.split()
        try:
            exp = Experiment.from_id(rid)
        except Exception as e:
            raise ValueError(f"{path}:{ln}: {e}") from None
        if bad := [o for o in overrides if "=" not in o]:
            raise ValueError(f"{path}:{ln}: overrides need key=value: {bad}")
        cur.runs.append((exp.id, overrides))
    return sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "runs"])
    ap.add_argument("section", nargs="?")
    ap.add_argument("--file", default=str(JOBS))
    a = ap.parse_args()
    sections = parse(Path(a.file))
    if a.cmd == "runs":
        if a.section not in sections:
            sys.exit(f"no section [{a.section}] in {a.file} (have: {', '.join(sections)})")
        for rid, ov in sections[a.section].runs:
            print(" ".join([rid, *ov]))
        return
    root = Path(default_results())
    for s in sections.values():
        print(f"[{s.name}]")
        for rid, ov in s.runs:
            d = root / Experiment.from_id(rid).path
            st = "done" if (d / "final.json").exists() else "started" if (d / "config.json").exists() else "-"
            print(f"  {st:8s} {rid} {' '.join(ov)}")


if __name__ == "__main__":
    main()
