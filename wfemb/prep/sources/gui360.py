"""GUI-360 (vyokky/GUI-360): an agent's successful Word, Excel and PowerPoint trajectories on Windows.

Its train and test folders are both used. How its GUI and API actions become steps: docs/DATA_PREP.md.
"""

from __future__ import annotations

import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from .common import TERMINATE, SourceWriter, call, copy_images, unit

DATASET = "gui360"
REPOSITORY = "vyokky/GUI-360"
SPLITS = ("train", "test")
APPLICATIONS = {"word": "Microsoft Word", "excel": "Microsoft Excel", "ppt": "Microsoft PowerPoint"}
FALLBACK_SIZE = (1040, 736)
API_CALL = "computer.api()"  # an action through the application's API: no pointer or keyboard input to record


def download_patterns(raw_directory: Path) -> list[list[str]]:
    return [[f"{split}/*" for split in SPLITS]]


def convert(raw_directory: Path, workers: int) -> None:
    writer = SourceWriter(DATASET)
    failures: Counter = Counter()
    sessions = []
    with Pool(workers) as pool:
        for split in SPLITS:
            trajectory_paths = sorted((raw_directory / split / "data").glob("*/*/success/*.jsonl"))
            split_sessions = []
            for session, failure in pool.imap(read_session, trajectory_paths, chunksize=16):
                if failure:
                    failures[failure] += 1
                else:
                    split_sessions.append(session)
            wanted = {step["member"]: step["image"] for session in split_sessions for step in session["steps"]}
            found = copy_images([raw_directory / split / "image.tar.gz"], wanted, writer, pool)
            print(f"{split}: {found}/{len(wanted)} screenshots of {len(split_sessions)} sessions", flush=True)
            sessions += split_sessions
    for session in sessions:
        writer.add_session(record_of(session, writer))
    writer.close({"failures": dict(failures)})
    print(f"{writer.n_sessions} sessions; {dict(failures)}", flush=True)


def read_session(trajectory_path: Path) -> tuple[dict | None, str | None]:
    """The fields of a trajectory file that the conversion needs (each line also holds the whole UI tree)."""
    application, category = trajectory_path.parts[-4], trajectory_path.parts[-3]
    with open(trajectory_path) as f:
        rows = sorted((json.loads(line) for line in f), key=lambda row: row["step_id"])
    if not rows:
        return None, "empty"
    if str(rows[0]["evaluation"].get("complete")).lower() != "yes":
        return None, "task_not_completed"
    session_id = f"{DATASET}-{category}-{rows[0]['execution_id']}"
    steps = []
    for row in rows:
        screenshot = row["step"]["screenshot_clean"]
        steps.append(
            {
                "member": f"image/{application}/{category}/{screenshot}",
                "image": f"{session_id}-{Path(screenshot).name}",
                "action": {
                    name: row["step"]["action"].get(name)
                    for name in ("function", "args", "coordinate_x", "coordinate_y")
                },
            }
        )
    return {
        "id": session_id,
        "instruction": rows[0]["request"],
        "application": APPLICATIONS[application],
        "steps": steps,
    }, None


def record_of(session: dict, writer: SourceWriter) -> dict:
    sizes = [writer.image_size(step["image"]) or FALLBACK_SIZE for step in session["steps"]]
    return {
        "task_id": session["id"],
        "instruction": session["instruction"],
        "traj": [
            {"index": index, "image": step["image"], "value": {"code": "\n".join(action_calls(step["action"], size))}}
            for index, (step, size) in enumerate(zip(session["steps"], sizes))
        ],
        "meta": {
            "applications": [session["application"]],
            "system": "Windows",
            "origin": "agent",
            "screen_width": sizes[0][0],
            "screen_height": sizes[0][1],
        },
    }


def action_calls(action: dict, size: tuple[int, int]) -> list[str]:
    """Positions are pixels of the step's own screenshot (the application window)."""
    function, args = action.get("function") or "", action.get("args") or {}
    width, height = size

    def point(x, y) -> dict:
        return {} if x is None or y is None else {"x": unit(x, width), "y": unit(y, height)}

    target = point(action.get("coordinate_x"), action.get("coordinate_y"))
    if function == "click":
        if str(args.get("button") or "left").lower() == "right":
            return [call("rightClick", **target)]
        return [call("doubleClick" if args.get("double") else "click", **target)]
    if function == "type":
        calls = [call("click", **target)] if target else []
        return calls + ([call("write", str(args["text"]))] if args.get("text") else [])
    if function == "drag":
        start, end = point(args.get("start_x"), args.get("start_y")), point(args.get("end_x"), args.get("end_y"))
        return [call("moveTo", **start), call("dragTo", **end)] if start and end else []
    if function == "wheel_mouse_input":
        return [call("scroll", clicks=args.get("wheel_dist") or 0, **target)]
    if function in ("", "summary"):
        return [TERMINATE]
    return [API_CALL]
