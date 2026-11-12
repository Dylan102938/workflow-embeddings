"""ScaleCUA-Data (OpenGVLab/ScaleCUA-Data): the human-collected desktop and web trajectories.

The release has one annotation row per (instruction wording, step); how sessions are rebuilt: docs/DATA_PREP.md.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

from .common import MODIFIERS, NAMED_KEYS, TERMINATE, SourceWriter, call, copy_images, key_name

DATASET = "scalecua"
REPOSITORY = "OpenGVLab/ScaleCUA-Data"
ANNOTATION_NAME = re.compile(r"data_(\d{8})_(mac|ubuntu|windows|web)_navigation_\d+_boost_instruction\.jsonl")
TASK = re.compile(r"Task: (.*?)\n\nPrevious operations:(.*)", re.S)
PREVIOUS_STEP = re.compile(r"^Step \d+:", re.M)
ACTION = re.compile(r"<action>\n(.*?)\n</action>", re.S)
SYSTEMS = {"mac": "Darwin", "ubuntu": "Ubuntu", "windows": "Windows", "web": "Web"}
SWIPE_SCROLL_UNITS = 10  # a swipe of the full screen height counts as this many scroll clicks
END_CALLS = {"terminate", "response", "success"}
KEY_CALLS = {"press", "keyDown", "keyUp"}

APPLICATIONS = {
    "vs_code": "Visual Studio Code",
    "chrome": "Google Chrome",
    "vlc": "VLC media player",
    "os_windows": "Microsoft Windows",
    "os": "Ubuntu",
    "desktop": "Ubuntu",
    "map": "Maps",
    "email": "Mail",
    "photoshop": "Adobe Photoshop",
    "excel": "Microsoft Excel",
}
PLATFORM_APPLICATIONS = {("mac", "settings"): "System Settings", ("windows", "settings"): "Windows Settings"}
WEBSITES = {
    "docs_google_com": "Google Docs",
    "mail_google_com": "Gmail",
    "github_com": "GitHub",
    "openstreetmap_org": "OpenStreetMap",
}


def download_patterns(raw_directory: Path) -> Iterator[list[str]]:
    """The annotations first: their names say which screenshot archives belong to the human batches."""
    yield ["annotations/*_navigation_*_boost_instruction.jsonl"]
    yield [
        f"data/data_{match.group(1)}/{match.group(2)}.tar.gz.part-*"
        for match in map(
            ANNOTATION_NAME.fullmatch, sorted(path.name for path in (raw_directory / "annotations").glob("*"))
        )
        if match
    ]


def convert(raw_directory: Path, workers: int) -> None:
    sessions, failures = [], Counter()
    for annotation_path in sorted((raw_directory / "annotations").glob("*.jsonl")):
        match = ANNOTATION_NAME.fullmatch(annotation_path.name)
        if match:
            sessions += read_sessions(annotation_path, *match.groups(), failures)
    print(f"{len(sessions)} annotated sessions; {dict(failures)}", flush=True)

    writer = SourceWriter(DATASET)
    released = []
    with Pool(workers) as pool:
        for date, platform in sorted({(session["date"], session["platform"]) for session in sessions}):
            members = [session for session in sessions if (session["date"], session["platform"]) == (date, platform)]
            parts = sorted((raw_directory / "data" / f"data_{date}").glob(f"{platform}.tar.gz.part-*"))
            if not parts:
                failures["archive_not_released"] += len(members)
                continue
            wanted = {step["member"]: step["image"] for session in members for step in session["steps"]}
            found = copy_images(parts, wanted, writer, pool)
            released += members
            print(f"{date} {platform}: {found}/{len(wanted)} screenshots of {len(members)} sessions", flush=True)
    for session in released:
        writer.add_session(record_of(session))
    writer.close({"failures": dict(failures)})
    print(f"{writer.n_sessions} sessions; {dict(failures)}", flush=True)


# ------------------------------------------------------------------------------------------------- annotations
def read_sessions(annotation_path: Path, date: str, platform: str, failures: Counter) -> list[dict]:
    """One session per screenshot folder. Its first wording is the collector's instruction; the others are rewrites."""
    rows_of_folder: dict[str, list[dict]] = {}
    with open(annotation_path) as f:
        for line in f:
            row = json.loads(line)
            rows_of_folder.setdefault(row["image"].rsplit("/", 1)[0], []).append(row)

    sessions = []
    for folder, rows in rows_of_folder.items():
        wordings: dict[str, None] = {}
        steps: dict[int, dict] = {}
        for row in rows:
            task, previous = TASK.search(row["conversations"][0]["value"]).groups()
            wordings.setdefault(task.strip())
            action = ACTION.search(row["conversations"][1]["value"])
            if task.strip() != next(iter(wordings)) or action is None:
                continue
            index = len(PREVIOUS_STEP.findall(previous))
            steps.setdefault(
                index, {"path": row["image"], "code": action.group(1), "size": (row["width"], row["height"])}
            )
        if sorted(steps) != list(range(len(steps))):
            failures["missing_steps"] += 1
            continue
        session = session_of(date, platform, folder, list(wordings), [steps[index] for index in range(len(steps))])
        if session is None:
            failures["marked_failed"] += 1
        else:
            sessions.append(session)
    return sessions


def session_of(date: str, platform: str, folder: str, wordings: list[str], raw_steps: list[dict]) -> dict | None:
    """None if the collector marked the attempt as failed. The session ends at its first terminating call."""
    session_id = f"{DATASET}-{date[4:]}-{platform}-{re.sub(r'[^A-Za-z0-9]+', '_', folder.removesuffix('/images'))}"
    steps = []
    for raw_step in raw_steps:
        lines, outcome = normalize_code(raw_step["code"], *raw_step["size"])
        if outcome == "failed":
            return None
        steps.append(
            {
                "member": f"{platform}/images/{raw_step['path']}",
                "image": f"{session_id}-{Path(raw_step['path']).name}",
                "code": "\n".join(lines),
            }
        )
        if outcome == "ended":
            break
    width, height = raw_steps[0]["size"]
    return {
        "id": session_id,
        "date": date,
        "platform": platform,
        "folder": folder,
        "instruction": wordings[0],
        "rewrites": wordings[1:],
        "steps": steps,
        "screen": (width, height),
    }


def record_of(session: dict) -> dict:
    applications, websites = applications_of(session["platform"], session["folder"])
    return {
        "task_id": session["id"],
        "instruction": session["instruction"],
        "traj": [
            {"index": index, "image": step["image"], "value": {"code": step["code"]}}
            for index, step in enumerate(session["steps"])
        ],
        "meta": {
            "applications": applications,
            "websites": websites,
            "system": SYSTEMS[session["platform"]],
            "origin": "human",
            "screen_width": session["screen"][0],
            "screen_height": session["screen"][1],
            "instruction_rewrites": session["rewrites"],
        },
    }


def applications_of(platform: str, folder: str) -> tuple[list[str], list[str]]:
    if platform != "web":
        name = folder.split("/")[0]
        application = PLATFORM_APPLICATIONS.get((platform, name)) or APPLICATIONS.get(name) or name.replace("_", " ")
        return [application], []
    parts = folder.split("/")
    host = re.sub(r"_\d+$", "", parts[1]) if len(parts) > 1 else ""
    is_named_site = bool(re.fullmatch(r"[a-z][a-z0-9_]*_[a-z]{2,}", host))
    return ["Google Chrome"], ([WEBSITES.get(host, host.replace("_", "."))] if is_named_site else [])


# ------------------------------------------------------------------------------------------------ action code
def normalize_code(code: str, width: int, height: int) -> tuple[list[str], str]:
    """The step's calls as AgentNet writes them, and "failed", "ended" or "continues"."""
    lines, outcome = [], "continues"
    for line in code.splitlines():
        parsed = parse_call(line)
        if parsed is None:
            lines.append(line)
            continue
        name, args, kwargs = parsed
        if name == "failure" or (name == "terminate" and str(kwargs.get("status", "success")).startswith("fail")):
            return [], "failed"
        if name in END_CALLS:
            lines.append(TERMINATE)
            outcome = "ended"
        elif name == "swipe":
            lines.append(swipe_call(kwargs))
        elif name == "hotkey":
            lines += hotkey_calls([key_name(str(key).lower()) for key in args])
        elif name in KEY_CALLS:
            keys = kwargs.get("keys") or kwargs.get("key") or (args[0] if args else None)
            lines.append(call(name, key_name(keys.lower()) if isinstance(keys, str) else keys))
        else:
            for axis, extent in (("x", width), ("y", height)):
                if isinstance(kwargs.get(axis), (int, float)) and kwargs[axis] > 1:
                    kwargs[axis] = round(kwargs[axis] / extent, 4)
            lines.append(call(name, *args, **kwargs))
    return lines, outcome


def parse_call(line: str) -> tuple[str, list, dict] | None:
    try:
        node = ast.parse(line.strip(), mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        name = node.func.id
        args = [ast.literal_eval(argument) for argument in node.args]
        kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
    except (SyntaxError, ValueError):
        return None
    return name[0].lower() + name[1:], args, kwargs


def swipe_call(kwargs: dict) -> str:
    """Swiping up moves the content up, which is scrolling down."""
    clicks = max(1, round(float(kwargs.get("amount") or 0) * SWIPE_SCROLL_UNITS))
    direction = str(kwargs.get("direction"))
    if direction in ("left", "right"):
        return call("hscroll", clicks if direction == "left" else -clicks)
    return call("scroll", -clicks if direction == "up" else clicks)


def hotkey_calls(keys: list[str]) -> list[str]:
    """The recorder logged typed characters and repeated keys as hotkeys: hotkey('1', '2', '8', '0') is typing."""
    if not keys:
        return []
    if any(key in MODIFIERS for key in keys):
        return [call("hotkey", *keys)]
    if all(len(key) == 1 for key in keys):
        return [call("write", "".join(keys))] if len(keys) > 1 else [call("press", keys[0])]
    if all(key in NAMED_KEYS for key in keys):
        return [call("press", key) for key in keys]
    return [call("hotkey", *keys)]
