"""VideoCUA (ServiceNow/VideoCUA): human screen recordings with a timestamped input log, one zip per application.

How raw log entries become steps: docs/DATA_PREP.md.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

from .common import TERMINATE, ConvertedImage, SourceWriter, call, image_stats, is_key_combination, key_tokens, unit

DATASET = "videocua"
REPOSITORY = "ServiceNow/VideoCUA"
SECONDS_BEFORE_ACTION = 0.1  # a step's screenshot is the video frame this long before its first log entry
END_TYPES = {"AFTER_LAST_ACTION", "TERMINATE", "TERMINATE_SUCCESS"}
BUTTONS = {"left", "right", "middle"}
SEEKS_PER_COMMAND = 16


@dataclass
class Step:
    timestamp: float | None  # None: the last frame of the video
    calls: list[str] = field(default_factory=list)


def download_patterns(raw_directory: Path) -> list[list[str]]:
    return [["raw_data/*"]]


def convert(raw_directory: Path, workers: int) -> None:
    jobs = [
        (str(archive_path), task_directory)
        for archive_path in sorted((raw_directory / "raw_data").glob("*.zip"))
        for task_directory in task_directories(archive_path)
    ]
    writer = SourceWriter(DATASET)
    failures: dict[str, int] = {}
    with Pool(workers) as pool:
        for done, (record, images, failure) in enumerate(pool.imap(convert_task, jobs, chunksize=2)):
            if failure:
                failures[failure] = failures.get(failure, 0) + 1
            else:
                for image in images:
                    writer.add_image(image)
                writer.add_session(record)
            if done % 500 == 0:
                print(f"{done}/{len(jobs)} tasks", flush=True)
    writer.close({"tasks": len(jobs), "failures": failures})
    print(f"{writer.n_sessions} sessions; failures: {failures}", flush=True)


def task_directories(archive_path: Path) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        return sorted(
            name[: -len("action_log.json")] for name in archive.namelist() if name.endswith("action_log.json")
        )


def convert_task(job: tuple[str, str]) -> tuple[dict | None, list[ConvertedImage], str | None]:
    archive_path, task_directory = job
    application = Path(archive_path).stem
    try:
        with zipfile.ZipFile(archive_path) as archive:
            log = json.loads(archive.read(task_directory + "action_log.json"))
            video_name = next(
                name
                for name in archive.namelist()
                if name.startswith(task_directory + "video/") and name.endswith(".mp4")
            )
            with tempfile.TemporaryDirectory() as scratch:
                video_path = Path(scratch) / "video.mp4"
                with archive.open(video_name) as source, open(video_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                width, height, frame_rate = video_properties(video_path)
                steps, unsupported = merge_actions(log["action_log"], width, height)
                if not steps:
                    return None, [], "no_actions"
                frames = extract_frames(video_path, [step.timestamp for step in steps], frame_rate, Path(scratch))
    except StopIteration:
        return None, [], "no_video"
    except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError):
        return None, [], "unreadable"

    session = f"{DATASET}-{slug(application)}-{log['task_id']}"
    images = [
        ConvertedImage(f"{session}-{index:03d}.jpg", data, image_stats(data)) for index, data in enumerate(frames)
    ]
    record = {
        "task_id": session,
        "instruction": with_application(str(log.get("task_instruction") or "").strip(), application),
        "traj": [
            {"index": index, "image": image.name, "value": {"code": "\n".join(step.calls)}}
            for index, (step, image) in enumerate(zip(steps, images))
        ],
        "meta": {
            "applications": [application],
            "origin": "human",
            "screen_width": width,
            "screen_height": height,
            "original_instruction": log.get("task_instruction"),
            "unsupported_actions": unsupported,
        },
    }
    return record, images, None


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name)


def with_application(instruction: str, application: str) -> str:
    """VideoCUA instructions rarely name their application; AgentNet's usually do."""
    if not instruction or application.lower() in instruction.lower():
        return instruction
    first_word = instruction.split()[0]
    if first_word[1:].islower():
        instruction = instruction[0].lower() + instruction[1:]
    return f"In {application}, {instruction}"


# ------------------------------------------------------------------------------------------------ log -> steps
def merge_actions(log: list[dict], width: int, height: int) -> tuple[list[Step], int]:
    """Steps as AgentNet records them (a click is one step, not a move and a click), and the entries not understood.

    The last step is always the end of the recording, without actions."""
    steps: list[Step] = []
    unsupported = 0
    position = 0

    def kind(offset: int = 0) -> str | None:
        index = position + offset
        return str(log[index].get("action_type", "")).upper() if index < len(log) else None

    def parameters(offset: int = 0) -> dict:
        return log[position + offset].get("action_params") or {}

    def point(values: dict, x_name: str = "x", y_name: str = "y") -> dict | None:
        if values.get(x_name) is None or values.get(y_name) is None:
            return None
        return {"x": unit(values[x_name], width), "y": unit(values[y_name], height)}

    while position < len(log):
        action, values = kind(), parameters()
        step = Step(float(log[position].get("timestamp") or 0.0))
        consumed = 1
        if action in END_TYPES:
            break
        if action == "MOVE_TO":
            target = point(values)
            if target is None:
                position += 1
                continue
            if kind(1) == "CLICK" and point(parameters(1)):
                step.calls.append(click_call(parameters(1), point(parameters(1))))
                consumed = 2
            else:
                step.calls.append(call("moveTo", **target))
                drag_end = point(values, "xEnd", "yEnd")
                if drag_end:
                    step.calls.append(call("dragTo", **drag_end))
                elif kind(1) == "MOUSE_DOWN":
                    drag_calls, drag_length = drag_from(log, position + 1, point)
                    step.calls += drag_calls
                    consumed = 1 + drag_length
                elif kind(1) == "SCROLL":
                    step.calls.append(scroll_call(parameters(1)))
                    consumed = 2
        elif action == "CLICK":
            step.calls.append(click_call(values, point(values)))
        elif action == "MOUSE_DOWN":
            step.calls, consumed = drag_from(log, position, point)
        elif action == "DRAG_TO":
            target = point(values)
            if target:
                step.calls.append(call("dragTo", **target))
        elif action == "SCROLL":
            step.calls.append(scroll_call(values))
        elif action in ("TYPING", "TYPE", "TEXT"):
            step.calls.append(call("write", str(values.get("text") or "")))
            if kind(1) == "PRESS":
                step.calls += key_calls(parameters(1))
                consumed = 2
        elif action in ("PRESS", "HOTKEY"):
            step.calls += key_calls(values)
        elif action in ("KEY_DOWN", "KEY_UP"):
            keys = key_tokens(str(values.get("key") or values.get("text") or ""))
            released_at_once = action == "KEY_DOWN" and kind(1) == "KEY_UP" and parameters(1) == values
            if released_at_once:
                step.calls += key_calls(values)
                consumed = 2
            elif is_key_combination(keys):
                step.calls += [call("keyDown" if action == "KEY_DOWN" else "keyUp", key) for key in keys]
        elif action != "MOUSE_UP":
            unsupported += 1
        if step.calls:
            steps.append(step)
        position += consumed
    if steps:
        steps.append(Step(None, [TERMINATE]))
    return steps, unsupported


def click_call(values: dict, target: dict | None) -> str:
    button = str(values.get("text") or "left").lower()
    clicks = int(values.get("numClicks") or 1)
    arguments = dict(target or {})
    if button in BUTTONS and button != "left":
        return call("rightClick" if button == "right" else "middleClick", **arguments)
    if clicks >= 3:
        return call("tripleClick", **arguments)
    return call("doubleClick" if clicks == 2 else "click", **arguments)


def drag_from(log: list[dict], start: int, point) -> tuple[list[str], int]:
    """MOUSE_DOWN, DRAG_TO ..., MOUSE_UP -> one dragTo to the last position (a click if the pointer never moved)."""
    calls: list[str] = []
    press_point = point(log[start].get("action_params") or {})
    if press_point:
        calls.append(call("moveTo", **press_point))
    end, index = None, start + 1
    while index < len(log) and str(log[index].get("action_type", "")).upper() == "DRAG_TO":
        end = point(log[index].get("action_params") or {}) or end
        index += 1
    if index < len(log) and str(log[index].get("action_type", "")).upper() == "MOUSE_UP":
        index += 1
    calls.append(call("dragTo", **end) if end else call("click"))
    return calls, index - start


def scroll_call(values: dict) -> str:
    """The log counts scrolling down as positive, pyautogui as negative."""
    amount = values.get("scrollY", values.get("scroll", 0)) or 0
    return call("scroll", -amount)


def key_calls(values: dict) -> list[str]:
    text = str(values.get("key") or values.get("text") or "")
    keys = key_tokens(text)
    if not keys:
        return []
    if not is_key_combination(keys):
        return [call("write", text)]
    return [call("press", keys[0]) if len(keys) == 1 else call("hotkey", *keys)]


# ------------------------------------------------------------------------------------------------------ frames
def video_properties(video_path: Path) -> tuple[int, int, float]:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate"]
        + ["-of", "json", str(video_path)],
        check=True,
        capture_output=True,
    ).stdout
    stream = json.loads(output)["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return int(stream["width"]), int(stream["height"]), float(numerator) / float(denominator)


def extract_frames(video_path: Path, timestamps: list[float | None], frame_rate: float, scratch: Path) -> list[bytes]:
    """One JPEG per timestamp. A timestamp past the end of the video, and None, give the video's last frame.

    Each frame is reached by seeking (the recordings have a key frame every second), not by decoding the video."""
    last_path = scratch / "last.jpg"
    run_ffmpeg("-sseof", "-1", "-i", str(video_path), "-update", "1", "-q:v", "2", str(last_path))
    if not last_path.exists():
        run_ffmpeg("-i", str(video_path), "-update", "1", "-q:v", "2", str(last_path))
    last_frame = last_path.read_bytes()

    numbers = [
        None if time is None else int(max(0.0, time - SECONDS_BEFORE_ACTION) * frame_rate) for time in timestamps
    ]
    wanted = sorted({number for number in numbers if number is not None})
    frame_of = {}
    for start in range(0, len(wanted), SEEKS_PER_COMMAND):
        chunk = wanted[start : start + SEEKS_PER_COMMAND]
        inputs, outputs = [], []
        for position, number in enumerate(chunk):
            # seeking stops at the first frame at or after the time: half a frame early lands on frame `number`
            inputs += ["-threads", "1", "-ss", f"{max(0.0, (number - 0.5) / frame_rate):.4f}", "-i", str(video_path)]
            outputs += ["-map", f"{position}:v:0", "-frames:v", "1", "-q:v", "2", str(scratch / f"f{number}.jpg")]
        run_ffmpeg(*inputs, *outputs, check=False)
        for number in chunk:
            frame_path = scratch / f"f{number}.jpg"
            if frame_path.exists() and frame_path.stat().st_size:
                frame_of[number] = frame_path.read_bytes()
    return [frame_of.get(number, last_frame) for number in numbers]


def run_ffmpeg(*arguments: str, check: bool = True) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *arguments], check=check, capture_output=True)
