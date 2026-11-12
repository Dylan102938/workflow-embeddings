"""What every converter writes: DATA/sources/<dataset>/{sessions.jsonl, images.zip, image_stats.json}.

Formats: docs/DATA_PREP.md.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import zipfile
from multiprocessing.pool import Pool
from pathlib import Path
from typing import BinaryIO, NamedTuple

import numpy as np
from PIL import Image

from ...storage import paths

Image.MAX_IMAGE_PIXELS = None
STATS_BATCH = 64

KEY_ALIASES = {
    "control": "ctrl",
    "command": "cmd",
    "⌘": "cmd",
    "option": "alt",
    "windows": "win",
    "super": "win",
    "meta": "win",
    "return": "enter",
    "escape": "esc",
    "del": "delete",
    "spacebar": "space",
    "space bar": "space",
    "comma": ",",
    "period": ".",
    "plus": "+",
    "minus": "-",
    "page up": "pageup",
    "page down": "pagedown",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "caps lock": "capslock",
    "prt sc": "print_screen",
    "prtsc": "print_screen",
    "caps_lock": "capslock",
    "print screen": "print_screen",
    "printscreen": "print_screen",
}
MODIFIERS = {"ctrl", "alt", "shift", "cmd", "win", "fn"}
NAMED_KEYS = MODIFIERS | {
    "enter",
    "esc",
    "tab",
    "space",
    "backspace",
    "delete",
    "insert",
    "home",
    "end",
    "pageup",
    "pagedown",
    "up",
    "down",
    "left",
    "right",
    "capslock",
    "print_screen",
    "num_lock",
    *(f"f{number}" for number in range(1, 25)),
}


class ConvertedImage(NamedTuple):
    name: str
    data: bytes
    stats: dict | None  # None: the image does not decode


class SourceWriter:
    """Collects one dataset's sessions and screenshots; nothing is visible under its final name until close()."""

    def __init__(self, dataset: str):
        self.directory = paths.SOURCES / dataset
        self.directory.mkdir(parents=True, exist_ok=True)
        self._archive = zipfile.ZipFile(self._partial("images.zip"), "w", zipfile.ZIP_STORED, allowZip64=True)
        self._records = open(self._partial("sessions.jsonl"), "w")
        self._image_stats: dict[str, dict | None] = {}
        self.n_sessions = 0

    def add_image(self, image: ConvertedImage) -> None:
        if image.name in self._image_stats:
            return
        self._image_stats[image.name] = image.stats
        if image.stats is not None:
            self._archive.writestr(image.name, image.data)

    def add_session(self, record: dict) -> None:
        self._records.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.n_sessions += 1

    def image_size(self, name: str) -> tuple[int, int] | None:
        stats = self._image_stats.get(name)
        return tuple(stats["size"]) if stats else None

    def close(self, conversion_stats: dict) -> None:
        self._archive.close()
        self._records.close()
        with open(self._partial("image_stats.json"), "w") as f:
            json.dump(self._image_stats, f)
        for name in ("images.zip", "sessions.jsonl", "image_stats.json"):
            self._partial(name).rename(self.directory / name)
        with open(self.directory / "conversion.json", "w") as f:
            json.dump({"sessions": self.n_sessions, "images": len(self._image_stats), **conversion_stats}, f, indent=1)

    def _partial(self, name: str):
        return self.directory / f"{name}.partial"


def image_stats(data: bytes) -> dict | None:
    """Size, a digest of the file and the grey-level spread of a screenshot; None if it does not decode."""
    try:
        image = Image.open(io.BytesIO(data))
        grey = image.convert("L")
    except Exception:
        return None
    return {
        "size": list(image.size),
        "digest": hashlib.md5(data).hexdigest(),
        "spread": round(float(np.asarray(grey.resize((64, 64), Image.BILINEAR)).std()), 2),
    }


def key_tokens(text: str) -> list[str]:
    """'Ctrl + Shift+I', '"Cmd" + "V"', 'Ctrl-A', 'Right arrow' -> lowercase key names as AgentNet writes them."""
    text = re.sub(r"[\"'`]", "", text).strip().lower()
    text = re.sub(r"^(ctrl|control|alt|shift|cmd|command|option|win)\s*-\s*(?=\S)", r"\1+", text)
    plus_key = text == "+" or text.replace(" ", "").endswith("++")
    keys = [key_name(part.strip()) for part in text.split("+") if part.strip()]
    return keys + ["+"] if plus_key else keys


def key_name(name: str) -> str:
    name = re.sub(r"\s+(arrow\s+)?key$", "", name)
    arrow = re.fullmatch(r"(?:arrow\s+)?(up|down|left|right)(?:\s+arrow)?", name)
    if arrow:
        return arrow.group(1)
    name = KEY_ALIASES.get(name, name)
    return name.replace(" ", "") if name.replace(" ", "") in NAMED_KEYS else name


def is_key_combination(keys: list[str]) -> bool:
    """False for free text that an annotator put into a key field."""
    return bool(keys) and all(key in NAMED_KEYS or len(key) == 1 for key in keys)


def call(name: str, *args, **kwargs) -> str:
    """One line of AgentNet action code: pyautogui.<name>(...), which wfemb.prep.actions parses."""
    arguments = [json.dumps(value, ensure_ascii=False) for value in args]
    arguments += [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in kwargs.items()]
    return f"pyautogui.{name}({', '.join(arguments)})"


TERMINATE = "computer.terminate(status='success')"


def unit(value: float, extent: float) -> float:
    return round(min(max(value / extent, 0.0), 1.0), 4)


# ------------------------------------------------------------------------------------------ split tar archives
class JoinedParts:
    """The parts of a split archive as one readable stream."""

    def __init__(self, parts: list[Path]):
        self._parts = iter(parts)
        self._current: BinaryIO | None = None

    def read(self, size: int = -1) -> bytes:
        while True:
            if self._current is None:
                part = next(self._parts, None)
                if part is None:
                    return b""
                self._current = open(part, "rb")
            data = self._current.read(size)
            if data:
                return data
            self._current.close()
            self._current = None


def copy_images(parts: list[Path], wanted: dict[str, str], writer: SourceWriter, pool: Pool) -> int:
    """Copies the wanted members (archive path -> image name) into the writer; returns how many were found."""
    found = 0
    batch: list[tuple[str, bytes]] = []

    def flush() -> None:
        for (name, data), stats in zip(batch, pool.map(image_stats, [data for _, data in batch])):
            writer.add_image(ConvertedImage(name, data, stats))
        batch.clear()

    with tarfile.open(fileobj=JoinedParts(parts), mode="r|gz") as archive:
        for member in archive:
            name = wanted.get(member.name)
            if name is None or not member.isfile():
                continue
            batch.append((name, archive.extractfile(member).read()))
            found += 1
            if len(batch) == STATS_BATCH:
                flush()
            if found == len(wanted):
                break
    flush()
    return found
