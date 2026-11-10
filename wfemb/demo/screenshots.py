"""Screenshots of AgentNet test sessions, as the JPEGs the viewer shows: from a cache folder or the split archives."""

from __future__ import annotations

import io
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from ..storage import paths
from .assets import CACHE_DIR, load_sessions

DEFAULT_CACHE = CACHE_DIR / "screens"  # <session id>/<step>.jpg for every test session
MAX_WIDTH = 1280
JPEG_QUALITY = 78
MAX_FAILURES_LISTED = 20
PROGRESS_EVERY = 200  # sessions


def to_jpeg(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    if image.width > MAX_WIDTH:
        image = image.resize((MAX_WIDTH, round(image.height * MAX_WIDTH / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


class Archives:
    """The AgentNet screenshot archives (paths.ZIPS), opened on first use. Safe to read from several threads."""

    def __init__(self):
        self._thread = threading.local()
        self._member_of: dict[str, dict[str, str]] = {}  # archive -> file name -> member path
        self._lock = threading.Lock()

    def read(self, archive: str, file_name: str) -> bytes | None:
        """The file's bytes, or None if the archive does not have it."""
        reader = self._reader(archive)
        member = self._member_of[archive].get(file_name)
        return None if member is None else reader.read(member)

    def _reader(self, archive: str):
        from ..prep.splitzip import SplitZip

        # One SplitZip per thread: it seeks and reads on file handles it keeps open, so threads sharing one
        # get each other's bytes (zlib errors, or silently another file's content).
        readers = self._thread.__dict__.setdefault("readers", {})
        if archive not in readers:
            readers[archive] = SplitZip(paths.ZIPS[archive])
            with self._lock:
                if archive not in self._member_of:
                    self._member_of[archive] = readers[archive].basename_index()
        return readers[archive]


class Screenshots:
    """A test session's screenshots as JPEG bytes: from the cache folder if it has them, else from the archives."""

    def __init__(self, cache_dir: Path | None = DEFAULT_CACHE):
        self.cache_dir = cache_dir
        self.archives = Archives()

    def get(self, session: dict, step: int) -> bytes | None:
        """None if neither the cache nor the archive has the screenshot."""
        if self.cache_dir is not None:
            cached = self.cache_dir / session["id"] / f"{step}.jpg"
            if cached.exists():
                return cached.read_bytes()
        original = self.archives.read(session["source"], session["images"][step])
        return None if original is None else to_jpeg(Image.open(io.BytesIO(original)))


def write_bundle_frames(bundle_dir: Path, cache_dir: Path = DEFAULT_CACHE):
    """Writes <bundle>/frames/<session id>/<step>.jpg for the bundle's AgentNet workflows (existing files are kept)."""
    screenshots = Screenshots(cache_dir)
    session_of = {session["id"]: session for session in load_sessions()}
    for workflow_path in sorted((bundle_dir / "workflows").glob("*.json")):
        if workflow_path.name == "index.json":
            continue
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        if not workflow.get("agentnet"):
            continue
        frames_dir = bundle_dir / "frames" / workflow["id"]
        frames_dir.mkdir(parents=True, exist_ok=True)
        n_missing = 0
        for step in range(workflow["n_steps"]):
            frame_path = frames_dir / f"{step}.jpg"
            if frame_path.exists():
                continue
            jpeg = screenshots.get(session_of[workflow["id"]], step)
            if jpeg is None:
                n_missing += 1
            else:
                frame_path.write_bytes(jpeg)
        missing_note = f" ({n_missing} missing)" if n_missing else ""
        print(f"{workflow['id']}: {workflow['n_steps']} frames{missing_note}", flush=True)


def cache_test_screenshots(cache_dir: Path = DEFAULT_CACHE, n_workers: int = 16) -> bool:
    """Writes every test session's screenshots to cache_dir/<session id>/<step>.jpg, reading the archives.

    Existing files are kept; a screenshot that cannot be read or decoded is reported and skipped.
    Returns whether the cache holds any screenshot afterwards.
    """
    sessions = [session for session in load_sessions() if session["split"] == "test"]
    screenshots = Screenshots(cache_dir=None)
    started = time.time()
    counts = Counter()
    with ThreadPoolExecutor(n_workers) as executor:
        results = executor.map(lambda session: cache_session(screenshots, cache_dir, session), sessions)
        for n_done, (session_counts, failures) in enumerate(results, start=1):
            for failure in failures[: max(0, MAX_FAILURES_LISTED - counts["failed"])]:
                print(f"FAILED {failure}", flush=True)
            counts += session_counts
            if n_done % PROGRESS_EVERY == 0:
                print(f"{n_done}/{len(sessions)} sessions, {time.time() - started:.0f}s", flush=True)
    if counts["failed"] > MAX_FAILURES_LISTED:
        print(f"... and {counts['failed'] - MAX_FAILURES_LISTED} more failures")
    print(
        f"{len(sessions)} test sessions' screenshots in {cache_dir} ({time.time() - started:.0f}s): "
        f"{counts['written']} written, {counts['already cached']} already cached, "
        f"{counts['missing']} missing from the archives, {counts['failed']} failed"
    )
    return counts["written"] + counts["already cached"] > 0


def cache_session(screenshots: Screenshots, cache_dir: Path, session: dict) -> tuple[Counter, list[str]]:
    """(number of screenshots written / already cached / missing / failed, one line per failure)."""
    counts, failures = Counter(), []
    session_dir = cache_dir / session["id"]
    session_dir.mkdir(parents=True, exist_ok=True)
    for step, file_name in enumerate(session["images"]):
        path = session_dir / f"{step}.jpg"
        if path.exists():
            counts["already cached"] += 1
            continue
        try:
            jpeg = screenshots.get(session, step)
        except Exception as error:
            counts["failed"] += 1
            failures.append(f"{session['id']} step {step} ({file_name}): {type(error).__name__}: {error}")
            continue
        if jpeg is None:
            counts["missing"] += 1
            continue
        # Renamed into place, so a killed job never leaves a truncated file that later runs would keep.
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(jpeg)
        temporary.rename(path)
        counts["written"] += 1
    return counts, failures
