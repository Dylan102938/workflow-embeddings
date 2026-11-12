"""Raw workflows: screenshots plus action code from outside the dataset, encoded as the training data was.

Input format and what "as the training data" means: docs/DEMO.md.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..data import Batch
from ..prep import actions
from ..prep.actions import TYPE_ID, parse_code
from ..storage import paths
from .assets import DemoAssets, SessionEvents
from .background import BackgroundMap
from .screenshots import to_jpeg
from .workflow import build_workflow, safe_file_name

SCREEN_ENCODER = "google/siglip2-so400m-patch16-naflex"
TEXT_ENCODER = "Qwen/Qwen3-Embedding-0.6B"
MAX_PATCHES = 1024
ENCODER_BATCH_SIZE = 32
MAX_TEXT_TOKENS = 256
SCROLL_LIMIT = 1000
UNKNOWN_KEY = 1  # index of <unk> in the key vocabulary


def export_raw_workflow(
    assets: DemoAssets,
    background_map: BackgroundMap,
    path: Path,
    bundle_dir: Path,
    images_dir: Path | None = None,
    channel_pca_path: Path | None = None,
    single_file: bool = False,
    top_k: int = 5,
) -> dict:
    """The record of the raw workflow at path. Its screenshots go to <bundle>/frames, or into the record itself."""
    metadata, steps, images_dir = read_raw_workflow(path, images_dir)
    screenshots = [Image.open(resolve_image(step["image"], images_dir)).convert("RGB") for step in steps]
    events, new_texts = encode_events(steps, [screenshot.size for screenshot in screenshots], assets)

    text_embeddings = torch.from_numpy(assets.text_embeddings)
    if new_texts:
        new_embeddings = torch.from_numpy(encode_texts(new_texts, assets.device))
        text_embeddings = torch.cat([text_embeddings.float(), new_embeddings.float()])
    model = assets.load_model(text_embeddings)
    frames = encode_screenshots(
        screenshots,
        assets.device,
        channel_pca_path or default_channel_pca_path(assets.experiment),
        assets.experiment.patch_grid_size,
    )
    directions, confidences = embed_workflow(model, assets.device, frames, events)

    if single_file:
        images = ["data:image/jpeg;base64," + base64.b64encode(to_jpeg(shot)).decode() for shot in screenshots]
    else:
        frames_dir = bundle_dir / "frames" / safe_file_name(metadata["id"])
        frames_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for step, screenshot in enumerate(screenshots):
            (frames_dir / f"{step}.jpg").write_bytes(to_jpeg(screenshot))
            images.append(f"frames/{frames_dir.name}/{step}.jpg")
    return build_workflow(
        assets,
        background_map,
        {**metadata, "source": "raw"},
        directions,
        confidences,
        events=events,
        typed_texts=assets.typed_texts + new_texts,
        images=images,
        top_k=top_k,
    )


def read_raw_workflow(path: Path, images_dir: Path | None) -> tuple[dict, list[dict], Path]:
    """(metadata, steps [{"image", "code"}], folder of the screenshots) from a workflow JSON or an AgentNet record."""
    text = path.read_text()
    record = json.loads(text.splitlines()[0] if path.suffix == ".jsonl" else text)
    if "traj" in record:
        steps = [
            {"image": step["image"], "code": step["value"].get("code") or ""}
            for step in sorted(record["traj"], key=lambda step: step["index"])
        ]
        workflow_id = record.get("task_id")
    else:
        steps = record["steps"]
        workflow_id = record.get("id")
    metadata = {
        "id": workflow_id or path.stem,
        "instruction": record.get("instruction"),
        "app": record.get("app"),
        "os": record.get("os"),
    }
    return metadata, steps, images_dir or path.parent


def resolve_image(image: str, images_dir: Path) -> Path:
    return Path(image) if Path(image).is_absolute() else images_dir / image


def encode_events(
    steps: list[dict], screen_sizes: list[tuple[int, int]], assets: DemoAssets
) -> tuple[SessionEvents, list[str]]:
    """The workflow's event arrays, built as prep/prepare.py builds them, and its typed texts.

    Typed texts are indexed after the vocabulary's: events.texts == len(assets.typed_texts) + position in the list.
    """
    n_steps = len(steps)
    max_events, max_keys = assets.events["ev_keys"].shape[1:]
    key_ids = {name: index for index, name in enumerate(assets.key_names)}
    events = SessionEvents(
        types=np.zeros((n_steps, max_events), np.int8),
        points=np.full((n_steps, max_events, 4), np.nan, np.float32),
        scrolls=np.zeros((n_steps, max_events), np.float32),
        keys=np.zeros((n_steps, max_events, max_keys), np.int16),
        texts=np.full((n_steps, max_events), -1, np.int32),
        focus=np.full((n_steps, 2), 0.5, np.float32),
    )
    position_of_text: dict[str, int] = {}
    cursor = None
    for step, (record, screen_size) in enumerate(zip(steps, screen_sizes)):
        code = record.get("code") or ""
        with pixel_coordinates(*screen_size) if uses_pixel_coordinates(code) else contextlib.nullcontext():
            step_events, cursor = parse_code(code, cursor)
        step_events = [event for event in step_events if event.type != "terminate"][:max_events]
        last_point = None
        for slot, event in enumerate(step_events):
            events.types[step, slot] = TYPE_ID[event.type]
            events.points[step, slot] = [
                np.nan if value is None else value for value in (event.x, event.y, event.x2, event.y2)
            ]
            if event.scroll is not None:
                events.scrolls[step, slot] = float(np.clip(event.scroll, -SCROLL_LIMIT, SCROLL_LIMIT))
            keys = [key_ids.get(key, UNKNOWN_KEY) for key in event.keys[:max_keys]]
            events.keys[step, slot, : len(keys)] = keys
            if event.text:
                position = position_of_text.setdefault(event.text, len(position_of_text))
                events.texts[step, slot] = len(assets.typed_texts) + position
            if event.point is not None:
                last_point = event.point
        focus = last_point if last_point is not None else cursor
        if focus is not None:
            events.focus[step] = focus
    return events, list(position_of_text)


@contextlib.contextmanager
def pixel_coordinates(width: int, height: int):
    """While active, parse_code divides x / y arguments by the screen size (for code written in pixels)."""
    original_get = actions._get

    def get_scaled(args, kwargs, position, *names):
        value = original_get(args, kwargs, position, *names)
        if is_number(value) and names and names[0] in ("x", "y"):
            value = value / (width if names[0] == "x" else height)
        return value

    actions._get = get_scaled
    try:
        yield
    finally:
        actions._get = original_get


def uses_pixel_coordinates(code: str) -> bool:
    """Whether any call has an x or y above 1: AgentNet code gives coordinates as fractions of the screen."""
    try:
        calls = list(actions._calls_from_ast(code))
    except SyntaxError:
        return False
    for _, _, _, args, kwargs in calls:
        for position, name in ((0, "x"), (1, "y")):
            value = actions._get(args, kwargs, position, name)
            if is_number(value) and value > 1.0:
                return True
    return False


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@torch.no_grad()
def embed_workflow(
    model, device: torch.device, frames: np.ndarray, events: SessionEvents
) -> tuple[np.ndarray, np.ndarray]:
    """frames [L, n_patches, D] bfloat16 bits -> every step's direction [L, D_label] float32 and confidence [L]."""
    n_steps = len(frames)

    def one_session(array: np.ndarray, dtype) -> torch.Tensor:
        return torch.from_numpy(array.astype(dtype)[None]).to(device)

    batch = Batch(
        session_indices=torch.zeros(1, dtype=torch.long),
        frames=torch.from_numpy(frames.view(np.int16)[None]).to(device).view(torch.bfloat16),
        event_types=one_session(events.types, np.int64),
        event_points=one_session(events.points, np.float32),
        event_scrolls=one_session(events.scrolls, np.float32),
        event_keys=one_session(events.keys, np.int64),
        event_texts=one_session(events.texts, np.int64),
        valid=torch.ones(1, n_steps, dtype=torch.bool, device=device),
        lengths=torch.tensor([n_steps], device=device),
    )
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        directions, confidences = model(batch)
    return directions[0].float().cpu().numpy(), confidences[0].squeeze(-1).float().cpu().numpy()


def default_channel_pca_path(experiment) -> Path:
    return paths.BULK / f"dense_pca_g{experiment.patch_grid_size}_d512.npz"


@torch.no_grad()
def encode_screenshots(
    screenshots: list[Image.Image], device: torch.device, channel_pca_path: Path, grid_size: int
) -> np.ndarray:
    """[L, grid_size ** 2, 512] bfloat16 bits, as `prep/cache_frames.py --mode dense` caches training frames."""
    from ..prep.cache_frames import pool_grid, to_bf16_bits

    processor, vision_model, mean, components = load_screen_encoder(device, channel_pca_path)
    encoded = []
    for start in range(0, len(screenshots), ENCODER_BATCH_SIZE):
        inputs = [
            processor(images=[screenshot], max_num_patches=MAX_PATCHES, return_tensors="pt")
            for screenshot in screenshots[start : start + ENCODER_BATCH_SIZE]
        ]
        spatial_shapes = torch.cat([item["spatial_shapes"] for item in inputs])
        hidden = vision_model(
            pixel_values=torch.cat([item["pixel_values"] for item in inputs]).to(device, torch.bfloat16),
            attention_mask=torch.cat([item["pixel_attention_mask"] for item in inputs]).to(device),
            spatial_shapes=spatial_shapes.to(device),
        ).last_hidden_state
        pooled = pool_grid(hidden, spatial_shapes, (grid_size, grid_size))
        encoded.append(to_bf16_bits((pooled - mean) @ components))
    return np.concatenate(encoded)


@functools.cache
def load_screen_encoder(device: torch.device, channel_pca_path: Path):
    """(image processor, SigLIP2 vision model, channel-PCA mean, channel-PCA components)."""
    from transformers import AutoImageProcessor, Siglip2VisionModel

    processor = AutoImageProcessor.from_pretrained(SCREEN_ENCODER)
    model = Siglip2VisionModel.from_pretrained(SCREEN_ENCODER, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    channel_pca = np.load(channel_pca_path)
    mean = torch.from_numpy(channel_pca["mean"]).to(device)
    components = torch.from_numpy(channel_pca["comps"]).to(device)
    return processor, model.vision_model.to(device).eval(), mean, components


@torch.no_grad()
def encode_texts(texts: list[str], device: torch.device) -> np.ndarray:
    """[n, D_text] float16, unit length, as prep/text_embed.py embeds the vocabulary's typed texts."""
    return (
        load_text_encoder(device)
        .encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        .astype(np.float16)
    )


@functools.cache
def load_text_encoder(device: torch.device):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(TEXT_ENCODER, device=str(device), model_kwargs={"torch_dtype": torch.bfloat16})
    encoder.max_seq_length = MAX_TEXT_TOKENS
    return encoder
