"""One workflow's record for the viewer: per step the actions, the intent belief and the position on the map.

File formats: docs/DEMO.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..prep.actions import EVENT_TYPES
from .assets import DemoAssets, SessionEvents, app_name
from .background import BackgroundMap
from .scoring import StepScores, score_steps
from .selection import Partners

N_NEIGHBORS = 8  # test workflows nearest to the final embedding
N_NEAREST_PER_STEP = 3  # test workflows nearest to each step's embedding
EXCLUDED = -2  # a similarity below every cosine: keeps a workflow out of its own neighbors
MAX_TYPED_LABEL_LENGTH = 80
NO_PARTNERS = Partners([], [])

CLICK_LABELS = {
    "click": "Click",
    "doubleClick": "Double-click",
    "tripleClick": "Triple-click",
    "rightClick": "Right-click",
    "middleClick": "Middle-click",
}
KEY_LABELS = {"press": "Press", "keyDown": "Hold", "keyUp": "Release"}
OTHER_LABELS = {
    "moveTo": "Move pointer",
    "dragTo": "Drag",
    "mouseDown": "Mouse down",
    "mouseUp": "Mouse up",
    "wait": "Wait",
    "other": "Other action",
}


def workflow_of_test_session(
    assets: DemoAssets, background_map: BackgroundMap, partners: list[Partners], session_index: int, top_k: int
) -> dict:
    """The record of an AgentNet test session, from the run's saved test embeddings."""
    session = assets.sessions[session_index]
    directions, confidences = assets.test_embeddings[session_index]
    map_position = int(np.nonzero(background_map.session_indices == session_index)[0][0])
    metadata = {
        "id": session["id"],
        "source": "agentnet-test",
        "instruction": session["instruction"],
        "app": app_name(session),
        "os": session["os"],
        "agentnet": {"archive": session["source"], "images": session["images"]},
    }
    return build_workflow(
        assets,
        background_map,
        metadata,
        directions,
        confidences,
        events=assets.session_events(session_index),
        typed_texts=assets.typed_texts,
        images=[f"frames/{session['id']}/{step}.jpg" for step in range(len(directions))],
        top_k=top_k,
        map_position=map_position,
        partners=partners[map_position],
    )


def build_workflow(
    assets: DemoAssets,
    background_map: BackgroundMap,
    metadata: dict,
    directions: np.ndarray,
    confidences: np.ndarray,
    events: SessionEvents,
    typed_texts: list[str],
    images: list[str],
    top_k: int = 5,
    map_position: int | None = None,
    partners: Partners = NO_PARTNERS,
) -> dict:
    """images: per step a path inside the bundle or a data URL. typed_texts: what events.texts indexes.

    map_position: the workflow's own point on the background map, if it is one of the test sessions.
    """
    n_steps = len(directions)
    instruction = metadata.get("instruction")
    scores = score_steps(assets.intents, directions, confidences, top_k, instruction)
    in_bank = scores.true_ranks is not None
    true_ranks = [int(rank) for rank in scores.true_ranks] if in_bank else [None] * n_steps
    true_probabilities = rounded(scores.true_probabilities, 5) if in_bank else [None] * n_steps

    unit_directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    map_points = background_map.project(unit_directions)
    step_clusters = (unit_directions @ background_map.cluster_centers.T).argmax(1)

    map_embeddings = background_map.embeddings.astype(np.float32)
    final_similarities = map_embeddings @ unit_directions[-1]  # [n_test]
    step_similarities = unit_directions @ map_embeddings.T  # [L, n_test]
    if map_position is not None:
        final_similarities[map_position] = EXCLUDED
        step_similarities[:, map_position] = EXCLUDED
    neighbors = np.argsort(-final_similarities)[:N_NEIGHBORS]
    nearest_per_step = np.argsort(-step_similarities, axis=1)[:, :N_NEAREST_PER_STEP]

    steps = [
        {
            "t": step,
            "image": images[step],
            "actions": step_actions(events, step, assets.key_names, typed_texts),
            "focus": rounded(events.focus[step], 4),
            "kappa": round(float(confidences[step]), 3),
            "entropy": round(float(scores.entropies[step]), 3),
            "true_rank": true_ranks[step],
            "true_p": true_probabilities[step],
            "top": top_intents(assets.intents.texts, scores, step, instruction),
            "xy": rounded(map_points[step], 4),
            "cluster": int(step_clusters[step]),
            "near": [[int(near), round(float(step_similarities[step, near]), 4)] for near in nearest_per_step[step]],
        }
        for step in range(n_steps)
    ]
    return {
        "version": 1,
        "run": assets.run_id,
        **metadata,
        "n_steps": n_steps,
        "bank_size": len(assets.intents),
        "true_in_bank": in_bank,
        "bg_index": map_position,
        "partners": [int(position) for position in partners.same_workflow],
        "related": [int(position) for position in partners.related],
        "neighbors": [{"bg": int(near), "cos": round(float(final_similarities[near]), 4)} for near in neighbors],
        "steps": steps,
    }


def top_intents(intent_texts: list[str], scores: StepScores, step: int, instruction: str | None) -> list[list]:
    """[intent text, probability, whether it is the true instruction], most probable first."""
    return [
        [
            intent_texts[int(intent)],
            round(float(probability), 5),
            instruction is not None and intent_texts[int(intent)] == instruction,
        ]
        for probability, intent in zip(scores.top_probabilities[step], scores.top_intents[step])
    ]


def step_actions(events: SessionEvents, step: int, key_names: list[str], typed_texts: list[str]) -> list[dict]:
    actions = []
    for slot in range(events.types.shape[1]):
        type_id = int(events.types[step, slot])
        if type_id == 0:
            continue
        event_type = EVENT_TYPES[type_id]
        x, y, x2, y2 = (round(float(value), 4) if np.isfinite(value) else None for value in events.points[step, slot])
        keys = [key_names[int(key)] for key in events.keys[step, slot] if int(key) > 0]
        text_index = int(events.texts[step, slot])
        text = typed_texts[text_index] if text_index >= 0 else None
        scroll = float(events.scrolls[step, slot])
        actions.append(
            {
                "type": event_type,
                "x": x,
                "y": y,
                "x2": x2,
                "y2": y2,
                "keys": keys,
                "text": text,
                "scroll": scroll,
                "label": action_label(event_type, keys, text, scroll),
            }
        )
    return actions


def action_label(event_type: str, keys: list[str], text: str | None, scroll: float) -> str:
    """What the viewer shows for an action: 'Click', 'Type “hello”', 'Shortcut Ctrl+C', 'Scroll ↓ 3'."""
    if event_type in CLICK_LABELS:
        return CLICK_LABELS[event_type]
    if event_type == "write":
        typed = (text or "").replace("\n", "⏎")
        return f"Type “{typed[:MAX_TYPED_LABEL_LENGTH]}{'…' if len(typed) > MAX_TYPED_LABEL_LENGTH else ''}”"
    key_labels = [key.capitalize() if len(key) > 1 else key.upper() for key in keys]
    if event_type in KEY_LABELS:
        return KEY_LABELS[event_type] + " " + " ".join(key_labels)
    if event_type == "hotkey":
        return "Shortcut " + "+".join(key_labels)
    if event_type in ("scroll", "hscroll"):
        amount = float(scroll or 0)
        if event_type == "scroll":
            arrow = "↑" if amount > 0 else "↓"
        else:
            arrow = "→" if amount > 0 else "←"
        return f"Scroll {arrow} {abs(amount):g}"
    return OTHER_LABELS.get(event_type, event_type)


def write_workflow(bundle_dir: Path, workflow: dict, single_file: bool = False) -> Path:
    """Writes workflows/<id>.json and puts the workflow's entry at the end of workflows/index.json."""
    workflows_dir = bundle_dir / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    name = safe_file_name(workflow["id"])
    path = workflows_dir / f"{name}.json"
    path.write_text(json.dumps(workflow, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

    index_path = workflows_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    index = [entry for entry in index if entry["id"] != workflow["id"]]
    index.append(
        {
            "id": workflow["id"],
            "file": f"workflows/{name}.json",
            "title": workflow.get("instruction") or workflow["id"],
            "app": workflow.get("app"),
            "os": workflow.get("os"),
            "n_steps": workflow["n_steps"],
            "source": workflow["source"],
            "final_rank": workflow["steps"][-1]["true_rank"],
            "thumb": None if single_file else workflow["steps"][0]["image"],  # no data URLs in the index
        }
    )
    index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def safe_file_name(workflow_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", workflow_id)


def rounded(values: np.ndarray, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]
