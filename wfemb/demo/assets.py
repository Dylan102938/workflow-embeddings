"""What the demo commands read: the prepared sessions and events, the intent bank, and a run's results folder."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from ..config import DataConfig, Experiment, default_results_root
from ..model import WorkflowTransformer, load_checkpoint
from ..storage import paths
from ..storage.frames import load_numpy, read_file
from .scoring import IntentBank

CACHE_DIR = paths.DATA / "demo"  # what the demo derives: <run id>_pca.npz, screens/

# SessionEvents field -> array in prepared/events.npz
EVENT_ARRAYS = {
    "types": "ev_type",
    "points": "ev_xy",
    "scrolls": "ev_scroll",
    "keys": "ev_keys",
    "texts": "ev_text",
    "focus": "focus",
}


@dataclass
class SessionEvents:
    """One session's events: L steps, E event slots per step (the layout of prepared/events.npz)."""

    types: np.ndarray  # [L, E] index into prep.actions.EVENT_TYPES, 0 = empty slot
    points: np.ndarray  # [L, E, 4] (x, y, x2, y2) as fractions of the screen, nan = absent
    scrolls: np.ndarray  # [L, E]
    keys: np.ndarray  # [L, E, K] index into the key vocabulary, 0 = none
    texts: np.ndarray  # [L, E] index into the typed texts, -1 = none
    focus: np.ndarray  # [L, 2] where the step's action lands on the screen


def load_sessions() -> list[dict]:
    return pickle.load(read_file(paths.PREP / "sessions.pkl"))


def app_name(session: dict) -> str:
    return session["app"].split(":", 1)[-1]


class DemoAssets:
    """The prepared data, the intent bank and one run's results folder."""

    def __init__(self, run_id: str, checkpoint_name: str = "best.pt", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.experiment = Experiment.from_id(run_id)
        self.run_id = self.experiment.id
        self.run_dir = Path(default_results_root()) / self.experiment.relative_dir
        self.checkpoint_path = self.run_dir / checkpoint_name

        self.sessions = load_sessions()
        self.session_index_of = {session["id"]: index for index, session in enumerate(self.sessions)}
        self.splits = np.array([session["split"] for session in self.sessions])
        self.test_session_indices = np.nonzero(self.splits == "test")[0]
        events = load_numpy(paths.PREP / "events.npz")
        self.events = {name: events[name] for name in events.files}
        vocabulary = json.load(open(paths.PREP / "vocab.json"))
        self.key_names: list[str] = vocabulary["key_vocab"]
        self.typed_texts: list[str] = vocabulary["texts"]

        data = DataConfig()
        labels_dir = Path(data.root) / data.label_set
        label_keys = load_numpy(labels_dir / "keys.npy").astype(np.float32)
        self.text_embeddings = load_numpy(labels_dir / "text_vecs.npy")
        self.intents = IntentBank(self.sessions, self.splits, label_keys, self.device)

    @cached_property
    def test_embeddings(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Session index -> (directions [L, D] float32, confidences [L]) for every test session."""
        path = self.run_dir / "test_embeddings.npz"
        if not path.exists():
            raise SystemExit(
                f"{path} missing: fetch it (python -m wfemb.storage.fetch rnn|baseten --embeddings), or write it "
                f"where the frame cache is: python -m wfemb.eval id={self.run_id} save_embeddings=True"
            )
        saved = np.load(path)
        # An NpzFile decompresses an array on every access: read the two large ones once.
        directions, confidences = saved["directions"], saved["confidences"]
        return {
            int(session_index): (
                directions[offset : offset + length].astype(np.float32),
                confidences[offset : offset + length],
            )
            for session_index, offset, length in zip(saved["session_indices"], saved["step_offsets"], saved["lengths"])
        }

    def session_events(self, session_index: int) -> SessionEvents:
        offset = int(self.events["sess_offset"][session_index])
        length = int(self.events["sess_len"][session_index])
        return SessionEvents(
            **{field: self.events[array_name][offset : offset + length] for field, array_name in EVENT_ARRAYS.items()}
        )

    def load_model(self, text_embeddings: torch.Tensor) -> WorkflowTransformer:
        """The run's model in eval mode, with the given typed-text embedding table."""
        if not self.checkpoint_path.exists():
            raise SystemExit("--raw needs the run's checkpoint: python -m wfemb.storage.fetch ... --checkpoints")
        model, _ = load_checkpoint(self.checkpoint_path, text_embeddings, self.device)
        return model
