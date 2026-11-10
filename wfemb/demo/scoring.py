"""Intent beliefs: softmax(confidence * cos(direction, key)) over the intent bank, as in the training loss."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


class IntentBank:
    """The candidate intents: the unique instructions of the training and test sessions, as in evaluation."""

    def __init__(self, sessions: list[dict], splits: np.ndarray, label_keys: np.ndarray, device: torch.device):
        first_session_with = {}
        for session_index in np.nonzero(np.isin(splits, ["train", "test"]))[0]:
            first_session_with.setdefault(sessions[session_index]["instruction"], int(session_index))
        self.texts = list(first_session_with)
        self.position_of = {text: position for position, text in enumerate(self.texts)}
        keys = label_keys[np.array(list(first_session_with.values()))]
        self.keys = F.normalize(torch.from_numpy(keys), dim=-1).to(device)  # [N, D]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class StepScores:
    """Per step of one workflow. The true_* fields are None when the true instruction is not in the bank."""

    top_intents: np.ndarray  # [L, top_k] positions in the bank, most probable first
    top_probabilities: np.ndarray  # [L, top_k]
    entropies: np.ndarray  # [L]
    true_ranks: np.ndarray | None  # [L], 1 = most probable
    true_probabilities: np.ndarray | None  # [L]


def score_steps(
    bank: IntentBank,
    directions: np.ndarray,
    confidences: np.ndarray,
    top_k: int = 5,
    true_instruction: str | None = None,
) -> StepScores:
    """directions [L, D], confidences [L]."""
    device = bank.keys.device
    unit_directions = F.normalize(torch.from_numpy(directions).to(device), dim=-1)
    logits = torch.from_numpy(confidences).to(device)[:, None] * (unit_directions @ bank.keys.T)
    log_probabilities = logits.log_softmax(-1)
    probabilities = log_probabilities.exp()
    top_probabilities, top_intents = probabilities.topk(top_k, dim=-1)
    scores = StepScores(
        top_intents=top_intents.cpu().numpy(),
        top_probabilities=top_probabilities.cpu().numpy(),
        entropies=-(probabilities * log_probabilities).sum(-1).cpu().numpy(),
        true_ranks=None,
        true_probabilities=None,
    )
    true_position = bank.position_of.get(true_instruction) if true_instruction else None
    if true_position is not None:
        true_log_probabilities = log_probabilities[:, true_position]
        n_more_probable = (log_probabilities > true_log_probabilities[:, None]).sum(-1)
        scores.true_ranks = (n_more_probable + 1).cpu().numpy()
        scores.true_probabilities = true_log_probabilities.exp().cpu().numpy()
    return scores
