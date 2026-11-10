"""What the key-set builders (text_embed.py, openai_keys.py) share: paraphrase ownership, drift filter, files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SIMILARITY_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def deduplicate(texts: list[str]) -> tuple[list[str], list[int]]:
    """The sorted unique texts, and each text's position among them (vectors[positions] undoes the dedup)."""
    unique_texts = sorted(set(texts))
    position = {text: i for i, text in enumerate(unique_texts)}
    return unique_texts, [position[text] for text in texts]


def session_paraphrases(sessions: list[dict], paraphrases: dict[str, list[str]]) -> tuple[list[str], np.ndarray]:
    """Every session's paraphrases, flattened in session order, and the index of the session that owns each."""
    texts, owners = [], []
    for session_index, session in enumerate(sessions):
        for paraphrase in paraphrases.get(session["instruction"], []):
            texts.append(paraphrase)
            owners.append(session_index)
    return texts, np.asarray(owners)


def drift_similarities(paraphrase_vectors: np.ndarray, owners: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Cosine of each paraphrase to its session's instruction key (all vectors are l2-normalized)."""
    return (paraphrase_vectors * keys[owners]).sum(1)


def threshold_keeping(similarities: np.ndarray, fraction: float) -> float:
    """The similarity threshold above which `fraction` of the paraphrases lie."""
    return float(np.quantile(similarities, 1 - fraction))


def write_paraphrase_keys(
    key_dir: Path,
    paraphrase_vectors: np.ndarray,
    owners: np.ndarray,
    similarities: np.ndarray,
    threshold: float,
    session_count: int,
) -> dict:
    """Writes para_keys.npy and para_index.npz for the paraphrases above the threshold; returns the meta.json stats."""
    kept = similarities > threshold
    # owners is sorted (session order), so each session's kept rows are contiguous: offset / count index them
    count = np.bincount(owners[kept], minlength=session_count).astype(np.int32)
    offset = np.zeros(session_count, np.int64)
    offset[1:] = np.cumsum(count)[:-1]
    np.save(key_dir / "para_keys.npy", paraphrase_vectors[kept].astype(np.float16))
    np.savez(key_dir / "para_index.npz", offset=offset, count=count)
    return {
        "theta_para": threshold,
        "para_total": int(len(similarities)),
        "para_kept": int(kept.sum()),
        "para_sim_quantiles": {
            quantile: float(np.quantile(similarities, quantile)) for quantile in SIMILARITY_QUANTILES
        },
        "sessions_with_paraphrase": int((count > 0).sum()),
    }
