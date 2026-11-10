"""The background map: a 2D PCA of every test session's final embedding, with k-means clusters."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .assets import CACHE_DIR, DemoAssets, app_name
from .scoring import score_steps

# BackgroundMap field -> array in <CACHE_DIR>/<run id>_pca.npz
ARRAY_NAMES = {
    "mean": "mu",
    "axes": "W",
    "explained_variance": "explained",
    "embeddings": "E",
    "session_indices": "sidx",
    "clusters": "cluster",
    "cluster_centers": "cent",
    "first_step_ranks": "first_rank",
    "last_step_ranks": "last_rank",
    "first_step_entropies": "ent0",
    "last_step_entropies": "entT",
}


@dataclass
class BackgroundMap:
    """The map of one run. Row p of every per-session array is the test session session_indices[p]."""

    mean: np.ndarray  # [D] mean of the embeddings
    axes: np.ndarray  # [D, 2] first two principal directions
    explained_variance: np.ndarray  # [2] fraction of the variance along each axis
    embeddings: np.ndarray  # [n_test, D] float16, unit length: each session's last-step direction
    session_indices: np.ndarray  # [n_test]
    clusters: np.ndarray  # [n_test] k-means cluster, 0 = the largest
    cluster_centers: np.ndarray  # [n_clusters, D] unit length
    first_step_ranks: np.ndarray  # [n_test] rank of the true intent after the first step
    last_step_ranks: np.ndarray  # [n_test]
    first_step_entropies: np.ndarray  # [n_test] entropy of the intent belief after the first step
    last_step_entropies: np.ndarray  # [n_test]

    def project(self, unit_embeddings: np.ndarray) -> np.ndarray:
        """[..., D] -> [..., 2] map coordinates."""
        return (unit_embeddings - self.mean) @ self.axes

    def save(self, run_id: str):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path(run_id), **{ARRAY_NAMES[field.name]: getattr(self, field.name) for field in fields(self)})

    @classmethod
    def load(cls, run_id: str) -> BackgroundMap:
        saved = np.load(cache_path(run_id))
        return cls(**{field: saved[array_name] for field, array_name in ARRAY_NAMES.items()})


def cache_path(run_id: str) -> Path:
    return CACHE_DIR / f"{run_id}_pca.npz"


def write_background(assets: DemoAssets, bundle_dir: Path, n_clusters: int):
    """Writes <bundle>/background.json and caches the map for `export`."""
    session_indices = assets.test_session_indices
    sessions = [assets.sessions[index] for index in session_indices]
    embeddings = final_embeddings(assets)
    mean, axes, explained_variance = fit_projection(embeddings)
    clusters, centers = cluster_embeddings(embeddings, n_clusters)
    first_ranks, last_ranks, first_entropies, last_entropies = first_and_last_step_scores(assets)

    background_map = BackgroundMap(
        mean=mean,
        axes=axes,
        explained_variance=np.array(explained_variance),
        embeddings=embeddings.astype(np.float16),
        session_indices=session_indices,
        clusters=clusters,
        cluster_centers=centers / np.linalg.norm(centers, axis=1, keepdims=True),
        first_step_ranks=np.array(first_ranks),
        last_step_ranks=np.array(last_ranks),
        first_step_entropies=np.array(first_entropies),
        last_step_entropies=np.array(last_entropies),
    )
    background_map.save(assets.run_id)

    points = background_map.project(embeddings)
    background = {
        "run": assets.run_id,
        "n": len(sessions),
        "explained": [round(fraction, 4) for fraction in explained_variance],
        "clusters": describe_clusters(sessions, embeddings, clusters, centers, background_map),
        "points": [[round(float(x), 4), round(float(y), 4), int(cluster)] for (x, y), cluster in zip(points, clusters)],
        "items": [
            {
                "id": session["id"],
                "app": app_name(session),
                "os": session["os"],
                "instruction": session["instruction"][:200],
            }
            for session in sessions
        ],
    }
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "background.json").write_text(json.dumps(background, separators=(",", ":")))
    print(
        f"background: {len(sessions)} test workflows, PCA explains {explained_variance[0]:.3f} + "
        f"{explained_variance[1]:.3f} of variance, {n_clusters} clusters; "
        f"final true-intent rank 1: {np.mean(background_map.last_step_ranks == 1):.3f}, "
        f"<=5: {np.mean(background_map.last_step_ranks <= 5):.3f}"
    )


def final_embeddings(assets: DemoAssets) -> np.ndarray:
    """[n_test, D] float32, unit length: every test session's direction at its last step."""
    embeddings = np.stack([assets.test_embeddings[int(index)][0][-1] for index in assets.test_session_indices])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def fit_projection(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """PCA to 2D: (mean [D], axes [D, 2], fraction of the variance along each axis)."""
    mean = embeddings.mean(0)
    _, singular_values, components = np.linalg.svd(embeddings - mean, full_matrices=False)
    axes = components[:2].T.copy()
    # An SVD leaves each axis's sign open: make its largest entry positive, so the map does not flip between runs.
    axes *= np.sign(axes[np.abs(axes).argmax(0), [0, 1]])
    explained_variance = (singular_values[:2] ** 2 / (singular_values**2).sum()).tolist()
    return mean, axes, explained_variance


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """k-means: (cluster of each embedding [n], cluster means [n_clusters, D]), clusters numbered largest first."""
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(embeddings).labels_
    sizes = np.bincount(labels, minlength=n_clusters)
    rank_by_size = {label: rank for rank, label in enumerate(np.argsort(-sizes))}
    clusters = np.array([rank_by_size[label] for label in labels])
    centers = np.stack([embeddings[clusters == cluster].mean(0) for cluster in range(n_clusters)])
    return clusters, centers


def describe_clusters(
    sessions: list[dict],
    embeddings: np.ndarray,
    clusters: np.ndarray,
    centers: np.ndarray,
    background_map: BackgroundMap,
) -> list[dict]:
    """Per cluster: size, most common apps, the instruction of its most central session, and its map position."""
    descriptions = []
    for cluster, center in enumerate(centers):
        members = np.nonzero(clusters == cluster)[0]
        top_apps = Counter(app_name(sessions[member]) for member in members).most_common(3)
        most_central = members[np.argmax(embeddings[members] @ center)]
        descriptions.append(
            {
                "id": cluster,
                "n": int(len(members)),
                "apps": [[app, count] for app, count in top_apps],
                "label": " / ".join(app for app, _ in top_apps[:2]),
                "example": sessions[most_central]["instruction"][:160],
                "xy": background_map.project(center / np.linalg.norm(center)).round(4).tolist(),
            }
        )
    return descriptions


def first_and_last_step_scores(assets: DemoAssets) -> tuple[list[int], list[int], list[float], list[float]]:
    """Per test session: the true intent's rank and the belief's entropy, after the first and after the last step."""
    first_ranks, last_ranks, first_entropies, last_entropies = [], [], [], []
    for session_index in assets.test_session_indices:
        directions, confidences = assets.test_embeddings[int(session_index)]
        scores = score_steps(
            assets.intents,
            directions[[0, -1]],
            confidences[[0, -1]],
            true_instruction=assets.sessions[session_index]["instruction"],
        )
        first_ranks.append(int(scores.true_ranks[0]))
        last_ranks.append(int(scores.true_ranks[1]))
        first_entropies.append(float(scores.entropies[0]))
        last_entropies.append(float(scores.entropies[1]))
    return first_ranks, last_ranks, first_entropies, last_entropies
