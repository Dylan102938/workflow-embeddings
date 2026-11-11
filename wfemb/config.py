"""Run configuration: the experiment variables (which define a run and its id) and the fixed recipe.

Reference: docs/EXPERIMENTS.md.
  python -m wfemb.config id=<run id>     # print the canonical id, results folder and frame cache
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import chz

REPO = Path(__file__).resolve().parent.parent
ARCHITECTURES = ("transf", "late-history-transf", "gru")

# Run-id key -> Experiment field, in id order.
ID_KEYS = {
    "arch": "architecture",
    "dm": "d_model",
    "dh": "d_head",
    "nl": "n_layers",
    "nh": "n_heads",
    "drop": "dropout",
    "smooth": "label_smoothing",
    "patch": "n_patches",
    "lr": "learning_rate",
    "bs": "batch_size",
    "ep": "n_epochs",
    "beta": "hard_negative_beta",
    "pca": "pca_dim",
    "seed": "seed",
}


@chz.chz
class Experiment:
    """The variables that define a run. Defaults are the best model so far."""

    architecture: str = "late-history-transf"
    d_model: int = 512
    d_head: int = 64
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.3
    label_smoothing: float = 0.2
    n_patches: int = 576
    learning_rate: float = 5e-4
    batch_size: int = 64
    n_epochs: int = 45
    hard_negative_beta: float = 0.0
    pca_dim: int = 512  # channels the screen features are PCA-projected to; 0 = unprojected
    seed: int = 0

    @chz.validate
    def _validate(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {self.architecture!r}")
        if self.architecture != "transf" and self.n_layers < 2:
            raise ValueError(f"architecture {self.architecture} needs n_layers >= 2")
        if math.isqrt(self.n_patches) ** 2 != self.n_patches:
            raise ValueError(f"n_patches must be a square number (256, 576, ...), got {self.n_patches}")
        if self.pca_dim < 0:
            raise ValueError(f"pca_dim must be 0 (unprojected) or a channel count, got {self.pca_dim}")

    @property
    def id(self) -> str:
        return "_".join(f"{key}-{format_id_value(getattr(self, field))}" for key, field in ID_KEYS.items())

    @property
    def relative_dir(self) -> Path:
        """Results folder below the results root: one level per id segment."""
        return Path(*self.id.split("_"))

    @classmethod
    def from_id(cls, run_id: str) -> Experiment:
        """Accepts an id or its folder form; variables missing from the id take their defaults."""
        defaults = cls()
        values = {}
        for segment in run_id.strip().strip("/").replace("/", "_").split("_"):
            key, separator, text = segment.partition("-")
            if not separator or key not in ID_KEYS:
                raise ValueError(f"bad id segment {segment!r} in {run_id!r}")
            field = ID_KEYS[key]
            values[field] = type(getattr(defaults, field))(text)
        return cls(**values)

    @property
    def patch_grid_size(self) -> int:
        return math.isqrt(self.n_patches)

    @property
    def frame_cache_name(self) -> str:
        return frame_cache_name(self.patch_grid_size, self.pca_dim)


def frame_cache_name(grid_size: int, pca_dim: int) -> str:
    grid = "" if grid_size == 16 else f"_g{grid_size}"
    projection = {512: "", 0: "_raw"}.get(pca_dim, f"_pca{pca_dim}")
    return f"frames_dense{grid}{projection}"


def format_id_value(value) -> str:
    """Canonical text of a value in a run id: 512, 0.3, 5e-4, 2 (for 2.0). Values never contain '_'."""
    if isinstance(value, str):
        if "_" in value:
            raise ValueError(f"id values cannot contain '_': {value!r}")
        return value
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    if abs(value) < 0.01:
        mantissa, exponent = f"{value:e}".split("e")
        return f"{float(mantissa):g}e{int(exponent)}"
    return f"{value:g}"


@chz.chz
class TrainConfig:
    """Training recipe shared by all runs (not part of the run id)."""

    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    screen_token_dropout: float = 0.4
    step_dropout: float = 0.4
    smoothing_temperature: float = 0.05
    hard_negative_ramp: float = 0.3  # fraction of training over which beta ramps up from 0
    duplicate_threshold: float = 0.8  # candidates this similar to the target are not negatives
    confidence_min: float = 1.0
    confidence_max: float = 50.0
    confidence_init: float = 5.0
    mlp_ratio: int = 4
    max_frames_per_batch: int = 2048
    activation_checkpointing: bool = True
    max_steps: int = 0  # debugging: stop early (0 = full schedule)


@chz.chz
class EvalConfig:
    eval_steps: int = 500
    duplicate_threshold: float = 0.8  # fixed, so val/test metrics compare across runs
    n_neighbors: int = 100
    save_test_embeddings: bool = True
    batch_size: int = 64
    max_frames_per_batch: int = 2048


@chz.chz
class DataConfig:
    root: str = chz.field(
        default_factory=lambda: os.environ.get("WFEMB_DATA", "/nas/ucb/dfeng/data/workflow-embeddings")
    )
    bulk_root: str = chz.field(
        default_factory=lambda: os.environ.get("WFEMB_BULK", "/nas/ttl=60d/dfeng/workflow-embeddings")
    )
    label_set: str = "keys_openai"
    frame_cache: str = ""  # folder override; default <bulk_root>/<experiment.frame_cache_name>
    prefetch_threads: int = 4
    prefetch_depth: int = 4

    def frame_cache_dir(self, experiment: Experiment) -> Path:
        return Path(self.frame_cache) if self.frame_cache else Path(self.bulk_root) / experiment.frame_cache_name


def default_results_root() -> str:
    return os.environ.get("WFEMB_RESULTS", str(REPO / "results"))


@chz.chz
class RunConfig:
    experiment: Experiment = chz.field(default_factory=Experiment)
    train: TrainConfig = chz.field(default_factory=TrainConfig)
    eval: EvalConfig = chz.field(default_factory=EvalConfig)
    data: DataConfig = chz.field(default_factory=DataConfig)
    results_root: str = chz.field(default_factory=default_results_root)
    launcher: str = chz.field(default_factory=lambda: os.environ.get("WFEMB_LAUNCHER", "local"))
    overwrite: bool = False

    @property
    def run_dir(self) -> Path:
        return Path(self.results_root) / self.experiment.relative_dir


def expand_run_id(arguments: list[str]) -> list[str]:
    """Replaces an `id=<run id>` argument by the `experiment.<field>=<value>` arguments it stands for."""
    expanded = []
    for argument in arguments:
        if argument.startswith("id="):
            experiment = Experiment.from_id(argument.removeprefix("id="))
            expanded += [f"experiment.{field}={getattr(experiment, field)}" for field in ID_KEYS.values()]
        else:
            expanded.append(argument)
    return expanded


def parse_run_config(arguments: list[str] | None = None) -> RunConfig:
    arguments = sys.argv[1:] if arguments is None else arguments
    return chz.Blueprint(RunConfig).apply_from_argv(expand_run_id(arguments)).make()


def main():
    config = parse_run_config()
    print(config.experiment.id)
    print("results:", config.run_dir)
    print("frames: ", config.data.frame_cache_dir(config.experiment))


if __name__ == "__main__":
    main()
