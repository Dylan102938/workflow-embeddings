"""Experiment definition and run configuration (docs/EXPERIMENTS.md).

Experiment holds the variables that define a run; its id (`arch-..._dm-..._..._seed-...`) names the run everywhere: in
the results tree, in job files, in summaries. TrainConfig / EvalConfig / DataConfig hold the fixed recipe of the best
model so far; they are recorded with every run but do not define a distinct experiment. RunConfig is everything
train.py needs. Command line (chz):

  python -m wfemb.train id=arch-late-history-transf_dm-512_..._seed-0     # a run given by its id
  python -m wfemb.train exp.beta=2 exp.seed=1                            # the best model with changes
  python -m wfemb.train id=<id> train.max_steps=50 results=/tmp/r        # any recipe field can be overridden
  python -m wfemb.config exp.beta=2                 # print the id, results folder and frame cache of an experiment
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import chz

REPO = Path(__file__).resolve().parent.parent
ARCHS = ("transf", "late-history-transf", "gru")
QKV_MODES = ("shared", "sep-q", "sep-qkv")


def fmt_value(v) -> str:
    """Canonical text of a variable's value in a run id: 512, 0.3, 5e-4, 2 (for 2.0), off (for None)."""
    if v is None:
        return "off"
    if isinstance(v, str):
        if "_" in v:
            raise ValueError(f"id values cannot contain '_': {v!r}")
        return v
    if isinstance(v, bool):
        raise TypeError("no boolean experiment variables")
    if isinstance(v, int) or float(v).is_integer():
        return str(int(v))
    if abs(v) < 0.01:
        mant, exp = f"{v:e}".split("e")
        return f"{float(mant):g}e{int(exp)}"
    return f"{v:g}"


@chz.chz
class Experiment:
    """The variables that define a run. Defaults = the best model so far. Field order = order in the id; variables
    added beyond the original list (dupmask, qkv) come before seed."""
    arch: str = "late-history-transf"   # transf | late-history-transf | gru: what the final block does
    dm: int = 512                       # d_model
    dh: int = 64                        # d_head
    nl: int = 4                         # n_layers, including the final block
    nh: int = 8                         # n_heads
    drop: float = 0.3                   # dropout
    smooth: float = 0.2                 # label smoothing eps (training only)
    patch: int = 576                    # screen patch tokens per frame: 256 (16x16) or 576 (24x24)
    lr: float = 5e-4                    # peak learning rate
    bs: int = 64                        # batch size (sessions)
    ep: int = 45                        # epochs
    beta: float = 0.0                   # hard-negative beta (0 = off)
    dupmask: float | None = 0.8         # added: training near-duplicate mask threshold (None = off, groups only)
    qkv: str = "shared"                 # added: shared | sep-q | sep-qkv (per-token-type projections)
    seed: int = 0

    @chz.validate
    def _check(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        if self.qkv not in QKV_MODES:
            raise ValueError(f"qkv must be one of {QKV_MODES}, got {self.qkv!r}")
        if self.nl < (2 if self.arch != "transf" else 1):
            raise ValueError(f"arch {self.arch} needs nl >= 2")
        if math.isqrt(self.patch) ** 2 != self.patch:
            raise ValueError(f"patch must be a square grid size (256, 576, ...), got {self.patch}")

    # ---------------------------------------------------------------- ids
    @staticmethod
    def keys() -> list[str]:
        return list(chz.chz_fields(Experiment))

    @property
    def id(self) -> str:
        return "_".join(f"{k}-{fmt_value(getattr(self, k))}" for k in self.keys())

    @property
    def path(self) -> Path:
        """Relative results folder: one level per id segment."""
        return Path(*self.id.split("_"))

    @classmethod
    def from_id(cls, run_id: str) -> "Experiment":
        """Inverse of .id. Ids written before an added variable existed parse with that variable's default."""
        fields = chz.chz_fields(cls)
        vals = {}
        for seg in run_id.strip().strip("/").replace("/", "_").split("_"):
            k, sep, v = seg.partition("-")
            if not sep or k not in fields:
                raise ValueError(f"bad id segment {seg!r} in {run_id!r}")
            vals[k] = _parse(k, v)
        return cls(**vals)

    @property
    def side(self) -> int:
        return math.isqrt(self.patch)

    @property
    def frames_name(self) -> str:
        """Frame cache folder (under the bulk root) holding this patch count's SigLIP2 dense features."""
        return "frames_dense" if self.side == 16 else f"frames_dense_g{self.side}"


def _parse(key: str, text: str):
    if key == "dupmask":
        return None if text == "off" else float(text)
    typ = {"arch": str, "qkv": str, "drop": float, "smooth": float, "lr": float, "beta": float}.get(key, int)
    return typ(text)


@chz.chz
class TrainConfig:
    """Fixed training recipe (the best model's); not part of the run id."""
    wd: float = 0.1                     # AdamW weight decay (matrices only; no decay on biases, norms, embeddings)
    adam_betas: tuple[float, float] = (0.9, 0.98)
    warmup: int = 500                   # linear warmup steps, then cosine decay to 0
    grad_clip: float = 1.0
    ema: float = 0.999                  # EMA of the weights; the EMA model is evaluated and checkpointed
    tok_drop: float = 0.4               # fraction of a frame's screen tokens dropped in training
    step_drop: float = 0.4              # each intermediate step dropped in training (the final screen is kept)
    smooth_tau: float = 0.05            # temperature of the label-smoothing distribution over similar keys
    hard_neg_ramp: float = 0.3          # beta ramps 0 -> beta over this fraction of training
    k_min: float = 1.0                  # confidence kappa = softplus(.) + k_min, clamped to k_max
    k_max: float = 50.0
    k_init: float = 5.0
    mlp_ratio: int = 4
    max_frames: int = 2048              # cap on sessions x steps per batch: batches of long sessions are split
    activation_checkpointing: bool = True   # same results, less GPU memory, ~30% slower
    max_steps: int = 0                  # debug: stop after this many steps (0 = full schedule)


@chz.chz
class EvalConfig:
    eval_steps: int = 500               # validation every this many steps (and at the last step)
    dupmask: float = 0.8                # false-negative threshold in evaluation, fixed so val/test metrics compare
    knn: int = 100                      # neighbors for avg_log_dist_100NN
    batch: int = 64
    max_frames: int = 2048              # cap on sessions x steps per evaluation batch


@chz.chz
class DataConfig:
    root: str = chz.field(default_factory=lambda: os.environ.get(
        "WFEMB_DATA", "/nas/ucb/dfeng/data/workflow-embeddings"))     # prepared/, keys*/ (small files)
    bulk: str = chz.field(default_factory=lambda: os.environ.get(
        "WFEMB_BULK", "/nas/ttl=60d/dfeng/workflow-embeddings"))      # frame caches (large)
    keys: str = "keys_openai"           # label set: OpenAI text-embedding-3-large keys of instructions + paraphrases
    frames: str = ""                    # frame cache folder; default <bulk>/<frames_name of the experiment>
    prefetch_threads: int = 4
    prefetch_depth: int = 4

    def frames_dir(self, exp: Experiment) -> Path:
        return Path(self.frames) if self.frames else Path(self.bulk) / exp.frames_name


def default_results() -> str:
    return os.environ.get("WFEMB_RESULTS", str(REPO / "results"))


@chz.chz
class RunConfig:
    id: str = ""                        # run id; when given it defines the experiment (exp.* must stay default)
    exp: Experiment = chz.field(default_factory=Experiment)
    train: TrainConfig = chz.field(default_factory=TrainConfig)
    eval: EvalConfig = chz.field(default_factory=EvalConfig)
    data: DataConfig = chz.field(default_factory=DataConfig)
    results: str = chz.field(default_factory=default_results)       # results root
    launcher: str = chz.field(default_factory=lambda: os.environ.get("WFEMB_LAUNCHER", "local"))
    overwrite: bool = False             # rerun even if the run already has final.json

    @chz.validate
    def _check(self):
        if self.id and self.exp != Experiment():
            raise ValueError("give either id=... or exp.* overrides, not both")

    @property
    def experiment(self) -> Experiment:
        return Experiment.from_id(self.id) if self.id else self.exp

    @property
    def run_dir(self) -> Path:
        return Path(self.results) / self.experiment.path


def _describe(id: str = "", exp: Experiment = Experiment(), data: DataConfig = DataConfig()):
    exp = Experiment.from_id(id) if id else exp
    print(exp.id)
    print("results:", Path(default_results()) / exp.path)
    print("frames: ", data.frames_dir(exp))


if __name__ == "__main__":
    chz.entrypoint(_describe)
