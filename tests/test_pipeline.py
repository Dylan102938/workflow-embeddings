"""Run ids, and end-to-end training on synthetic data (tests/make_synthetic.py)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import main as make_synthetic  # noqa: E402

from wfemb.config import (
    DataConfig,
    EvalConfig,
    Experiment,
    RunConfig,
    TrainConfig,
    format_id_value,  # noqa: E402
    parse_run_config,
)

BEST = (
    "arch-late-history-transf_dm-512_dh-64_nl-4_nh-8_drop-0.3_smooth-0.2"
    "_patch-576_lr-5e-4_bs-64_ep-45_beta-0_pca-512_data-agentnet_seed-0"
)
SMALL = dict(d_model=64, d_head=16, n_heads=4, n_layers=2, n_patches=16, batch_size=16, n_epochs=2, learning_rate=1e-3)


def test_ids_round_trip():
    assert Experiment().id == BEST
    assert str(Experiment().relative_dir) == BEST.replace("_", "/")
    experiments = [
        Experiment(hard_negative_beta=2, seed=1),
        Experiment(architecture="gru", learning_rate=3e-4, dropout=0.1, n_patches=256),
        Experiment(learning_rate=1.5e-3, label_smoothing=0.0, n_layers=6),
        Experiment(n_patches=1024, pca_dim=0),
    ]
    for experiment in experiments:
        assert Experiment.from_id(experiment.id) == experiment
        assert Experiment.from_id(str(experiment.relative_dir)) == experiment
    assert [format_id_value(v) for v in (5e-4, 1.5e-3, 0.3, 2.0, 0.0)] == ["5e-4", "1.5e-3", "0.3", "2", "0"]


@pytest.mark.parametrize("bad_id", ["arch-foo", "dm512", "arch-gru_nl-1", "patch-300", "qkv-shared_seed-0"])
def test_bad_ids_are_rejected(bad_id):
    with pytest.raises(ValueError):
        Experiment.from_id(bad_id)


def test_ids_without_a_later_variable_take_its_default():
    assert Experiment.from_id(BEST.replace("_pca-512", "")) == Experiment()


def test_frame_cache_names():
    names = {(256, 512): "frames_dense", (576, 512): "frames_dense_g24", (1024, 512): "frames_dense_g32"}
    names |= {(256, 0): "frames_dense_raw", (576, 0): "frames_dense_g24_raw", (576, 768): "frames_dense_g24_pca768"}
    for (n_patches, pca_dim), name in names.items():
        assert Experiment(n_patches=n_patches, pca_dim=pca_dim).frame_cache_name == name


def test_command_line_accepts_an_id_and_overrides():
    config = parse_run_config([f"id={BEST.replace('beta-0', 'beta-2')}", "train.max_steps=5", "experiment.seed=3"])
    assert config.experiment == Experiment(hard_negative_beta=2, seed=3)
    assert config.train.max_steps == 5


@pytest.fixture(scope="module")
def data_config(tmp_path_factory) -> DataConfig:
    root = tmp_path_factory.mktemp("data")
    make_synthetic(root)
    return DataConfig(root=str(root), bulk_root=str(root / "bulk"))


@pytest.mark.parametrize("architecture,beta", [("late-history-transf", 2.0), ("transf", 0.0), ("gru", 0.0)])
def test_training_writes_the_results_files(data_config, tmp_path, architecture, beta):
    from wfemb.data import WorkflowDataset
    from wfemb.train import training_loop

    experiment = Experiment(architecture=architecture, hard_negative_beta=beta, **SMALL)
    config = RunConfig(
        experiment=experiment,
        train=TrainConfig(max_steps=24, warmup_steps=5),
        eval=EvalConfig(eval_steps=10, n_neighbors=20),
        data=data_config,
        results_root=str(tmp_path),
    )
    final = training_loop(config)

    run_dir = tmp_path / experiment.relative_dir
    assert {path.name for path in run_dir.iterdir()} >= {
        "config.json",
        "steps.jsonl",
        "evals.jsonl",
        "final.json",
        "best.pt",
        "last.pt",
        "test_embeddings.npz",
    }
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
    assert [row["step"] for row in steps] == list(range(1, 25))
    assert {"loss", "rigid_loss", "avg_dist", "lr", "grad_norm", "beta", "epoch", "time"} <= set(steps[0])
    assert steps[-1]["beta"] == pytest.approx(beta)
    evals = [json.loads(line) for line in (run_dir / "evals.jsonl").read_text().splitlines()]
    assert [row["step"] for row in evals] == [10, 20, 24]
    assert {"val_loss", "val_r@1", "val_r@5", "val_avg_log_dist_20NN"} <= set(evals[0])
    assert {"best_step", "test_loss", "test_r@1", "test_r@5", "test_avg_log_dist_20NN"} <= set(final)
    assert "val_unseen_apps_loss" not in final
    groups = WorkflowDataset(data_config, experiment).test_groups(np.arange(255, 300))
    assert 0 < groups["unseen_apps"].sum() < 45 and (groups["unseen_apps"] ^ groups["seen_app"]).all()
    for metric in ("loss", "r@1", "r@5", "avg_log_dist_20NN"):
        of_groups = sum(final[f"test_{group}_{metric}"] * members.sum() for group, members in groups.items()) / 45
        assert of_groups == pytest.approx(final[f"test_{metric}"], rel=1e-4, abs=1e-5)
    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["id"] == experiment.id and Experiment(**saved["experiment"]) == experiment

    assert training_loop(config) is None, "a finished run is not rerun"


def test_eval_command_and_migration(data_config, tmp_path):
    from wfemb.storage.results import iter_runs, migrate
    from wfemb.train import training_loop

    experiment = Experiment(**SMALL)
    training_loop(
        RunConfig(
            experiment=experiment,
            train=TrainConfig(max_steps=12, warmup_steps=2),
            eval=EvalConfig(eval_steps=6),
            data=data_config,
            results_root=str(tmp_path),
        )
    )

    command = [
        sys.executable,
        "-m",
        "wfemb.eval",
        f"id={experiment.id}",
        f"results_root={tmp_path}",
        f"data.root={data_config.root}",
        f"data.bulk_root={data_config.bulk_root}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "test" in completed.stdout and "r@5" in completed.stdout

    misplaced = tmp_path / "old" / "place"
    misplaced.parent.mkdir(parents=True)
    (tmp_path / experiment.relative_dir).rename(misplaced)
    migrate(tmp_path)
    assert [run_dir for _, run_dir in iter_runs(tmp_path)] == [tmp_path / experiment.relative_dir]
