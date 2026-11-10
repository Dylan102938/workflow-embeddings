"""Run ids, and an end-to-end training run on synthetic data (tests/make_synthetic.py): results files, metric names,
checkpoint reload, eval CLI, migration.  pytest tests/test_pipeline.py"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import main as make_synthetic  # noqa: E402

from wfemb.config import DataConfig, EvalConfig, Experiment, RunConfig, TrainConfig, fmt_value  # noqa: E402

BEST = "arch-late-history-transf_dm-512_dh-64_nl-4_nh-8_drop-0.3_smooth-0.2_patch-576_lr-5e-4_bs-64_ep-45_beta-0_seed-0"


def test_ids():
    assert Experiment().id == BEST
    assert str(Experiment().path) == BEST.replace("_", "/")
    for e in (Experiment(beta=2, seed=1), Experiment(arch="gru", lr=3e-4, drop=0.1, patch=256),
              Experiment(lr=1.5e-3, smooth=0.0, nl=6)):
        assert Experiment.from_id(e.id) == e
        assert Experiment.from_id(str(e.path)) == e
    assert [fmt_value(v) for v in (5e-4, 1.5e-3, 0.3, 2.0, 0.0)] == ["5e-4", "1.5e-3", "0.3", "2", "0"]
    for bad in ("arch-foo", "dm512", "arch-gru_nl-1", "patch-300", BEST.replace("_seed", "_qkv-shared_seed")):
        with pytest.raises(Exception):
            Experiment.from_id(bad)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    make_synthetic(root)
    return root


SMALL = dict(dm=64, dh=16, nh=4, nl=2, patch=16, bs=16, ep=2, lr=1e-3)


@pytest.mark.parametrize("arch,beta", [("late-history-transf", 2.0), ("transf", 0.0), ("gru", 0.0)])
def test_train_end_to_end(synth, tmp_path, arch, beta):
    from wfemb.train import training_loop
    exp = Experiment(arch=arch, beta=beta, **SMALL)
    cfg = RunConfig(exp=exp, train=TrainConfig(max_steps=24, warmup=5),
                    eval=EvalConfig(eval_steps=10, knn=20),
                    data=DataConfig(root=str(synth), bulk=str(synth / "bulk")), results=str(tmp_path))
    final = training_loop(cfg)
    d = tmp_path / exp.path
    assert {p.name for p in d.iterdir()} >= {"config.json", "steps.jsonl", "evals.jsonl", "final.json", "best.pt",
                                            "test_embeddings.npz"}
    steps = [json.loads(l) for l in (d / "steps.jsonl").read_text().splitlines()]
    assert [s["step"] for s in steps] == list(range(1, 25))
    assert {"loss", "rigid_loss", "avg_dist", "lr", "grad_norm", "beta", "epoch", "time"} <= set(steps[0])
    assert steps[-1]["beta"] == pytest.approx(beta)          # ramp done
    if beta == 0:
        assert all(s["loss"] >= 0 for s in steps)
    evals = [json.loads(l) for l in (d / "evals.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [10, 20, 24]
    assert {"val_loss", "val_r@1", "val_r@5", "val_avg_log_dist_20NN", "val_judged_r@5"} <= set(evals[0])
    for k in ("best_step", "test_loss", "test_r@1", "test_r@5", "test_avg_log_dist_20NN", "test_judged_ndcg@10"):
        assert k in final
    c = json.loads((d / "config.json").read_text())
    assert c["id"] == exp.id and Experiment(**c["experiment"]) == exp
    # rerunning a finished run is a no-op
    assert training_loop(cfg) is None


def test_eval_cli_and_migrate(synth, tmp_path):
    from wfemb.train import training_loop
    exp = Experiment(**SMALL)
    data = DataConfig(root=str(synth), bulk=str(synth / "bulk"))
    training_loop(RunConfig(exp=exp, train=TrainConfig(max_steps=12, warmup=2), eval=EvalConfig(eval_steps=6),
                            data=data, results=str(tmp_path)))
    r = subprocess.run([sys.executable, "-m", "wfemb.eval", f"id={exp.id}", f"results={tmp_path}",
                        f"data.root={synth}", f"data.bulk={synth / 'bulk'}"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "test" in r.stdout and "r@5" in r.stdout
    # migrate: a run filed under an outdated folder moves to the folder its config implies
    old = tmp_path / "old" / "place"
    old.parent.mkdir(parents=True)
    (tmp_path / exp.path).rename(old)
    from wfemb.storage.results import iter_runs, migrate
    migrate(tmp_path)
    assert [d for _, d in iter_runs(tmp_path)] == [tmp_path / exp.path]
