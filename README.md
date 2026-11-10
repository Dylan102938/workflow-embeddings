# workflow-embeddings

Session embeddings for computer-use workflows, trained on AgentNet (OpenCUA). A per-step transformer reads one
(screenshot, input events) step at a time and carries a history token from step to step. At each step it outputs a
direction in a frozen text-embedding space (the intent it currently believes) and a confidence κ; training is
per-step InfoNCE against every training instruction's frozen text key.

- **Experiments (variables, run ids, metrics, results layout):** [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- Earlier write-ups (pre-refactor run names): [docs/REPORT.md](docs/REPORT.md), [docs/RESULTS.md](docs/RESULTS.md)

## Layout

```
wfemb/
  config.py      experiment variables + run ids (Experiment), fixed recipe (TrainConfig, EvalConfig, DataConfig), CLI
  data.py        sessions, splits, batches for train / val / test, training label set
  train.py       training_step, training_loop
  eval.py        evaluation on val / test, pretty-printed metrics
  metrics.py     the objective and every tracked metric
  model/         transformer.py (workflow encoder), events.py (event tokens), gru.py (GRU final block)
  storage/       paths.py (data layout), frames.py (frame-cache reader), results.py (results tree), fetch.py
  prep/          one-time data preparation (sessions, actions, frame caches, text embeddings, keys, judged pools)
  demo/          exports Intent Replay bundles for the dashboard
dashboard/       local results dashboard (runs, curves, seed aggregation, ablations, job queue, demo viewer)
jobs.txt         the run queue: run ids in named sections, one section per launch
runners/         baseten/ (data download, frame-cache build, training) and slurm/ (rnn jobs, legacy import)
results/         raw results of every run, one folder level per id segment (gitignored)
```

## Running

```bash
uv sync
python -m wfemb.config exp.beta=2                         # id, results folder and frame cache of an experiment
python -m wfemb.train id=<run id>                         # train one run (or exp.<var>=... overrides)
python -m wfemb.eval id=<run id>                          # test metrics of its best checkpoint

python -m wfemb.storage.jobs list                         # jobs.txt sections and each run's status
WFEMB_JOB=<section> WFEMB_GPUS=2 truss train push runners/baseten/train_config.py --job-name <section>   # on Baseten
runners/slurm/submit.sh <section>                         # on rnn (Slurm), runs packed 3 per GPU job

python -m wfemb.storage.fetch baseten                     # results -> ./results  (or: rnn)
python dashboard/server.py                                # http://127.0.0.1:8765
pytest tests/                                             # ids + end-to-end training on synthetic data
```
