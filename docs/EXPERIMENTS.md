# Experiments: variables, run ids, results, metrics

The contract between the training code (`wfemb/`), the run queue (`jobs.txt`), the launchers (`runners/`), the
results tree (`results/`) and the dashboard (`dashboard/`). The code side is `wfemb/config.py` (variables, ids,
recipe) and `wfemb/metrics.py` (metrics). How the model and training work: [ARCHITECTURE.md](ARCHITECTURE.md).

## Experiment variables

A run is defined by these variables only (`config.Experiment`). Each has a short key, used only in run ids.

| id key | field | best model | meaning |
|---|---|---|---|
| `arch` | `architecture` | `late-history-transf` | `transf`, `late-history-transf` or `gru`: what the final block does ([ARCHITECTURE.md](ARCHITECTURE.md)) |
| `dm` | `d_model` | `512` | model width |
| `dh` | `d_head` | `64` | attention head width |
| `nl` | `n_layers` | `4` | blocks, including the final block |
| `nh` | `n_heads` | `8` | attention heads per block |
| `drop` | `dropout` | `0.3` | dropout on attention and MLP outputs and on the state |
| `smooth` | `label_smoothing` | `0.2` | share of the target probability spread over similar instructions (training only) |
| `patch` | `n_patches` | `576` | screen tokens per frame: 256 (16x16 grid), 576 (24x24) or 1024 (32x32) |
| `lr` | `learning_rate` | `5e-4` | peak learning rate |
| `bs` | `batch_size` | `64` | sessions per batch |
| `ep` | `n_epochs` | `45` | passes over the training sessions |
| `beta` | `hard_negative_beta` | `0` | strength of hard-negative weighting (0 = off) |
| `data` | `training_data` | `agentnet` | datasets trained on, joined by `+` (`agentnet+videocua`, ...); `all` = AgentNet and every added dataset ([DATA_PREP.md](DATA_PREP.md#added-datasets)). Validation and test are always AgentNet's. The default reproduces earlier runs; the best model uses `all` (added 2026-11-11) |
| `seed` | `seed` | `0` | seed of initialization, batch order, dropout and paraphrase draws |

Everything else is the fixed recipe, shared by all runs: `TrainConfig`, `EvalConfig` and `DataConfig` in
`wfemb/config.py`. It is recorded in each run's `config.json` but is not part of the id.

| recipe field | value | meaning |
|---|---|---|
| `train.weight_decay` | 0.1 | AdamW, on matrices only (not biases, norms, embedding tables) |
| `train.warmup_steps` | 500 | linear warmup, then cosine decay to 0 |
| `train.grad_clip` | 1.0 | gradient norm clip |
| `train.ema_decay` | 0.999 | the EMA of the weights is what is evaluated and checkpointed |
| `train.screen_token_dropout` | 0.4 | fraction of a frame's screen tokens dropped in training |
| `train.step_dropout` | 0.4 | each step but the last is dropped with this probability in training |
| `train.smoothing_temperature` | 0.05 | temperature of the label-smoothing distribution |
| `train.hard_negative_ramp` | 0.3 | beta rises linearly from 0 over this fraction of training |
| `train.duplicate_threshold` | 0.8 | candidates with cosine above this to the target are not used as negatives |
| `train.confidence_min / max / init` | 1 / 50 / 5 | range and initial value of the confidence |
| `train.max_frames_per_batch` | 2048 | a batch with more sessions x steps is split |
| `train.activation_checkpointing` | on | same results, less GPU memory, about 30% slower |
| `eval.eval_steps` | 500 | validation interval |
| `eval.duplicate_threshold` | 0.8 | as in training; fixed so metrics compare across runs |
| `eval.n_neighbors` | 100 | neighbors in `avg_log_dist_100NN` |
| `data.label_set` | `keys_openai` | OpenAI text-embedding-3-large keys of instructions and paraphrases |

Adding a variable (to ablate a recipe setting): add the field to `Experiment` and its key to `ID_KEYS` before `seed`,
with the default that reproduces earlier runs; document it here; move existing result folders with
`python -m wfemb.storage.results migrate`. A variable whose alternatives are no better than the best model is removed
again, together with its code and the results that used it: delete the field and its key, record the value every
kept run had in `config.REMOVED_VARIABLES` (ids and result folders written with that value are still read), and run
`migrate`, which also deletes the runs that had another value. Removed so far: turning the near-duplicate mask off,
separate Q/K/V projections per token type, and `pca` (screen features without the PCA projection to 512 channels:
worse on validation at 2.25 times the cache size). Added after the original thirteen: `data`.

`patch` chooses the frame cache, `frames_dense[_g<grid>]` (`Experiment.frame_cache_name`); a new grid needs its
cache built first ([DATA_PREP.md](DATA_PREP.md)). A run with added datasets also needs that
cache for the added set (`BULK/added/`).

`data` changes the training sessions and the training candidates only. `ep` counts passes over the run's own
training sessions, so a run with more data takes more optimizer steps. Evaluation always scores against AgentNet's
training instructions plus the evaluated split's, so `val_*` and `test_*` metrics compare across `data` values.

## Command line

```
python -m wfemb.train id=<run id>                                      # a run by its id
python -m wfemb.train experiment.hard_negative_beta=2 experiment.seed=1   # the best model with changes
python -m wfemb.train id=<run id> train.max_steps=200 results_root=/tmp/r # any recipe field can be overridden
python -m wfemb.eval id=<run id> [split=val] [finalize=True]
python -m wfemb.config id=<run id>                                     # canonical id, results folder, frame cache
```

## Run ids

`arch-<v>_dm-<v>_dh-<v>_nl-<v>_nh-<v>_drop-<v>_smooth-<v>_patch-<v>_lr-<v>_bs-<v>_ep-<v>_beta-<v>_data-<v>_seed-<v>`

Values never contain `_`; a key is everything before the first `-` of a segment. Numbers are canonical: integers as
integers (`beta-0`, `beta-2`), floats below 0.01 in short scientific form (`lr-5e-4`, `lr-1.5e-3`), other floats with
`%g` (`drop-0.3`). `wfemb.config.Experiment.id` / `Experiment.from_id` round-trip exactly; an id written before a variable was added
is read with that variable's default. The best model so far is

`arch-late-history-transf_dm-512_dh-64_nl-4_nh-8_drop-0.3_smooth-0.2_patch-576_lr-5e-4_bs-64_ep-45_beta-0_data-all_seed-0`

Results are always reported and named with these ids.

## Results tree

Every run writes raw data only (no summaries; the dashboard computes those) to
`<results root>/<one folder per id segment>/`, e.g.
`results/arch-late-history-transf/dm-512/dh-64/nl-4/nh-8/drop-0.3/smooth-0.2/patch-576/lr-5e-4/bs-64/ep-45/beta-0/data-agentnet/seed-0/`:

| file | kind | content |
|---|---|---|
| `config.json` | json | `{"id", "experiment": {...}, "train": {...}, "eval": {...}, "data": {...}, "model": {...}, "meta": {...}}` (meta: git commit, host, launcher, start time, n_params, steps_per_epoch, total_steps; `legacy` / `legacy_name` for imported pre-refactor runs) |
| `steps.jsonl` | jsonl, one row per training step | `step, epoch, time, loss, rigid_loss, avg_dist, lr, grad_norm, beta, data_s, step_s` |
| `evals.jsonl` | jsonl, one row every `eval_steps` (and at the last step) | `step, epoch, time, val_loss, val_r@1, val_r@5, val_avg_log_dist_100NN, max_mem_gb` |
| `final.json` | json, written once at the end | `best_step, selected_on` and every `val_*` / `test_*` metric at the chosen checkpoint (the test metrics also per group: `test_unseen_apps_*`, `test_seen_app_*`), `train_minutes` |
| `test_embeddings.npz` | npz, written once at the end | every test step's direction (float16) and confidence at the chosen checkpoint: `session_indices`, `step_offsets`, `lengths` (one entry per session) and `directions`, `confidences` (one row per step; session i owns rows `step_offsets[i]` to `step_offsets[i] + lengths[i] - 1`). The demo is built from it. About 360 MB; fetched with `--embeddings` |
| `best.pt`, `last.pt` | checkpoints | EMA weights and the model configuration; fetched with `--checkpoints` |

`time` is seconds since the run started. A run without `final.json` is still running (or died). The results root is
`$WFEMB_RESULTS`, default `<repo>/results` (gitignored). Runs on rnn write to `$WFEMB_DATA/results`, runs on Baseten to
the job's checkpoint directory; `python -m wfemb.storage.fetch rnn|baseten` copies them into the local `results/`.

## Run queue

`jobs.txt` (repo root) lists every run by id, in named sections; one section is one launch (a Baseten job, or one
rnn submission packed onto GPUs). Format: `[section]` lines, then one `<run id> [overrides]` per line, `#` comments.
`python -m wfemb.storage.jobs list` shows each run's status; the dashboard's Queue view shows the same. Launching a
section: `WFEMB_JOB=<section> WFEMB_GPUS=2 truss train push runners/baseten/train_config.py --job-name <section>` on
Baseten, `runners/slurm/submit.sh <section>` on rnn. A run that already has `final.json` is not rerun. If a run
trained fully but lost its final evaluation, `python -m wfemb.eval id=<run id> finalize=True` redoes it
(`runners/baseten/evaluate.sh` does this on Baseten).

## Metrics (wfemb/metrics.py)

A session's workflow embedding is its direction at its last step. Distances are cosine distances, 1 - cos. Candidates in training are the keys of all N training instructions; in evaluation, the training keys plus the
evaluated split's keys (val/test instructions never occur in training).

- `loss`: the optimized objective on the batch. Per step: cross-entropy of softmax(confidence * cos(direction, key))
  over the candidates (each shown with a random wording: the instruction or one of its paraphrases), false negatives
  (same instruction group, or cosine above 0.8 to the target) removed, label-smoothed target, hard-negative weights.
  Averaged over a session's steps, then over sessions.
- `rigid_loss`: the same forward pass and batch scored with a one-hot target: no label smoothing and no hard-negative
  weights (so it means the same in every run; with beta = 0 it is exactly the training loss without smoothing).
- `avg_dist`: mean distance between each session's workflow embedding and the instruction keys of the other sessions in
  the batch, excluding its own label (its key and those of batch sessions with the same instruction).
- `val_loss` / `test_loss`: per-step cross-entropy with one-hot targets over the evaluation candidates, false negatives
  masked at the fixed threshold 0.8 (so runs compare), averaged over steps then sessions; every `eval_steps` on the
  whole validation set; on the test set once, at the chosen checkpoint (lowest `val_loss`).
- `val_r@1`, `val_r@5` (and `test_*`): fraction of sessions whose own instruction key is ranked first / in the top 5
  among the candidates at the session's last step.
- `val_avg_log_dist_100NN` (and `test_*`): for each session, the 100 candidates nearest its workflow embedding, minus
  its own label if it is among them; the mean log distance to them; averaged over sessions.
- `test_unseen_apps_<metric>`, `test_seen_app_<metric>`: each test metric again over two groups of test sessions. A
  test session has at least one held-out application, but it may use others too. `unseen_apps` are the sessions
  none of whose applications occurs in AgentNet's training split (1,697 of 3,050); `seen_app` are those with at
  least one that does (1,353). The groups are the same sessions for every run, whatever `data` it trains on. The
  overall metric is their session-weighted mean. Validation has no such groups: it holds out instructions, not
  applications.
