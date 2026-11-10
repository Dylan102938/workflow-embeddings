# Experiments: variables, run ids, metrics, results layout

This is the contract between the training code (`wfemb/`), the run queue (`jobs.txt`), the launchers (`runners/`), the
local
results tree (`results/`) and the dashboard (`dashboard/`). Definitions live in code in `wfemb/config.py`
(variables, ids) and `wfemb/metrics.py` (metrics).

## Experiment variables

A run is defined by these variables only. Everything else is the fixed recipe of the best model so far
(`TrainConfig` / `EvalConfig` / `DataConfig` in `wfemb/config.py`); it is recorded in each run's `config.json` but does
not define a distinct experiment.

| key | name | default (best model) | meaning |
|---|---|---|---|
| `arch` | architecture | `late-history-transf` | what the final block does (see below) |
| `dm` | d_model | `512` | model width (screen, event and history tokens all live in it) |
| `dh` | d_head | `64` | attention head width |
| `nl` | n_layers | `4` | blocks, including the final block |
| `nh` | n_heads | `8` | attention heads per block |
| `drop` | dropout | `0.3` | dropout on attention/MLP outputs and on the state |
| `smooth` | label smoothing | `0.2` | eps of the similarity-based label smoothing (training only) |
| `patch` | n_patches | `576` | screen patch tokens per frame: 256 = 16x16 grid, 576 = 24x24 grid (frozen SigLIP2 so400m, channel PCA 1152 -> 512) |
| `lr` | learning rate | `5e-4` | peak AdamW lr (500 warmup steps, cosine decay to 0) |
| `bs` | batch size | `64` | sessions per batch (very long sessions split a batch, see `train.max_frames`) |
| `ep` | epochs | `45` | passes over the training sessions |
| `beta` | hard-negative beta | `0` | negatives weighted by exp(beta * cos(target key, negative key)), normalized to mean 1; beta ramps in linearly over the first 30% of training |
| `seed` | seed | `0` | torch / numpy seed (init, batch order, dropout, paraphrase draws) |

To ablate a recipe setting, add it as a variable: register it in `wfemb/config.py:Experiment` before `seed`, with the
default that reproduces earlier runs, document it here (marked *added*), and move existing result folders with
`python -m wfemb.storage.results migrate` (it re-derives every path from the run's `config.json`). A variable whose
alternatives turn out no better than the best model is removed again, with its code. Removed so far: the near-duplicate
mask threshold (off was no better than 0.8; fixed at `train.dupmask = 0.8`) and per-token-type Q/K/V projections
(`sep-q` was worse than shared projections).

### Architectures

Every step t is one transformer pass over `[screen patch tokens | one token per input event | history token]` (no causal
mask; padding masks only). The model's state h_t is read from the history position; the heads map it to a unit
direction d_t in the frozen 3072-d text-key space and a confidence kappa_t.

- `transf`: the history token h_{t-1} takes part in every block.
- `late-history-transf`: the first `nl - 1` blocks see only the step's screen and event tokens (they run for all frames
  in parallel); h_{t-1} joins in the last block, which computes only the history row.
- `gru`: `nl - 1` transformer blocks with the history token in every block, then a GRU cell as the final block:
  h_t = LayerNorm(GRU(LayerNorm(output at the history position), h_{t-1})).

## Run ids

`arch-<v>_dm-<v>_dh-<v>_nl-<v>_nh-<v>_drop-<v>_smooth-<v>_patch-<v>_lr-<v>_bs-<v>_ep-<v>_beta-<v>_seed-<v>`

Values never contain `_`; a key is everything before the first `-` of a segment. Numbers are canonical: integers as
integers (`beta-0`, `beta-2`), floats below 0.01 in short scientific form (`lr-5e-4`, `lr-1.5e-3`), other floats with
`%g` (`drop-0.3`). `wfemb.config.Experiment.id` / `Experiment.from_id` round-trip exactly. The best model so far is

`arch-late-history-transf_dm-512_dh-64_nl-4_nh-8_drop-0.3_smooth-0.2_patch-576_lr-5e-4_bs-64_ep-45_beta-0_seed-0`

Results are always reported and named with these ids.

## Results tree

Every run writes raw data only (no summaries; the dashboard computes those) to
`<results root>/<one folder per id segment>/`, e.g.
`results/arch-late-history-transf/dm-512/dh-64/nl-4/nh-8/drop-0.3/smooth-0.2/patch-576/lr-5e-4/bs-64/ep-45/beta-0/seed-0/`:

| file | kind | content |
|---|---|---|
| `config.json` | json | `{"id", "experiment": {...}, "train": {...}, "eval": {...}, "data": {...}, "meta": {...}}` (meta: git commit, host, launcher, start time, n_params, steps_per_epoch, total_steps; `legacy` / `legacy_name` for imported pre-refactor runs) |
| `steps.jsonl` | jsonl, one row per training step | `step, epoch, time, loss, rigid_loss, avg_dist, lr, grad_norm, beta, data_s, step_s` |
| `evals.jsonl` | jsonl, one row every `eval_steps` (and at the last step) | `step, epoch, time, val_loss, val_r@1, val_r@5, val_avg_log_dist_100NN, val_judged_r@1, val_judged_r@5, val_judged_ndcg@10, max_mem_gb` |
| `final.json` | json, written once at the end | `best_step, selected_on` and every `val_*` / `test_*` metric at the chosen checkpoint, `train_minutes` |
| `best.pt`, `last.pt` | checkpoints | model (EMA) weights + config; not downloaded by default |

`time` is seconds since the run started. A run without `final.json` is still running (or died). The results root is
`$WFEMB_RESULTS`, default `<repo>/results` (gitignored). Runs on rnn write to `$WFEMB_DATA/results`, runs on Baseten to
the job's checkpoint directory; `python -m wfemb.storage.fetch rnn|baseten` copies them into the local `results/`.

## Run queue

`jobs.txt` (repo root) lists every run by id, in named sections; one section is one launch (a Baseten job, or one
rnn submission packed onto GPUs). Format: `[section]` lines, then one `<run id> [overrides]` per line, `#` comments.
`python -m wfemb.storage.jobs list` shows each run's status; the dashboard's Queue view shows the same.

## Metrics (wfemb/metrics.py)

Notation: a session's workflow embedding e is its direction at its last step, d_T. Distances are cosine distances
1 - cos. Candidates in training are the keys of all N training instructions; in evaluation, the training keys plus the
evaluated split's keys (val/test instructions never occur in training).

- `loss`: the optimized objective on the batch. Per step t: cross-entropy of softmax(kappa_t * cos(d_t, key_j)) over
  the candidates (each shown with a random wording: the instruction or one of its paraphrases), false negatives
  (same instruction group, or cos > 0.8 to the target) removed, label-smoothed target (`smooth`), hard-negative
  weights (`beta`). Averaged over a session's steps, then over sessions.
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
- `val_judged_r@1`, `val_judged_r@5`, `val_judged_ndcg@10` (and `test_*`): session-to-session retrieval of the
  workflow embeddings across apps, with pooled LLM-judged relevance (`keys/judged_<split>.jsonl`).
