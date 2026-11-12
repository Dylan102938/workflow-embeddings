# Architecture: data, model, training, evaluation

How the code under `wfemb/` works and why. Variables, run ids, metric definitions and file formats are in
[EXPERIMENTS.md](EXPERIMENTS.md); how the input files are produced is in [DATA_PREP.md](DATA_PREP.md).

## The task

A session is one recorded computer-use workflow: a task instruction and T steps. Sessions come from AgentNet and,
as extra training data, from the datasets added to it ([DATA_PREP.md](DATA_PREP.md#added-datasets)). Each step is the
screenshot the user saw and the input events they performed on it; the last step is the final screen, without events.
The model reads a session step by step and, at every step, outputs

- a **direction**: a unit vector in the frozen text-embedding space of the instructions (what it believes the task is),
- a **confidence**: a positive scalar, the inverse temperature of its belief.

A session's **workflow embedding** is its direction at the last step. Nothing in the text-embedding space is trained:
instructions are embedded once with OpenAI text-embedding-3-large (their *keys*, 3072-d).

## Data (`wfemb/data.py`)

**Splits** are fixed once by `prep/prepare.py`, grouped so nothing leaks:

- *test*: every session of a held-out set of applications, plus any session that shares an instruction group with one;
- *val*: the sessions of a random fraction of the remaining instruction groups;
- *train*: the rest.

Sessions with the same normalized instruction form a *group*; a group never spans two splits.

The splits are AgentNet's. The added datasets are training data only and a second prepared set on disk; a run whose
`data` variable names any of them loads both sets as one (`stack`): the added sessions follow AgentNet's, and the
training sessions of datasets the run does not name are left unused.

**Inputs per step**, all frozen and precomputed:

- *screen tokens* `[n_patches, 512]`: SigLIP2 so400m patch features of the screenshot, pooled to a 16x16, 24x24 or
  32x32 grid, channel PCA 1152 -> 512 (variable `pca`; 0 keeps all 1152 channels), stored as exact bfloat16;
- *events*, up to 8 per step: type, up to two screen points, scroll amount, up to 4 key tokens, and typed text (an
  index into the frozen Qwen3 embeddings of every typed string).

**Batches.** `Batch` holds B sessions padded to T steps; padding repeats a session's last step and is masked by
`valid`. Training batches (`train_batches`): each epoch the training sessions are shuffled, sorted by length within
buckets of 50 batches so a batch pads little, and cut into batches of `batch_size` sessions. A batch with more than
`max_frames_per_batch` sessions x steps is split rather than truncating sessions; split batches count as steps, so an
epoch has slightly more optimizer steps than sessions / batch_size. Each step but a session's last is dropped with
probability `step_dropout`. Evaluation batches (`eval_batches`) visit every session of a split once, nothing dropped.

**Candidates.** In training, every step is scored against the keys of all N training instructions of the run
(`TrainingCandidates`). Each step, every candidate is shown with a random wording: the instruction's key or the key of
one of its paraphrases, so wording style cannot separate the target from the negatives. In evaluation the candidates
are AgentNet's training keys plus the evaluated split's keys, whatever the run trained on (`evaluation_candidates`):
more training data must not make the evaluation harder. Val and test instructions never occur in training.

## Model (`wfemb/model/`)

At every step the model runs a transformer over one sequence:

    [ screen tokens | one token per event | history token ]

- Screen tokens are the projected patch features plus a learned position per patch. In training a random
  `screen_token_dropout` fraction of them is removed (fewer tokens, not masked ones).
- Each event is its own token (`model/events.py`): the sum of embeddings of its type, points (Fourier features),
  scroll, key tokens and typed text, through a residual MLP, plus a learned position for its order within the step.
  A step without events gets a learned no-event token.
- The history token carries the state from the previous step. The state after this step is read at its position.
- Every token gets a learned token-type embedding. There is no causal mask: a step sees only its own screen, its own
  events and the state. The only masks hide empty event slots.

The heads map the state to the direction (linear, normalized) and the confidence (softplus, clamped to
`[confidence_min, confidence_max]`). After a session's last step its state is frozen.

The `architecture` variable is what the final block does:

| architecture | blocks | where the history token is |
|---|---|---|
| `transf` | `n_layers` transformer blocks | in every block |
| `late-history-transf` | `n_layers` transformer blocks | only in the last block; the earlier blocks see screen and event tokens only, so they run for all steps of the batch at once |
| `gru` | `n_layers - 1` transformer blocks, then a GRU cell | in every block; the GRU combines the last block's output at the history position with the previous state |

The last transformer block computes only the history position's output (nothing else is used). Blocks are pre-norm
with GPT-2 style initialization. With `activation_checkpointing`, block activations are recomputed in the backward
pass: identical results, much less memory, about 30% slower.

## Training (`wfemb/train.py`)

`training_step` is one optimizer update:

1. Targets: each session's own instruction among the candidates. The **false-negative mask** removes candidates that
   are not real negatives for the session: those of its instruction group, and those whose key has cosine above
   `duplicate_threshold` to the target's key.
2. Forward pass, then logits `confidence * cos(direction, candidate key)` for every step and candidate.
3. Loss: per-step cross-entropy, averaged over a session's valid steps, then over sessions (InfoNCE against all
   training instructions). Two modifications apply in training only:
   - **label smoothing**: `label_smoothing` of the target probability is spread over the other candidates in
     proportion to softmax(key similarity / `smoothing_temperature`);
   - **hard negatives**: negative j gets weight exp(beta * cos(target key, key j)), normalized to mean 1, added to the
     logits as a bias. Beta rises linearly from 0 over the first `hard_negative_ramp` of training.
4. Backward, gradient clipping, AdamW step, EMA update of the weights, learning-rate schedule step.

Weight decay applies to matrices only. Parameters whose name contains `embedding` are excluded; the position tables
(`screen_positions`, `event_positions`) are decayed. Keep that in mind when renaming parameters.

`training_loop` evaluates the EMA model on the validation set every `eval_steps`, keeps the checkpoint with the lowest
validation loss (`best.pt`), and at the end evaluates that checkpoint on validation and test (`final.json`,
`test_embeddings.npz`).

## Evaluation (`wfemb/eval.py`)

`evaluate` embeds every session of a split, then scores each step against the candidates with the false-negative mask
at the fixed `eval.duplicate_threshold`, with one-hot targets (no smoothing, no hard-negative weights). Metric
definitions: [EXPERIMENTS.md](EXPERIMENTS.md#metrics-wfembmetricspy).

## Reading the frame cache (`wfemb/storage/frames.py`)

The cache for 24x24 grids is 225 GB (100 GB for 16x16), shared by the runs on a machine and often on NFS. These rules
were each learned from a stalled or crashed job; changing them needs a test on the cluster:

- **preadv, not memory maps.** Several long-running multithreaded processes mapping the same file stall on page
  faults (seconds per batch) and after about an hour wedge on the mmap lock.
- **One read per session, not per frame.** A session's steps are contiguous in the cache. Per-frame reads hand the
  GIL back and forth with the training thread and were about 50x slower.
- **At most 1 MB per read call.** Larger single reads livelocked the NFS client on memory-pressured nodes. The same
  applies to loading the small arrays (`load_numpy`).
- **No large per-batch allocations.** Frames are read into a fixed pool of pinned buffers and copied to the GPU
  synchronously, after which the buffer is reused. Fresh multi-hundred-MB allocations from several threads also
  wedged processes.
- **Frames stay on the CPU in exact bfloat16.** The cache does not fit on a GPU, and a lossy GPU copy was rejected.

`train.setup_process` lowers the interpreter's thread switch interval for the same reason (prefetch threads starve
behind the training loop at the default), and arms a watchdog that dumps every thread's stack if a run makes no
progress for 15 minutes.

Packing runs: two or three runs can share a GPU and the page cache of the frame files. Leave GPU memory headroom for
the final test evaluation; two runs evaluating at once on an 80 GB GPU have run out of memory.
