# Data preparation

`wfemb/prep/` turns the raw AgentNet download into the files training reads. Every step is run once, offline.
This page lists the steps in order, what each one writes, and the design choices that are not obvious from the code.

## Where the data lives

`wfemb/storage/paths.py` builds every path from two roots.

| Root | Environment variable | Holds |
|---|---|---|
| `DATA` | `WFEMB_DATA` | everything that must persist (on rnn: the NAS) |
| `BULK` | `WFEMB_BULK` | large caches that can be rebuilt (on rnn: the 60-day TTL share, so they stay out of the home quota) |

```
DATA/agentnet_raw/     the AgentNet download: two jsonl files, meta_data_merged.jsonl, split image archives
DATA/prepared/         sessions.pkl, events.npz, vocab.json, summary.json                  (prepare.py)
DATA/keys/             the default key set (Qwen3-Embedding-0.6B) and the files every key set shares:
                       paraphrases.json, judged_<split>.jsonl, text_vecs.npy
DATA/keys_openai/      the label key set used for training (OpenAI text-embedding-3-large)   (openai_keys.py)
BULK/frames_dense[_g<grid>]/    frozen screen features, one folder per grid size             (cache_frames.py)
BULK/dense_pca_g<grid>_d<dim>.npz   the PCA basis of that cache
```

Training results are not stored here; see `wfemb/storage/results.py`.

## Pipeline

Each step has a Slurm launcher in `runners/slurm/`; arguments after the launcher name are passed to the script.

| # | Step | Command | Needs | Writes |
|---|---|---|---|---|
| 0 | Download AgentNet | `sbatch runners/slurm/download_agentnet.sbatch` | | `DATA/agentnet_raw/` |
| 1 | Sessions, splits, events | `sbatch runners/slurm/prepare.sbatch` | 0 | `DATA/prepared/` |
| 2 | Default text keys | `sbatch runners/slurm/text_embed.sbatch` | 1 | `DATA/keys/keys.npy`, `text_vecs.npy`, `meta.json` |
| 3 | Paraphrases | `sbatch runners/slurm/paraphrase.sbatch` | 1 | `DATA/keys/paraphrases.json` |
| 4 | Label key set | `sbatch runners/slurm/openai_keys.sbatch` | 2, 3 | `DATA/keys_openai/` |
| 5 | Judged relevance pools | `sbatch runners/slurm/judge.sbatch --split val --mode batch` (and `--split test`) | 1, 2 | `DATA/keys/judged_<split>.jsonl` |
| 6 | Frame cache: PCA basis | `sbatch --array=0 runners/slurm/cache_frames.sbatch --mode pca_fit --grid 24` | 1 | `BULK/dense_pca_g24_d512.npz` |
| 7 | Frame cache: shards | `sbatch runners/slurm/cache_frames.sbatch --mode dense --grid 24` (array of 8 shards) | 6 | `BULK/frames_dense_g24/` |

Steps 2 to 5 and steps 6 to 7 are independent of each other. On Baseten, `runners/baseten/build.sh` runs step 7
from PCA bases fitted on rnn.

**Everything after step 1 is indexed by position in `sessions.pkl`** (session index) or by global step index.
Re-running `prepare.py` with different data, filters or seed invalidates the key sets, the judged pools and the
frame caches.

To run a script directly: `python -m wfemb.prep.<module> --help`.

## 1. `prepare.py`: sessions, splits, events

**Filters.** A trajectory is dropped if its task id was already seen, if its screenshot sequence was already seen,
if the AgentNet metadata marks the task description as low or not aligned, or if it has fewer than 2 steps.

**App keys.** Each session gets keys `app:<name>` for its non-browser applications and `web:<site>` for its websites.
Its primary key is the first non-browser application, else the first website that is not a search engine, else the
first website, else its browser, else `domain:<domain>`.

**Splits.** Nothing leaks between splits:

- *test* holds out whole apps. Primary app keys are shuffled and taken until they cover `--test_frac` of the
  sessions; apps with more than 1.5% of all sessions are never held out. A session is in test if any of its app
  keys is a test app.
- *val* holds out whole instruction groups (sessions whose instructions are equal ignoring case and whitespace),
  taken until they cover `--val_frac` of the remaining sessions.
- A session whose instruction group has a member in test is moved to test.

**Actions.** `actions.py` parses each step's code (`pyautogui.*` / `computer.*` calls) into events. Coordinates are
normalized to [0, 1]. A pointer position is carried across calls and steps.

| Call | Event fields |
|---|---|
| `click`, `doubleClick`, `tripleClick`, `rightClick`, `middleClick`, `mouseDown`, `mouseUp` | `x`, `y` (the pointer if not given) |
| `moveTo` | `x`, `y` |
| `dragTo` | `x`, `y` = start (the pointer), `x2`, `y2` = end |
| `scroll`, `hscroll` | `scroll` amount, `x`, `y` (the pointer if not given) |
| `press`, `hotkey`, `keyDown`, `keyUp` | `keys` (lowercase key tokens) |
| `write`, `typewrite` | `text` |
| `wait`, `terminate`, anything else (`other`) | none |

`terminate` events are dropped so the model gets no artificial end-of-session cue. The terminate step itself stays,
without events: its screenshot is the real final screen.

### Files

`sessions.pkl`: a list of dicts in global session order (train, then val, then test; by id within a split).

| Key | Value |
|---|---|
| `id`, `source` | AgentNet task id; `ubuntu` or `win_mac` (which jsonl and image archive) |
| `instruction`, `group` | the task instruction; hash shared by sessions with the same instruction |
| `app`, `app_keys` | primary app key; all app keys, sorted |
| `os`, `domain`, `task_completed`, `screen` | from the AgentNet metadata; `screen` is (width, height) |
| `split` | `train`, `val` or `test` |
| `images` | screenshot file name of each step |

`events.npz`: one row per step, in global step order (sessions in order, steps by index). Session `i` owns rows
`sess_offset[i] : sess_offset[i] + sess_len[i]`. A step keeps at most 8 events; an event keeps at most 4 keys.

| Array | Shape, dtype | Content |
|---|---|---|
| `ev_type` | [N, 8] int8 | index into `actions.EVENT_TYPES`; 0 = empty slot |
| `ev_xy` | [N, 8, 4] float16 | `x, y, x2, y2`; NaN = not set |
| `ev_scroll` | [N, 8] float16 | scroll amount, clipped to ±1000 |
| `ev_keys` | [N, 8, 4] int16 | index into `key_vocab`; 0 = pad, 1 = unknown |
| `ev_text` | [N, 8] int32 | index into `texts` (= row of `text_vecs.npy`); -1 = none |
| `focus` | [N, 2] float32 | where the step acts: its last event with a position, else the pointer, else (0.5, 0.5) |
| `sess_offset`, `sess_len` | [S] int64, int32 | first row and number of steps of each session |

`vocab.json`: `key_vocab` (`<pad>`, `<unk>`, then the key tokens seen at least `--key_min_count` times, sorted),
`texts` (every typed string, in first-seen order), `e_max` = 8, `k_max` = 4.

`summary.json`: filter counts, split sizes, the held-out test apps, per-split OS counts.

## 2 to 4. Key sets

A key set is a directory of frozen text embeddings from one embedder:

| File | Shape, dtype | Content |
|---|---|---|
| `keys.npy` | [S, D] float32 | L2-normalized instruction key of each session |
| `para_keys.npy` | [M, D] float16 | keys of the paraphrases that pass the drift filter, grouped by session |
| `para_index.npz` | `offset` [S] int64, `count` [S] int32 | session `i` owns `para_keys[offset[i] : offset[i] + count[i]]` |
| `text_vecs.npy` | [n_texts, D_text] float16 | embedding of each typed string in `vocab.json` (a model input) |
| `meta.json` | | embedder, dimension, drift-filter threshold and counts |

There are two builders:

- `text_embed.py` embeds with a local sentence-transformers model (default Qwen3-Embedding-0.6B) into `WFEMB_KEYS`,
  which defaults to `DATA/keys`. It refuses to overwrite an existing default key set unless `--overwrite_default`
  is passed. Its first run (before step 3) writes no paraphrase keys; run it again afterwards if the default set
  needs them.
- `openai_keys.py` embeds instructions and paraphrases with the OpenAI API into `--out` (default
  `DATA/keys_openai`). These keys are training and evaluation **labels** only. Typed-text vectors are model
  **inputs**, so `text_vecs.npy` there is a symlink to the default set's file. `raw_instr.*` and `raw_para.*` cache
  the embeddings of the unique strings, so changing the drift filter costs no API calls.

**Paraphrases** (`paraphrase.py`). A local instruction-tuned LLM (vLLM) rewrites every unique instruction `--n`
times: same task, different wording, with app names and concrete values kept. Rewrites equal to the instruction are
dropped. `paraphrases.json` holds `model`, `n`, `paraphrases` (instruction → rewrites) and `raw` (instruction → the
LLM's reply). vLLM pins its own torch, so the launcher runs this script in a separate environment.

**Drift filter.** A paraphrase is kept only if the cosine between its key and the original instruction's key is
above a threshold; this removes rewrites that changed the meaning. `text_embed.py` uses the fixed `--theta_para`
(0.85, tuned for Qwen3-Embedding-0.6B). Similarity scales differ between embedders, so `--para_keep_frac F` sets
the threshold at the quantile that keeps the fraction `F` instead; `openai_keys.py` always works this way and
defaults to the fraction the Qwen set keeps. The filter runs when a key set is built, so `paraphrases.json` stays
unfiltered and is shared by all key sets.

`text_embed.py --placeholders` embeds instructions with file names, quoted strings and numbers replaced by
`<FILE>` / `<VALUE>`.

## 5. `judge_pool.py`: judged relevance pools

The pools pick the judged same-workflow partners shown by the demo (`wfemb/demo/selection.py`). They are not a
training metric: the val pool has only 60 same-workflow queries, and unjudged pairs would count as irrelevant.

1. **Pool.** For every session of the split, candidates from the same split are proposed by four retrievers:
   `key` (nearest sessions by default instruction key, `--k_key` 4), `action` (nearest by TF-IDF over the
   session's actions, `--k_action` 2), `app` (random sessions of the same primary app, `--k_app` 1) and `dup`
   (all sessions of the same instruction group). An action token is the event type, the screen cell on a 4×4
   grid and the key ids; TF-IDF uses 1- to 3-grams.
2. **Judge.** An LLM labels each unordered pair from both sessions' instruction, application, OS and the
   natural-language action of every step (read from the raw jsonl). Which session is shown first is random.

   | Label | Meaning |
   |---|---|
   | 2 | same workflow: same goal and essentially the same procedure in the same application |
   | 1 | related: same application and a partly overlapping procedure |
   | 0 | different |

3. **Use.** Pairs outside the pool count as not relevant (standard IR pooling). `wfemb/metrics.py` reads the file
   and adds duplicate-instruction groups as label 2.

`judged_<split>.jsonl` has one line per judged pair: `a`, `b` (session indices, `a < b`), `label`, `rationale`,
`sources` (the retrievers that proposed the pair) and `usage` (`in` / `out` tokens).

| `--mode` | Does |
|---|---|
| `dry` (default) | prints pool sizes, a token estimate and two prompts; calls no API |
| `sync` | judges with concurrent requests; `--limit N` judges a random sample (pilot) |
| `batch` | submits one Message Batch, waits for it and writes the results; the batch id is saved in `judged_<split>_batch.json` |
| `collect` | waits for an already submitted batch (`--batch_id`, else the saved id) and writes its results |

Pairs already in the file are skipped, and refused or failed requests are not written, so `sync` and `batch` can
be re-run to fill the gaps. `collect` appends whatever the batch returned: run it only for a batch whose results
were not written yet, or pairs are duplicated.

## 6 to 7. `cache_frames.py`: frame cache

Every step's screenshot is encoded once with the frozen SigLIP2 so400m NaFlex encoder (aspect ratio kept, at most
`--global_patches` patches). The patch grid is adaptive-average-pooled to `--grid` × `--grid` tokens and every
token is projected from 1152 to `--pca_dim` channels. The number of tokens per frame, grid², is the experiment
variable `patch`.

`--mode pca_fit` fits the projection on the tokens of `--pca_frames` random **training** frames and writes
`dense_pca_g<grid>_d<pca_dim>.npz`: `mean` [1152], `comps` [1152, pca_dim], `cum_var`, `n_tokens`.

`--mode dense` writes one shard. Shard `k` of `n` covers the global steps `[lo, hi)`:

| File | Content |
|---|---|
| `global_<kk>of<nn>.npy` | [hi - lo, grid², pca_dim] uint16: the bit patterns of bfloat16 values (numpy has no bfloat16) |
| `meta_<kk>of<nn>.json` | `shard`, `num_shards`, `lo`, `hi`, `dim`, `global_tokens`, `grid`, `encoder`, `global_patches`, `pca`, `missing` |
| `progress_<kk>of<nn>.json` | `next`: the step an interrupted shard resumes from |

`meta_*.json` is written last, and `wfemb/storage/frames.py` finds shards through it, so an unfinished shard is
never read. A screenshot that is missing from the archive or unreadable is encoded as a black frame and counted in
`missing` (after a resume, the count covers only the frames encoded since the resume).

### Split archives are read in place

AgentNet ships its screenshots as split ZIP archives (`images.z01` … `images.zNN`, `images.zip`). Python's
`zipfile` refuses multi-disk archives, and merging the parts needs a second copy of roughly 200 GB that the NAS
quota cannot hold. `splitzip.SplitZip` therefore reads members straight from the parts: it treats them as one
concatenated file and converts the archive's (disk, offset) positions, which count from the start of each part.
It supports stored and deflated members and ZIP64, and nothing else.
