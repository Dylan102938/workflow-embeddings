# Data preparation

`wfemb/prep/` turns the raw AgentNet download into the files training reads. Every step is run once, offline.
This page lists the steps in order, what each one writes, and the design choices that are not obvious from the code.
The last section, [Added datasets](#added-datasets), covers the training data from other datasets.

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
BULK/frames_dense[_g<grid>][_raw]/   frozen screen features, one folder per grid size and projection   (cache_frames.py)
BULK/dense_pca_g<grid>_d<dim>.npz   the PCA basis of that cache
DATA/added/, BULK/added/   the same layout for the datasets added to AgentNet (see "Added datasets")
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
token is projected from 1152 to `--pca_dim` channels (`--pca_dim 0`: no projection, all 1152 channels are stored).
The number of tokens per frame, grid², is the experiment variable `patch`; `--pca_dim` is the variable `pca`. The
cache folder is `frames_dense[_g<grid>][_raw]` (`config.frame_cache_name`). On Baseten:
`WFEMB_SPECS="siglip2:32 siglip2:16:0" truss train push runners/baseten/build_config.py`.

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

## Added datasets

AgentNet has 18,081 training sessions. `wfemb/prep/sources/` converts other computer-use datasets to AgentNet's
record format, `add_datasets.py` cleans them, and they become extra **training** data. The validation and test
splits stay AgentNet's, so results remain comparable with earlier runs.

| Dataset | What it is | License | Used part |
|---|---|---|---|
| `videocua` | [VideoCUA](https://huggingface.co/datasets/ServiceNow/VideoCUA): people doing tasks in 87 desktop applications, as 30 fps screen video with a timestamped input log | MIT | everything |
| `scalecua` | [ScaleCUA-Data](https://huggingface.co/datasets/OpenGVLab/ScaleCUA-Data): trajectories on six platforms | none stated on the dataset (its code repository is Apache-2.0) | the trajectories collected by people on Ubuntu, macOS, Windows and the web |
| `gui360` | [GUI-360](https://huggingface.co/datasets/vyokky/GUI-360): an agent doing tasks in Word, Excel and PowerPoint on Windows | MIT | the successful trajectories (its `train` and `test` folders) |

Not used: ScaleCUA's Android and iOS trajectories (phone screens), its trajectories from automated exploration
(their instructions were written afterwards to describe where a random walk ended), its grounding data, and
GUI-360's failed trajectories. ScaleCUA did not release the screenshots of its 2025-06-12 desktop batch, so those
691 sessions cannot be used.

### Layout

The added datasets are a **second prepared set** with the same layout as AgentNet's, below `DATA/added/` and
`BULK/added/`. Nothing of AgentNet's is rewritten, so code and runs that know nothing about the added data are
unaffected.

```
DATA/added/sources/<dataset>/   the download, and its conversion: sessions.jsonl, images.zip, image_stats.json,
                                conversion.json
DATA/added/prepared/            sessions.pkl, events.npz, vocab.json, summary.json, rejections.json, validation.json
DATA/added/keys/                paraphrases.json, text_vecs.npy (and the default key set)
DATA/added/keys_openai/         the label key set
BULK/added/frames_dense[...]/   frame caches
```

`WFEMB_SET=added` points `paths.DATA`, `paths.BULK` and everything built from them at this copy of the layout, so
the key-set, paraphrase and frame-cache steps above run on the added set unchanged. Four things are taken from
AgentNet so that both sets live in one input space:

- the key vocabulary (`vocab.json` `key_vocab`): a key name AgentNet does not have becomes `<unk>`;
- the PCA basis of each frame cache (`BULK/dense_pca_*.npz`);
- the drift-filter threshold of the label key set (`keys_openai/meta.json` `theta_para`);
- the embedders (Qwen3-Embedding-0.6B for typed text, OpenAI text-embedding-3-large for labels).

A run uses the added data when its `data` variable names any of the datasets
([EXPERIMENTS.md](EXPERIMENTS.md)). `wfemb/data.py` then stacks the two sets: the added sessions, steps, typed
texts and paraphrase keys are numbered on from AgentNet's.

### Pipeline

| # | Step | Command | Writes |
|---|---|---|---|
| 0 | Download | `hf download <repository> --repo-type dataset --local-dir DATA/added/sources/<dataset>` (ScaleCUA: `annotations/*navigation*` and the `data/data_<date>/{mac,ubuntu,windows,web}.tar.gz.part-*` archives of the human batches; GUI-360: `train/*`, `test/*`) | the raw files |
| 1 | Convert | `python -m wfemb.prep.sources <dataset>` | `sources/<dataset>/{sessions.jsonl, images.zip, image_stats.json, conversion.json}` |
| 2 | Clean and prepare | `python -m wfemb.prep.add_datasets` | `added/prepared/` |
| 3 | Validate | `python -m wfemb.prep.validate [--decode -1] [--contact_sheets DIR]` | `added/prepared/validation.json` |
| 4 | Typed-text vectors, default keys | `WFEMB_SET=added python -m wfemb.prep.text_embed` | `added/keys/` |
| 5 | Paraphrases | `WFEMB_SET=added python -m wfemb.prep.paraphrase` (on Baseten: `runners/baseten/paraphrase.sh added`) | `added/keys/paraphrases.json` |
| 6 | Label key set | `WFEMB_SET=added python -m wfemb.prep.openai_keys` | `added/keys_openai/` |
| 7 | Frame caches | `WFEMB_SET=added python -m wfemb.prep.cache_frames --mode dense --grid 24` per shard (on Baseten: `WFEMB_SET=added WFEMB_SPECS="siglip2:24" truss train push runners/baseten/build_config.py`) | `BULK/added/frames_dense_g24/` |

Step 3 is run again after steps 4 to 7: it also checks the key sets and frame caches that exist. Baseten reads the
added set from the private data repository: `hf upload <repository> DATA/added added --repo-type dataset` (without
the raw downloads), then `runners/baseten/download.sh`.

### 1. Converters

A converted session is one line of `sessions.jsonl` in AgentNet's format (`task_id`, `instruction`,
`traj: [{index, image, value: {code}}]`) plus a `meta` object (`applications`, `websites`, `system`, `origin`:
`human` or `agent`, screen size). `code` is `pyautogui.*` / `computer.*` calls with positions in [0, 1], so
`actions.py` parses every dataset. `images.zip` holds every screenshot under a unique flat name, stored without
compression. `image_stats.json` maps each screenshot to its size, the MD5 digest of its file and the spread
(standard deviation) of its grey levels; `null` marks a screenshot that does not decode. `conversion.json` counts
what the converter itself left out.

**VideoCUA** logs raw input: a pointer move, then a click; a button press, moves, a release. `videocua.py` merges
log entries into the steps AgentNet records:

| Log entries | Step |
|---|---|
| `MOVE_TO`, `CLICK` | one click (double, triple, right or middle by the click's count and button) |
| `MOVE_TO`, `MOUSE_DOWN`, `DRAG_TO`..., `MOUSE_UP` | `moveTo` + `dragTo` to the last drag position |
| `MOUSE_DOWN`, `MOUSE_UP` with no drag between | a click at the pointer |
| `MOVE_TO`, `SCROLL` | `moveTo` + `scroll` |
| `MOVE_TO` followed by anything else | `moveTo` |
| `TYPING`, `PRESS` | `write` + `press` |
| `PRESS`, `HOTKEY` | `press` (one key) or `hotkey`; free text in the key field becomes `write` |
| `KEY_DOWN`, `KEY_UP` of one key in a row | `press`; otherwise `keyDown` / `keyUp` |

Key names are annotator-written (`Ctrl + V`, `"Cmd" + "V"`, `Right arrow`) and are normalized to AgentNet's
(`common.key_tokens`). A step's screenshot is the video frame 0.1 s before its first log entry (as in VideoCUA's own
converter); every session ends with the last video frame as an eventless step, like AgentNet's final step.
Instructions rarely name their application, while AgentNet's usually do, so `In <application>, ` is put in front of
an instruction that does not mention it (the original is kept in `meta.original_instruction`). The application
name is the name of the zip, not the log's hand-typed `platform`.

**ScaleCUA** ships one annotation row per (instruction wording, step, description wording). `scalecua.py` rebuilds
sessions: a session is one screenshot folder; its first wording is the collector's instruction (the others are
model-written rewrites and are not used); a row's step index is the number of previous operations it lists. The
action calls already use AgentNet's format and normalized positions. Normalized: `swipe` becomes `scroll`
(swiping up scrolls down; a full-screen swipe counts as 10 clicks), a `hotkey` of plain characters becomes `write`
(the recorder logged typed digits that way), `response` and `success` end the session like `terminate`, pixel
positions are divided by the screen size. A session with `failure()` or a failed `terminate` is left out, as is one
whose step indices have gaps.

**GUI-360** records an agent (UFO) that acts through the mouse and keyboard and through application APIs.
`gui360.py` keeps the application-window screenshot of each step (`screenshot_clean`; the release has no
full-desktop screenshots) and converts positions from its pixels. `click`, `type` (a click on the control, then
`write`), `drag` and `wheel_mouse_input` become pointer and keyboard events. An API action
(`select_table_range`, `insert_excel_table`, ...) has no input to record: its step keeps its screenshot and gets an
event of type `other`. A step with an empty action or a `summary` is an eventless step (the agent's closing step).

### 2. Cleaning (`clean.py`)

`add_datasets.py` goes through the datasets in the order given and through each dataset's sessions by id, and
keeps a session unless it breaks a rule. The first broken rule is counted (`summary.json`) and the session id is
listed under it (`rejections.json`).

| Rule | A session is dropped if | Why |
|---|---|---|
| `duplicate_id` | its id was already accepted | |
| `bad_instruction` | its instruction has fewer than 2 or more than 150 words, or no letters | no usable label |
| `held_out_application` | one of its applications or websites is a held-out test application of AgentNet (names compared after dropping vendor words and suffixes: `Mozilla Thunderbird` = `domain:thunderbird`) | the test split measures unseen applications |
| `mentions_held_out_application` | its instruction names a held-out application (not checked for names that are ordinary words, such as `calendar` or `pages`, and not inside mail addresses) | the application is unknown for some web sessions; this errs on the side of dropping |
| `evaluation_instruction` | its instruction (case and whitespace ignored) is a validation or test instruction | no leak of evaluation labels |
| `near_evaluation_instruction` | its instruction has a character 3-to-5-gram TF-IDF cosine of 0.9 or more to a validation or test instruction | near-verbatim copies leak too |
| `missing_screenshot` | a screenshot is missing or does not decode | |
| `too_short`, `too_long` | it has fewer than 2 or more than 100 steps | AgentNet's range is 2 to 129 |
| `portrait_screen` | a screenshot is more than 1.25 times as tall as wide | phone screens are not part of this data (VideoCUA has a few recordings of mobile apps) |
| `few_input_actions` | fewer than half of its steps (before the final screenshot) have a pointer or keyboard event | mostly API actions or waiting: little to learn the workflow from |
| `pointer_off_screen` | more than 30% of its positions lie exactly on the screen border | clipped positions: the coordinates did not belong to the screenshot |
| `blank_screenshots` | its first screenshot, or more than 20% of them, is blank (grey-level spread below 0.5; real blank frames have 0, dark-themed screens 1 to 2) | failed captures |
| `identical_screenshots` | all its screenshots are the same file | the capture did not follow the actions |
| `duplicate_screenshots` | an accepted session has the same sequence of screenshot files | the same recording listed twice, or two questions answered from the same screens |
| `duplicate_demonstration` | an accepted session has the same instruction and the same actions (types, keys, text, positions on an 8x8 grid) | a repeated recording adds nothing |
| `instruction_repeated_too_often` | 8 sessions with its instruction were already accepted | one instruction must not dominate |

Sessions that pass become `added/prepared/`, all with `split: "train"`, in the order they were accepted. Their
session records have two more fields than AgentNet's: `dataset` and `origin`.

### 3. Validation (`validate.py`)

`validate.py` recomputes its checks from the written files, not from the cleaning code's bookkeeping:

- structure: one events row per step, consecutive steps per session, one screenshot per step, valid event types,
  positions on the screen, key ids within AgentNet's vocabulary, typed-text ids valid and all used, unique ids (also
  against AgentNet);
- leakage: no validation or test instruction, none near one, no held-out application;
- duplicates: no repeated demonstration, no repeated screenshot sequence, at most 8 sessions per instruction;
- screenshots: each one is in its archive; a sample (`--decode N`, all with `-1`) decodes and is not blank;
- derived files, once built: one unit-length key per session with AgentNet's dimension, a consistent paraphrase
  index, one vector per typed text, frame caches that cover every step with no missing screenshot.

It also writes per-dataset statistics next to AgentNet's (steps per session, instruction length, event shares, share
of unknown keys, applications) into `validation.json`, and with `--contact_sheets DIR` one image per sampled
session with every step's screenshot and its pointer positions, for checking by eye that positions land on the
controls they should.

### Result (2026-11-11)

26,926 added training sessions with 185,731 steps, next to AgentNet's 18,081 training sessions (22,083 sessions and
408,384 steps over all splits). Sessions left out by the converters: VideoCUA 41 tasks without a video and 1 without
actions; ScaleCUA 691 whose screenshots were not released, 23 marked as failed, 2 with missing steps.

| | `videocua` | `scalecua` | `gui360` |
|---|---|---|---|
| converted sessions | 9,513 | 2,718 | 17,189 |
| `bad_instruction` | 1 | 1 | 0 |
| `held_out_application` | 335 | 236 | 0 |
| `mentions_held_out_application` | 17 | 73 | 24 |
| `near_evaluation_instruction` | 0 | 0 | 1 |
| `too_short` | 0 | 13 | 635 |
| `portrait_screen` | 5 | 0 | 1 |
| `few_input_actions` | 0 | 6 | 821 |
| `pointer_off_screen` | 1 | 0 | 3 |
| `blank_screenshots` | 13 | 1 | 0 |
| `identical_screenshots` | 0 | 1 | 27 |
| `duplicate_screenshots` | 5 | 75 | 1 |
| `duplicate_demonstration` | 3 | 5 | 173 |
| `instruction_repeated_too_often` | 0 | 0 | 17 |
| **kept sessions** | **9,133** | **2,307** | **15,486** |
| steps | 45,079 | 15,158 | 125,494 |

The held-out applications that cost the most sessions: OBS Studio (235), Brave (54) and IntelliJ IDEA (46) in
VideoCUA; Microsoft Windows (73), Clock (52), GitHub (38), Google Docs (24) and Gmail (22) in ScaleCUA.

How the kept sessions compare with AgentNet (`validation.json`):

| | AgentNet | `videocua` | `scalecua` | `gui360` |
|---|---|---|---|---|
| recorded by | people | people | people | an agent |
| steps per session, median (5% to 95%) | 17 (8 to 34) | 4 (2 to 10) | 5 (2 to 16) | 6 (2 to 22) |
| instruction words, median (5% to 95%) | 30 (10 to 71) | 11 (6 to 19) | 11 (3 to 30) | 13 (8 to 23) |
| applications | 1,377 | 84 | 26 | 3 |
| distinct instructions | 21,819 | 9,114 | 2,262 | 14,718 |
| steps without events | 5% | 20% | 15% | 18% |
| key presses unknown to the vocabulary | 0.3% | 1.4% | 3.5% | 0% |
| clicks / typing among events | 63% / 8% | 67% / 9% | 64% / 11% | 82% / 9% |

The added sessions are much shorter than AgentNet's and their instructions are shorter. Their share of eventless
steps is higher mostly because every session has one closing step and has few steps.

All validation checks pass, with every one of the 185,731 screenshots decoded. Contact sheets of samples from each
dataset were checked by eye: pointer positions land on the controls the instructions call for. The inspection of
rejected sessions changed three rules from their first version: a 16x16 perceptual hash treated screens with small
changes as unchanged (replaced by file digests), the blank threshold caught dark-themed editors (lowered from 2 to
0.5), and `gmail` inside mail addresses counted as a mention of Gmail.
