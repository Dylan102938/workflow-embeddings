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
BULK/frames_dense[_g<grid>]/   frozen screen features, one folder per grid size   (cache_frames.py)
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
token is projected from 1152 to `--pca_dim` channels (512). The number of tokens per frame, grid², is the
experiment variable `patch`. The cache folder is `frames_dense[_g<grid>]` (`config.frame_cache_name`). On Baseten:
`WFEMB_SPECS="siglip2:24 siglip2:32" truss train push runners/baseten/build_config.py`.

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

### Rules for the test set

The test split measures workflows in applications the model never trained on, and no validation or test
instruction is seen in training. Added data must not weaken either. `wfemb/prep/held_out.py` holds the rules:
`python -m wfemb.prep.held_out` prints each one with the number of sessions that break it and exits with 1 if any
does. `add_datasets.py` runs the same check and writes nothing if it fails.

What is held out:

- **Test applications**: the 157 app keys that `prepare.py` sampled (`summary.json` `test_apps`).
- **Held-out applications**: the test applications, and every other application that only validation or test
  sessions use. Compared by name, without the key prefix, vendor words and suffixes (`Mozilla Thunderbird` =
  `domain:thunderbird`, `QGIS` = `app:qgis desktop`): 369 names, 153 of them from the test applications.
- **Evaluation instructions**: those of the validation and test sessions, ignoring case and whitespace.

| Rule | Enforced by |
|---|---|
| AgentNet: every test session has a test application, or the instruction of a session that has one | `prepare.py` `assign_splits` |
| AgentNet: no training or validation session has a test application | `assign_splits`: a session with any test app key goes to test |
| AgentNet: no instruction is in two splits | `assign_splits`: validation takes whole instruction groups, and a group with a member in test moves to test |
| Added: every session is training data | `add_datasets.py` writes `split: "train"`, so validation and test are AgentNet's 952 and 3,050 sessions in every run |
| Added: no session has a held-out application | cleaning rule `held_out_application` |
| Added: no instruction names a held-out application | cleaning rule `mentions_held_out_application` |
| Added: no instruction is an evaluation instruction, or a near copy of one | cleaning rules `evaluation_instruction`, `near_evaluation_instruction` |

Evaluation is the same for every run: the candidates are AgentNet's training keys plus the split's own, whatever
the run trains on (`data.evaluation_candidates`).

What the rules do not cover:

- **AgentNet's own split compares app keys exactly.** 907 of its 18,081 training sessions have an application with
  the name of a test application under another key: `app:discord` (117 sessions; `web:discord` is held out),
  `app:intellij idea` (88; `app:intellij idea community edition`), `app:steam` (32; `web:steam`),
  `app:mozilla thunderbird` and `app:thunderbird` (21; `domain:thunderbird`). Some are different products with one
  name: `web:google maps` (262) and `web:google calendar` (182) against `app:maps` and `app:calendar`. The check
  prints this number. Added data is held to the stricter comparison by name.
- **A test session can also use applications that training has**: 1,353 of the 3,050 do. Test metrics are therefore
  also reported for each group ([EXPERIMENTS.md](EXPERIMENTS.md)).
- **A near copy is a near-verbatim one** (character 3-to-5-gram TF-IDF cosine of 0.9 or more). A reworded test
  instruction is not detected.

### Cleaning rules (`clean.py`)

`add_datasets.py` goes through the datasets in the order given and through each dataset's sessions by id. A
session is dropped by the first rule that is true for it, in the order of the three tables below, which are the
three tables in `clean.py`. `summary.json` counts the dropped sessions per rule and `rejections.json` lists them.

`HELD_OUT_RULES`: the session touches what the validation and test splits hold out (see above).

| Rule | A session is dropped if |
|---|---|
| `held_out_application` | one of its applications or websites is a held-out application |
| `mentions_held_out_application` | its instruction names a test application. Not checked for names of fewer than 4 letters, names that are ordinary words (`calendar`, `pages`), or inside mail addresses. Catches sessions whose application is not recorded |
| `evaluation_instruction` | its instruction is an evaluation instruction |
| `near_evaluation_instruction` | its instruction is a near copy of one |

`QUALITY_RULES`: the recording is not a usable demonstration.

| Rule | A session is dropped if | Why |
|---|---|---|
| `bad_instruction` | its instruction has fewer than 2 or more than 150 words, or no letters | no usable label |
| `missing_screenshot` | a screenshot is missing or does not decode | |
| `too_short`, `too_long` | it has fewer than 2 or more than 100 steps | AgentNet's range is 2 to 129 |
| `portrait_screen` | a screenshot is more than 1.25 times as tall as wide | phone screens (VideoCUA has a few recordings of mobile apps) |
| `few_input_actions` | fewer than half of its steps before the final screenshot have a pointer or keyboard event | mostly API actions or waiting |
| `pointer_off_screen` | more than 30% of its positions lie exactly on the screen border | clipped positions: the coordinates did not belong to the screenshot |
| `blank_screenshots` | its first screenshot, or more than 20% of them, is blank (grey-level spread below 0.5; blank frames have 0, dark-themed screens 1 to 2) | failed captures |
| `identical_screenshots` | all its screenshots are the same file | the capture did not follow the actions |

`DUPLICATE_RULES`: a session that was already kept makes this one redundant.

| Rule | A session is dropped if a kept session has | Why |
|---|---|---|
| `duplicate_id` | the same id | |
| `duplicate_screenshots` | the same sequence of screenshot files | the same recording listed twice |
| `duplicate_demonstration` | the same instruction and the same actions (types, keys, text, positions on an 8x8 grid) | a repeated recording adds nothing |
| `instruction_repeated_too_often` | its instruction, 8 times | one instruction must not dominate |

AgentNet's own filters are in [section 1](#1-preparepy-sessions-splits-events).

### Layout

The added datasets are a **second prepared set** with the same layout as AgentNet's, below `DATA/added/` and
`BULK/added/`. Nothing of AgentNet's is rewritten, so code and runs that know nothing about the added data are
unaffected.

```
DATA/added/sources/<dataset>/   the download, and its conversion: sessions.jsonl, images.zip, image_stats.json,
                                conversion.json
DATA/added/prepared/            sessions.pkl, events.npz, vocab.json, summary.json, rejections.json
DATA/added/keys/                paraphrases.json, text_vecs.npy (and the default key set)
DATA/added/keys_openai/         the label key set
BULK/added/frames_dense[...]/   frame caches
```

Every session in `sessions.pkl` has `split: "train"` and two more fields than AgentNet's: `dataset` and `origin`
(`human` or `agent`). `WFEMB_SET=added` points `paths.DATA`, `paths.BULK` and everything built from them at this
copy of the layout, so the key-set, paraphrase and frame-cache steps above run on the added set unchanged. Four
things are taken from AgentNet so that both sets live in one input space:

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
| 1 | Download and convert | `python -m wfemb.prep.sources <dataset> --download` (without `--download`: convert what is already in `sources/<dataset>/`) | the raw files, then `sources/<dataset>/{sessions.jsonl, images.zip, image_stats.json, conversion.json}` |
| 2 | Clean and prepare | `python -m wfemb.prep.add_datasets` | `added/prepared/` |
| 3 | Check the rules for the test set | `python -m wfemb.prep.held_out` (step 2 runs it too) | |
| 4 | Typed-text vectors, default keys | `WFEMB_SET=added python -m wfemb.prep.text_embed` | `added/keys/` |
| 5 | Paraphrases | `WFEMB_SET=added python -m wfemb.prep.paraphrase` (on Baseten: `runners/baseten/paraphrase.sh added`) | `added/keys/paraphrases.json` |
| 6 | Label key set | `WFEMB_SET=added python -m wfemb.prep.openai_keys` | `added/keys_openai/` |
| 7 | Frame caches | `WFEMB_SET=added python -m wfemb.prep.cache_frames --mode dense --grid 24` per shard (on Baseten: `WFEMB_SET=added WFEMB_SPECS="siglip2:24" truss train push runners/baseten/build_config.py`) | `BULK/added/frames_dense_g24/` |

Baseten reads the added set from the private data repository: upload `added/prepared/`,
`added/keys/text_vecs.npy`, `added/keys_openai/` and `added/sources/<dataset>/images.zip` there (about 37 GB,
nearly all of it screenshots), then run `runners/baseten/download.sh`.

### Converters

A converted session is one line of `sessions.jsonl` in AgentNet's format (`task_id`, `instruction`,
`traj: [{index, image, value: {code}}]`) plus a `meta` object (`applications`, `websites`, `system`, `origin`,
screen size). `code` is `pyautogui.*` / `computer.*` calls with positions in [0, 1], so `actions.py` parses every
dataset. `images.zip` holds every screenshot under a unique flat name. `image_stats.json` maps each screenshot to
its size, the MD5 digest of its file and the spread of its grey levels (`null`: it does not decode); the cleaning
rules read it. `conversion.json` counts what the converter itself left out.

- **VideoCUA** logs raw input, which `videocua.py` merges into the steps AgentNet records: a pointer move and a
  click become one click; a button press, moves and a release become `moveTo` + `dragTo`; typing followed by a key
  press becomes `write` + `press`. Annotator-written key names (`Ctrl + V`, `"Cmd" + "V"`, `Right arrow`) are
  normalized to AgentNet's. A step's screenshot is the video frame 0.1 s before its first log entry, and the last
  video frame closes the session as an eventless step. `In <application>, ` is put in front of an instruction that
  does not name its application, as AgentNet's usually do (the original is kept in `meta.original_instruction`).
- **ScaleCUA** ships one annotation row per instruction wording and step. A session is one screenshot folder; its
  first wording is the collector's instruction (the others are model-written rewrites and are not used). `swipe`
  becomes `scroll`, a `hotkey` of plain characters becomes `write` (the recorder logged typed digits that way), and
  `response` and `success` end the session like `terminate`. Sessions marked as failed are left out.
- **GUI-360** records an agent that acts through the mouse and keyboard and through application APIs. Its `click`,
  `type`, `drag` and `wheel_mouse_input` actions become pointer and keyboard events on the application-window
  screenshot of the step. An API action (`select_table_range`, ...) has no input to record: its step keeps its
  screenshot and gets an event of type `other`.

### Result (2026-11-11)

26,926 added training sessions with 185,731 steps, next to AgentNet's 18,081 training sessions (22,083 sessions and
408,384 steps over all splits). Sessions left out by the converters: VideoCUA 41 tasks without a video and 1 without
actions; ScaleCUA 691 whose screenshots were not released, 23 marked as failed, 2 with missing steps. The rules for
the test set hold (`python -m wfemb.prep.held_out`).

| | `videocua` | `scalecua` | `gui360` |
|---|---|---|---|
| converted sessions | 9,513 | 2,718 | 17,189 |
| `held_out_application` | 335 | 236 | 0 |
| `mentions_held_out_application` | 17 | 73 | 24 |
| `near_evaluation_instruction` | 0 | 0 | 1 |
| `bad_instruction` | 1 | 1 | 0 |
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

How the kept sessions compare with AgentNet:

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

These statistics, a decode of all 185,731 screenshots, and contact sheets of sampled sessions (pointer positions
land on the controls the instructions call for) come from a one-off script, `wfemb/prep/validate.py` at commit
`2931a41`. Inspecting rejected sessions changed three rules from their first version: a 16x16 perceptual hash
treated screens with small changes as unchanged (replaced by file digests), the blank threshold caught dark-themed
editors (lowered from 2 to 0.5), and `gmail` inside mail addresses counted as a mention of Gmail.
