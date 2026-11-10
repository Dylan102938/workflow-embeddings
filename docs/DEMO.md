# Demo bundles

The dashboard's Demo tab (`dashboard/static/demo.html`, the Intent Replay viewer) plays a workflow step by step: the
screenshot and actions, the intents the model currently believes, and where its embedding sits among all test
workflows. `wfemb/demo/` writes the files the viewer reads: one **bundle** per run, in `dashboard/bundles/<run id>/`.

## Commands

```
python -m wfemb.demo bundle --id <run id>          # everything below, with defaults
python -m wfemb.demo background --id <run id>      # the map of all test workflows
python -m wfemb.demo export --id <run id> --auto 8                      # choose illustrative test workflows
python -m wfemb.demo export --id <run id> --session <session id> [...]  # these test sessions
python -m wfemb.demo export --id <run id> --raw workflow.json [--single_file]   # a workflow from outside the dataset
python -m wfemb.demo frames --id <run id>          # screenshots of the bundle's test workflows
python -m wfemb.demo screens --screens <folder>    # once, where the AgentNet archives are: the screenshot cache
```

`--id` defaults to the best model's id (`Experiment()`), `--out` to `dashboard/bundles/<run id>`.
`export` needs the map that `background` caches, and adds to the bundle: run it as often as needed.

`bundle` is `background`, then `export --auto 8` plus every `--session`, then `frames`. If the folder next to `--out`
holds the bundle of the `--like` run (default: the best model), its test workflows are exported too, so that two runs
can be compared on the same workflows.

## What the commands read

| What | Where | How it gets there |
|---|---|---|
| Sessions, events, vocabularies | `$WFEMB_DATA/prepared/` | data prep ([DATA_PREP.md](DATA_PREP.md)) |
| Instruction keys, typed-text embeddings | `$WFEMB_DATA/keys_openai/` | data prep |
| Judged pairs of test workflows | `$WFEMB_DATA/keys/judged_test.jsonl` | data prep |
| Every test step's embedding | `<results>/<run folder>/test_embeddings.npz` | the final test evaluation writes it; `python -m wfemb.storage.fetch rnn\|baseten --embeddings` copies it here |
| Screenshots of the test sessions | `$WFEMB_DATA/demo/screens/<session id>/<step>.jpg`, or `--screens` | `screens`, then `fetch baseten --screenshots` |
| The map, between `background` and `export` | `$WFEMB_DATA/demo/<run id>_pca.npz` | `background` |
| Checkpoint (`--raw` only) | `<results>/<run folder>/best.pt`, or `--ckpt` | `fetch ... --checkpoints` |

Test workflows need no GPU, checkpoint or frame cache: their embeddings were saved by the evaluation. Only `--raw`
runs the model.

**Screenshots.** `frames` copies from the screenshot cache; for a screenshot the cache does not have it reads the
AgentNet split archives (`paths.ZIPS`), which exist only on rnn and Baseten. `screens` fills a cache from the archives
for every test session (JPEG, at most 1280 px wide). It keeps files that exist, lists screenshots that could not be
read or decoded and carries on, prints how many were written / already cached / missing / failed, and fails only if
the cache ends up empty. Each thread opens the archives itself: a `SplitZip` is not safe to share between threads.

**Intents.** The candidates ("intent bank") are the unique instructions of the training and test sessions, as in
evaluation. At a step with direction `d` and confidence `κ`, the probability of an intent with key `u` is
`softmax(κ · cos(d, u))` over the bank, as in the loss.

## Bundle layout

```
background.json                  the map
workflows/index.json             the workflows in the bundle, in export order
workflows/<workflow id>.json     one workflow
frames/<workflow id>/<step>.jpg  its screenshots
```

A *map position* below is an index into `points` / `items` of `background.json`.

### background.json

The last-step embedding of every test session, unit length, projected onto its first two principal directions, with
k-means clusters (`--k`, default 10; cluster 0 is the largest).

```
{"run": "<run id>", "n": 3050, "explained": [0.074, 0.053],        variance fraction of each axis
 "points": [[x, y, cluster], ...],                                  one per test session
 "items": [{"id", "app", "os", "instruction"}, ...],                same order as points
 "clusters": [{"id", "n", "apps": [[app, count], ...], "label", "example", "xy"}, ...]}
```

### workflows/index.json

```
[{"id", "file": "workflows/<id>.json", "title", "app", "os", "n_steps", "source", "final_rank", "thumb"}, ...]
```

`source` is `agentnet-test` or `raw`; `final_rank` is the true intent's rank at the last step; `thumb` is the first
screenshot. Exporting a workflow again moves its entry to the end.

### workflows/&lt;id&gt;.json

```
{"version": 1, "run", "id", "source", "instruction", "app", "os",
 "agentnet": {"archive", "images"},      test sessions only: where the original screenshots are
 "n_steps", "bank_size",
 "true_in_bank": bool,                   whether the instruction is one of the candidate intents
 "bg_index": map position | null,        the workflow's own point on the map (null for raw workflows)
 "partners": [map position, ...],        test workflows judged the same workflow, or in the same group
 "related": [map position, ...],         judged related
 "neighbors": [{"bg": map position, "cos"}, ...],     8 nearest test workflows to the final embedding
 "steps": [{
   "t": 0,
   "image": "frames/<id>/0.jpg",         or a data URL with --single_file
   "actions": [{"type", "x", "y", "x2", "y2", "keys", "text", "scroll", "label"}, ...],
   "focus": [x, y],                      where the step's action lands
   "kappa": 50.0, "entropy": 5.745,      confidence, and entropy of the intent belief
   "true_rank": 223, "true_p": 0.00058,  the true intent (null if it is not in the bank)
   "top": [[intent text, probability, is the true intent], ...],    --topk, default 5
   "xy": [x, y], "cluster": 4,           this step's embedding on the map, and its nearest cluster
   "near": [[map position, cos], ...]    3 nearest test workflows to this step's embedding
 }, ...]}
```

Coordinates in `actions` and `focus` are fractions of the screen. A workflow is left out of its own `neighbors` and
`near`.

## Which workflows `--auto` chooses

Test workflows of 8 to 22 steps with an instruction of at most 150 characters. `--auto N --hard H` takes `N - H`
workflows whose belief sharpens onto the true intent (rank above 5 at the first step, 3 or better at the last), one
per app and balanced over operating systems, preferring a large entropy drop and workflows that have a same-workflow
partner; then `H` hard cases (final rank above 30). The choice is deterministic for a given run.

## Raw workflows

`export --raw` takes a workflow from outside the dataset: per step, the screenshot the user saw and the action code
run on it.

```json
{"id": "optional (default: the file name)", "instruction": "optional ground truth", "app": "optional", "os": "optional",
 "steps": [{"image": "shot_000.png", "code": "pyautogui.click(x=0.41, y=0.22)"}, ...]}
```

- `image` is absolute, or relative to `--images` (default: the JSON's folder).
- `code` is pyautogui-style, as in AgentNet, with coordinates as fractions of the screen. A step whose code has an
  `x` or `y` above 1 is taken to be in pixels and divided by its screenshot's size.
- An AgentNet record is accepted as well: `{"task_id", "instruction", "traj": [{"index", "image", "value": {"code"}}]}`
  (of a `.jsonl` file, the first line).

The workflow is encoded as the training data was: events as in `prep/prepare.py`, screenshots with the frozen SigLIP2
encoder and the channel PCA fitted on training frames (`prep/cache_frames.py`; the basis is
`$WFEMB_BULK/dense_pca_g<grid>_d512.npz`, or `--frames_pca`), typed text with Qwen3-Embedding-0.6B
(`prep/text_embed.py`). This needs the run's checkpoint and both encoders (a GPU, in practice).

With `--single_file` the screenshots are embedded in the workflow JSON, which can then be opened in the viewer or
dropped on it without a bundle folder.

## Code

```
wfemb/demo/
  cli.py          the commands
  assets.py       DemoAssets: prepared data, intent bank, a run's saved embeddings and checkpoint
  scoring.py      intent bank and per-step intent probabilities
  background.py   the map: PCA, clusters, background.json, the cached BackgroundMap
  selection.py    --auto's choice, partners from the judged pairs
  workflow.py     a workflow's record and its files in the bundle
  raw.py          raw workflows: events from code, screenshot and text encoders, the model
  screenshots.py  screenshot cache and archives, frames and screens
```
