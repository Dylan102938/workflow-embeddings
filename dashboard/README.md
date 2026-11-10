# Dashboard

A local, wandb-style view of the results tree (`docs/EXPERIMENTS.md`) plus the Intent Replay demo viewer. Python
standard library only; plain HTML/CSS/JS, no build step, no CDN.

```
python dashboard/server.py                       # http://127.0.0.1:8765/
python dashboard/server.py --results results --jobs jobs --bundles dashboard/bundles --port 8765 --host 127.0.0.1
```

Defaults are relative to the repo root (`--results` defaults to `$WFEMB_RESULTS`, else `results/`). A run without
`final.json` is *running* if its files changed in the last `--stale-minutes` (15), else *stale*. The best model's id
(read from `docs/EXPERIMENTS.md`, override with `--reference-id`) is only used to order variables and to bold the
parts of an id that differ from it.

Views (all state is in the URL hash, so a view can be bookmarked):

- **Runs**: one row per run, one column per varying variable; variables equal across the filtered runs collapse into
  the "fixed:" line. Filters per variable, on the id (`arch-gru seed-0 !dm-384`), on status; sorting; shift-click
  ranges. *Merge seeds* turns runs that differ only in `seed` into one row (mean ± std, n).
- **Compare**: the selected runs' training curves (EMA smoothing in steps, faint raw line, lr, grad_norm) and
  validation curves (ring = the checkpoint `final.json` chose), crosshair readout, drag to zoom (all charts),
  final-metrics table with the best value per column highlighted. With *Merge seeds*: mean line, min–max band.
- **Group by**: a final metric (or best val_loss) by one variable, optionally × a second one, mean ± std over runs.
- **Queue**: every `jobs/*.txt` (lines: `<run id> [overrides]`, `#` comments) with each id's status, plus runs that
  are in results but in no queue file.
- **Demo**: Intent Replay (`static/demo.html?bundle=/bundles/<run id>/`) for the selected run's bundle in
  `dashboard/bundles/<run id>/`, or the commands that make one.

API: `/api/runs`, `/api/run?id=`, `/api/series?id=&max_points=2000[&minmax=1][&which=steps,evals][&cols=...]`
(bucket means; `max_points=0` = every row), `/api/queue`, `/bundles/<run id>/<file>`. Parsed files are cached by
(path, mtime, size); a growing `steps.jsonl` is parsed incrementally.

Try it on fake data:

```
python tests/make_fake_results.py /tmp/fake
python dashboard/server.py --results /tmp/fake/results --jobs /tmp/fake/jobs
```
