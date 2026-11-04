# Session-embedding model (v1): training results

Status as of 2026-11-04. All training ran on the CHAI `rnn` cluster through Slurm, on single A6000s.
Data lives on NAS under `/nas/ucb/dfeng/data/workflow-embeddings` (runs in `runs/`, logs in `logs/`).

## Headline

| | val loss | val key R@5 | test loss (held-out apps) | test key R@5 |
|---|---|---|---|---|
| Round 1: the pseudocode's v1 config | 5.986 | 0.248 | 7.387 | 0.136 |
| Round 1 best (no paraphrases) | 5.366 | 0.350 | 6.897 | 0.232 |
| **Current best, 2 seeds** | **4.078 / 4.106** | **0.586 / 0.576** | **5.152 / 5.139** | **0.431 / 0.434** |
| Non-recurrent baseline (ridge regression on mean/last frame + action TF-IDF) | – | 0.273 | – | 0.180 |

- **Loss:** per-step InfoNCE against all frozen instruction keys. Val candidates are the 18,081 train keys plus the
  952 val keys; test candidates are the train keys plus the 3,050 test keys. Near-duplicates are masked at a fixed
  θ = 0.9 at eval, so runs are comparable.
- **Key R@k:** where the session's own instruction ranks at the last step.
- **Test split:** whole apps held out (157 apps).

Seed noise for the best config is about 0.02–0.03 on val loss and 0.05–0.09 on test loss.

### Trajectory (best run per round)

| round | what changed | val loss | test loss |
|---|---|---|---|
| 1 | v1 as specified; crop, dropout, paraphrase ablations | 5.366 | 6.897 |
| 2 | crop tokens everywhere; fixed paraphrase augmentation; dropout + token dropout + wd | 4.988 | 6.246 |
| 3 | smaller recurrent state (d384/h512) + step dropout | 4.645 | 5.887 |
| 4 | κ cap 50 | 4.473 | 5.614 |
| 5 | (lr, EMA, queries, epochs: each within noise) | 4.416 | 5.532 |
| 6 | merged round-5 settings + similarity label smoothing | 4.218 | 5.321 |
| 7 | (regularization / smoothing variants: plateau) | 4.195 | 5.210 |
| 8 | frame-pool queries also conditioned on the previous action | **4.078** | 5.152 |
| 9–12 | separate prev-action queries, κ cap 75, keep-biased GRU gate, 2-layer GRU, width, next-frame crops, mixture readout | ≥ 4.068 (noise) | ≥ 5.074 (noise) |

## Update: rounds 13–16 (A100s, no click-centered crops)

Direction changes from the user (2026-11-04):
- all GPU jobs on A100s at the `high` QOS;
- test-set relevance labels;
- stop click-centered crops; try history-conditioned visual attention and learned compression of dense whole-screen
  features (Perceiver pattern);
- training/eval labels from OpenAI's best embedding model, with the model's typed-text inputs kept on Qwen3-0.6B.

**Test-set judged pool.** 16,885 pairs judged by Claude Opus 5.5 ($86.41): 357 same-workflow and 3,666 related. 525
test sessions have a same-workflow partner, versus 60 in val, so test is now the reliable retrieval benchmark.

| run (best-val-loss checkpoint) | labels | screen features | val loss | test loss | test R@5 | test MRR | test nDCG@10 |
|---|---|---|---|---|---|---|---|
| r8 best *with* click crops (reference) | Qwen | 8×8 + crop | 4.078 | 5.152 | 0.813 | 0.707 | 0.474 |
| r14_nocrop | Qwen | 8×8 averaged | 4.242 | 5.331 | – | – | 0.452 |
| r14_nocrop_hist / r15 seed 1 | Qwen | 8×8 + h[t−1] queries | 4.197 / 4.223 | 5.321 / 5.277 | 0.783 / 0.773 | 0.691 / 0.690 | 0.460 / 0.463 |
| r14_dense_avg8 | Qwen | dense 16×16 → fixed avg 8×8 | 4.276 | 5.458 | – | – | 0.447 |
| r14_dense_direct | Qwen | dense 16×16 (256 tokens) | 4.241 | 5.345 | – | – | 0.445 |
| r14_dense_hist / r15 seed 1 | Qwen | dense + h[t−1] queries | 4.189 / 4.211 | 5.260 / 5.290 | 0.766 / 0.766 | 0.674 / 0.689 | 0.459 / 0.453 |
| Perceiver (4 variants: 32/64/128 latents, 1–2 blocks) | Qwen | dense → learned latents | 4.35–4.42 | 5.56–5.73 | – | – | 0.43–0.45 |
| r16_oai_dense_hist (2 seeds) | OpenAI | dense + h[t−1] queries | 3.646 / 3.663* | 4.660 / 4.666* | 0.802 / 0.783 | 0.697 / 0.694 | 0.461 / 0.459 |
| r16_oai_nocrop_hist (2 seeds) | OpenAI | 8×8 + h[t−1] queries | 3.644 / 3.699* | 4.667 / 4.633* | 0.794 / 0.802 | 0.687 / 0.699 | 0.461 / 0.465 |

\* Losses under OpenAI labels are not comparable to Qwen-label losses: it's a different target space, and OpenAI keys
are more spread out (median cosine 0.16 vs 0.35). Compare label sets on the judged columns.

Findings:
- **History-conditioned attention helps a little, and the gain replicates.** Readout queries from h[t−1] plus static
  whole-screen queries improve val by about 0.03–0.04 and test by 0.03–0.05 on both feature sets and both seeds.
  Pooling runs inside the recurrence (keys/values precomputed in parallel), and training is about 2× slower
  (19 vs 36 steps/s).
- **Dense tokens beat fixed averaging of the same features** (4.241 vs 4.276 val; 5.35 vs 5.46 test). The
  1152→512 channel PCA keeps 91.5% of the variance.
- **Learned compression (Perceiver) doesn't help.** All four variants are worse than attending the dense tokens
  directly. Likely reason: latents compressed before the action- and history-conditioned readout can no longer
  isolate the region those queries need.
- **Removing click crops costs about 0.16 val.** History queries and dense tokens recover about a third of it.
- **OpenAI text-embedding-3-large labels:** judged test R@5 is +2.3 points over Qwen labels (0.795 vs 0.772,
  4 runs each), and MRR/R@1/nDCG are slightly better. OS is a bit more decodable from the embedding (probe −0.10 vs
  −0.15). As a target space alone (instruction-similarity oracle on judged test pairs), OpenAI gives R@1 0.707 vs
  0.671 and MRR 0.790 vs 0.759, but nDCG@10 0.686 vs 0.707. THETA_FN = 0.8 in the OpenAI space. Cost $0.87.

Infrastructure:
- **Quota incident.** Model downloads filled the ~2 TiB home quota on 2026-11-04. HF_HOME and the 8×8 frame cache
  now live on `/nas/ttl=60d` (with a symlink at the old path).
- **flash-attn.** A prebuilt 2.7.4 wheel matching torch 2.5.1/cu121 makes the Qwen2.5-VL vision cache about 5× faster
  (`scripts/install_flash_attn.sh`; install without dependencies).
- **Round 17 (Qwen2.5-VL-7B vision tower; not completed).** The pipeline works end to end: the smoke tests pass,
  and 512 PCA channels keep 97.6% of the variance on a 256-frame sample. The full cache was stopped when the user
  asked to wrap up. It was CPU-bound in the data loader (Qwen2VLImageProcessor on a shared node, 2–5 frames/s at 0%
  GPU utilization), so the cache would have needed several more hours. To finish it, move resize/normalize/patchify
  onto the GPU, or use more CPU per shard, then run `sweeps/r17.txt`.

## Best configuration

```
python -m wfemb.train --run NAME --use_crop --para_negatives --dropout 0.3 --tok_drop 0.4 --wd 0.1 \
  --d 384 --h 512 --step_drop 0.4 --k_max 50 --lr 5e-4 --ema 0.999 --n_queries 8 --epochs 45 \
  --label_smooth 0.2 --prev_query
```

- **Model:** 8.3M parameters. Runs about 25 steps/s on one A6000; a full run takes about 25 minutes including evals.
- **Checkpoints:** `runs/r8_prevq_reg_ls20/best.pt` (seed 0) and `runs/r9_best_seed1/best.pt` (seed 1). Selection is
  on val loss and happens around step 8,000 of 12,690.
- **Uncertainty behaves as intended.** On val, entropy over instructions falls from 5.5 to 3.9 nats along the
  session, true-key R@5 rises from 0.16 to 0.61, and ECE is 0.06–0.09.
- **Shortcut check passes.** OS is less decodable from the session embedding than from raw frame features: probe
  accuracy 0.83–0.87 vs 0.99, a gap of −0.12 to −0.16.
- **Session-to-session retrieval (val, judged pool):** cross-app nDCG@10 is 0.61–0.63. For comparison: mean-frame
  features 0.50, action TF-IDF 0.22, text-key oracle 0.71.

## Departures from the pseudocode, and why

1. **The paraphrase augmentation as written is a shortcut.** Replacing only the positive column with an LLM
   paraphrase means positives come from "LLM style" while every negative is a human-written instruction. The model
   learned that style direction: κ hit the cap at step 0, and val was much worse than without paraphrases (5.99 vs
   5.37). Fix (`--para_negatives`): every candidate column shows a randomly drawn wording of its instruction. With
   the fix, paraphrases help (val 5.24 vs 5.30, test 6.57 vs 6.68, both with crop tokens).
2. **THETA_FN = 0.9.** Under Qwen3-Embedding keys, different tasks on the same site sit at cosine 0.85–0.95, and
   near-verbatim duplicates at ≥ 0.95. 0.85 would have masked exactly the same-app negatives the design wants to
   keep.
3. **Action-crop tokens from the start.** This is the first half of upgrade 2, built straight into the cache: a
   high-res crop at the action point, 16 extra tokens per frame. It was the largest single win in round 1.
4. **Regularization, a κ cap, label smoothing and EMA were added.** The dominant failure mode is memorizing
   training sessions from their first screenshot: κ at step 0 climbs over training while val loss rises. Remedies
   that worked:
   - a smaller recurrent state;
   - step dropout, which drops intermediate steps and always keeps the final screen;
   - frame-token dropout;
   - K_MAX = 50;
   - label smoothing toward similar instruction keys, excluding the true key itself;
   - EMA of the weights.
5. **The frame pool is also queried by the previous action**, so the model can see what the last action did in the
   current frame. It's causal and costs O(1) extra state. Worth about 0.05 val and 0.1 test, on two seeds.
6. **Upgrade 1 (mixture readout) was implemented and tested: it hurts.** Val loss: K=2 4.25, K=4 4.33, K=8 4.68,
   against 4.08 single-direction. The extra readout capacity overfits.
7. **The data has no separate final screenshot.** The `computer.terminate` step's screenshot is the real final
   screen; its terminate event is dropped, so the final step is event-free as specified.
8. **Duplicate-instruction retrieval is uninformative as a metric.** Raw mean-frame features get R@1 = 1.0, because
   repeated tasks were recorded on the same machine. The judged pool replaces it.

## Pipeline

| stage | module | output |
|---|---|---|
| download AgentNet (≈200 GB split zips) | `scripts/download_agentnet.sbatch` | `agentnet_raw/` |
| parse, filter, dedupe, grouped splits, flat event arrays | `wfemb/prepare.py` | `prepared/` |
| paraphrases (Qwen2.5-7B-Instruct, vLLM) | `wfemb/paraphrase.py` | `keys/paraphrases.json` |
| frozen text keys (Qwen3-Embedding-0.6B), drift filter, typed-text vectors | `wfemb/text_embed.py`, `inspect_keys.py` | `keys/` |
| frozen frame cache: SigLIP2-so400m NaFlex, 8×8 global grid + 4×4 action crop, read straight from split zips | `wfemb/cache_frames.py`, `splitzip.py` | `frames/` (71 GB) |
| trace-judged relevance pool (Claude Opus 5.5, Message Batches) | `wfemb/judge_pool.py` | `keys/judged_val.jsonl` |
| model / loss / training / eval | `model.py`, `loss.py`, `train.py`, `evaluate.py` | `runs/<name>/` |
| baselines, run comparison | `baselines.py`, `summarize.py` | `runs/baselines.json` |

**Data after filtering:** 22,083 sessions (408K steps). Train 18,081, val 952 (5% of instruction groups), test
3,050 (157 held-out apps).

**Infrastructure notes:**
- The frame cache is held on the GPU as fp8 (30–38 GB), so a run needs one A6000 and 48 GB of host RAM. The exact
  bf16 path (`--frames_mode memmap`) matched it (val 6.08 vs 5.99 at the v1 config).
- `/nas/ucb/dfeng` has a ~2 TiB quota with about 24 GiB free, so merged copies of the zips are never written. The
  regenerable next-frame-crop cache lives on `/nas/ttl=60d`.

## Reproduce

```bash
sbatch scripts/download_agentnet.sbatch            # raw data (CPU)
sbatch scripts/prepare.sbatch                      # sessions, splits, events (CPU)
sbatch scripts/paraphrase.sbatch                   # vLLM paraphrases (1 GPU)
sbatch scripts/text_embed.sbatch                   # keys + paraphrase keys + typed text (1 GPU)
sbatch scripts/cache_frames.sbatch                 # frame cache, 8-way array (8 GPUs, ~20 min)
sbatch scripts/judge.sbatch --split val --mode batch   # judged pool (needs ANTHROPIC_API_KEY in .env)
scripts/submit_sweep.sh sweeps/r8.txt              # one single-GPU job per line
python -m wfemb.summarize r8_                      # comparison table
```

## Open items

- **Richer visual input is the most likely next gain.** Rounds 8–12 are flat, and the evidence curve stops rising
  halfway through sessions. More or higher-resolution tokens don't fit an A6000 next to the on-GPU cache. Options:
  run on A100-80GB, or stream frames from host RAM / NAS, which is about 10× slower per run.
- **Test-set judged pool:** about $80 at batch price. Only val is judged so far ($25.49).
- **Push to GitHub.** Both SSH keys started failing mid-session. Commits are on branch `train-v1` in the NAS bare
  repo `rnn:/nas/ucb/dfeng/code/workflow-embeddings.git`.
