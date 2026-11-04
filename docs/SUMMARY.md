The project trains a gated recurrent unit (GRU) to turn screenshots and input events into a running intent direction and confidence κ, using per-step contrastive cross-entropy (InfoNCE) against frozen instruction embeddings. AgentNet/OpenCUA supplies 22,083 filtered sessions and 408,384 steps: 18,081 train, 952 validation, and 3,050 test sessions, with instruction groups separated and 157 whole apps held out; frozen encoders are SigLIP2-so400m NaFlex for images and Qwen3-Embedding-0.6B for text, with OpenAI text-embedding-3-large targets added later. The archive contains 95 runs, 93 completed, on single A6000s and later A100s through Slurm; research is closed, with the round-17 image-encoder comparison unfinished.

Results below use [runs.json](data/runs.json), preferring corrected `reeval_*` over `final_*`, at each run's best-validation-loss checkpoint. Loss averages per-step InfoNCE equally across sessions, against train plus evaluated-split keys; near-duplicate masking uses θ=0.9 for Qwen and 0.8 for OpenAI. Loss and terminal instruction-key recall (`key_R@5`) are comparable **only within a label space**.

Judged retrieval is label-space independent: Claude Opus 5.5 read both session traces and assigned 2=same workflow, 1=related, 0=different; duplicate-instruction pairs also receive 2. Recall and mean reciprocal rank (MRR) use queries with a label-2 partner; normalized discounted cumulative gain (nDCG@10) uses graded gains over queries with any positive label. Unjudged pairs count as 0. Below, `_cross_app` means an unrestricted candidate pool, including same-app sessions; `_within_app` restricts it. Validation has 60 same-workflow queries; test has 525.

Qwen-label results follow. The no-crop validation winner is `r15_dense_hist8`; its advantage over the replicated history-query configurations is within seed noise. “Best” refers to validation loss, not retrospective test selection.

| Regime / run | Val loss | Test loss | Key R@5 val / test | Judged test R@5 / MRR / nDCG@10 |
|---|---:|---:|---:|---:|
| Spec: `r1_v1` | 5.986 | 7.387 | .248 / .136 | unavailable |
| Crops: `r8_prevq_reg_ls20` | 4.078 | 5.152 | .586 / .431 | .813 / .707 / .474 |
| Replicate: `r9_best_seed1` | 4.106 | 5.139 | .576 / .433 | .789 / .698 / .478 |
| No crops, dense: `r15_dense_hist8` | 4.180 | 5.287 | .569 / .422 | .771 / .679 / .461 |
| No crops, dense: `r14_dense_hist` | 4.189 | 5.260 | .565 / .427 | .766 / .674 / .459 |
| Replicate: `r15_dense_hist_seed1` | 4.211 | 5.290 | .572 / .417 | .766 / .689 / .453 |
| No crops, 8×8: `r14_nocrop_hist` | 4.197 | 5.321 | .560 / .409 | .783 / .691 / .460 |
| Replicate: `r15_nocrop_hist_seed1` | 4.223 | 5.277 | .545 / .404 | .773 / .690 / .463 |

The crop recipe uses internal width/state 384/512, eight action queries incorporating the previous action, dropout/token-dropout/step-dropout .3/.4/.4, weight decay .1, κ≤50, batch 64, learning rate .0005, exponential moving average (EMA) .999, similarity-label smoothing .2, and 45 epochs. No-crop history variants add four history and four static queries; `hist8` adds eight each. OpenAI runs retain Qwen typed-text inputs and use 3,072-dimensional instruction targets.

| OpenAI labels, no crops / run | Val loss | Test loss | Key R@5 val / test | Judged test R@5 / MRR / nDCG@10 |
|---|---:|---:|---:|---:|
| `r16_oai_nocrop_hist` | 3.644 | 4.667 | .657 / .506 | .794 / .687 / .461 |
| `r16_oai_nocrop_hist_seed1` | 3.699 | 4.633 | .659 / .511 | .802 / .699 / .465 |
| `r16_oai_dense_hist` | 3.646 | 4.660 | .659 / .516 | .802 / .697 / .461 |
| `r16_oai_dense_hist_seed1` | 3.663 | 4.666 | .667 / .513 | .783 / .694 / .459 |

[Baselines](data/baselines.json) lack judged **test** metrics; their `session_*` fields use duplicate instructions and must not substitute for `jsession_*`. TF-IDF means term frequency–inverse document frequency. The oracle reads the instruction, unavailable at inference.

| Baseline | Qwen key R@5 val / test | Judged val nDCG@10 |
|---|---:|---:|
| Ridge, mean/last frame + action TF-IDF, α=10 | .273 / .180 | unavailable |
| Action TF-IDF | unavailable | .220 |
| Mean frame | unavailable | .500 |
| Last frame | unavailable | .392 |
| Qwen text-key oracle | unavailable | .710 |

Across 17 configuration-matched seed pairs, median absolute differences are .022 validation loss and .043 test loss; maxima are .055 and .108. Seven pairs have judged test metrics: median differences are .013 R@5, .012 MRR, and .004 nDCG. These are empirical variability estimates, not significance thresholds; for example, `r8_prevq_reg_ls20` versus `r9_best_seed1` differs by .025 judged R@5.

The strongest findings are:

- **Repair the paraphrase shortcut.** Removing positive-only paraphrases (`r1_v1`→`r1_nopara`) improves val/test loss by .620/.490. Randomizing every candidate's wording (`r2_crop_nopara`→`r2_crop_paraneg`) then improves .055/.114. Both comparisons are single-seed; the proposed explanation is a positive-versus-negative wording-style shortcut.
- **Crops and regularization help.** `r1_v1`→`r1_crop` improves .466/.353; removing crops later (`r8_prevq_reg_ls20`→`r14_nocrop`) costs .164/.179, although hardware also changes. Token dropout alone (`r2_crop_paraneg`→`r2_crop_paraneg_td25`) improves .047/.171, single-seed. Smaller width/state plus step dropout improves .343/.360 and .395/.209 across `r2_crop_paraneg_reg`→`r3_small_sd30` and `r3_reg_seed1`→`r4_small_sd30_seed1`. Step dropout alone hurts the larger model (`r3_sd30`).
- **κ cap and training settings replicate.** Capping κ at 50 improves .172/.273 and .141/.460 (`r3_small_sd30`→`r4_small_sd30_k50`; `r4_small_sd30_seed1`→`r5_seed1`). The combined learning-rate/EMA/query-count/schedule/step-dropout changes improve .193/.247 and .179/.122 (`r4_small_sd30_k50`→`r6_combo`; `r5_seed1`→`r6_combo_seed1`); individual round-5 gains are not convincingly separated from noise. Smoothing adds .062/.046 and .086/.111 (`r6_combo`→`r6_combo_ls10`; `r6_combo_seed1`→`r7_seed1`).
- **Previous-action queries replicate.** Loss improves .048/.125 and .044/.070 (`r6_combo_ls10`→`r8_prevq`; `r7_seed1`→`r8_prevq_seed1`).
- **History queries offer smaller gains.** History plus static queries improve validation .018–.051 and test .010–.085: compare `r14_nocrop_hist`/`r15_nocrop_hist_seed1` with `r14_nocrop`, and `r14_dense_hist`/`r15_dense_hist_seed1` with `r14_dense_direct`. Treatment seeds agree directionally, but controls lack seed-1 replicas. Direct dense attention beats fixed averaging in loss by .035/.114 (`r14_dense_avg8`→`r14_dense_direct`), single-seed, without improving judged nDCG.
- **OpenAI targets modestly improve retrieval.** The four round-16 runs versus their corresponding round-14/15 history runs average .795 versus .772 test R@5: all four matched comparisons improve. MRR gains .008 and nDCG .003 are near seed variability. The operating-system nuisance-probe gap worsens from −.149 to −.100; more negative means less machine information relative to raw frames.

Negative results constrain further work. Mixture readout, spec upgrade 1, worsens validation loss by .167/.254/.598 for `r12_mix2`, `r12_mix4`, and `r12_mix8` versus `r8_prevq_reg_ls20`; the four-component failure repeats in `r12_mix4_seed1` versus `r9_best_seed1`. Effect crops show no meaningful test gain (`r11b_effect`/`r11_effect_seed1` versus those same crop references). Perceiver compression loses .115–.179 validation loss against `r14_dense_direct` (`r14_dense_perc32`, `r15_perc64`, `r15b_perc128_b2`, `r15b_perc64_td20`); `r14_dense_perc32_hist` and its `_seed1` replicate also lose against `r14_dense_hist`/`r15_dense_hist_seed1`. Larger states (`r6_h768` versus `r4_small_sd30_k50`) do not help; deeper recurrence (`r10_gru2` versus `r8_prevq_reg_ls20`) hurts. Batch 32 (`r6_b32`) raises validation loss to 6.095 versus 4.473 for `r4_small_sd30_k50`.

Departures from the spec address memorization and evaluation quality: symmetric paraphrase sampling, regularization, confidence capping, smoothing, EMA, causal contextual queries, and validation-loss checkpoint selection replace the original recipe. The terminate-step screenshot supplies the final event-free frame. User direction later removed click crops, moved compute to A100s, added test judgments, and changed targets.

Limitations remain. Judgments are incomplete large-language-model labels; small validation pools and many single-seed ablations limit claims. Dense features use principal component analysis, 1,152→512 channels retaining a reported 91.5% variance, confounding comparisons with the original cache. OpenAI targets also change output dimensionality and masking. For `r8_prevq_reg_ls20`/`r9_best_seed1`, test terminal top-key expected calibration error is .055/.066, but κ already starts at 47.0/46.4 and ends near 50: learned uncertainty is not established merely by calibration. Entropy curves are absent from the export. Qwen2.5-VL caching was CPU-bound and stopped at the user's wrap-up request; round 17 supplies no encoder comparison.

Reconciliation with [RESULTS.md](RESULTS.md): `runs.json` takes precedence; “only val is judged” and “all A6000” are stale, crops were not round 1's largest gain, history gains are overstated, `r15_dense_hist8` is omitted, and quoted test-oracle results are absent from `baselines.json`.

Recommended next steps, in priority order:

- Expand and human-audit validation relevance; evaluate all baselines on the same judged test pool.
- Finish the Qwen2.5-VL cache after fixing CPU preprocessing, then compare frozen encoders with matched settings.
- Add matched control seeds for history/static queries, dense averaging, and target-space changes.
- Test prefix ambiguity and confidence directly; retain the compact recurrent recipe unless new evidence supports added capacity.
