# Research summary

## Aim and setup

Can a model infer a user's workflow as a computer-use session unfolds? The original specification proposed a gated recurrent unit (GRU) state machine that reads each screenshot and its input events, then emits an intent direction and confidence κ. Per-step contrastive cross-entropy (InfoNCE) trains each prediction against frozen text embeddings of all training instructions, with κ scaling similarity scores. Inference uses only screenshots and events.

AgentNet supplies about 22,000 filtered sessions: 18,081 training, 952 validation, and 3,050 test. Instruction groups remain separate, and test holds out 157 whole applications. Initial experiments used frozen SigLIP2 image features and Qwen3 instruction embeddings.

## Repairing the shortcuts

Implementing the specification exposed an asymmetric augmentation: only the positive candidate could receive a large language model (LLM) paraphrase; negatives remained human-written. This offered a wording-style shortcut. The trained model's κ was nearly at its cap of 100 even on the first screenshot, and removing paraphrases reduced validation loss from 5.99 to 5.37. Randomly sampling a wording for every candidate removed the asymmetry and modestly improved both validation and test loss over no paraphrases. These single-seed results support, but do not directly prove, the style-shortcut explanation.

Evaluation exposed another shortcut. Raw mean-frame features retrieved duplicate-instruction partners perfectly on validation; the notes attribute this to repeated tasks recorded on the same machine. We therefore used Claude Opus 5.5 to judge pooled session pairs from their step traces as same workflow, related, or different. Test includes about 17,000 judged pairs. Only 60 validation queries have a same-workflow partner, versus 525 test queries. Reported retrieval searches all sessions in the split, including same-app candidates. These judgments provide a common benchmark when the training target space changes.

## Controlling memorization

After repairing augmentation, the main evidence pointed to memorization of sessions from their opening screenshot. During one run, first-step κ climbed from 62 to 88 after validation loss bottomed out, while loss worsened from 5.37 to 6.02. More training was strengthening recognition without improving generalization.

We reduced recurrent capacity and added step and token dropout, capped κ at 50, smoothed targets toward similar instructions, and used an exponential moving average (EMA) of weights. Smaller capacity plus step dropout, the κ cap, and similarity smoothing each improved validation and test loss in two matched seeds. An EMA-containing bundle of training changes also replicated; EMA alone and token dropout were not seed-replicated. Step dropout alone hurt the larger model, so the successful recipe depends on the combination.

Across 17 matched seed pairs, median absolute differences were about 0.02 validation loss and 0.04 test loss: useful context for small gains, not significance thresholds.

## Looking at action effects

Regularization brought validation loss to roughly 4.2, then progress slowed. Conditioning visual queries on the previous action let the model inspect its effect in the current screenshot. The gain repeated across two seeds; with stronger regularization and smoothing, the crop-based reference reached 4.08 validation and 5.15 test loss.

More elaborate alternatives did not establish a better recipe. The specification's mixture readout worsened validation loss, including a replicated four-component variant, so multiple intent directions did not resolve the plateau. Crops of the following frame around the preceding action added no convincing test benefit beyond previous-action queries. Larger states offered no clear gain, and deeper recurrence hurt: added capacity was not the missing ingredient in these tests.

## What uncertainty looks like

Across relative-position bins in the crop reference, validation entropy fell overall from about 5.5 to 3.9 nats, while true-instruction recall among the top five rose from 16% to 61%. The second seed showed the same pattern. Entropy flattened and rose slightly late in sessions, rather than falling monotonically.

Yet κ already averaged roughly 47 at the first step and approached 50 at the last. This suggests that changing alignment between the intent direction and candidate instructions carries much of the changing uncertainty, despite little change in κ. It does not establish the specification's intended confidence mechanism or calibrated uncertainty on ambiguous prefixes.

## Working without click crops

The supervisor then removed click-centered crops from scope. Removing them cost about 0.16 validation loss, although the comparison also changed hardware. Within this constraint, adding queries from the previous recurrent state, h[t−1], alongside static queries gave small, directionally consistent improvements. Both treatment seeds improved, but the controls lacked second seeds.

Direct attention over dense whole-screen tokens beat fixed averaging of the same features in loss in one seed, without improving normalized discounted cumulative gain (nDCG@10), the graded retrieval metric. Perceiver-style learned compression consistently worsened loss across tested sizes and with history queries. One plausible explanation is that compression before action- and history-conditioned attention discards location information needed by those queries; this mechanism remains untested. The best no-crop Qwen validation result was 4.18, with 5.29 test loss, essentially tied with nearby history-query variants at the observed seed variability.

## Changing the targets

We next replaced instruction targets with OpenAI text-embedding-3-large, keeping typed-text inputs unchanged. Losses across target spaces are not comparable, so the relevant result is judged session retrieval. Recall at five (R@5), measured over queries with a same-workflow partner, improved in all four matched architecture/seed comparisons, averaging 77.2% to 79.5% on test. The 2.3-point gain is an average; individual gains varied. Average changes in top-one recall, mean reciprocal rank (MRR), and nDCG@10 were comparable to seed variability.

## Limits and next steps

The current result is a compact recurrent model with modest benefits from contextual attention and richer text targets. Its evaluation depends on incomplete LLM judgments, with unjudged pairs counted as irrelevant, a small validation relevance pool, and many single-seed controls. Dense features also use principal component analysis (PCA), confounding comparisons with the original cache; changing targets also changes dimensionality and negative masking.

The Qwen2.5-VL image-encoder comparison remains unfinished: image preprocessing bottlenecked caching, which stopped at the requested wrap-up before comparative training. Next, human-audit and expand relevance judgments and score all baselines on judged test data; replicate missing controls; repair preprocessing and finish the encoder comparison; and test deliberately ambiguous prefixes to distinguish directional uncertainty from κ-based confidence.

## Evidence

Numbers use [runs.json](data/runs.json), taking `reeval_val`/`reeval_test` before `final_val`/`final_test` and `best_val` where needed. Evaluations use the checkpoint selected by validation loss. Loss averages steps within sessions, then sessions, against training plus evaluated-split instruction keys. Qwen/OpenAI near-duplicate masking thresholds are 0.9/0.8. Judged metrics use `jsession_*_cross_app`, an unrestricted candidate pool; `session_*` instead measures duplicate-instruction retrieval. Recall/MRR require a same-workflow partner; nDCG uses graded relevance over queries with any positive label. Duplicate-instruction pairs receive the highest relevance label.

- **Setup:** [original specification](PSEUDOCODE_v1.md), [dataset summary](data/dataset_summary.json), and `r1_v1`. Encoder names and experimental chronology: [running notes](RESULTS.md).
- **Paraphrase asymmetry:** `r1_v1` → `r1_nopara` gives validation/test loss 5.986/7.387 → 5.366/6.897. Symmetric sampling: `r2_crop_nopara` → `r2_crop_paraneg`, 5.295/6.681 → 5.240/6.567. [Curves](data/curves.json), `r1_v1`: first-step κ ≈99 at the selected checkpoint. Mechanism and repair: specification §4, running notes, [round 2](../sweeps/r2.txt).
- **Evaluation repair:** [baselines](data/baselines.json), `val/mean_frame`: duplicate-instruction top-one recall 1.0 on 29 queries. Same-machine explanation and 16,885 test pairs: running notes. Judge and pooling protocol: [judge_pool.py](../wfemb/judge_pool.py). `r8_prevq_reg_ls20` corrected metrics confirm 60/525 validation/test same-workflow queries. The repository export does not establish the proposed ≈5,000 validation-pair count.
- **Memorization:** curves' `kappa_first_over_training` entries are [training step, first-step κ, validation loss]. `r1_nopara`: [4500, 62.4, 5.366] → [8460, 87.7, 6.015]; `r2_crop_paraneg_reg`: [4500, 55.7, 4.988] → [8460, 70.4, 5.190]. These support an overfitting diagnosis, not a direct identification of memorized visual cues.
- **Replicated remedies:** smaller capacity + step dropout: `r2_crop_paraneg_reg` → `r3_small_sd30`, `r3_reg_seed1` → `r4_small_sd30_seed1`; κ cap: `r3_small_sd30` → `r4_small_sd30_k50`, `r4_small_sd30_seed1` → `r5_seed1`; training bundle: `r4_small_sd30_k50` → `r6_combo`, `r5_seed1` → `r6_combo_seed1`; smoothing: `r6_combo` → `r6_combo_ls10`, `r6_combo_seed1` → `r7_seed1`. Isolated token dropout: `r2_crop_paraneg` → `r2_crop_paraneg_td25`; EMA: `r4_small_sd30_k50` → `r5_ema`; harmful step dropout on the larger model: `r2_crop_paraneg_reg` → `r3_sd30`. Bundle settings: [round 6](../sweeps/r6.txt).
- **Seed variability:** 17 configuration-matched pairs in runs.json yield median absolute validation/test loss differences 0.022/0.043. Seven pairs have judged test metrics: median absolute differences are 0.011 top-one recall, 0.013 R@5, 0.012 MRR, and 0.004 nDCG@10.
- **Previous-action attention:** `r6_combo_ls10` → `r8_prevq` improves validation/test loss by 0.048/0.125; `r7_seed1` → `r8_prevq_seed1` by 0.044/0.070. Final crop reference: `r8_prevq_reg_ls20` and `r9_best_seed1`; hypothesis: [round 8](../sweeps/r8.txt).
- **Unsuccessful extensions:** `r12_mix2`, `r12_mix4`, `r12_mix8` versus `r8_prevq_reg_ls20`; `r12_mix4_seed1` versus `r9_best_seed1`. Effect crops: `r11b_effect`/`r11_effect_seed1` versus the same respective references. Larger state: `r6_h768` versus `r4_small_sd30_k50`; deeper recurrence: `r10_gru2` versus `r8_prevq_reg_ls20`.
- **Uncertainty:** curves for `r8_prevq_reg_ls20`/`r9_best_seed1`, validation `entropy_vs_relpos`, `true_key_R@5_vs_relpos`, `kappa_first`, and `kappa_last`; test curves show the same broad pattern. Relative-position bins are not exact session endpoints. κ is exported only at the first/last session step, so these data do not establish a flat trajectory throughout; dense Qwen history variants start lower, around 44.
- **No-crop pivot:** `r8_prevq_reg_ls20` → `r14_nocrop` costs 0.164 validation loss; hardware change documented in running notes. `r14_nocrop_hist`/`r15_nocrop_hist_seed1` versus `r14_nocrop`, and `r14_dense_hist`/`r15_dense_hist_seed1` versus `r14_dense_direct`, support the qualified history-plus-static-query gain. Dense versus averaged: `r14_dense_direct` versus `r14_dense_avg8`, loss gains 0.035/0.114, test nDCG@10 0.445 versus 0.447. No-crop validation minimum: `r15_dense_hist8`, 4.180/5.287.
- **Compression:** `r14_dense_perc32`, `r15_perc64`, `r15b_perc128_b2`, `r15b_perc64_td20` all lose in validation/test loss to `r14_dense_direct`. `r14_dense_perc32_hist`/`r14_dense_perc32_hist_seed1` lose to `r14_dense_hist`/`r15_dense_hist_seed1`. The location-information explanation is a hypothesis from running notes, not a measured mechanism.
- **Target change:** matched pairs are `r14_nocrop_hist` → `r16_oai_nocrop_hist`, `r15_nocrop_hist_seed1` → `r16_oai_nocrop_hist_seed1`, `r14_dense_hist` → `r16_oai_dense_hist`, `r15_dense_hist_seed1` → `r16_oai_dense_hist_seed1`. Test R@5 gains: 1.14, 2.86, 3.62, 1.71 percentage points; mean 2.33. Mean top-one recall/MRR/nDCG@10 gains: 0.006/0.008/0.003. [Round 16](../sweeps/r16.txt) records unchanged typed-text inputs and changed masking.
- **Limits and unfinished comparison:** running notes report dense-feature PCA from 1,152 to 512 channels retaining 91.5% variance; this does not confound the comparison between direct and averaged versions of that same cache. They also record processor-bound Qwen2.5-VL caching and the wrap-up stop. [Round 17](../sweeps/r17.txt) specifies the uncompleted comparison; runs.json contains no round-17 results. Baselines.json has no judged test baseline metrics.
