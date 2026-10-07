# Results
Detailed report of the reproduction of Kirchenbauer et al., *A Watermark for Large Language Models* (ICML 2023), and of two extensions. The [README](README.md) has the overview; every number below comes from a file in `results/summary/`, rebuilt by `watermark.ipynb`.

Environment: Apple M5 Pro, 24 GB unified memory (`hw.memsize` = 25769803776), macOS 27.0.1, device `mps`, dtype fp16 (logits finite; argmax agreement with fp32 = 0.9996 on 4 prompts, `results/summary/phase0_check.jsonl`). T5-Large in bf16. Python 3.12.5, torch 2.14.1, transformers 5.18.0, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, extra package `tabulate` (pandas markdown tables).

Processor order check: in transformers 5.18.0 `GenerationMixin._get_logits_processor` builds the built-in processors, then merges the user `logits_processor` list (`_merge_criteria_processor_list`), and only then appends `TemperatureLogitsWarper`, top-k, top-p (sampling only), and finally the `watermarking_config` processor. So our KGW processor adds δ before division by τ = 0.7, exactly like the paper's code (effective bias δ/τ).

Green-list equivalence (`results/summary/greenlist_equivalence.txt`): for 1,000 random previous-token ids (OPT, V = 50,265) and 200 (Qwen, V = 151,665), γ ∈ {0.25, 0.5}, our green lists equal `WatermarkBase._get_greenlist_ids` of the official code (same ids, same order): PASS.

**Confidence intervals.** Square brackets are 95% intervals: Wilson score intervals for every proportion (TPR, FPR, kept shares, "never reaches z = 4"), percentile bootstrap (2,000 resamples) for medians (tokens to reach z = 4), mean ± 1.96·SE for means. "Paper inside / outside our CI" compares the paper's point with our interval; when it is outside we also build a Wilson interval for the paper from its own count (≈500 in Table 8, 487 in Table 9) and call the difference **different** only if the two intervals do not overlap, otherwise **borderline** (`vs_paper_*` columns in `results/summary/table8.md`, `table9_t5.md`).

## Summary
- **Reproduction matches the paper.** §4.1: spike entropy 0.813 (paper 0.807), Thm 4.2 bound 143.2 (142.2), TPR at z = 4 0.986 [0.971, 0.993] (0.984), no false positive on 1,000 human texts (FPR ≤ 0.004). Table 8 (12 rows): 11 agree with the paper within our 95% intervals; only m-nom δ = 1, γ = 0.5 is weaker than the paper (z = 5: 0.37 [0.31, 0.42] vs 0.50). Fig. 2, 3, 4, 7 reproduce (Fig. 7 bound 0.637 / 0.716 / 0.762 vs ≈ 0.63 / 0.71 / 0.76). Green lists are bit-identical to the official code.
- **T5 attack reproduced with the paper's one-word rule**: TPR at z = 4 at ε = 0.3 / 0.5 / 0.7 is 0.24 [0.17, 0.33] / 0.02 [0.01, 0.07] / 0.01 [0.00, 0.05] (paper 0.353 / 0.094 / 0.039: borderline / different / consistent), AUC within 0.05, PPL 16.0 / 24.8 / 33.5 (21.2 / 28.4 / 33.9); at ε = 0.1 our watermark is more robust (0.99 vs 0.82). Letting T5 fill multi-word spans weakens the attack (0.495 at ε = 0.3) because texts grow and z grows with √T.
- **1a**: on the distribution the code actually samples from (τ = 0.7, bias δ/τ), Theorem 4.2 stays valid in all 22 runs but is *looser* for δ ≥ 2; the paper's large-δ gap is the bound's conservatism, not the temperature.
- **1b, entropy decides**: median tokens to z = 4 is 24 [20, 27] for news and 188 [135, never] for code, where 49% [42, 57] never reach it in 200 tokens; TriviaQA answers are essentially never detectable (at γ = 0.5 a 16-token text cannot exceed z = 4). In code, repeated token pairs break calibration (unwatermarked mean z −2.9); ignoring repeats fixes it (−0.45) and doubles Qwen's TPR at 1% FPR (0.24 [0.19, 0.32] → 0.51 [0.44, 0.59]).
- **1c, base vs instruct**: instruction tuning lowers entropy everywhere (news 2.52 → 1.96 nats), yet news detection is indistinguishable (TPR at 1% FPR 0.995 [0.97, 1.00] vs 0.990 [0.96, 1.00]) because the watermark pushes the instruct model off its confident path (1.96 → 2.31 nats), at a higher quality cost (PPL 5.7 → 9.5). On code there is no consistent difference; QA accuracy is unaffected (EM 0.202 [0.17, 0.24] → 0.212 [0.18, 0.25]).
- **Length side effect, key-specific**: under our key EOS is green after "." while newline is red, so δ = 2 makes 16% [14, 18] of generations stop before 50 tokens vs 6% [4, 7] without watermark (kept share 0.50 vs 0.70); three other keys are indistinguishable from no watermark.
- **2a, a 2026 attacker**: at the fixed threshold z = 4, a light Qwen paraphrase and the paper's T5 attack are indistinguishable at matched meaning (0.22 [0.17, 0.28] vs 0.24 [0.17, 0.33] at similarity ≈ 0.85). At a calibrated 1% FPR the paraphraser is clearly stronger (0.435 [0.37, 0.50] vs 0.81 [0.72, 0.88]), and to reach TPR ≈ 0.1 T5 must sacrifice more meaning (similarity 0.774 vs 0.837 for a full paraphrase). Removal still costs meaning, but now takes one call to a free 1.5B model instead of 60-140 T5 calls.
- **2b, copy-paste**: a 60-token watermarked span in a 600-token human document is caught by the full-text test 11.5% [7.8, 16.7] of the time at 1% FPR and by WinMax (sliding window, authors' 2024 follow-up) 86.5% [81.1, 90.6].

## Paper vs ours
### §4.1 headline (`results/summary/headline.md`, `figures/head_z_hist.png`, examples in `results/summary/examples.md`)
γ = 0.5, δ = 2, multinomial (τ = 0.7), T = 200 ± 5.

| quantity | paper | ours |
|---|---|---|
| generations kept | 507 | 497 (of 1,000 generated; kept share 0.497 [0.466, 0.528]) |
| average spike entropy S | 0.807 | 0.813 |
| Thm 4.2 lower bound on E\|s\|_G | 142.2 | 143.2 |
| observed mean green count | 159.5 | 158.3 [157.5, 159.2]: the paper's 159.5 is just above our interval |
| Thm 4.2 upper bound on sd | 6.41 | 6.38 |
| observed sd of green count | – | 9.85 (above the bound: the bound holds per sequence at fixed entropy; across sequences the entropy varies) |
| predicted detection at z = 4 (Gaussian from the bounds) | 98.6% | 99.0% (the same formula on the paper's own bounds gives 98.5%) |
| observed TPR at z = 4 | 98.4% | 98.6% [97.1, 99.3]: consistent |
| 4-way beam search TPR at z = 4 | 99.6% | 100% [97.5, 100] (n = 150): consistent |
| above the 25th entropy percentile | 100% (375/500) | 100% [99.0, 100] (373/497) |
| FPR at z = 4, 1,000 human baselines | 0 | 0 [0, 0.4%] |
| mean z drop after decode → re-encode | – | 0.005 |
| TPR at z = 4 ignoring repeated (context, token) pairs | – | 97.2% [95.3, 98.3] |

### Table 8 (`results/summary/table8.md`; negatives: 1,000 human baselines, and 500 δ = 0 generations)

FPR at z = 4 is 0.000 [0, 0.004] on the human baselines in every row.

| sampling | δ | γ | count | TPR z=4 [95% CI] (paper) | vs paper | TPR z=5 [95% CI] (paper) | vs paper | FPR z=4 on δ=0 |
|---|---|---|---|---|---|---|---|---|
| m-nom | 1 | 0.50 | 300 | 0.700 [0.646, 0.749] (0.767) | borderline | 0.367 [0.314, 0.423] (0.504) | **different** | 0.002 |
| m-nom | 1 | 0.25 | 300 | 0.777 [0.726, 0.820] (0.729) | consistent | 0.537 [0.480, 0.592] (0.482) | consistent | 0.002 |
| m-nom | 2 | 0.50 | 497 | 0.986 [0.971, 0.993] (0.984) | consistent | 0.980 [0.963, 0.989] (0.978) | consistent | 0.002 |
| m-nom | 2 | 0.25 | 300 | 1.000 [0.987, 1.000] (0.994) | consistent | 1.000 [0.987, 1.000] (0.988) | consistent | 0.002 |
| m-nom | 5 | 0.50 | 300 | 1.000 [0.987, 1.000] (0.996) | consistent | 1.000 [0.987, 1.000] (0.992) | consistent | 0.002 |
| m-nom | 5 | 0.25 | 300 | 1.000 [0.987, 1.000] (1.000) | consistent | 1.000 [0.987, 1.000] (0.998) | consistent | 0.002 |
| 8-beams | 1 | 0.50 | 100 | 0.850 [0.767, 0.907] (0.873) | consistent | 0.750 [0.657, 0.825] (0.812) | consistent | – |
| 8-beams | 1 | 0.25 | 100 | 0.820 [0.733, 0.883] (0.819) | consistent | 0.750 [0.657, 0.825] (0.770) | consistent | – |
| 8-beams | 2 | 0.50 | 150 | 0.993 [0.963, 0.999] (0.992) | consistent | 0.987 [0.953, 0.996] (0.984) | consistent | – |
| 8-beams | 2 | 0.25 | 150 | 0.993 [0.963, 0.999] (0.994) | consistent | 0.987 [0.953, 0.996] (0.990) | consistent | – |
| 8-beams | 5 | 0.50 | 100 | 1.000 [0.963, 1.000] (1.000) | consistent | 1.000 [0.963, 1.000] (1.000) | consistent | – |
| 8-beams | 5 | 0.25 | 100 | 1.000 [0.963, 1.000] (1.000) | consistent | 1.000 [0.963, 1.000] (1.000) | consistent | – |

The only real discrepancy is m-nom δ = 1, γ = 0.5, where our watermark is weaker than the paper's (z = 5: 0.37 vs 0.50, intervals do not overlap). At δ = 1 detection depends on a few frequent token pairs whose green status is fixed by the key (see the key-specific null calibration under *Limitations*), so a different key can plausibly move this row; we have not tested that. The other 11 rows agree with the paper within our intervals.

The δ = 0 false positives (`results/summary/delta0_false_positives.md`) are repetitive generations; the γ = 0.5 one (idx 417, a list of jazz names repeated) has z = 4.38 with repeats and 0.40 ignoring repeated pairs.

### Fig. 4 / 8 / 9: AUC and oracle PPL (`results/summary/table8.md`, `results/summary/ppl_baselines.md`, `figures/fig4_roc.png`)

| sampling | δ | γ | AUC (paper) | PPL mean (paper) | PPL median |
|---|---|---|---|---|---|
| m-nom | 1 | 0.50 | 0.997 (0.985) | 5.72 (5.4) | 5.66 |
| m-nom | 1 | 0.25 | 0.992 (0.989) | 5.56 (5.5) | 5.45 |
| m-nom | 2 | 0.50 | 0.999 (0.998) | 6.53 (6.2) | 6.43 |
| m-nom | 2 | 0.25 | 1.000 (0.998) | 6.69 (6.6) | 6.73 |
| m-nom | 5 | 0.50 | 1.000 (1.000) | 9.23 (9.1) | 9.33 |
| m-nom | 5 | 0.25 | 1.000 (1.000) | 12.40 (10.7) | 12.42 |
| 8-beams | 1 | 0.50 | 0.991 (0.987) | 1.57 (1.2) | 1.45 |
| 8-beams | 1 | 0.25 | 0.986 (0.977) | 1.54 (1.2) | 1.43 |
| 8-beams | 2 | 0.50 | 1.000 (0.999) | 1.63 (1.2) | 1.56 |
| 8-beams | 2 | 0.25 | 0.998 (1.000) | 1.70 (1.3) | 1.62 |
| 8-beams | 5 | 0.50 | 1.000 (1.000) | 1.74 (1.2) | 1.58 |
| 8-beams | 5 | 0.25 | 1.000 (1.000) | 1.68 (1.3) | 1.57 |
| no watermark, m-nom (n = 500) | | | | 5.29 (5.1) | 5.16 |
| no watermark, 8 beams (n = 100) | | | | 1.44 (1.2) | 1.39 |
| human baselines (n = 1,000) | | | | 11.06 | 9.93 |

Our perplexities sit slightly above the paper's (+0.1 to +0.3 for multinomial, about +0.25 to +0.5 for 8 beams), except δ = 5, γ = 0.25 (+1.7). Same ordering everywhere.

### Fig. 2 (`results/summary/fig2_tradeoff.md`, `figures/fig2_tradeoff.png`)
Left, multinomial, δ ∈ {0, 1, 2, 5, 10} × γ ∈ {0.1, 0.25, 0.5, 0.75, 0.9} (n = 100-500 per point): smaller γ gives a higher z at the same PPL (δ = 2: γ = 0.1 / 0.25 / 0.5 / 0.75 / 0.9 → z = 10.8 / 11.1 / 8.3 / 5.3 / 3.3 at PPL 6.3 / 6.7 / 6.5 / 5.8 / 5.5; γ = 0.1, δ = 5: z = 32.6 at PPL 13.7), consistent with γ = 0.1 being Pareto-best. Right, γ = 0.5, n = 100 (150 at δ = 2): 8-beam search reaches z = 11.5 at δ = 2 with PPL 1.63 (no watermark 1.44); greedy reaches z = 8.4 at PPL 2.11 (no watermark 1.81). As in the paper, beam search buys a large z at almost no PPL cost. 4 beams: only δ = 2 (z = 11.2, PPL 1.74).

### Fig. 3 (`results/summary/fig3_z_vs_T.md`, `figures/fig3_z_vs_T.png`)
First T at which the mean z exceeds 4: δ = 5 multinomial, γ = 0.1 / 0.25 / 0.5 / 0.75 / 0.9 → 4 / 8 / 22 / 60 / 176 tokens; γ = 0.25, δ = 1 / 2 / 5 / 10 → 124 / 30 / 8 / 6 tokens. 8 beams, γ = 0.25 (n = 100, 150 at δ = 2), δ = 0.5 / 1 / 2 / 5 / 10 → 200 / 58 / 16 / 6 / 6 tokens; mean z > 5 from 88 / 24 / 10 / 10 tokens at δ = 1 / 2 / 5 / 10 (paper: about 35 tokens at δ = 2). As in the paper, beam search detects with far fewer tokens than multinomial sampling at the same δ (δ = 2: 16 vs 30 tokens).

### Table 9 / Fig. 5: T5 span attack (`results/summary/table9_t5.md`, `figures/fig5_t5_attack.png`)
γ = 0.5, δ = 2, multinomial. AUC against the 1,000 unattacked human baselines; PPL = OPT-2.7B on the attacked text given the prompt.

Multi-word variant (T5's fill may be several words), 200 texts:

| ε | TPR z=4 [95% CI] (paper) | TPR z=5 [95% CI] (paper) | AUC (paper) | PPL (paper) | OPT tokens after attack |
|---|---|---|---|---|---|
| 0 (re-encoded) | 1.000 [0.981, 1.000] (0.984) | 0.985 [0.957, 0.995] (0.977) | 1.000 (0.998) | 6.6 (6.3) | 200 |
| 0.1 | 0.955 [0.917, 0.976] (0.819) **different** | 0.890 [0.839, 0.926] (0.577) **different** | 1.000 (0.988) | 8.1 (13.1) | 232 |
| 0.3 | 0.495 [0.426, 0.564] (0.353) **different** | 0.295 [0.236, 0.362] (0.127) **different** | 0.992 (0.954) | 9.9 (21.2) | 295 |

Single-word variant (each replacement is exactly one word, as the paper describes), first 100 of the same texts:

| ε | TPR z=4 [95% CI] (paper), verdict | TPR z=5 [95% CI] (paper), verdict | AUC (paper) | PPL (paper) | OPT tokens after attack | words replaced |
|---|---|---|---|---|---|---|
| 0 (re-encoded) | 1.000 [0.963, 1.000] (0.984) consistent | 0.980 [0.930, 0.994] (0.977) consistent | 1.000 (0.998) | 6.7 (6.3) | 200 | 0 |
| 0.1 | 0.990 [0.946, 0.998] (0.819) **different** | 0.870 [0.790, 0.922] (0.577) **different** | 1.000 (0.988) | 9.2 (13.1) | 201 | 20.0 |
| 0.3 | 0.240 [0.167, 0.332] (0.353) borderline | 0.090 [0.048, 0.162] (0.127) consistent | 0.996 (0.954) | 16.0 (21.2) | 202 | 60.0 |
| 0.5 | 0.020 [0.006, 0.070] (0.094) **different** | 0.000 [0, 0.037] (0.029) consistent | 0.868 (0.838) | 24.8 (28.4) | 206 | 99.9 |
| 0.7 | 0.010 [0.002, 0.054] (0.039) consistent | 0.000 [0, 0.037] (0.012) consistent | 0.652 (0.696) | 33.5 (33.9) | 210 | 130.4 (cap of 3·ε·T = 420 attempts reached by some texts) |

**Reading.** Our attack is implemented from the paper's description (no attack code was released). With single-word replacements it reproduces the *shape* of Table 9 and Fig. 5 and most numbers: at ε = 0.3 the TPR at z = 5 and at ε = 0.5-0.7 the TPR at z = 5 agree with the paper within our intervals, ε = 0.7 agrees at z = 4, AUC is within 0.05 and PPL within 0.4-5.2. Two differences are outside the intervals: at ε = 0.1 our watermark survives better (0.99 [0.95, 1.00] vs 0.82), and at ε = 0.5 (z = 4) our attack removes slightly more (0.02 [0.01, 0.07] vs 0.094); ε = 0.3 at z = 4 is borderline (0.24 [0.17, 0.33] vs 0.353). With multi-word fills the attack lowers the green fraction just as much (0.79 → 0.62 at ε = 0.3) but the texts grow to 295 tokens, so z, which grows with √T at a fixed green fraction, stays high; its PPL cost is also far lower (9.9 vs 16.0). The replacement unit, not the detector, explains the earlier gap.

### Fig. 7 (`results/summary/fig7_theory.md`, `figures/fig7_theory.png`)
γ = 0.5: observed green fraction at δ = 0, 0.5, 1, 2, 5, 10 = 0.487, 0.586, 0.661, 0.792, 0.948, 0.997 (paper, read off the plot: ≈ 0.53, 0.60, 0.68, 0.80, 0.94, 0.99). Thm 4.2 bound (S on the raw distribution, α = e^δ) = 0.500, 0.575, 0.637, 0.716, 0.762, 0.742 (paper ≈ 0.50, 0.59, 0.63, 0.71, 0.76, 0.76). Tight for small δ, conservative for large δ, as in the paper.

### Table 10 / TriviaQA (`figures/ext1b_triviaqa.md`)
γ = 0.5, δ = 2, greedy, first 500 validation questions. EM without → with watermark: FLAN-UL2 (paper) 0.374 → 0.336; BLOOMZ (paper) 0.296 → 0.259; OPT-1.3B 0.000 → 0.004 (cannot do the task with this prompt); Qwen2.5-1.5B 0.072 [0.052, 0.098] → 0.070 [0.051, 0.096]; Qwen2.5-1.5B-Instruct 0.202 [0.169, 0.239] → 0.212 [0.178, 0.250]: both differences are well inside the intervals, i.e. no measurable EM change, while the paper reports a drop of ≈0.04. Mean z without → with: paper −0.007 → 0.402 (FLAN-UL2); ours (Qwen-Instruct) −0.05 → 0.80, answers average 5.8 tokens.

## Extension 1 / 2
### 1a. Theorem 4.2 on the distribution actually sampled (`results/summary/ext1a_bounds.md`, `figures/ext1a_bound_vs_observed.png`)
The code samples from softmax((l + δ·green)/τ), which is the watermark with bias δ/τ applied to softmax(l/τ). The bound recomputed for that distribution (S_sampled, α_eff = e^(δ/τ)) is a valid lower bound on the mean green fraction in all 22 multinomial runs, and so is the paper-style bound. The corrected bound is marginally tighter only at the smallest biases (γ = 0.1, δ = 1: 0.175 vs 0.172; γ = 0.25 and 0.5 at δ = 0.5: +0.002 and +0.001) and looser everywhere else, clearly so for δ ≥ 2 (γ = 0.5, δ = 2: 0.675 vs 0.716, observed 0.792; δ = 10: 0.665 vs 0.742, observed 0.997). Temperature 0.7 lowers the spike entropy (0.714 vs 0.813 at δ = 2) more than the larger effective bias raises the coefficient.
**Interpretation**: the paper's version does not use the distribution the theorem assumes, but it happens to give a slightly higher (still valid) number. The large-δ gap in Fig. 7 is the bound's own conservatism, not the temperature.

### 1b + 1c. Entropy decides (`results/summary/ext1b_cells.md`; figures `ext1b_z_vs_entropy.png` (z ignoring repeats), `ext1b_z_vs_entropy_repeats.png`, `ext1b_tokens_to_detect.png`, `ext1b_tokens_to_detect_norep.png`, `ext1b_theory_vs_observed.png`, `ext1c_entropy_hist.png`)
KGW γ = 0.25, δ = 2, τ = 0.7 for news/code; QA greedy γ = 0.5, δ = 2. TPR at 1% FPR uses each cell's own δ = 0 generations as negatives (empirical threshold). H = mean Shannon entropy of the unwatermarked model's next-token distribution. Two z versions: repeats counted (paper) and repeated (context, token) pairs ignored.

| domain | model | n | mean T | H (δ = 0) | S_sampled | mean z: repeats / ignored | δ = 0 mean z: repeats / ignored | TPR at 1% FPR: repeats / ignored | median tokens to z = 4 (never) |
|---|---|---|---|---|---|---|---|---|---|
| news | OPT-1.3B | 200 | 180 | 2.41 | 0.53 | 10.3 / 8.8 | 0.03 / −0.02 | 0.990 / 0.990 | 24 (1%) |
| code | OPT-1.3B | 164 | 192 | 0.66 | 0.35 | 3.8 / 2.5 | **−2.88 / −0.45** | 0.524 / 0.616 | 188 (49%) |
| QA | OPT-1.3B | 500 | 16 | 2.06 | 0.77* | 1.4 / 1.3 | −0.41 / −0.38 | 0.518 / 0.698 | – (99%) |
| news | Qwen2.5-1.5B | 198 | 176 | 2.52 | 0.55 | 10.5 / 9.2 | 0.06 / 0.15 | 0.995 / 0.995 | 25 (2%) |
| code | Qwen2.5-1.5B | 164 | 132 | 0.59 | 0.36 | 3.4 / 2.3 | 0.02 / −0.10 | 0.244 / 0.512 | – (60%) |
| QA | Qwen2.5-1.5B | 500 | 5.5 | 1.74 | 0.77* | 1.2 / 1.2 | −0.24 / −0.20 | 0.142 / 0.156 | – (100%) |
| news | Qwen2.5-1.5B-Instruct | 200 | 184 | 1.96 | 0.54 | 10.1 / 10.0 | 0.12 / 0.11 | 0.990 / 0.990 | 27 (4%) |
| code | Qwen2.5-1.5B-Instruct | 164 | 145 | 0.43 | 0.35 | 2.6 / 2.4 | −0.12 / −0.00 | 0.457 / 0.488 | – (73%) |
| QA | Qwen2.5-1.5B-Instruct | 500 | 5.8 | 1.18 | 0.72* | 0.8 / 0.8 | −0.05 / −0.06 | 0.054 / 0.052 | – (100%) |
| news, δ = 4 | Qwen2.5-1.5B-Instruct | 200 | 184 | 1.96 | 0.55 | 18.7 / 18.5 | 0.12 / 0.11 | 0.990 / 0.990 | 8 (1%) |

**95% intervals for the 1b/1c proportions and medians** (`results/summary/ext1b_cells.md`, columns `*_lo`, `*_hi`; error bars in `ext1b_tokens_to_detect*.png`):

| domain | model | TPR at z = 4 | TPR at 1% FPR (repeats) | TPR at 1% FPR (repeats ignored) | median tokens to z = 4 | never reach z = 4 |
|---|---|---|---|---|---|---|
| news | OPT-1.3B | 0.975 [0.943, 0.989] | 0.990 [0.964, 0.997] | 0.990 [0.964, 0.997] | 24 [20, 27] | 0.010 [0.003, 0.036] |
| code | OPT-1.3B | 0.463 [0.389, 0.540] | 0.524 [0.448, 0.599] | 0.616 [0.540, 0.687] | 188 [135, never] | 0.494 [0.418, 0.570] |
| QA | OPT-1.3B | 0.000 [0, 0.008] | 0.518 [0.474, 0.561] | 0.698 [0.656, 0.737] | never | 0.990 [0.977, 0.996] |
| news | Qwen2.5-1.5B | 0.975 [0.942, 0.989] | 0.995 [0.972, 0.999] | 0.995 [0.972, 0.999] | 25 [24, 27] | 0.015 [0.005, 0.044] |
| code | Qwen2.5-1.5B | 0.329 [0.262, 0.404] | 0.244 [0.185, 0.315] | 0.512 [0.436, 0.588] | never | 0.598 [0.521, 0.670] |
| QA | Qwen2.5-1.5B | 0.000 [0, 0.008] | 0.142 [0.114, 0.175] | 0.156 [0.127, 0.190] | never | 0.998 [0.989, 1.000] |
| news | Qwen2.5-1.5B-Instruct | 0.950 [0.910, 0.973] | 0.990 [0.964, 0.997] | 0.990 [0.964, 0.997] | 27 [24, 29] | 0.040 [0.020, 0.077] |
| code | Qwen2.5-1.5B-Instruct | 0.220 [0.163, 0.289] | 0.457 [0.383, 0.534] | 0.488 [0.412, 0.564] | never | 0.732 [0.659, 0.794] |
| QA | Qwen2.5-1.5B-Instruct | 0.000 [0, 0.008] | 0.054 [0.037, 0.077] | 0.052 [0.036, 0.075] | never | 0.998 [0.989, 1.000] |
| news, δ = 4 | Qwen2.5-1.5B-Instruct | 0.990 [0.964, 0.997] | 0.990 [0.964, 0.997] | 0.990 [0.964, 0.997] | 8 [8, 8] | 0.010 [0.003, 0.036] |

The TPR-at-1%-FPR intervals cover only the sampling of the positives; the threshold itself is estimated from 164-500 negatives and adds uncertainty, so treat those columns as slightly too narrow. QA TPR at 1% FPR is inflated by discreteness (16-token z takes few values, so the 99th-percentile threshold is a tie point).

\* greedy decoding, τ = 1. "–" = more than half of the sequences never reach z = 4 (tokens counted with repeats; with repeats ignored code never reaches a median either, `ext1b_tokens_to_detect_norep.png`).

- **Domains (1b)**: detectability tracks entropy. Code has about two thirds of the spike entropy of news and needs ~8x more tokens; short factual answers are not detectable (at γ = 0.5 a 16-token answer has a maximum z of exactly 4).
- **Repetition breaks the z-test on code**: counting repeated (context, token) pairs, unwatermarked OPT code has mean z −2.88 (it can swing either way: the green status of one repeated pair, e.g. newline → indentation, is counted dozens of times). Ignoring repeats restores calibration (−0.45) and raises TPR at 1% FPR for code: clearly for Qwen (0.24 [0.19, 0.32] → 0.51 [0.44, 0.59]), within the intervals for OPT (0.52 [0.45, 0.60] → 0.62 [0.54, 0.69]). For code, use the ignore-repeats z (or empirical thresholds).
- **Theory vs observed**: per sequence, the Thm 4.2 prediction from the sequence's own mean spike entropy is below the observed green fraction on average in every cell (news OPT: observed 0.584 vs corrected prediction 0.453).
- **Base vs instruct (1c)**: instruction tuning lowers the unwatermarked entropy in every domain (news 2.52 → 1.96, code 0.59 → 0.43, QA 1.74 → 1.18 nats). On news, detection is indistinguishable at δ = 2 (TPR at 1% FPR 0.995 [0.972, 0.999] vs 0.990 [0.964, 0.997]; median tokens 25 [24, 27] vs 27 [24, 29]; intervals overlap) because under the watermark the instruct model's entropy rises (1.96 → 2.31 nats) while the base model's does not (2.52 → 2.48). The cost is quality: oracle PPL (base Qwen as judge) 5.67 → 9.55 at δ = 2 and 25.6 at δ = 4 (for comparison OPT news 5.20 → 6.89 with OPT-2.7B). δ = 4 cuts the tokens needed for z = 4 from 27 to 8. On code, base vs instruct depends on the metric: the instruct model has a lower mean z (2.6 vs 3.4) and a lower TPR at z = 4 (0.22 [0.16, 0.29] vs 0.33 [0.26, 0.40], overlapping), but a *higher* TPR at 1% FPR with repeats counted (0.46 [0.38, 0.53] vs 0.24 [0.19, 0.32], because the base model's unwatermarked code has a heavier upper tail) and an indistinguishable one with repeats ignored (0.49 vs 0.51). No consistent base-vs-instruct difference on code.

### Length side effect: the watermark changes when generations stop, depending on the key (`results/summary/kept_counts.md`, `keys_eos.md`, `eos_after_dot.json`, `eos_end_contexts.json`)
- Under our key, γ = 0.5, δ = 2: 50% [47, 53] of generations reach 195 tokens vs 70% [67, 74] without watermark; 16% [14, 18] stop before 50 tokens vs 6% [4, 7] (n = 1,000 and 711; intervals far apart). At δ = 5-10 and γ = 0.5-0.75, 29-37% stop before 50 tokens. At γ ≤ 0.25 generations stop *less* often than without watermark (2-3% before 50 tokens).
- Mechanism: 71% of natural endings happen right after "." (`eos_end_contexts.json`). Under our key, after "." the end-of-text token is green for γ ≥ 0.5 while the main continuation (newline, 43% of continuations) is red, so δ = 2 multiplies P(EOS) by about 2.3 (estimated with the unwatermarked continuation mix, before temperature).
- Check on 3 other keys (γ = 0.5, δ = 2, same first 100 prompts):

| hash key | EOS green after "." | estimated P(EOS) multiplier after "." | kept share (195..205) [95% CI] | stopped before 50 tokens [95% CI] |
|---|---|---|---|---|
| 15485863 (ours) | yes | 2.29 | 0.53 [0.43, 0.63] | 13% [8, 21] |
| 15485867 | no | 0.17 | 0.80 [0.71, 0.87] | 3% [1, 8] |
| 32452843 | yes | 1.18 | 0.76 [0.67, 0.83] | 1% [0, 5] |
| 49979687 | no | 0.32 | 0.87 [0.79, 0.92] | 1% [0, 5] |
| no watermark | – | 1 | 0.75 [0.66, 0.83] | 4% [2, 10] |

Only the key where EOS is green after "." *and* its competitors are mostly red stops early: on these 100 prompts its kept share (0.53 [0.43, 0.63]) is clearly below the unwatermarked 0.75 [0.66, 0.83]; its early-stop share (13% [8, 21]) overlaps the unwatermarked 4% [2, 10] at n = 100, but the full-sample comparison above (16% vs 6%, n = 1,000 / 711) is unambiguous. EOS being green alone is not enough: key 32452843 has EOS green, but 82% of the competing continuation mass is green too. The other three keys are indistinguishable from no watermark on both measures (all intervals overlap). The effect is real but key-specific; the paper's T = 200 ± 5 filter (and ours) hides it in the detection tables.

### 2a. A 2026 attacker: paraphrase vs T5 (`results/summary/ext2a_attacks.md`, `figures/ext2a_removal_frontier.png`)
Same 200 watermarked texts (and their 200 human baselines, attacked the same way for calibration).

| attack | n (wm / human) | meaning kept (MiniLM cos) [95% CI] | OPT tokens changed | green fraction | TPR at z = 4 [95% CI] | TPR at 1% FPR [95% CI] | AUC | PPL |
|---|---|---|---|---|---|---|---|---|
| none (re-encoded) | 200 / 200 | 1.000 | 0.00 | 0.79 | 1.000 [0.981, 1.000] | 1.000 [0.981, 1.000] | 1.000 | 6.6 |
| T5 multi-word, ε = 0.1 | 200 / 200 | 0.951 [0.947, 0.955] | 0.22 | 0.72 | 0.955 [0.917, 0.976] | 1.000 [0.981, 1.000] | 1.000 | 8.1 |
| T5 multi-word, ε = 0.3 | 200 / 200 | 0.881 [0.874, 0.888] | 0.65 | 0.62 | 0.495 [0.426, 0.564] | 0.925 [0.880, 0.954] | 0.991 | 9.9 |
| T5 single-word, ε = 0.1 | 100 / 100 | 0.953 [0.946, 0.960] | 0.11 | 0.72 | 0.990 [0.946, 0.998] | 0.990 [0.946, 0.998] | 1.000 | 9.2 |
| T5 single-word, ε = 0.3 | 100 / 100 | 0.856 [0.843, 0.868] | 0.34 | 0.62 | 0.240 [0.167, 0.332] | 0.810 [0.722, 0.875] | 0.995 | 16.0 |
| T5 single-word, ε = 0.5 | 100 / 100 | 0.774 [0.757, 0.791] | 0.57 | 0.55 | 0.020 [0.006, 0.070] | 0.110 [0.063, 0.186] | 0.824 | 24.8 |
| T5 single-word, ε = 0.7 | 100 / 100 | 0.719 [0.700, 0.738] | 0.75 | 0.51 | 0.010 [0.002, 0.054] | 0.010 [0.002, 0.054] | 0.607 | 33.5 |
| Qwen paraphrase, light | 200 / 200 | 0.850 [0.828, 0.872] | 0.65 | 0.60 | 0.220 [0.168, 0.282] | 0.435 [0.368, 0.504] | 0.881 | 30.1 |
| Qwen paraphrase, medium | 200 / 200 | 0.848 [0.834, 0.862] | 0.81 | 0.55 | 0.035 [0.017, 0.070] | 0.205 [0.155, 0.266] | 0.773 | 21.2 |
| Qwen paraphrase, full | 200 / 200 | 0.837 [0.822, 0.852] | 0.86 | 0.53 | 0.015 [0.005, 0.043] | 0.100 [0.066, 0.149] | 0.705 | 20.6 |

AUC and TPR at 1% FPR use the attacked human texts of the same attack and level as negatives (1% FPR threshold = 99th percentile of 100-200 negatives, so the threshold itself is noisy; the Wilson intervals cover only the positives). FPR at z = 4 on attacked human text: 0 or 1 text per attack (upper 95% bound 0.019-0.037).

**Reading** (`figures/ext2a_removal_frontier.png`: TPR at 1% FPR; `figures/ext2a_removal_frontier_z4.png`: TPR at z = 4; both with Wilson error bars).
- *At the fixed threshold z = 4*, the paraphraser and the paper-faithful single-word T5 attack are indistinguishable at matched meaning: light paraphrase 0.22 [0.17, 0.28] at similarity 0.850 vs single-word T5 ε = 0.3 0.24 [0.17, 0.33] at 0.856 (both intervals overlap, and so do the similarities). Medium/full paraphrases (0.035 / 0.015) remove more at a similar similarity (0.85 / 0.84).
- *At a calibrated 1% FPR*, which is what a deployed detector would use, the paraphraser is clearly stronger: at matched meaning (≈0.85), light paraphrase leaves 0.435 [0.37, 0.50] detected vs 0.81 [0.72, 0.88] for single-word T5 at ε = 0.3 (intervals far apart). To bring TPR at 1% FPR down to ≈0.1, T5 needs ε = 0.5 and keeps 0.774 [0.757, 0.791] of the meaning, a full paraphrase keeps 0.837 [0.822, 0.852] (intervals do not overlap). The reason: T5-attacked texts keep many z-scores between the empirical threshold (≈2.4) and 4, the paraphrased ones do not.
- The multi-word T5 variant is the weakest attack on both measures.
- So the paper's claim survives a 2026 attacker in its form (removing the watermark still costs meaning: about 15% of MiniLM similarity for near-complete removal), but the modern attack is both cheaper (one prompt to a free 1.5B model instead of 60-140 sequential T5 calls) and, at a calibrated threshold, more meaning-efficient. An instruction to "lightly edit" already rewrites 65% of the tokens.
- The paraphrases cost fluency as judged by OPT-2.7B given the original prompt (PPL 21-30, vs 16.0 / 24.8 for single-word T5 at ε = 0.3 / 0.5): the paraphraser drops the continuation's link to the prompt (texts often restart a sentence), which the oracle penalizes.

### 2b. Copy-paste dilution (`results/summary/ext2b_dilution.md`, `results/summary/ext2b_thresholds.json`, `figures/ext2b_dilution.png`)
600-token documents of human C4 text (baselines of prompts 1,000+) with one watermarked span (γ = 0.5, δ = 2) at a random position; 200 documents per share; 1% FPR thresholds from 1,000 human-only documents (full-text z > 2.20, WinMax > 4.02).

| watermarked share | full-text z, TPR at 1% FPR [95% CI] | WinMax, TPR at 1% FPR [95% CI] |
|---|---|---|
| 10% (60 tokens) | 0.115 [0.078, 0.167] | 0.865 [0.811, 0.906] |
| 25% (150 tokens) | 0.815 [0.755, 0.863] | 0.995 [0.972, 0.999] |
| 50% (300 tokens) | 1.000 [0.981, 1.000] | 1.000 [0.981, 1.000] |

The WinMax advantage at 10% and 25% is far outside the intervals.

The sliding-window maximum (WinMax, from the authors' follow-up: Kirchenbauer et al., *On the Reliability of Watermarks for Large Language Models*, ICLR 2024) answers the paper's open question for spans of ~60+ tokens; the plain full-text test does not.

## Deviations
- **Green lists** use a CPU `torch.Generator` (as the reference `watermark_processor.py` does when run on CPU). The paper's runs used a CUDA generator, so the concrete green lists differ from the paper's; the scheme and its statistics are identical. The legacy `experiments/watermark.py` took green = complement of a "blacklist" of size `bl_proportion·V`; we follow the root reference `randperm[:int(γV)]` (validated bit-for-bit, `results/summary/greenlist_equivalence.txt`).
- **Prompts**: last 200 tokens = human baseline; prompt = the rest (BOS + last 1847 tokens if longer). The paper's script truncated each document to its first 2048 tokens before splitting; this only matters for documents longer than 2048 tokens.
- **Length filter / sample size**: the paper's script kept drawing prompts until 500 rows passed its length check. We do the same for the multinomial news reproduction configs: `n` = rows with T in 195..205, prompts drawn in order from the first 1,000 (prompts 1,000+ are reserved as human text for Ext 2b). The paper required both the watermarked and the unwatermarked output of a prompt to pass; we filter each config on its own length (`results/summary/kept_counts.md`). The headline config ran out of prompts at 497 kept rows (of 1,000 generated).
- **Greedy decoding on news** suppresses end-of-text like beam search, so every output has 200 tokens. QA greedy (Table 10 style) keeps EOS.
- **Perplexity** is computed on token ids (OPT-1.3B and OPT-2.7B share the tokenizer) instead of re-tokenized text; attacked texts are re-tokenized with OPT first. When prompt + attacked text exceeds 2,048 positions (texts lengthened by the attack), the prompt is cut from the left.
- **Spike entropy for greedy QA** uses τ = 1 (no temperature), so `S_sampled = S_paper` there.
- **T5 attack**: run in bfloat16 on MPS (fp32 was 2x slower; same first replacements and same success count on a 32-text check, `results/summary/benchmark.json`). Words are `\w+` runs (punctuation kept). Windows are ±60 T5 tokens padded to a fixed length (no MPS recompiles); 50-beam search with 20 returns, at most 10 new tokens, each beam stops at `<extra_id_1>`; a candidate must be a complete span. The multi-word run accepts multi-word fills; the single-word run only accepts a candidate that is one `\w+` word. The ε levels are nested: one replacement process per text, snapshotted the first time it reaches ε·T successes or 3·ε·T attempts (marginally identical to independent runs per ε, but correlated across ε).
- **Qwen news prompts** (Ext 1c): the last ~150 words of the C4 prompt for both the base and the instruct model, so the two models see the same content.
- **Paraphraser sampling**: pure temperature sampling, τ = 0.7 with `top_k=0, top_p=1, repetition_penalty=1`, not Qwen's chat defaults (top_p 0.8, top_k 20). Three paraphrases were near-empty; their z is scored as 0.
- **Empty or short texts**: rows with T = 0 are dropped from z statistics (2 of 1,000 in the headline config).
- **Sample sizes**: 8-beam rows use N = 150 at δ = 2 and N = 100 elsewhere; Fig. 2 grid points use N = 100 kept rows where no larger run exists.
- **Not run**: the optional SynthID-Text experiment, the multi-word T5 attack at ε = 0.5 / 0.7 and the T5 attack on 8-beam texts.

## Limitations
- **One weaker Table 8 row**: multinomial δ = 1, γ = 0.5 detects less than in the paper (z = 5: 0.367 [0.314, 0.423] vs 0.504; intervals do not overlap). Plausibly key-specific (next point); not tested with other keys.
- **T5 attack**: implemented from the paper's description (no attack code was released). With the single-word rule it reproduces Table 9 within our intervals from ε = 0.3 at z = 5; at z = 4 our attack is slightly stronger (ε = 0.5: 0.020 vs 0.094) and at ε = 0.1 our watermark is more robust (0.990 vs 0.819). Letting T5 fill multi-word spans makes the attack much weaker because the text grows.
- **Noisy 1%-FPR thresholds in 2a**: each threshold is the 99th percentile of 100-200 attacked human texts, and the Wilson intervals cover only the positives; the AUC column, which uses every negative, supports the main comparison: at matched meaning (≈0.85) the light paraphrase has AUC 0.881 against 0.995 for single-word T5 at ε = 0.3.
- **Null calibration is key-specific** (`results/summary/null_mean_by_key.json`): for our key the mean z of unwatermarked text is −0.20 (human) / −0.37 (δ = 0) at γ = 0.5; across 6 keys it ranges from −0.52 to +0.71, with sd 1.03-1.36 (repeated pairs). The FPR at z = 4 is still 0 on human text. So z is N(0, 1) over random keys but only approximately for a fixed key; thresholds for deployment should be calibrated empirically, as we do for every 1%-FPR number.

## Compute
Apple M5 Pro, 24 GB unified memory, `mps`, fp16 (T5-Large in bf16). Measured before the runs (`results/summary/benchmark.json`), seconds per sequence: OPT-1.3B multinomial 0.59 (batch 16) to 0.84 (batch 8); 4-beam 7.1; 8-beam 9.2 (batch 4, typical prompts); oracle PPL 0.31; Qwen generation 0.52-0.53; Qwen paraphrase 0.63; one T5 round over 32 texts 9.5 s in bf16 (19 s fp32).

Approximate wall-clock per step of `run.py` (from the run logs): `headline` 25 min (4-beam 15 min); `table8` 1.5 h (the two 8-beam rows ≈ 1 h); `fig2_grid` 35 min; `fig2_beams` 2 h; `fig3c_beams` 3 h; `ext1b` 3 min; `ext1c` 8 min; `ext1c_qa` 3 min; `keys` 4 min; `para` 10 min; `t5` 2.1 h; `t5w` 6.8 h (only ≈60% of masks yield a one-word candidate); `score`, `dilution` 1 min each; `ppl` about 1 h in total; `ppl_qwen` under 1 min. Total ≈ 20 hours, dominated by the T5 attacks (≈9 h) and 8-beam search (≈6 h). Beam search and the perplexity pass need `torch.mps.empty_cache()` after every batch and the token budgets in `params.py` to stay out of swap on 24 GB.
