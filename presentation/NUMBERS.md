# Numbers on the slides and where they come from

Slide = frame number printed in the footer (backup slides: B-number and frame number).
Paths are relative to the repository root; raw files are `.jsonl` in the working project and `.jsonl.gz` in the public release. "Formula" means computed by hand from the formula shown on the slide; "illustrative" means a made-up input used only to explain the mechanism.
Rounding is to the precision shown on the slide.

## Main deck

| Slide | Number | Source |
|---|---|---|
| 1 | ICML 2023 | the paper (`paper/kirchenbauer23a.pdf`) |
| 1 | October 2026 | talk date (brief) |
| 2 | A (my email) and B (AI email) | B: Qwen2.5-1.5B-Instruct with our watermark (gamma 0.25, delta 2, temperature 0.7, top-k 0), given the same points as A, sample 44 of 64 (`presentation/snippets/emails.json`, made by `make_figures.py` `generate_emails()`); A: `EMAIL_HUMAN` in `make_figures.py`; side drawn at random |
| 2 (note) | 64 samples | `snippets/emails.tex` (`\emailSamples`) |
| 3 | 26% AI text caught, 9% human text flagged | OpenAI blog, "New AI classifier for indicating AI-written text" (31 Jan 2023, updated 20 Jul 2023) |
| 3 | January 2023, July 2023 | same OpenAI blog post |
| 3 (note) | about one honest email in eleven | 1 / 0.09 = 11.1 |
| 4 | five properties | the paper, Sec. 1 |
| 4 | 26%, 9% (2023) | OpenAI blog, as on slide 3 |
| 4 | 98.4% | the paper, Sec. 4.1 (z = 4, T = 200, gamma 0.5, delta 2) |
| 4 | about 0.003% (3 in 100,000) | formula: one-sided p-value at z = 4, 1 - Phi(4) = 3.2e-5 |
| 5 | a quarter of the vocabulary is green; a human lands on green about 1 in 4 | gamma = 0.25 |
| 5 | A: 16 green of 55 tokens, chance gives 14 (z = 0.7, note) | `snippets/emails.tex`: detector (`core.detect`) on EMAIL_HUMAN, Qwen tokenizer, our key, gamma 0.25; chance = 0.25 x 55 = 13.75 |
| 5 | B: 30 green of 53 tokens, chance gives 13, z = 5.3 | `snippets/emails.tex`: detector on sample 44; chance = 0.25 x 53 = 13.25 |
| 5 | by chance: 1 in 19 million | one-sided p-value of z = 5.31 (`make_figures.py` `odds()`) |
| 5 (note) | 44 of 64 samples above z = 4, median z 4.6 | `snippets/emails.tex` (`\emailSamplesAbove`, `\emailSamplesMedian`), from `snippets/emails.json` |
| 5 | August 2026 (Anthropic watermarks Claude) | the brief (Anthropic's public statement); not checked against a primary source here |
| 6 | 50,265 tokens | OPT vocabulary size, `results/summary/greenlist_equivalence.txt` (V = 50265) |
| 6 (note) | 15,485,863; h = 1 | `core.py` / official `watermark_processor.py` (simple_1 seeding), RESULTS.md Deviations |
| 6 (figure) | 0.167 -> 0.333 (green); 0.167 -> 0, banned (red); 50% -> 100% | `make_figures.py` `ban_example("dinner")`: illustrative inputs (same as the push example), red set to 0, green renormalised |
| 7 | probability 1/2 | Algorithm 1 with gamma = 0.5 |
| 7 | z > 4; p-value 3.2 x 10^-5 | formula, 1 - Phi(4) |
| 7 | 16 tokens reach z = 4 | formula: all-green text, z = sqrt(T) = 4 at T = 16 |
| 7 | 200 of 1,000 tokens; at most 400 red; 600 - 500; z = 6.3 | the paper, Sec. 2 example; z = 2 (600 - 500)/sqrt(1000) = 6.32 |
| 7 (figure) | 1,000 human texts; N(0, 1); none above 4; flag z > 4, p = 3.2e-5; attacker z = 6.3 | `make_figures.py` `z_null()`: stored `z` of `results/raw/human/c4` (idx < 1000, gamma 0.5, our key: mean -0.20, sd 1.04, 0 above 4); p-value and 6.3 as on this slide |
| 9 (figure) | Obama 0.990 -> 0, banned; Hussein 0.004 -> 0.571; and 0.003 -> 0.429; was 0.003 -> 0; 0.7% -> 100% | `make_figures.py` `ban_example("barack")`: illustrative inputs (same as the push example) |
| 9 | about 99% on "Obama"; the ban forces "Hussein" or "and" | illustrative (same inputs as the push example, slide 12) |
| 11 | x 7.4 at delta = 2 | formula: e^2 = 7.39 |
| 11 | gamma 0.25 or 0.5; delta 2 | the paper's settings (Sec. 4, Table 2) |
| 12 | 0.990 -> 0.948; 0.004 -> 0.028; 0.003 -> 0.021; 0.003 -> 0.003 | `make_figures.py` `push_example()`: illustrative inputs, watermarked values computed with the slide-10 formula at delta = 2 |
| 12 | 0.167 -> 0.294 (green); 0.167 -> 0.040 (red) | same, six words at 1/6 each |
| 12 | chance of a green token 0.7% -> 5%; 50% -> 88% | same |
| 13 | T = 200, gamma = 0.5, delta = 2 | the paper's main setting (Sec. 4.1) |
| 13 | 100 +/- 7 green tokens | formula: gamma T = 100, sd = sqrt(200 x 0.25) = 7.07 |
| 13 | z = 4 at about 128 green tokens | formula: 100 + 4 x 7.07 = 128.3 |
| 13 | 159.5; z = 8.4 | the paper, Sec. 4.1 (observed mean); z = 59.5/7.07 = 8.4 |
| 14 | 0.88 | formula: gamma alpha/(1 + (alpha - 1) gamma) at gamma 0.5, alpha = e^2: 0.881 |
| 14 | 0.807; 142.2; 159.5; >= 98.6%; 98.4% | the paper, Sec. 4.1 |
| 14 (note) | 100% above the 25th entropy percentile | the paper, Sec. 4.1 (375/500); ours 373/497 in `results/summary/headline.csv` |
| 16 | T5 (2020) | Raffel et al. 2020, the model the paper's attack uses |
| 17 | last 200 tokens; OPT-1.3B; tau = 0.7; T = 200 +/- 5; OPT-2.7B | the paper (Sec. 6) and RESULTS.md Deviations |
| 17 | about 500 texts per setting | the paper's Table 8 counts (495 to 507) |
| 17 | 100 to 500 per setting; 497 for the headline; 1,000 human texts | `results/summary/table8.csv` (`count`), `kept_counts.csv`, `ppl_baselines.csv` |
| 17 | about 20 hours | RESULTS.md compute log (Oct 4, 02:15 to 22:06, with pauses) |
| 17 | 1,000 random contexts; 200 for Qwen | `results/summary/greenlist_equivalence.txt` |
| 18 | detected at z = 4: 98.4% / 98.6% [97.1, 99.3] | paper Sec. 4.1 / `results/summary/headline.csv` (TPR 0.9859 [0.971, 0.993]) |
| 18 | human texts flagged: 0 / 0 of 1,000 | paper Table 8 (FPR 0.0) / `headline.csv` (FPR 0 [0, 0.004]) |
| 18 | mean green tokens 159.5 / 158.3 | paper / `headline.csv` (158.34 [157.5, 159.2]) |
| 18 | Theorem 4.2 bound 142.2 / 143.2 | paper / `headline.csv` (143.20) |
| 18 | 4-beam search 99.6% / 100% | paper / `headline.csv` (1.0 [0.975, 1.0], n = 150) |
| 18 | 497 texts, 150 with 4 beams, 1,000 human texts | `headline.csv`, `kept_counts.csv` |
| 18 (figure) | human 1,000; no watermark 500; watermarked 497; z = 4 line | `make_figures.py` `z_hist()`: `results/raw/human/c4` (idx < 1000), `results/raw/gen/opt-news-m-d0` and `opt-news-m-d2-g0.5` (kept rows, first 500) |
| 19 | 11 of 12 settings within our 95% intervals | `results/summary/table8.csv` (`vs_paper_z4`, `vs_paper_z5`: 11 rows consistent at both thresholds) |
| 19 | consistent from 30% of words at z = 5 | `results/summary/table9_t5.csv`, single-word rows (`vs_paper_z5`: 0.3, 0.5, 0.7 consistent) |
| 19 (figures) | all plotted TPRs, CIs and paper markers | `table8.csv` (`TPR_z4`, `TPR_z5`, `*_lo`, `*_hi`, `paper_TPR_*`), `table9_t5.csv` (single-word rows) |
| 20 | delta = 1, gamma = 0.5: 0.37 [0.31, 0.42] vs 0.50 | `table8.csv` (TPR_z5 0.367 [0.314, 0.423], paper 0.504) |
| 20 | 10%: 0.99 vs 0.82; from 30% on | `table9_t5.csv` (eps 0.1: TPR_z4 0.99, paper 0.819) |
| 20 | +0.1 to +0.3 under multinomial sampling; one setting +1.7 | `table8.csv` (`PPL_mean` vs `paper_PPL`: +0.06 to +0.33; delta 5, gamma 0.25: 12.39 vs 10.7) |
| 20 | exact match 0.202 to 0.212; paper about -0.04 | `figures/ext1b_triviaqa.md` (qwen-it 0.202 -> 0.212; FLAN-UL2 0.374 -> 0.336, BLOOMZ 0.296 -> 0.259) |
| 21 | news 24; code 188; 49% never in 200 tokens; short answers never | `results/summary/ext1b_cells.csv`, OPT rows (`tokens_to_z4_median`, `never_z4` = 0.494) |
| 21 (figure) | 24 [20, 27], never 1%; 188 [135, never], never 49%; short answers never (99%) | `ext1b_cells.csv` (`tokens_med_lo/hi`, `never_z4` = 0.01, 0.494, 0.99) |
| 21 | 200, 164, 500 texts; gamma 0.25 / 0.5; delta 2 | `ext1b_cells.csv` (`n`, `gamma`, `delta`) |
| 22 | same meaning kept 0.85 | `results/summary/ext2a_attacks.csv` (`meaning_kept`: t5w 0.3 = 0.856, para light = 0.850) |
| 22 | 81% vs 43.5% detected at 1% false positives | `ext2a_attacks.csv` (`TPR_1pctFPR`: t5w 0.3 = 0.81, para light = 0.435) |
| 22 | AUC 0.995 vs 0.881 | `ext2a_attacks.csv` (`AUC`: 0.99495, 0.8806) |
| 22 | model calls 60 to 140 vs 1 | RESULTS.md 2a ("60-140 sequential T5 calls"); one call per attempted swap, `table9_t5.csv` words_replaced 60.0 at 30% |
| 22 (figure) | meaning kept, TPR at 1% FPR and their CIs for every attack | `ext2a_attacks.csv` (`meaning_kept`, `meaning_lo/hi`, `TPR_1pctFPR`, `TPR_1pct_lo/hi`) |
| 22 (note) | 0.84 kept by a full rewrite; 0.22 vs 0.24 at z = 4 | `ext2a_attacks.csv` (para full 0.837; `TPR_z4` 0.22, 0.24) |
| 23 | 60-token passage, 600-token text | RESULTS.md 2b; `ext2b_dilution.csv` (share 0.1) |
| 23 | 11.5% vs 86.5% at 1% false alarms | `results/summary/ext2b_dilution.csv` (`TPR_full_z` 0.115, `TPR_winmax` 0.865) |
| 23 | ICLR 2024 | Kirchenbauer et al., On the Reliability of Watermarks for Large Language Models |
| 23 (note) | z > 2.20, WinMax > 4.02, 1,000 documents | `results/summary/ext2b_thresholds.json` |
| 24 | about 2.3 times more likely | `results/summary/eos_after_dot.json` (15485863 `eos_prob_multiplier` 2.29) |
| 24 | 71% of natural endings after a period | `results/summary/eos_end_contexts.json` (156 of 219), RESULTS.md |
| 24 | 16% vs 6% stop before 50 tokens; 1,000 and 711 texts | `results/summary/kept_counts.csv` (d2-g0.5 `ended_before_50` 0.157 [0.136, 0.181], generated 1000; d0 0.055 [0.040, 0.074], generated 711) |
| 24 (figure) | 0.53, 0.80, 0.76, 0.87, 0.75 reach 195 tokens | `results/summary/keys_eos.csv` (`kept_share`, `kept_lo/hi`) |
| 25 | 98.6% vs 98.4%; 11 of 12; 86.5% vs 11.5% | as on slides 17, 18, 22 |

## Backup

| Slide | Number | Source |
|---|---|---|
| B1 (27) | zeta* = 0.76, c = 0.881 | formula at gamma 0.5, alpha = e^2 |
| B1 | spread 0.807 / 0.813; bound 142.2 / 143.2; mean 159.5 / 158.3; sigma bound 6.41 / 6.38; observed sigma 9.85; predicted detection 98.6% / 99.0% | paper Sec. 4.1 / `results/summary/headline.csv`, RESULTS.md Sec. 4.1 |
| B1 (note) | 98.5% with the paper's bounds | `headline.csv`, RESULTS.md |
| B2 (28) | -0.52 to +0.71 (unwatermarked model text); -0.36 to +0.55 (human); sd 1.03 to 1.36; six keys | `results/summary/null_mean_by_key.json` (delta 0 means -0.521 to 0.709, human means -0.362 to 0.554, sd 1.03 to 1.363) |
| B2 | 0 of 1,000 false alarms at z = 4 | `headline.csv`, `calibration_negatives.md` |
| B2 | 16 tokens or fewer | QA generations, up to 16 new tokens (`ext1b_cells.csv` `T_mean` 16.0) |
| B2 | 0.005; 97.2% [95.3, 98.3]; 98.6% | `headline.csv` (re-encode drop 0.0048; norep TPR 0.9718 [0.953, 0.983]) |
| B3 (29) | delta 2: z 10.8 at gamma 0.1 vs 8.3 at gamma 0.5; perplexity 6.3 vs 6.5 | `results/summary/fig2_tradeoff.csv` (10.77 / 6.26; 8.25 / 6.53) |
| B3 | 8 beams z = 11.5 at perplexity 1.63; 1.44 without watermark | `fig2_tradeoff.csv` (11.52, 1.629); `ppl_baselines.csv` (1.442) |
| B3 (note) | 4 beams z 11.2, perplexity 1.74; 100 / 150 / 300-500 texts per point | `fig2_tradeoff.csv` (`n`) |
| B4 (30) | 4 / 8 / 22 / 60 / 176; 124 / 30 / 8 / 6; 16; z > 5 from 24; paper about 35 | `results/summary/fig3_z_vs_T.csv` (first T with z_mean > 4 or > 5); paper Sec. 7 |
| B5 (31) | all 12 rows: n, TPR at z = 4 and 5 with CIs, paper values, verdicts | `results/summary/table8.csv` |
| B5 | FPR 0 [0, 0.004] on 1,000 human texts; 0.002 on 500 unwatermarked texts | `table8.csv` (`FPR_z4`, `FPR_z4_hi`, `FPR_z4_delta0`) |
| B6 (32) | all AUC and perplexity values, paper values | `table8.csv` (`AUC`, `paper_AUC`, `PPL_mean`, `paper_PPL`) |
| B6 | no watermark 5.29 (5.1), 1.44 (1.2); human 11.06 | `results/summary/ppl_baselines.csv` |
| B6 | +0.1 to +0.3; +0.3 to +0.5 (8 beams); +1.7 | `table8.csv` differences, RESULTS.md Fig. 4 section |
| B7 (33) | temperature 0.7 | RESULTS.md (processor-order check) |
| B7 | 22 runs; 0.675 vs 0.716; observed 0.792 | `results/summary/ext1a_bounds.csv` (gamma 0.5, delta 2 row), `fig7_theory.csv` |
| B7 (figure) | observed mean and IQR, both bounds, paper markers 0.53 / 0.60 / 0.68 / 0.80 / 0.94 / 0.99 | `fig7_theory.csv`; paper markers read off the paper's Fig. 7 (RESULTS.md Fig. 7) |
| B7 (note) | 0.175 vs 0.172; 0.714 vs 0.813 | `ext1a_bounds.csv`, RESULTS.md 1a |
| B8 (34) | all rows: TPR with CIs and verdicts, AUC, perplexity, tokens after, words replaced, paper values | `results/summary/table9_t5.csv` (rows "m-nom, single-word T5") |
| B8 | 100 texts; 1,000 human texts | `table9_t5.csv` (`count`), RESULTS.md Table 9 |
| B8 | multi-word: 200 texts, 295 tokens at 30%, 0.495 vs 0.24 | RESULTS.md Table 9 multi-word, `ext2a_attacks.csv` (t5 0.3: T_mean 294.7, TPR_z4 0.495) |
| B8 (note) | green fraction 0.79 -> 0.62; perplexity 9.9 vs 16.0 | `ext2a_attacks.csv` (`green_frac`, `PPL_mean`) |
| B9 (35) | EM and mean z, all five models | `figures/ext1b_triviaqa.md` |
| B9 | 500 questions; 16 new tokens; gamma 0.5, delta 2 | `figures/ext1b_triviaqa.md` header |
| B9 | 0.202 [0.169, 0.239] -> 0.212 [0.178, 0.250]; 5.8 tokens | RESULTS.md Table 10; `ext1b_cells.csv` (qa qwen-it `T_mean` 5.77) |
| B10 (36) | hit: spread 0.875, z = 9.62; miss: spread 0.571, z = 0.57 | `results/summary/examples.csv` (idx 699 and 562, `S`, `z`) |
| B11 (37) | z 4.01 / 3.13; 4.08 / 2.94; 4.38 / 0.40; out of 500 | `results/summary/delta0_false_positives.md` (idx 56, 208, 417), `calibration_negatives.md` (n 500) |
| B11 | human 0 of 1,000; mean z -0.25 to +0.09; sd 1.04 to 1.10; unwatermarked sd 1.27 to 1.35 | `results/summary/calibration_negatives.md` |
| B12 (38) | 0.70 vs 0.77; 0.37 [0.31, 0.42] vs 0.50 | `table8.csv` |
| B12 | T5 verdicts; 0.02 vs 0.094 | `table9_t5.csv` |
| B12 | 12.4 vs 10.7 | `table8.csv` |
| B12 | about -0.04 | `figures/ext1b_triviaqa.md` (paper rows) |
| B12 | 158.3 [157.5, 159.2] vs 159.5 | `headline.csv` |
| B12 | 2,048 tokens, last 1,847; 497 of 1,000; +/-60-token windows; temperature 0.7, top-k 0, top-p 1; 150 words | RESULTS.md Deviations; `kept_counts.csv` |
| B13 (39) | every cell: n, entropy, TPR at 1% FPR (both), median tokens with CI, never | `results/summary/ext1b_cells.csv` (`n`, `H_nowm`, `TPR_1pctFPR`, `TPR_1pctFPR_norep`, `tokens_to_z4_median`, `tokens_med_lo/hi`, `never_z4`) |
| B13 | gamma 0.25 / 0.5, delta 2, tau 0.7 | `ext1b_cells.csv`, RESULTS.md 1b |
| B14 (40) | -2.88 / -0.45 | `ext1b_cells.csv` (code opt `z_mean_nowm`, `z_norep_mean_nowm`) |
| B14 | OPT 0.52 [0.45, 0.60] -> 0.62 [0.54, 0.69]; Qwen 0.24 [0.19, 0.32] -> 0.51 [0.44, 0.59] | `ext1b_cells.csv` (`TPR_1pctFPR`, `TPR_1pctFPR_norep`, `TPR_1pct_lo/hi`, `TPR_1pct_norep_lo/hi`) |
| B15 (41) | every cell: observed, predicted (both), share of texts below (both) | `ext1b_cells.csv` (`green_frac`, `pred_paper`, `pred_corrected`, `seq_below_paper`, `seq_below_corrected`) |
| B15 | 0.584 vs 0.453 | `ext1b_cells.csv`, news opt |
| B16 (42) | entropy 2.52 -> 1.96; 0.59 -> 0.43; 1.74 -> 1.18 | `ext1b_cells.csv` (`H_nowm`, qwen vs qwen-it) |
| B16 | 99.5% vs 99.0% | `ext1b_cells.csv` (`TPR_1pctFPR` news qwen 0.995, qwen-it 0.990) |
| B16 | 1.96 -> 2.30 under the watermark | `ext1b_cells.csv` (news qwen-it `H_nowm` 1.963, `H` 2.305) |
| B16 | perplexity 5.7 -> 9.5; OPT 5.2 -> 6.9 | `ext1b_cells.csv` (`PPL_mean_nowm`, `PPL_mean`: 5.673 / 9.548; 5.204 / 6.888) |
| B16 | 27 -> 8 tokens; perplexity 25.6 | `ext1b_cells.csv` (news qwen-it delta 2 vs delta 4; `PPL_mean` 25.56) |
| B16 (figure) | means 2.52 and 1.96 | per-token `H` of `results/raw/gen/qwen-news-m-d0`, `qwen-it-news-m-d0` (legend uses `ext1b_cells.csv` `H_nowm`) |
| B17 (43) | every attack: meaning kept, tokens changed, green fraction, TPR at z = 4, TPR at 1% FPR, AUC, perplexity | `results/summary/ext2a_attacks.csv` |
| B17 | 65% of tokens | `ext2a_attacks.csv` (para light `tokens_changed` 0.650) |
| B18 (44) | 0.22 [0.17, 0.28] vs 0.24 [0.17, 0.33] | `ext2a_attacks.csv` (`TPR_z4`, `TPR_z4_lo/hi`) |
| B18 | calibrated threshold about 2.4 | `ext2a_attacks.csv` (`thr_1pctFPR`: t5w 0.3 2.38, para light 2.44) |
| B19 (45) | all shares: TPR with CIs, mean z, mean WinMax | `results/summary/ext2b_dilution.csv` |
| B19 | 200 documents per share | `ext2b_dilution.csv` (`n`) |
| B19 | z > 2.20, WinMax > 4.02, human means -0.37 and 3.03 | `results/summary/ext2b_thresholds.json` |
| B19 | at z = 4: 1%, 28.5%, 97.5% | `ext2b_dilution.csv` (`TPR_full_z_at_4`) |
| B20 (46) | EOS green, multipliers 2.29 / 0.17 / 1.18 / 0.32, kept shares, stops before 50 with CIs | `results/summary/keys_eos.csv`, `eos_after_dot.json`, RESULTS.md key table |
| B20 | 16% [14, 18] vs 6% [4, 7]; 1,000 and 711 | `kept_counts.csv` |
| B20 | 29% to 37% at delta >= 5, gamma 0.5 to 0.75 | `kept_counts.csv` (d5/d10 x g0.5/g0.75: 0.294, 0.310, 0.367, 0.360) |
| B20 | gamma 0.25: 2.8% (delta 2), 2.3% (delta 5) | `kept_counts.csv` (d2-g0.25 0.028, d5-g0.25 0.023) |
| B20 | "stop" red after "." at gamma 0.25 | `results/summary/eos_green_status.json` ("'.'" green_at 0.25: false) |
| B20 (note) | 82% of competing mass green (key 32452843) | `eos_after_dot.json` (`green_mass_of_other_continuations` 0.815) |
| B21 (47) | 0.59 to 0.84; 7.1; 9.2; 0.31; 0.52 to 0.53; 0.63 | `results/summary/benchmark_opt.json`, `benchmark.json` |
| B21 | T5 round of 32 texts 9.5 s | `results/summary/benchmark_opt_log.txt` (bf16) |
| B21 | about 20 hours; 24 GB; transformers 5.18 | RESULTS.md (environment, compute log) |
