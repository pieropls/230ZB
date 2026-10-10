# Numbers on the slides and where they come from

Slide = frame number printed in the footer (backup slides: B-number and frame number).
Paths are relative to the repository root; raw files are `.jsonl` in the working project and `.jsonl.gz` in the public release. "Formula" means computed by hand from the formula shown on the slide; "illustrative" means a made-up input used only to explain the mechanism.
Rounding is to the precision shown on the slide.

## Main deck

| Slide | Number | Source |
|---|---|---|
| 1 | ICML 2023 | the paper (`paper/kirchenbauer23a.pdf`) |
| 1 | October 2026 | talk date |
| 2 | A (my email) and B (AI email) | B: Qwen2.5-1.5B-Instruct with our watermark (gamma 0.5, delta 2, temperature 0.7, top-k 0), given the same points as A, sample 83 of 128 (`presentation/snippets/emails.json`, made by `make_figures.py` `generate_emails()`); A: `EMAIL_HUMAN` in `make_figures.py`; side drawn at random |
| 2 (note) | 128 samples | `snippets/emails.tex` (`\emailSamples`) |
| 3 | 26% AI text caught, 9% human text flagged | OpenAI blog, "New AI classifier for indicating AI-written text" (31 Jan 2023, updated 20 Jul 2023) |
| 3 | January 2023, July 2023 | same OpenAI blog post |
| 3 (note) | about one honest email in eleven | 1 / 0.09 = 11.1 |
| 4 | five properties | the paper, Sec. 1 |
| 4 | 26%, 9% (2023) | OpenAI blog, as on slide 3 |
| 4 | 98.4% | the paper, Sec. 4.1 (z = 4, T = 200, gamma 0.5, delta 2) |
| 4 | about 0.003% (3 in 100,000) | formula: one-sided p-value at z = 4, 1 - Phi(4) = 3.2e-5 |
| 5 | half the vocabulary is green; a human lands on green about 1 in 2 | gamma = 0.5 |
| 5 | A: 32 green of 55 tokens, chance gives 28 (z = 1.2, note) | `snippets/emails.tex`: detector (`core.detect`) on EMAIL_HUMAN, Qwen tokenizer, our key, gamma 0.5; chance = 0.5 x 55 = 27.5 |
| 5 | B: 50 green of 62 tokens, chance gives 31, z = 4.8 | `snippets/emails.tex`: detector on sample 83; chance = 0.5 x 62 = 31 |
| 5 | by chance: 1 in 1.4 million | one-sided p-value of z = 4.83 (`make_figures.py` `odds()`) |
| 5 (note) | 76 of 128 samples above z = 4, median z 4.3 | `snippets/emails.tex` (`\emailSamplesAbove`, `\emailSamplesMedian`), from `snippets/emails.json` |
| 5 | August 2026, Anthropic and Claude's watermark (a version of SynthID-Text) | anthropic.com/news/claude-text-watermark (14 Aug 2026). It announces that future Claude models will watermark their text with a version of SynthID-Text, and older models "over the coming months". The slide's "Since August 2026, Anthropic watermarks Claude's text" overstates this |
| 6 | integers from 0 to 50,264; 50,265 tokens | OPT vocabulary size, `results/summary/greenlist_equivalence.txt` (V = 50265) |
| 6 (note) | 15,485,863; h = 1 | `core.py` / official `watermark_processor.py` (simple_1 seeding), RESULTS.md Deviations |
| 6 (note) | green list of 25,132 ids | formula: int(0.5 x 50,265) |
| 6 (figure) | 0.167 -> 0.333 (green); 0.167 -> 0, banned (red); 50% -> 100% | `make_figures.py` `ban_example("dinner")`: illustrative inputs (same as the push example), red set to 0, green renormalised |
| 7 | green with probability 1/2 | Algorithm 1 with gamma = 0.5 |
| 7 | z > 4; probability 3.2 x 10^-5, about 3 human texts in 100,000 | formula, 1 - Phi(4) = 3.17e-5 |
| 7 | our 1,000 human texts: none above 4 | `results/summary/headline.csv` (FPR 0 of 1,000); `make_figures.py` `z_null()` |
| 7 | 16 tokens are enough | formula: all-green text, z = sqrt(T) = 4 at T = 16 (z = 4 exactly; one more green token passes z > 4) |
| 7 | one edit turns at most 2 tokens red; 200 edits on 1,000 tokens; z = 6.3 | the paper, Sec. 2 example; z = (600 - 500)/(sqrt(1000)/2) = 6.32 |
| 7 (note) | about a quarter of the text to get below 4 | formula: at T = 1,000, z < 4 needs fewer than 563 green, so more than 437 red, so at least 219 edits |
| 7 (figure) | 1,000 human texts; N(0, 1); none above 4; flag z > 4, p = 3.2e-5; attacker z = 6.3 | `make_figures.py` `z_null()`: stored `z` of `results/raw/human/c4` (idx < 1000, gamma 0.5, our key: mean -0.20, sd 1.04, 0 above 4); p-value and 6.3 as on this slide |
| 8 | about 99% on "Obama"; the ban forces "Hussein" or "and" | illustrative (same inputs as the push example, slide 10) |
| 8 (figure) | Obama 0.990 -> 0, banned; Hussein 0.004 -> 0.571; and 0.003 -> 0.429; was 0.003 -> 0; 0.7% -> 100% | `make_figures.py` `ban_example("barack")`: illustrative inputs (same as the push example) |
| 10 | 0.990 -> 0.948; 0.004 -> 0.028; 0.003 -> 0.021; 0.003 -> 0.003 | `make_figures.py` `push_example()`: illustrative inputs, watermarked values computed with the slide-9 formula at delta = 2 |
| 10 | 0.167 -> 0.294 (green); 0.167 -> 0.040 (red) | same, six words at 1/6 each |
| 10 | chance of a green token 0.7% -> 5%; 50% -> 88% (note: p_G = 0.007 and 0.5) | same |
| 11 | Berry-Esseen error about 4 times larger at gamma = 0.05 than at gamma = 1/2 | formula: (gamma^2 + (1 - gamma)^2)/sqrt(gamma (1 - gamma)) = 4.15 at gamma 0.05, 1 at gamma 0.5 |
| 11 | exact false-alarm rate at Z > 4, T = 200: 2.5e-5 at gamma = 1/2, 1.9e-4 at gamma = 0.05 | formula: exact binomial tail (scipy `binom.sf`): P(Bin(200, 0.5) >= 129) = 2.49e-5; P(Bin(200, 0.05) >= 23) = 1.90e-4 |
| 11 | six times the Gaussian value | 1.9e-4 / 3.2e-5 = 6.0 |
| 11 (note) | about 17 times more tokens; 7.6e-4 at gamma 0.05, T = 50 | formula: 4.15^2 = 17.2; P(Bin(50, 0.05) >= 9) = 7.56e-4 |
| 11 (note) | q = 0.80 in the paper's main setting | the paper, Sec. 4.1: 159.5 / 200 = 0.80 |
| 12 | S = 1/(1 + zeta*) = 0.57 | formula: zeta* = (1 - gamma)(alpha - 1)/(1 + (alpha - 1) gamma) = 0.76 at gamma 0.5, alpha = e^2; 1/1.76 = 0.57 |
| 12 | c = 0.88 | formula: gamma alpha/(1 + (alpha - 1) gamma) = 0.881 |
| 12 | gamma = 0.5, delta = 2: zeta* = 0.76, c = 0.88 | formula, as above |
| 12 | tokens needed for Z = 4: 45 vs 400; q = 0.8 and 0.6 | formula: T >= 16 gamma (1 - gamma)/(q - gamma)^2 = 44.4 at q = 0.8, 400 at q = 0.6 (illustrative q, gamma 0.5) |
| 12 (figure) | mean Z_T against T for q = 0.8 and q = 0.6; z = 4 line | formula: Z_T = sqrt(T)(q - gamma)/sqrt(gamma (1 - gamma)), illustrative q, drawn in `main.tex` |
| 12 | paper: S* = 0.807, at least 142.2 green tokens of 200; observed 159.5 | the paper, Sec. 4.1 |
| 12 (note) | Barack S = 0.57 gives 0.50; two options S = 0.72 gives 0.64; six options S = 0.89 gives 0.78 | formula: c S(p) with zeta* = 0.76, c = 0.881 |
| 12 (note) | ours: S* = 0.813, bound 143.2, observed 158.3; Extension 1: 24 tokens for news, 188 for code | `results/summary/headline.csv`; `ext1b_cells.csv` (OPT news and code) |
| 13 | attack table: paraphrase (T5), insert or delete, character tricks, emoji attack | the paper, Sec. 7 and App. B; emoji attack: Goodside (2023) |
| 13 (note) | paper T5 attack at 30%: AUC 0.954, about 15 perplexity points | the paper, Table 9 (AUC 0.954; PPL 21.2 vs 6.3 = +14.9) |
| 13 (note) | about T/4 edits | as on slide 7 |
| 14 | T5 (2020) | Raffel et al. 2020, the model the paper's attack uses |
| 15 | last 200 tokens; OPT-1.3B; tau = 0.7; T = 200 +/- 5; OPT-2.7B | the paper (Sec. 6) and RESULTS.md Deviations |
| 15 | about 500 texts per setting | the paper's Table 8 counts (495 to 507) |
| 15 | 100 to 500 per setting; 497 for the headline; 1,000 human texts | `results/summary/table8.csv` (`count`), `kept_counts.csv`, `ppl_baselines.csv` |
| 15 | about 20 hours | RESULTS.md compute log (Oct 4, 02:15 to 22:06, with pauses) |
| 15 | 1,000 random contexts; 200 for Qwen | `results/summary/greenlist_equivalence.txt` |
| 16 | detected at z = 4: 98.4% / 98.6% [97.1, 99.3] | paper Sec. 4.1 / `results/summary/headline.csv` (TPR 0.9859 [0.971, 0.993]) |
| 16 | human texts flagged: 0 / 0 of 1,000 | paper Table 8 (FPR 0.0) / `headline.csv` (FPR 0 [0, 0.004]) |
| 16 | mean green tokens 159.5 / 158.3 | paper / `headline.csv` (158.34 [157.5, 159.2]) |
| 16 | Theorem 4.2 bound 142.2 / 143.2 | paper / `headline.csv` (143.20) |
| 16 | 4-beam search 99.6% / 100% | paper / `headline.csv` (1.0 [0.975, 1.0], n = 150) |
| 16 | 497 texts, 150 with 4 beams, 1,000 human texts | `headline.csv`, `kept_counts.csv` |
| 16 (figure) | human 1,000; no watermark 500; watermarked 497; z = 4 line | `make_figures.py` `z_hist()`: `results/raw/human/c4` (idx < 1000), `results/raw/gen/opt-news-m-d0` and `opt-news-m-d2-g0.5` (kept rows, first 500) |
| 17 | 11 of 12 settings within our 95% intervals; the shaded one is the exception | `results/summary/table8.csv` (`vs_paper_z4`: 11 consistent, multinomial delta 1, gamma 0.5 borderline: 0.700 [0.646, 0.749] vs 0.767) |
| 17 | same shape; a light attack removes less, heavier ones slightly more | `results/summary/table9_t5.csv`, single-word rows, z = 4: 0.99 / 0.24 / 0.02 / 0.01 vs the paper's 0.819 / 0.353 / 0.094 / 0.039 |
| 17 (figures) | all plotted TPRs at z = 4, CIs and paper markers | `make_figures.py` `table8_dots_z4()` (`table8.csv`: `TPR_z4`, `TPR_z4_lo/hi`, `paper_TPR_z4`; gold band where `vs_paper_z5` = different) and `t5_attack_z4()` (`table9_t5.csv` single-word rows) |
| 18 | delta = 1, gamma = 0.5: 0.37 [0.31, 0.42] vs 0.50 | `table8.csv` (TPR_z5 0.367 [0.314, 0.423], paper 0.504) |
| 18 | 10%: 0.99 vs 0.82; from 30% on | `table9_t5.csv` (eps 0.1: TPR_z4 0.99, paper 0.819) |
| 18 | +0.1 to +0.3 under multinomial sampling; one setting +1.7 | `table8.csv` (`PPL_mean` vs `paper_PPL`: +0.06 to +0.33; delta 5, gamma 0.25: 12.39 vs 10.7) |
| 18 | exact match 0.202 to 0.212; paper about -0.04 | `figures/ext1b_triviaqa.md` (qwen-it 0.202 -> 0.212; FLAN-UL2 0.374 -> 0.336, BLOOMZ 0.296 -> 0.259) |
| 19 | news 24; code 188; 49% never in 200 tokens; short answers never | `results/summary/ext1b_cells.csv`, OPT rows (`tokens_to_z4_median`, `never_z4` = 0.494) |
| 19 (figure) | 24 [20, 27], never 1%; 188 [135, never], never 49%; short answers never (99%) | `ext1b_cells.csv` (`tokens_med_lo/hi`, `never_z4` = 0.01, 0.494, 0.99) |
| 19 | 200, 164, 500 texts; gamma 0.25 / 0.5; delta 2 | `ext1b_cells.csv` (`n`, `gamma`, `delta`) |
| 20 | same meaning kept 0.85 | `results/summary/ext2a_attacks.csv` (`meaning_kept`: t5w 0.3 = 0.856, para light = 0.850) |
| 20 | 81% vs 43.5% detected at 1% false positives | `ext2a_attacks.csv` (`TPR_1pctFPR`: t5w 0.3 = 0.81, para light = 0.435) |
| 20 | AUC 0.995 vs 0.881 | `ext2a_attacks.csv` (`AUC`: 0.99495, 0.8806) |
| 20 | model calls 60 to 140 vs 1 | RESULTS.md 2a ("60-140 sequential T5 calls"); one call per attempted swap, `table9_t5.csv` words_replaced 60.0 at 30% |
| 20 (figure) | meaning kept, TPR at 1% FPR and their CIs for every attack | `ext2a_attacks.csv` (`meaning_kept`, `meaning_lo/hi`, `TPR_1pctFPR`, `TPR_1pct_lo/hi`) |
| 20 (note) | 0.84 kept by a full rewrite; 0.22 vs 0.24 at z = 4 | `ext2a_attacks.csv` (para full 0.837; `TPR_z4` 0.22, 0.24) |
| 21 | 60-token passage, 600-token text | RESULTS.md 2b; `ext2b_dilution.csv` (share 0.1) |
| 21 | 11.5% vs 86.5% at 1% false alarms | `results/summary/ext2b_dilution.csv` (`TPR_full_z` 0.115, `TPR_winmax` 0.865) |
| 21 | ICLR 2024 | Kirchenbauer et al., On the Reliability of Watermarks for Large Language Models |
| 21 (note) | z > 2.20, WinMax > 4.02, 1,000 documents | `results/summary/ext2b_thresholds.json` |
| 22 | 98.6% vs 98.4%; 11 of 12; 86.5% vs 11.5% | as on slides 16, 17 and 21 |

## Backup

| Slide | Number | Source |
|---|---|---|
| B1 (24) | zeta* = 0.76, c = 0.881 | formula at gamma 0.5, alpha = e^2 |
| B1 | spread 0.807 / 0.813; bound 142.2 / 143.2; mean 159.5 / 158.3; sigma bound 6.41 / 6.38; observed sigma 9.85; predicted detection 98.6% / 99.0% | paper Sec. 4.1 / `results/summary/headline.csv`, RESULTS.md Sec. 4.1 |
| B1 (note) | 98.5% with the paper's bounds | `headline.csv`, RESULTS.md |
| B2 (25) | -0.52 to +0.71 (unwatermarked model text); -0.36 to +0.55 (human); sd 1.03 to 1.36; six keys | `results/summary/null_mean_by_key.json` (delta 0 means -0.521 to 0.709, human means -0.362 to 0.554, sd 1.03 to 1.363) |
| B2 | 0 of 1,000 false alarms at z = 4 | `headline.csv`, `calibration_negatives.md` |
| B2 | 16 tokens or fewer | QA generations, up to 16 new tokens (`ext1b_cells.csv` `T_mean` 16.0) |
| B2 | 0.005; 97.2% [95.3, 98.3]; 98.6% | `headline.csv` (re-encode drop 0.0048; norep TPR 0.9718 [0.953, 0.983]) |
| B3 (26) | delta 2: z 10.8 at gamma 0.1 vs 8.3 at gamma 0.5; perplexity 6.3 vs 6.5 | `results/summary/fig2_tradeoff.csv` (10.77 / 6.26; 8.25 / 6.53) |
| B3 | 8 beams z = 11.5 at perplexity 1.63; 1.44 without watermark | `fig2_tradeoff.csv` (11.52, 1.629); `ppl_baselines.csv` (1.442) |
| B3 (note) | 4 beams z 11.2, perplexity 1.74; 100 / 150 / 300-500 texts per point | `fig2_tradeoff.csv` (`n`) |
| B4 (27) | 4 / 8 / 22 / 60 / 176; 124 / 30 / 8 / 6; 16; z > 5 from 24; paper about 35 | `results/summary/fig3_z_vs_T.csv` (first T with z_mean > 4 or > 5); paper Sec. 7 |
| B5 (28) | all 12 rows: n, TPR at z = 4 and 5 with CIs, paper values, verdicts | `results/summary/table8.csv` |
| B5 | FPR 0 [0, 0.004] on 1,000 human texts; 0.002 on 500 unwatermarked texts | `table8.csv` (`FPR_z4`, `FPR_z4_hi`, `FPR_z4_delta0`) |
| B6 (29) | all AUC and perplexity values, paper values | `table8.csv` (`AUC`, `paper_AUC`, `PPL_mean`, `paper_PPL`) |
| B6 | no watermark 5.29 (5.1), 1.44 (1.2); human 11.06 | `results/summary/ppl_baselines.csv` |
| B6 | +0.1 to +0.3; +0.3 to +0.5 (8 beams); +1.7 | `table8.csv` differences, RESULTS.md Fig. 4 section |
| B7 (30) | temperature 0.7 | RESULTS.md (processor-order check) |
| B7 | 22 runs; 0.675 vs 0.716; observed 0.792 | `results/summary/ext1a_bounds.csv` (gamma 0.5, delta 2 row), `fig7_theory.csv` |
| B7 (figure) | observed mean and IQR, both bounds, paper markers 0.53 / 0.60 / 0.68 / 0.80 / 0.94 / 0.99 | `fig7_theory.csv`; paper markers read off the paper's Fig. 7 (RESULTS.md Fig. 7) |
| B7 (note) | 0.175 vs 0.172; 0.714 vs 0.813 | `ext1a_bounds.csv`, RESULTS.md 1a |
| B8 (31) | all rows: TPR with CIs and verdicts, AUC, perplexity, tokens after, words replaced, paper values | `results/summary/table9_t5.csv` (rows "m-nom, single-word T5") |
| B8 | 100 texts; 1,000 human texts | `table9_t5.csv` (`count`), RESULTS.md Table 9 |
| B8 | multi-word: 200 texts, 295 tokens at 30%, 0.495 vs 0.24 | RESULTS.md Table 9 multi-word, `ext2a_attacks.csv` (t5 0.3: T_mean 294.7, TPR_z4 0.495) |
| B8 (note) | green fraction 0.79 -> 0.62; perplexity 9.9 vs 16.0 | `ext2a_attacks.csv` (`green_frac`, `PPL_mean`) |
| B9 (32) | EM and mean z, all five models | `figures/ext1b_triviaqa.md` |
| B9 | 500 questions; 16 new tokens; gamma 0.5, delta 2 | `figures/ext1b_triviaqa.md` header |
| B9 | 0.202 [0.169, 0.239] -> 0.212 [0.178, 0.250]; 5.8 tokens | RESULTS.md Table 10; `ext1b_cells.csv` (qa qwen-it `T_mean` 5.77) |
| B10 (33) | hit: spread 0.875, z = 9.62; miss: spread 0.571, z = 0.57 | `results/summary/examples.csv` (idx 699 and 562, `S`, `z`) |
| B11 (34) | z 4.01 / 3.13; 4.08 / 2.94; 4.38 / 0.40; out of 500 | `results/summary/delta0_false_positives.md` (idx 56, 208, 417), `calibration_negatives.md` (n 500) |
| B11 | human 0 of 1,000; mean z -0.25 to +0.09; sd 1.04 to 1.10; unwatermarked sd 1.27 to 1.35 | `results/summary/calibration_negatives.md` |
| B12 (35) | 0.70 vs 0.77; 0.37 [0.31, 0.42] vs 0.50 | `table8.csv` |
| B12 | T5 verdicts; 0.02 vs 0.094 | `table9_t5.csv` |
| B12 | 12.4 vs 10.7 | `table8.csv` |
| B12 | about -0.04 | `figures/ext1b_triviaqa.md` (paper rows) |
| B12 | 158.3 [157.5, 159.2] vs 159.5 | `headline.csv` |
| B12 | 2,048 tokens, last 1,847; 497 of 1,000; +/-60-token windows; temperature 0.7, top-k 0, top-p 1; 150 words | RESULTS.md Deviations; `kept_counts.csv` |
| B13 (36) | every cell: n, entropy, TPR at 1% FPR (both), median tokens with CI, never | `results/summary/ext1b_cells.csv` (`n`, `H_nowm`, `TPR_1pctFPR`, `TPR_1pctFPR_norep`, `tokens_to_z4_median`, `tokens_med_lo/hi`, `never_z4`) |
| B13 | gamma 0.25 / 0.5, delta 2, tau 0.7 | `ext1b_cells.csv`, RESULTS.md 1b |
| B14 (37) | -2.88 / -0.45 | `ext1b_cells.csv` (code opt `z_mean_nowm`, `z_norep_mean_nowm`) |
| B14 | OPT 0.52 [0.45, 0.60] -> 0.62 [0.54, 0.69]; Qwen 0.24 [0.19, 0.32] -> 0.51 [0.44, 0.59] | `ext1b_cells.csv` (`TPR_1pctFPR`, `TPR_1pctFPR_norep`, `TPR_1pct_lo/hi`, `TPR_1pct_norep_lo/hi`) |
| B15 (38) | every cell: observed, predicted (both), share of texts below (both) | `ext1b_cells.csv` (`green_frac`, `pred_paper`, `pred_corrected`, `seq_below_paper`, `seq_below_corrected`) |
| B15 | 0.584 vs 0.453 | `ext1b_cells.csv`, news opt |
| B16 (39) | entropy 2.52 -> 1.96; 0.59 -> 0.43; 1.74 -> 1.18 | `ext1b_cells.csv` (`H_nowm`, qwen vs qwen-it) |
| B16 | 99.5% vs 99.0% | `ext1b_cells.csv` (`TPR_1pctFPR` news qwen 0.995, qwen-it 0.990) |
| B16 | 1.96 -> 2.30 under the watermark | `ext1b_cells.csv` (news qwen-it `H_nowm` 1.963, `H` 2.305) |
| B16 | perplexity 5.7 -> 9.5; OPT 5.2 -> 6.9 | `ext1b_cells.csv` (`PPL_mean_nowm`, `PPL_mean`: 5.673 / 9.548; 5.204 / 6.888) |
| B16 | 27 -> 8 tokens; perplexity 25.6 | `ext1b_cells.csv` (news qwen-it delta 2 vs delta 4; `PPL_mean` 25.56) |
| B16 (figure) | means 2.52 and 1.96 | per-token `H` of `results/raw/gen/qwen-news-m-d0`, `qwen-it-news-m-d0` (legend uses `ext1b_cells.csv` `H_nowm`) |
| B17 (40) | every attack: meaning kept, tokens changed, green fraction, TPR at z = 4, TPR at 1% FPR, AUC, perplexity | `results/summary/ext2a_attacks.csv` |
| B17 | 65% of tokens | `ext2a_attacks.csv` (para light `tokens_changed` 0.650) |
| B18 (41) | 0.22 [0.17, 0.28] vs 0.24 [0.17, 0.33] | `ext2a_attacks.csv` (`TPR_z4`, `TPR_z4_lo/hi`) |
| B18 | calibrated threshold about 2.4 | `ext2a_attacks.csv` (`thr_1pctFPR`: t5w 0.3 2.38, para light 2.44) |
| B19 (42) | all shares: TPR with CIs, mean z, mean WinMax | `results/summary/ext2b_dilution.csv` |
| B19 | 200 documents per share | `ext2b_dilution.csv` (`n`) |
| B19 | z > 2.20, WinMax > 4.02, human means -0.37 and 3.03 | `results/summary/ext2b_thresholds.json` |
| B19 | at z = 4: 1%, 28.5%, 97.5% | `ext2b_dilution.csv` (`TPR_full_z_at_4`) |
| B20 (43) | EOS green, multipliers 2.29 / 0.17 / 1.18 / 0.32, kept shares, stops before 50 with CIs | `results/summary/keys_eos.csv`, `eos_after_dot.json`, RESULTS.md key table |
| B20 | 16% [14, 18] vs 6% [4, 7]; 1,000 and 711 | `kept_counts.csv` |
| B20 | 29% to 37% at delta >= 5, gamma 0.5 to 0.75 | `kept_counts.csv` (d5/d10 x g0.5/g0.75: 0.294, 0.310, 0.367, 0.360) |
| B20 | gamma 0.25: 2.8% (delta 2), 2.3% (delta 5) | `kept_counts.csv` (d2-g0.25 0.028, d5-g0.25 0.023) |
| B20 | "stop" red after "." at gamma 0.25 | `results/summary/eos_green_status.json` ("'.'" green_at 0.25: false) |
| B20 (note) | 82% of competing mass green (key 32452843) | `eos_after_dot.json` (`green_mass_of_other_continuations` 0.815) |
| B21 (44) | 0.59 to 0.84; 7.1; 9.2; 0.31; 0.52 to 0.53; 0.63 | `results/summary/benchmark_opt.json`, `benchmark.json` |
| B21 | T5 round of 32 texts 9.5 s | `results/summary/benchmark_opt_log.txt` (bf16) |
| B21 | about 20 hours; 24 GB; transformers 5.18 | RESULTS.md (environment, compute log) |
