# Handoff · Extension 3 (the watermark as a filter for sentiment data)

Run on 2026-10-08 on the project machine (Apple M5 Pro, 24 GB, `mps`, fp16). Spec: `SPEC_ext3_filter.md`; plan and inventory: `PLAN_ext3.md`.

## What ran

| phase | what | where | time |
|---|---|---|---|
| 0 | plan, inventory, alternatives (approved) | `PLAN_ext3.md` | – |
| 1 | pilot: FinBERT on 50 texts per source, fp16 vs fp32 check, controls | `python run.py ext3 --n 50` | 11 s |
| 2 | per-text table: 7,064 (text, length) pairs; human and watermarked prefixes at 25 / 50 / 100 / 200 tokens, unwatermarked at 200, three paraphrase strengths whole | `python run.py ext3` → `results/raw/ext3_filter/texts.jsonl.gz` | 37 s |
| 3 | simulation, scenarios A, B, D (16 cells × 500 draws), summary, figures 1-3 | `ext3.pipeline()` from the notebook → `results/summary/ext3_filter.{md,csv}`, `ext3_controls.csv` | 31 s |
| 4 | Tier 2: threshold at 1% FPR (25, 50 tokens); scenario C (Ext. 2b documents rebuilt with text, FinBERT on two halves, whole-text vs WinMax) | `python run.py ext3_docs` → `results/raw/ext3_filter/docs.jsonl.gz`; `ext3_calibrated.csv`, `ext3_pasted.csv`, figure 4 | 90 s + 50 s |
| 5 | `RESULTS.md` section, README section and TL;DR point, notebook cells (38 → 44), this note | – | – |

Total compute under 4 minutes against the 90-minute budget. Timing rows: `results/summary/ext3_timing.jsonl` and the Timing table in `ext3_filter.md`.

**No new text was generated.** Every AI text comes from earlier phases (§4.1 watermarked texts, δ = 0 generations, Ext. 2a paraphrases, Ext. 2b documents). The only new model is the sentiment scorer `ProsusAI/finbert` (already cached).

## Design decisions taken (approved before Phase 1)

- Reuse of slanted AI texts across buckets accepted and stated (164 texts per slant for 525 slots at c = 0.3; 66 per slant for the paraphrases). Within a bucket: without replacement.
- Human pool = the 1,000 human baselines with idx < 1000 (the paper's negatives), with replacement across buckets.
- Scenario B runs all three paraphrase strengths; "full" is the H4 headline. Paraphrases keep their actual length (median 151 / 177 / 178 tokens); the human texts in B and D are the 200-token baselines.
- H5 is tested against two sentiment-blind predictions: random removal at the detection rate p (the spec's), and the detector's own flags with identities shuffled within each slant pool (same count). The second is the H5 test; the first is kept because the spec asks for it. The closed-form formula 1 − (1 − p)/(1 − cp) is reported next to both; it predicts bias, not RMSE, so it sits a few points higher.
- Scenario C: FinBERT reads 512 tokens, so a 600-token document's score is the mean of its two halves; AI pools slanted by the pasted span's own score.

## Results in one paragraph

Recovery equals the detection rate to within 0.03 everywhere: 0.95 at 200 tokens, 0.40 at 50, 0.07 at 25 (z cannot exceed 5 there), 0.15 after a light paraphrase, under 0.01 after a full one, exactly 0 for unwatermarked AI text. H1-H4 supported, H5 not rejected (all 16 intervals of actual − shuffled cover zero; point estimates slightly negative, at 50-100 tokens the escapers are if anything a bit more extreme than their pool). Calibrating the threshold to 1% FPR lifts the 25-token recovery to 0.63 for 20 lost human texts in 5,000. On 60-token pasted passages WinMax recovers 0.66 where the whole-text test recovers nothing (H6 supported).

## What did not run / caveats

- The spec's rule "generate more if a pool is short" was not triggered by decision (reuse accepted). With 66 texts per slant in scenario B and C the pools are small; the shuffled-flags prediction was added precisely because small pools with a fixed handful of escapers make the independent random-removal prediction too pessimistic (visible in scenario C: up to +0.13 against random at p, at most +0.15 and n.s. in 17 of 18 cells against shuffled).
- Scenario C's contamination is mild by construction (a 10% span leaves the document 90% human), so its recoveries have wide intervals.
- The realised FPR of the calibrated thresholds is 0.3-0.4%, not 1%, because z is discrete at 25-50 tokens (the threshold is the "higher" quantile, as in `stats.tpr_at_fpr`).
- The notebook cell runs the whole simulation (about 80 s); the figures and summary files are regenerated on every run, with fixed seeds, so the numbers in `RESULTS.md` are reproducible exactly.

## Open questions

- Would a campaign that *prompts* for slant (rather than selecting slanted texts) produce lower-entropy, harder-to-detect text? That is the mechanism in section 1 of the spec that this design cannot reach; the escapers table shows no such selection within our pools.
- A larger paraphrase pool (rerunning `para` on more texts, about 10 min) would tighten scenario B's intervals; not needed for the conclusions.
- FinBERT labels 71-74% of general news neutral; a general-domain sentiment model would give the slant more room, at the cost of leaving the finance framing.

## Files added or changed

`params.py` (EXT3_* knobs, `SENTIMENT_MODEL`), `core.py` (`load_sentiment`, `sentiment`, `EXT3_FILE`, `EXT3_DOCS`), `run.py` (`ext3`, `ext3_docs`, `dilution_docs` factored out of `dilution` with identical output), `ext3.py` (new), `plots.py` (4 figure functions), `watermark.ipynb` (6 cells), `RESULTS.md`, `README.md`, `PLAN_ext3.md`, `HANDOFF_ext3.md`, `results/raw/ext3_filter/`, `results/summary/ext3_*`, `figures/ext3_*.png`.
