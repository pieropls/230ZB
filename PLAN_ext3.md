# Plan · Extension 3: the watermark as a filter for sentiment data

Phase 0 deliverable for `SPEC_ext3_filter.md`. Approved 2026-10-08.

## Inventory (what is reused, nothing regenerated)

| Pool | File | Usable texts | Detection at z > 4 |
|---|---|---|---|
| Human news | `results/raw/human/c4.jsonl.gz` | 1,000 paper-style baselines (idx < 1000), all 200 tokens | 0 flagged at 25 / 50 / 100 / 200 tokens |
| Watermarked (A) | `results/raw/gen/opt-news-m-d2-g0.5.jsonl.gz` | 493 with T >= 200 | 0.103 / 0.533 / 0.939 / 0.986 at 25 / 50 / 100 / 200 |
| Paraphrased (B) | `results/raw/attacks/scored.jsonl.gz` | 200 per strength (light / medium / full); median T 151 / 177 / 178 | 0.22 / 0.035 / 0.015 |
| Unwatermarked (D) | `results/raw/gen/opt-news-m-d0.jsonl.gz` | 492 with T >= 200 | 0.0014 |
| Pasted spans (C, Tier 2) | `results/raw/dilution/docs.jsonl.gz` | 1,600 docs; only z / WinMax stored, ids rebuilt from the same seed | – |

Every stored text carries per-token green ranks, so z at any prefix length is a cumulative sum (`stats.z_of`, `stats.green`); no language model is loaded. FinBERT (`ProsusAI/finbert`) is already cached.

## Design decisions

1. **Reuse of AI texts across buckets is accepted and stated.** At c = 0.3 each attacked bucket takes 21 AI texts, so 25 "up" buckets draw 525 times from a top-third pool of 164 texts (about 3 uses per text). Within a bucket: without replacement. Across buckets: with replacement. Buckets are compared to their own truth and the 500 re-draws carry the induced correlation. No new generation.
2. **Human pool = the 1,000 human baselines of the paper's setup** (idx < 1000), the same negatives behind the headline FPR. 100 buckets x 50 texts = 5,000 draws from 1,000: without replacement within a bucket, with replacement across buckets.
3. **Scenario B runs all three paraphrase strengths**; "full" is the headline for H4. Human texts in B and D are the 200-token baselines; the paraphrases keep their actual length (reported).
4. **H5 gets a fourth filter**: a *random* filter that drops each AI text with probability p (the pool's detection rate at that length) regardless of sentiment, inside the same draws. Actual minus random recovery, with a percentile interval over draws, is the H5 test. The closed-form prediction 1 - (1 - p) / (1 - c p) is reported next to it.
5. **Scenario C (Tier 2)**: the 600-token documents exceed FinBERT's 512-token window; score = mean of the two 300-token halves. Slant by the pasted span's own sentiment. Flags: full-text z and WinMax at the 1 % FPR thresholds of Ext. 2b (`ext2b_thresholds.json`), plus z > 4.

## Phases

| Phase | Work | Compute |
|---|---|---|
| 1 | `run.py ext3 --n 50`: FinBERT pilot, texts/s, controls (neutral share, slant means, pool sizes) | ~1 min |
| 2 | `run.py ext3`: full per-text table `results/raw/ext3_filter/texts.jsonl.gz` (resumable) | a few minutes |
| 3 | `ext3.py` simulation (numpy, seconds), notebook cells, `results/summary/ext3_filter.md`, figures 1-3 | seconds |
| 4 | Tier 2: calibrated threshold at 25 / 50 tokens; scenario C | a few minutes |
| 5 | `RESULTS.md` section, notebook runs top to bottom, `HANDOFF_ext3.md`, commit | – |
