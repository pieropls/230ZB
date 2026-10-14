# SPEC · Extension 3: the watermark as a filter for sentiment data

Spec for Claude Code, to run inside the existing green-list watermark repo (MFE 230ZB, Kirchenbauer et al. reproduction). This replaces the earlier draft of this file, which tested a different question.

## 0. Read this first

**Goal.** A sentiment signal is built from many texts. Some of them are written by an AI model to push the signal one way. Measure **how much of the damage is undone by removing the texts the watermark detector flags**, and how that depends on text length and on attacks.

**Three rules that override everything below.**

1. **Reuse, do not rewrite, and do not regenerate.** Generation, the watermark processor, detection, the paraphrase attack, the copy-paste construction, confidence intervals and plotting already exist in `core.py`, `stats.py`, `plots.py`, `params.py`, `run.py`. Human, watermarked, unwatermarked and attacked texts from earlier phases are already on disk under `results/`. Read both before writing a line. Generate new text only where section 4 says nothing reusable exists.
2. **Propose before you build.** Phase 0 ends with a short written plan, an inventory of what is reusable, and any alternative design you think is better. Then **stop and wait for approval.**
3. **Every result is a recovery rate against two bounds:** no filter (worst) and a perfect filter (best). A filtered error reported alone is not a result.

**Budget.** About 90 minutes of compute on this machine (M5 Pro, `mps`, fp16). Time a pilot first. The simulation in section 5 needs no generation and should run in seconds; if it does not, the design is wrong.

## 1. Theory: what recovery should be

Let $c$ be the share of AI texts in the data, $\mu_H$ and $\mu_A$ the mean sentiment of human and AI texts.

- **Contaminated signal.** Mean sentiment is $(1-c)\mu_H + c\,\mu_A$, so the bias is $c(\mu_A - \mu_H)$.
- **After filtering.** The detector removes a share $p$ of AI texts (its detection rate) and almost no human text. The remaining AI share is $c' = c(1-p)/(1-cp)$ and the bias becomes $c'(\mu_A' - \mu_H)$, where $\mu_A'$ is the mean sentiment of the AI texts that **escaped**.
- **Prediction.** If escaping is unrelated to sentiment, $\mu_A' = \mu_A$ and recovery is a function of $p$ alone. Everything the repo already knows about $p$ (it falls for short text, after a rewrite, and for pasted passages) then translates directly into lost signal quality.
- **Where the prediction can fail.** The watermark is weak on constrained text. If constrained text also has a different sentiment (flat, factual sentences), the escapers are not a random sample and $\mu_A' \ne \mu_A$. That is the non-obvious thing this extension can find.

## 2. Hypotheses (fixed before running)

| | Hypothesis |
|---|---|
| H1 | Contamination harms the signal: error with AI texts is clearly above the clean noise floor |
| H2 | On long, unattacked AI text, the filter recovers almost all of the damage |
| H3 | Recovery falls as texts get shorter, in line with the detection rate |
| H4 | After a rewrite attack, recovery collapses |
| H5 | Recovery equals what the formula in section 1 predicts from $p$ alone (escaping is unrelated to sentiment) |
| H6 (Tier 2) | For a pasted passage, the window detector recovers what the whole-text detector misses |

A rejection of H5 is as interesting as a confirmation. Report either.

## 3. The signal and its error

**Sentiment score.** One classifier, separate from the generator. Default: `ProsusAI/finbert`. Score $S = P(\text{positive}) - P(\text{negative}) \in [-1, 1]$. Score each text **once**, at each length in section 4, and store the result.

**Buckets.** A bucket stands for one day of text about one subject. Draw $B = 100$ buckets of $n = 50$ human texts each, at random without replacement where the pool allows, with replacement otherwise (say which).

**Truth.** The bucket's mean sentiment over its human texts.

**Campaign.** Half of the buckets, chosen at random, are attacked. In an attacked bucket, AI texts are added until they make up a share $c$ of the bucket. The campaign pushes **up** in half of the attacked buckets and **down** in the other half.

**Slanted AI texts.** Take the pool of AI texts, rank by sentiment score, and use the top third for "up" campaigns and the bottom third for "down" campaigns. This is selection, not steering: no new prompt design.

**Error measures,** all against the truth, over buckets:

- RMSE of the bucket signal.
- Rank correlation (Spearman) between the bucket signal and the truth.

**Recovery.** $1 - \dfrac{\text{error}_{\text{filtered}} - \text{error}_{\text{clean}}}{\text{error}_{\text{contaminated}} - \text{error}_{\text{clean}}}$, on the RMSE. Clean error is zero by construction for the truth itself; keep the term so that the formula still holds if false positives remove human texts.

## 4. Scenarios

| | AI texts | Lengths (tokens) | Detector | Source |
|---|---|---|---|---|
| A | Watermarked, unattacked | 25, 50, 100, 200 | whole text, $z > 4$ | Prefixes of the existing §4.1 watermarked texts |
| B | Watermarked, then rewritten | 200 | whole text, $z > 4$ | Existing Extension 2a attacked texts |
| D | Not watermarked | 200 | whole text, $z > 4$ | Existing unwatermarked texts |
| C (Tier 2) | Watermarked passage pasted into human text | as in Extension 2b | whole text vs moving window | Existing Extension 2b texts |

- **Human texts** are the existing human news texts, cut to the same length as the AI texts in each scenario.
- **Scenario D is the sanity check.** The filter must remove nothing and recovery must be zero. It is also the honest limit of the whole idea: a model that does not embed the watermark is invisible.
- **Contamination share:** $c \in \{0.1, 0.3\}$ for every scenario.
- **Filters compared in every scenario:** none, watermark detector, perfect (removes every AI text).
- **Settings:** the paper's main watermark setting already used for §4.1.

**New generation** is needed only if a pool is too small: each slant (top or bottom third) must supply enough texts for the attacked buckets without heavy reuse. If it does not, generate more watermarked texts with the existing code, and log how many.

## 5. Procedure

1. **Build one table,** one row per text and length: id, source (human, watermarked, rewritten, unwatermarked), length, $z$, flagged or not, sentiment probabilities, $S$.
2. **Simulate from that table only.** For each scenario, share $c$ and filter: draw buckets, contaminate, filter, compute errors. Repeat the whole draw $R = 500$ times.
3. **Predicted recovery.** From the same table, compute $p$ and the section 1 formula with $\mu_A' = \mu_A$. Also compute the actual $\mu_A'$ of the escapers.

## 6. Controls

- **Classifier has room to move.** Report the label distribution on human text. If more than 80% is neutral, the slant has nothing to work with: flag it and stop before the simulation.
- **Slant is real.** Report $\mu_H$, $\mu_A$ for the up and down pools, per length.
- **False positives.** Report how many human texts the filter removes, per scenario. At $z > 4$ it should be about zero.
- **Short-text ceiling.** At 25 tokens the largest possible $z$ is barely above 4, so near-zero detection there is a property of the threshold, not a bug. State the largest attainable $z$ per length next to the detection rate.

## 7. Statistics

Reuse the conventions in `stats.py`.

- **Intervals.** Percentile intervals over the $R = 500$ simulation draws for RMSE, rank correlation and recovery. Wilson intervals for detection rates and false-positive rates.
- **H5.** Actual recovery minus predicted recovery, with its interval, per scenario and length. Also the difference $\mu_A' - \mu_A$ with a bootstrap interval over texts.
- **Tier 2, threshold.** Repeat scenario A at 25 and 50 tokens with a threshold calibrated to a 1% false-positive rate on human text of that length, and report the trade-off: more AI text removed against human text lost.

## 8. Phases

| Phase | Work | Ends with |
|---|---|---|
| 0 | Read the repo and `results/`. Inventory reusable texts and functions. Write the plan and any alternative. | **Stop. Wait for approval.** |
| 1 | Pilot: score 50 texts per source, time scoring (texts/s). Check pool sizes and the controls in section 6. | Timing table, go or no-go |
| 2 | Build the full table. Generate only if a pool is short. | Table on disk |
| 3 | Scenarios A, B, D: simulation, recovery, prediction, figures. | Summary files |
| 4 | Tier 2: scenario C, calibrated threshold. Only if budget remains. | Summary files |
| 5 | Write-up, notebook cells, commit. | Handoff note |

Commit at the end of each phase. Scoring must be resumable: write incrementally and skip what is on disk.

## 9. Deliverables

- **Table:** `results/ext3_filter/texts.jsonl` (the per-text table of section 5).
- **Summary:** `results/summary/ext3_filter.md` with, per scenario, length and share $c$: detection rate, RMSE and rank correlation under the three filters, recovery, predicted recovery, and the timing table.
- **Figures** in `figures/`, same style as the existing ones (bold title, default matplotlib fonts, sample size in the caption):
  1. Recovery against text length, scenario A, with the predicted curve.
  2. Signal error under the three filters, scenarios A (200 tokens), B and D side by side.
  3. Actual against predicted recovery, all scenarios (the H5 picture).
  4. (Tier 2) Whole-text against window detector on pasted passages.
- **Report:** a new section in `RESULTS.md`, "Extension 3 · The watermark as a filter for sentiment data", in the format of the existing sections.
- **Notebook:** cells appended to `watermark.ipynb` that load the summaries and show the tables and figures; the notebook must still run top to bottom.
- **Handoff:** update the existing progress file if the repo has one; otherwise create `HANDOFF_ext3.md` with what ran, what did not, timings, and open questions.

## 10. What to report honestly

- **The campaign is synthetic.** We inject the AI texts ourselves. The result is the value of the filter *if* providers embed the watermark and share the recipe, not evidence that it cleans real data today. Scenario D is that limit.
- **Slant by selection** is a simplification of a real campaign. Say so.
- **FinBERT on general news** is partly out of domain. Report the neutral share.
- **No link to returns.** "Signal quality" here means closeness to the clean sentiment signal, not predictive power for prices.

## 11. Before you finish: the rules again

1. **Reused existing code and existing texts.** Any new generation is logged with its count and the reason.
2. **Plan, inventory and alternatives were written and approved before Phase 1.**
3. **Every result is a recovery rate between the no-filter and perfect-filter bounds,** next to the recovery predicted from the detection rate.
4. **Everything is timed,** and the timing table is in the summary.
5. **Done means:** table, summary, figures, `RESULTS.md` section, notebook runs clean, repo committed, handoff note written. Then stop.
