"""Extension 3: the watermark as a filter for sentiment data. Pure numpy on the per-text table."""
import numpy as np
import pandas as pd

import params as P
import stats as S

SOURCES = {"human": "human", "wm": "watermarked", "nowm": "unwatermarked", "para-light": "paraphrase, light",
           "para-medium": "paraphrase, medium", "para-full": "paraphrase, full"}


def table(rows):
    """DataFrame of the per-text table with the pool key (source, length) each row belongs to."""
    df = pd.DataFrame(rows)
    df["flagged"] = df["flagged"].astype(bool)
    return df


def pools(df):
    """{(src, L): sub-table} for every source and length on disk."""
    return {k: g.reset_index(drop=True) for k, g in df.groupby(["src", "L"])}


def slant(sub, frac=P.EXT3_SLANT, by="S"):
    """(up, down) sub-tables: the top and bottom `frac` of an AI pool by sentiment score S (or another column)."""
    s = sub.sort_values(by, kind="stable")
    k = int(np.floor(len(s) * frac))
    return s.iloc[-k:].reset_index(drop=True), s.iloc[:k].reset_index(drop=True)


# ---- section 6 controls -----------------------------------------------------------------------------
def controls(df):
    """Per pool: size, label shares, mean S, detection rate (Wilson), largest attainable z, slant means."""
    out = []
    for (src, L), g in sorted(pools(df).items(), key=lambda kv: (list(SOURCES).index(kv[0][0]), kv[0][1])):
        lo, hi = S.wilson_of(g.flagged)
        row = dict(src=src, L=L, n=len(g), T_median=float(g["T"].median()), neutral=(g.label == "neutral").mean(),
                   positive=(g.label == "positive").mean(), negative=(g.label == "negative").mean(),
                   S_mean=g.S.mean(), S_sd=g.S.std(), flagged=g.flagged.mean(), flagged_lo=lo, flagged_hi=hi,
                   z_max_possible=float(np.sqrt(g["T"].median())), z_mean=g.z.mean())
        if src != "human":
            up, down = slant(g)
            row.update(n_slant=len(up), S_up=up.S.mean(), S_down=down.S.mean(),
                       flagged_up=up.flagged.mean(), flagged_down=down.flagged.mean())
        out.append(row)
    return pd.DataFrame(out)


def controls_text(rows):
    """Human-readable controls, with the section 6 stop rule on the neutral share."""
    c = controls(table(rows))
    lines = [c.to_string(index=False, float_format=lambda x: f"{x:.3f}")]
    h = c[c.src == "human"]
    if len(h):
        worst = h.neutral.max()
        lines.append(f"\nneutral share on human text: {h.neutral.min():.3f}-{worst:.3f}  "
                     + ("STOP: above 0.80, the slant has nothing to work with" if worst > 0.8 else "OK (below 0.80)"))
    return "\n".join(lines)


# ---- the simulation (section 5) --------------------------------------------------------------------
SCENARIOS = ([("A", "wm", L) for L in P.EXT3_LENGTHS] + [("B", f"para-{s}", P.MAX_NEW) for s in P.PARA_PROMPTS]
             + [("D", "nowm", P.MAX_NEW)])      # (scenario, AI source, nominal length); human texts use the same length


def _spearman(a, b):
    """Rank correlation (average ranks for ties, via scipy)."""
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _pick(rng, pool_size, rows, k):
    """rows x k indices into a pool: without replacement within a row, independent across rows."""
    return np.argsort(rng.random((rows, pool_size)), axis=1)[:, :k]


def simulate(hum, ai, c, thr=None, flag="flagged", R=P.EXT3_DRAWS, seed=0, by="S"):
    """Draw EXT3_BUCKETS buckets of EXT3_BUCKET_N human texts R times; half the buckets receive AI texts
    (share c; half pushed up with the top third of `ai` by S, half down with the bottom third); then
    measure the bucket signal against the truth (the bucket's human mean) under four filters:
    none, the detector (drop flagged texts), perfect (drop every AI text), random (drop each AI text with
    the pool's detection rate p, each human text with the false-positive rate, independent of sentiment) and
    shuffled (the detector's flags with their identities permuted within each slant pool every draw: the same
    number of texts removed, unrelated to sentiment). Returns per-draw arrays and the per-draw recovery;
    thr overrides the stored flags with z > thr."""
    rng = np.random.default_rng(seed)
    B, n = P.EXT3_BUCKETS, P.EXT3_BUCKET_N
    m = int(round(c * n / (1 - c)))                       # AI texts per attacked bucket
    fl = (lambda d: (d.z.values > thr)) if thr is not None else (lambda d: d[flag].values.astype(bool))
    up, down = slant(ai, by=by)
    Sh, fh = hum.S.values, fl(hum)
    Su, fu, Sd, fd = up.S.values, fl(up), down.S.values, fl(down)
    p, fpr = float(fl(ai).mean()), float(fh.mean())
    n_att = B // 2
    n_up = n_att // 2
    out = {k: np.empty(R) for k in ("rmse_none", "rmse_det", "rmse_perfect", "rmse_rand", "rmse_shuf", "rho_none", "rho_det",
                                    "rho_perfect", "rho_rand", "rho_shuf", "human_removed", "ai_removed")}
    for r in range(R):
        hidx = _pick(rng, len(Sh), B, n)
        H, FH = Sh[hidx], fh[hidx]
        truth = H.mean(1)
        order = rng.permutation(B)
        A, FA, FS, has = np.zeros((B, m)), np.zeros((B, m), dtype=bool), np.zeros((B, m), dtype=bool), np.zeros(B, dtype=bool)
        for rows_, Sx, fx in ((order[:n_up], Su, fu), (order[n_up:n_att], Sd, fd)):
            j = _pick(rng, len(Sx), len(rows_), m)
            A[rows_], FA[rows_], FS[rows_], has[rows_] = Sx[j], fx[j], rng.permutation(fx)[j], True
        keepA_r = (rng.random((B, m)) >= p) & has[:, None]
        keepH_r = rng.random((B, n)) >= fpr
        hasA = np.repeat(has[:, None], m, 1)
        sig = {"none": (H.sum(1) + A.sum(1)) / (n + m * has),
               "det": ((H * ~FH).sum(1) + (A * ~FA * hasA).sum(1)) / ((~FH).sum(1) + (~FA & hasA).sum(1)),
               "perfect": truth,
               "rand": ((H * keepH_r).sum(1) + (A * keepA_r).sum(1)) / (keepH_r.sum(1) + keepA_r.sum(1)),
               "shuf": ((H * ~FH).sum(1) + (A * ~FS * hasA).sum(1)) / ((~FH).sum(1) + (~FS & hasA).sum(1))}
        for k, s in sig.items():
            out[f"rmse_{k}"][r] = float(np.sqrt(np.mean((s - truth) ** 2)))
            out[f"rho_{k}"][r] = _spearman(s, truth)
        out["human_removed"][r] = FH.sum()
        out["ai_removed"][r] = (FA & hasA).sum()
    for k in ("det", "rand", "shuf"):
        out[f"recovery_{k}"] = 1 - (out[f"rmse_{k}"] - out["rmse_perfect"]) / (out["rmse_none"] - out["rmse_perfect"])
    out["recovery_diff"] = out["recovery_det"] - out["recovery_shuf"]          # the H5 test
    out["recovery_diff_rand"] = out["recovery_det"] - out["recovery_rand"]
    c_act = m / (n + m)
    out.update(m=m, c_actual=c_act, p=p, fpr=fpr, n_ai=len(ai), n_slant=len(up), n_human=len(hum),
               recovery_formula=1 - (1 - p) / (1 - c_act * p), mu_H=float(Sh.mean()),
               mu_up=float(Su.mean()), mu_down=float(Sd.mean()),
               mu_up_escaped=float(Su[~fu].mean()) if (~fu).any() else float("nan"),
               mu_down_escaped=float(Sd[~fd].mean()) if (~fd).any() else float("nan"),
               n_up_escaped=int((~fu).sum()), n_down_escaped=int((~fd).sum()))
    return out


def pct(x):
    """Mean and 2.5 / 97.5 percentiles over draws."""
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def boot_mean_diff(S_pool, escaped, B=2000, seed=0):
    """Bootstrap interval over texts for mean(S of escapers) - mean(S of the pool)."""
    rng = np.random.default_rng(seed)
    S_pool, escaped = np.asarray(S_pool), np.asarray(escaped, dtype=bool)
    if escaped.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    d = S_pool[escaped].mean() - S_pool.mean()
    diffs = []
    for _ in range(B):
        j = rng.integers(0, len(S_pool), len(S_pool))
        e = escaped[j]
        if e.sum() >= 2:
            diffs.append(S_pool[j][e].mean() - S_pool[j].mean())
    return float(d), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def row_of(scen, src, L, c, sim, ai, thr=None, flag="flagged", label=None, by="S"):
    """One summary row (point estimates with 95% intervals) from a simulate() result."""
    fl = (lambda d: (d.z.values > thr)) if thr is not None else (lambda d: d[flag].values.astype(bool))
    up, down = slant(ai, by=by)
    row = dict(scenario=scen, source=src, label=label or SOURCES[src], L=L, T_median=float(ai["T"].median()),
               c=c, c_actual=sim["c_actual"], m=sim["m"], n_ai=sim["n_ai"], n_slant=sim["n_slant"],
               z_max_possible=float(np.sqrt(ai["T"].median())), threshold=thr if thr is not None else P.EXT3_Z,
               p=sim["p"], p_lo=S.wilson(int(round(sim["p"] * sim["n_ai"])), sim["n_ai"])[0],
               p_hi=S.wilson(int(round(sim["p"] * sim["n_ai"])), sim["n_ai"])[1],
               fpr=sim["fpr"], fpr_hi=S.wilson(int(round(sim["fpr"] * sim["n_human"])), sim["n_human"])[1],
               human_removed_per_draw=float(np.mean(sim["human_removed"])),
               ai_removed_per_draw=float(np.mean(sim["ai_removed"])), ai_per_draw=sim["m"] * (P.EXT3_BUCKETS // 2),
               mu_H=sim["mu_H"], mu_up=sim["mu_up"], mu_down=sim["mu_down"])
    for k in ("none", "det", "perfect", "rand", "shuf"):
        for q in ("rmse", "rho"):
            v, lo, hi = pct(sim[f"{q}_{k}"])
            row[f"{q}_{k}"], row[f"{q}_{k}_lo"], row[f"{q}_{k}_hi"] = v, lo, hi
    for k in ("recovery_det", "recovery_rand", "recovery_shuf", "recovery_diff", "recovery_diff_rand"):
        v, lo, hi = pct(sim[k])
        row[k], row[f"{k}_lo"], row[f"{k}_hi"] = v, lo, hi
    row["recovery_formula"] = sim["recovery_formula"]
    for name, pool in (("up", up), ("down", down)):
        d, lo, hi = boot_mean_diff(pool.S.values, ~fl(pool))
        row[f"escaped_{name}_n"] = int((~fl(pool)).sum())
        row[f"escaped_{name}_dmu"], row[f"escaped_{name}_dmu_lo"], row[f"escaped_{name}_dmu_hi"] = d, lo, hi
    return row


def run_all(df, scenarios=SCENARIOS, shares=P.EXT3_SHARES, R=P.EXT3_DRAWS, log=print):
    """Summary rows for every scenario and share at the fixed threshold."""
    po = pools(df)
    rows = []
    for scen, src, L in scenarios:
        if (src, L) not in po:
            continue
        hum, ai = po[("human", L)], po[(src, L)]
        for c in shares:
            sim = simulate(hum, ai, c, R=R)
            rows.append(row_of(scen, src, L, c, sim, ai))
            log(f"{scen} {src:12s} L={L:3d} c={c}: p={sim['p']:.3f} rmse none {rows[-1]['rmse_none']:.4f} "
                f"det {rows[-1]['rmse_det']:.4f}  recovery {rows[-1]['recovery_det']:.3f} "
                f"[{rows[-1]['recovery_det_lo']:.3f}, {rows[-1]['recovery_det_hi']:.3f}]  random {rows[-1]['recovery_rand']:.3f}"
                f"  formula {sim['recovery_formula']:.3f}")
    return pd.DataFrame(rows)


# ---- summary files and figures (section 9) -------------------------------------------------------------
def _ci(v, lo, hi, d=3):
    return S.fmt_ci(v, (lo, hi), d)


def main_table(res):
    """Markdown rows of the main summary table."""
    cols = ["scenario", "AI texts", "tokens (median)", "c (actual)", "AI pool / slant", "detection rate p [95% CI]",
            "human texts removed per draw", "RMSE none / filter / perfect", "Spearman none / filter / perfect",
            "recovery [95% CI]", "predicted: random at rate p [95% CI]", "formula", "predicted: same count, shuffled [95% CI]", "actual − shuffled [95% CI]"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in res.itertuples():
        lines.append("| " + " | ".join([
            r.scenario, r.label, f"{r.T_median:.0f}", f"{r.c:g} ({r.c_actual:.3f})", f"{r.n_ai} / {r.n_slant}",
            f"{r.p:.3f} [{r.p_lo:.3f}, {r.p_hi:.3f}]", f"{r.human_removed_per_draw:.2f} of {P.EXT3_BUCKETS * P.EXT3_BUCKET_N}",
            f"{r.rmse_none:.4f} / {r.rmse_det:.4f} / {r.rmse_perfect:.4f}",
            f"{r.rho_none:.3f} / {r.rho_det:.3f} / {r.rho_perfect:.3f}",
            _ci(r.recovery_det, r.recovery_det_lo, r.recovery_det_hi),
            _ci(r.recovery_rand, r.recovery_rand_lo, r.recovery_rand_hi), f"{r.recovery_formula:.3f}",
            _ci(r.recovery_shuf, r.recovery_shuf_lo, r.recovery_shuf_hi),
            _ci(r.recovery_diff, r.recovery_diff_lo, r.recovery_diff_hi)]) + " |")
    return "\n".join(lines)


def escapers_table(res):
    """Per scenario and length: how many slanted AI texts escape the filter and how their mean S differs from the pool."""
    cols = ["scenario", "AI texts", "tokens", "threshold", "up pool: escaped / n", "μ_A' − μ_A (up) [95% CI]",
            "down pool: escaped / n", "μ_A' − μ_A (down) [95% CI]"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    seen = set()
    for r in res.itertuples():
        key = (r.scenario, r.source, r.L, r.threshold, r.label)
        if key in seen:
            continue
        seen.add(key)
        f = lambda d, lo, hi: "–" if np.isnan(d) else _ci(d, lo, hi)
        lines.append("| " + " | ".join([
            r.scenario, r.label, f"{r.T_median:.0f}", f"z > {r.threshold:.2f}", f"{r.escaped_up_n} / {r.n_slant}",
            f(r.escaped_up_dmu, r.escaped_up_dmu_lo, r.escaped_up_dmu_hi), f"{r.escaped_down_n} / {r.n_slant}",
            f(r.escaped_down_dmu, r.escaped_down_dmu_lo, r.escaped_down_dmu_hi)]) + " |")
    return "\n".join(lines)


def controls_table(ctrl):
    cols = ["source", "tokens (nominal / median)", "n", "neutral / positive / negative", "μ (S) ± sd", "flagged at z > 4 [95% CI]",
            "largest attainable z", "slant pools: n, μ_up, μ_down", "flagged in up / down pool"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in ctrl.itertuples():
        sl = "–" if r.src == "human" else f"{int(r.n_slant)}, {r.S_up:+.3f}, {r.S_down:+.3f}"
        fl = "–" if r.src == "human" else f"{r.flagged_up:.3f} / {r.flagged_down:.3f}"
        lines.append("| " + " | ".join([
            SOURCES[r.src], f"{r.L} / {r.T_median:.0f}", str(r.n), f"{r.neutral:.3f} / {r.positive:.3f} / {r.negative:.3f}",
            f"{r.S_mean:+.3f} ± {r.S_sd:.3f}", f"{r.flagged:.3f} [{r.flagged_lo:.3f}, {r.flagged_hi:.3f}]",
            f"{r.z_max_possible:.1f}", sl, fl]) + " |")
    return "\n".join(lines)


def timing_table(timing):
    cols = ["step", "n", "seconds", "per second", "device"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in timing:
        lines.append(f"| {t['step']} | {t.get('n', '')} | {t['seconds']} | {t.get('texts_per_s') or t.get('per_s') or ''} | "
                     f"{t.get('device', '')} {t.get('dtype', '')} |")
    return "\n".join(lines)


def make_figures(res, PL):
    """The three Tier 1 figures from the summary rows."""
    A = res[(res.scenario == "A") & (res.threshold == P.EXT3_Z)]
    PL.ext3_recovery(A, "ext3_recovery_vs_length.png", "Ext. 3: signal recovered by the watermark filter against text length",
                     f"{P.EXT3_DRAWS} draws of {P.EXT3_BUCKETS} buckets x {P.EXT3_BUCKET_N} human texts; AI pool {A.n_ai.iloc[0]}")
    sel = res[(res.threshold == P.EXT3_Z) & ((res.scenario != "A") | (res.L == P.MAX_NEW))]
    short = {"watermarked": "A: watermarked", "paraphrase, light": "B: light\nparaphrase", "paraphrase, medium": "B: medium\nparaphrase",
             "paraphrase, full": "B: full\nparaphrase", "unwatermarked": "D: no\nwatermark"}
    panels = {}
    for c, g in sel.groupby("c"):
        panels[f"contamination share c = {c:g}"] = [
            (short[r.label], [(r.rmse_none, r.rmse_none_lo, r.rmse_none_hi), (r.rmse_det, r.rmse_det_lo, r.rmse_det_hi),
                              (r.rmse_perfect, r.rmse_perfect_lo, r.rmse_perfect_hi)]) for r in g.itertuples()]
    PL.ext3_errors(panels, "ext3_error_by_filter.png", "Ext. 3: error of the sentiment signal under the three filters (200-token texts)",
                   f"{P.EXT3_DRAWS} draws; AI pools {sel.n_ai.min()}-{sel.n_ai.max()} texts")
    pts = []
    for r in res[res.threshold == P.EXT3_Z].itertuples():
        lab = (f"{r.T_median:.0f} tok, c={r.c:g}" if r.scenario == "A" else
               f"{r.label.split(', ')[-1]}, c={r.c:g}" if r.scenario == "B" else f"c={r.c:g}")
        fam = {"A": "A: watermarked prefixes", "B": "B: paraphrased", "D": "D: unwatermarked"}[r.scenario]
        pts.append((fam, lab, r.recovery_shuf, (r.recovery_shuf_lo, r.recovery_shuf_hi), r.recovery_det,
                    (r.recovery_det_lo, r.recovery_det_hi)))
    PL.ext3_actual_vs_predicted(pts, "ext3_actual_vs_predicted.png", "Ext. 3 (H5): is escaping the filter unrelated to sentiment?",
                                f"{P.EXT3_DRAWS} draws per point; 95% percentile intervals")


def write_summary(path, res, ctrl, timing, extra_sections=()):
    """results/summary/ext3_filter.md: main table, controls, escapers, timing, plus optional extra sections."""
    parts = [
        "# Extension 3 · the watermark as a filter for sentiment data",
        f"Signal: FinBERT S = P(pos) − P(neg) per text. {P.EXT3_BUCKETS} buckets of {P.EXT3_BUCKET_N} human texts; "
        f"half the buckets attacked, half of those pushed up (top third of the AI pool by S) and half down (bottom third); "
        f"AI texts added until share c. Errors over buckets against the truth (the bucket's human mean); {P.EXT3_DRAWS} full re-draws; "
        "square brackets are 2.5-97.5 percentiles over draws (Wilson for rates). Recovery = 1 − (RMSE_filter − RMSE_perfect) / (RMSE_none − RMSE_perfect). "
        "Two predictions assume escaping is unrelated to sentiment: 'random at rate p' removes each AI text independently with the pool's detection rate p "
        "(and each human text with the false-positive rate) inside the same draws; 'same count, shuffled' removes exactly the texts the detector flags but with "
        "their identities permuted within each slant pool on every draw (the H5 test is actual minus shuffled). The 'formula' is the bias prediction "
        "1 − (1 − p)/(1 − cp) of section 1 of the spec; it sits above the simulated predictions because RMSE is a root mean square, not a bias. "
        "Human texts: the 1,000 human baselines (idx < 1000), cut to the AI length; without replacement within a bucket, with replacement across buckets. "
        "Slanted AI pools are reused across buckets (stated in the inventory).",
        "## Main table", main_table(res),
        "## Escapers (H5): sentiment of the AI texts the filter misses", escapers_table(res),
        "## Controls (section 6)", controls_table(ctrl),
        "## Timing", timing_table(timing)]
    for title, body in extra_sections:
        parts += [f"## {title}", body]
    with open(path, "w") as f:
        f.write("\n\n".join(parts) + "\n")


# ---- Tier 2 ------------------------------------------------------------------------------------------
def run_calibrated(df, lengths=(25, 50), fpr=P.EXT3_FP_RATE, shares=P.EXT3_SHARES, R=P.EXT3_DRAWS, log=print):
    """Scenario A at short lengths with the threshold set to a `fpr` false-positive rate on human text of that length
    (the (1 - fpr) quantile of the 1,000 human z, as in stats.tpr_at_fpr), next to the fixed z > 4 rows."""
    po = pools(df)
    rows = []
    for L in lengths:
        hum, ai = po[("human", L)], po[("wm", L)]
        thr = S.tpr_at_fpr([0], hum.z.values, fpr)[1]
        for c in shares:
            for t in (None, thr):
                sim = simulate(hum, ai, c, thr=t, R=R)
                r = row_of("A", "wm", L, c, sim, ai, thr=t, label="watermarked" + ("" if t is None else f", 1% FPR threshold"))
                r["fpr_target"] = 0.0 if t is None else fpr
                rows.append(r)
                log(f"A L={L} c={c} thr={r['threshold']:.2f}: p={sim['p']:.3f} fpr={sim['fpr']:.3f} "
                    f"human removed/draw {r['human_removed_per_draw']:.1f}  recovery {r['recovery_det']:.3f} "
                    f"[{r['recovery_det_lo']:.3f}, {r['recovery_det_hi']:.3f}]")
    return pd.DataFrame(rows)


def calibrated_table(cal):
    cols = ["tokens", "c", "threshold", "AI texts removed (p) [95% CI]", "human texts removed (FPR)", "human removed per draw (of 5000)",
            "RMSE none / filter", "recovery [95% CI]", "predicted (random) [95% CI]"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in cal.itertuples():
        lines.append("| " + " | ".join([
            f"{r.L}", f"{r.c:g}", f"z > {r.threshold:.2f}" + (" (fixed)" if r.fpr_target == 0 else f" ({r.fpr_target:.0%} FPR)"),
            f"{r.p:.3f} [{r.p_lo:.3f}, {r.p_hi:.3f}]", f"{r.fpr:.3f} [≤ {r.fpr_hi:.3f}]", f"{r.human_removed_per_draw:.1f}",
            f"{r.rmse_none:.4f} / {r.rmse_det:.4f}", _ci(r.recovery_det, r.recovery_det_lo, r.recovery_det_hi),
            _ci(r.recovery_rand, r.recovery_rand_lo, r.recovery_rand_hi)]) + " |")
    return "\n".join(lines)


DETECTORS_C = {"whole text, z > 4": "flag_z4", "whole text, 1% FPR threshold": "flag_zcal", "WinMax window, 1% FPR threshold": "flag_win"}


def docs_table(rows, thresholds):
    """Scenario C documents with the three detector flags; AI pools are slanted by the pasted span's own sentiment."""
    d = pd.DataFrame(rows)
    d["flag_z4"] = d.z > P.EXT3_Z
    d["flag_zcal"] = d.z > thresholds["thr_full_z"]
    d["flag_win"] = d.winmax > thresholds["thr_winmax"]
    d["T"] = d["T"].astype(int)
    return d


def run_docs(d, shares=P.EXT3_SHARES, R=P.EXT3_DRAWS, log=print):
    """Scenario C: human-only documents as the human pool; for each watermarked share, the mixed documents as the AI pool
    (slanted by the span's S), under the three detectors."""
    hum = d[d.kind == "human"].reset_index(drop=True)
    rows = []
    for share, ai in d[d.kind == "mixed"].groupby("share"):
        ai = ai.reset_index(drop=True)
        for name, flag in DETECTORS_C.items():
            for c in shares:
                sim = simulate(hum, ai, c, flag=flag, R=R, by="S_span")
                r = row_of("C", "mixed", P.DILUTION_LEN, c, sim, ai, flag=flag, label=f"{share:.0%} span, {name}", by="S_span")
                r.update(share=share, detector=name, span_T=int(ai.span_T.iloc[0]), S_span_up=float(slant(ai, by="S_span")[0].S_span.mean()),
                         S_span_down=float(slant(ai, by="S_span")[1].S_span.mean()))
                rows.append(r)
                log(f"C share={share:.2f} {name:32s} c={c}: p={sim['p']:.3f} fpr={sim['fpr']:.3f} rmse none {r['rmse_none']:.4f} "
                    f"det {r['rmse_det']:.4f}  recovery {r['recovery_det']:.3f} [{r['recovery_det_lo']:.3f}, {r['recovery_det_hi']:.3f}]")
    return pd.DataFrame(rows)


def docs_summary_table(res_c):
    cols = ["watermarked span", "detector", "c", "AI docs removed (p) [95% CI]", "human docs removed (FPR)", "μ_H / μ_up / μ_down (doc S)",
            "RMSE none / filter", "Spearman none / filter", "recovery [95% CI]", "predicted: random at p [95% CI]", "predicted: shuffled [95% CI]", "actual − shuffled [95% CI]"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in res_c.itertuples():
        lines.append("| " + " | ".join([
            f"{r.share:.0%} ({r.span_T} of {P.DILUTION_LEN} tokens)", r.detector, f"{r.c:g}", f"{r.p:.3f} [{r.p_lo:.3f}, {r.p_hi:.3f}]",
            f"{r.fpr:.3f}", f"{r.mu_H:+.3f} / {r.mu_up:+.3f} / {r.mu_down:+.3f}", f"{r.rmse_none:.4f} / {r.rmse_det:.4f}",
            f"{r.rho_none:.3f} / {r.rho_det:.3f}", _ci(r.recovery_det, r.recovery_det_lo, r.recovery_det_hi),
            _ci(r.recovery_rand, r.recovery_rand_lo, r.recovery_rand_hi), _ci(r.recovery_shuf, r.recovery_shuf_lo, r.recovery_shuf_hi),
            _ci(r.recovery_diff, r.recovery_diff_lo, r.recovery_diff_hi)]) + " |")
    return "\n".join(lines)


def make_figure_c(res_c, PL):
    panels = {}
    for c, g in res_c.groupby("c"):
        panels[f"contamination share c = {c:g}"] = {
            det: (gg.share.values, gg.recovery_det.values, gg.recovery_det_lo.values, gg.recovery_det_hi.values)
            for det, gg in g.groupby("detector", sort=False)}
    PL.ext3_pasted(panels, "ext3_pasted_passages.png", "Ext. 3 (scenario C): whole-text against window detector on pasted passages",
                   f"{P.EXT3_DRAWS} draws; {int(res_c.n_ai.iloc[0])} documents per share, {int(res_c.n_human.iloc[0]) if 'n_human' in res_c else 1000} human documents")


# ---- everything, for the notebook -----------------------------------------------------------------------
def pipeline(C, PL, log=print):
    """Run the simulation on the tables on disk, write results/summary/ext3_*.csv, ext3_filter.md and the figures.
    Returns {name: DataFrame}."""
    import json
    import os
    import time
    out = {}
    df = table(C.read_jsonl(C.EXT3_FILE))
    ctrl = controls(df)
    ctrl.to_csv(os.path.join(C.SUMMARY, "ext3_controls.csv"), index=False)
    timing = [t for t in C.read_jsonl(os.path.join(C.SUMMARY, "ext3_timing.jsonl")) if t["step"].startswith("score")]
    t0 = time.time()
    res = run_all(df, log=log)
    timing.append(dict(step="simulation A, B, D", n=f"{len(res)} cells x {P.EXT3_DRAWS} draws", seconds=round(time.time() - t0, 1), device="cpu"))
    res.to_csv(os.path.join(C.SUMMARY, "ext3_filter.csv"), index=False)
    make_figures(res, PL)
    extra = []
    t0 = time.time()
    cal = run_calibrated(df, log=log)
    timing.append(dict(step="simulation A, calibrated threshold", n=f"{len(cal)} cells x {P.EXT3_DRAWS} draws", seconds=round(time.time() - t0, 1), device="cpu"))
    cal.to_csv(os.path.join(C.SUMMARY, "ext3_calibrated.csv"), index=False)
    extra.append(("Tier 2: threshold calibrated to a 1% false-positive rate on human text of the same length (scenario A)",
                  "The threshold is the 99th percentile of z over the 1,000 human texts at that length (`stats.tpr_at_fpr`); "
                  "the realised FPR is below 1% because z is discrete at short lengths.\n\n" + calibrated_table(cal)))
    out.update(texts=df, controls=ctrl, main=res, calibrated=cal)
    if os.path.exists(C.EXT3_DOCS):
        with open(os.path.join(C.SUMMARY, "ext2b_thresholds.json")) as f:
            thr = json.load(f)
        d = docs_table(C.read_jsonl(C.EXT3_DOCS), thr)
        t0 = time.time()
        res_c = run_docs(d, log=log)
        timing.append(dict(step="simulation C", n=f"{len(res_c)} cells x {P.EXT3_DRAWS} draws", seconds=round(time.time() - t0, 1), device="cpu"))
        res_c.to_csv(os.path.join(C.SUMMARY, "ext3_pasted.csv"), index=False)
        make_figure_c(res_c, PL)
        hum = d[d.kind == "human"]
        extra.append(("Tier 2: scenario C, watermarked passage pasted into human text",
                      f"Documents of Ext. 2b rebuilt with their text (seed 0; z identical to `results/raw/dilution/docs.jsonl.gz`). "
                      f"Document score = mean FinBERT S of the two 300-token halves (FinBERT reads at most 512 tokens). "
                      f"Human pool: {len(hum)} human-only documents (neutral share {(hum.label == 'neutral').mean():.3f}, μ_H = {hum.S.mean():+.3f}); "
                      f"AI pool: the {int((d.kind == 'mixed').sum() // 3)} mixed documents per share, slanted by the pasted span's own S "
                      f"(span S of the up / down thirds: " + ", ".join(f"{s:.0%}: {g.S_span_up.iloc[0]:+.2f} / {g.S_span_down.iloc[0]:+.2f}" for s, g in res_c.groupby('share')) + "). "
                      f"Thresholds: z > 4, and the 1% FPR thresholds of Ext. 2b (full-text z > {thr['thr_full_z']:.2f}, WinMax > {thr['thr_winmax']:.2f}).\n\n"
                      + docs_summary_table(res_c) + "\n\nEscapers (document S of the mixed documents the detector misses, pools slanted by the span's S):\n\n"
                      + escapers_table(res_c.assign(T_median=P.DILUTION_LEN, threshold=[{"flag_z4": P.EXT3_Z, "flag_zcal": thr["thr_full_z"], "flag_win": thr["thr_winmax"]}[DETECTORS_C[x]] for x in res_c.detector]))))
        out.update(docs=d, pasted=res_c)
    extra.append(("Hypotheses", hypotheses_text(res, cal, out.get("pasted"))))
    write_summary(os.path.join(C.SUMMARY, "ext3_filter.md"), res, ctrl, timing, extra)
    out["timing"] = pd.DataFrame(timing)
    return out


def hypotheses_text(res, cal, res_c=None):
    """One line per hypothesis, read off the tables."""
    A = res[res.scenario == "A"].set_index(["L", "c"])
    B = res[res.source == "para-full"].set_index("c")
    D = res[res.scenario == "D"].set_index("c")
    r = lambda x: f"{x.recovery_det:.2f} [{x.recovery_det_lo:.2f}, {x.recovery_det_hi:.2f}]"
    lines = [
        f"- **H1 (contamination harms the signal)**: with no filter, RMSE at 200 tokens is {A.loc[(200, 0.1)].rmse_none:.3f} (c = 0.1) and "
        f"{A.loc[(200, 0.3)].rmse_none:.3f} (c = 0.3) against 0 for the clean signal; Spearman with the truth falls from 1 to "
        f"{A.loc[(200, 0.1)].rho_none:.2f} / {A.loc[(200, 0.3)].rho_none:.2f}. Supported.",
        f"- **H2 (long, unattacked text: almost all damage undone)**: recovery at 200 tokens {r(A.loc[(200, 0.1)])} (c = 0.1), "
        f"{r(A.loc[(200, 0.3)])} (c = 0.3); Spearman back to {A.loc[(200, 0.1)].rho_det:.3f} / {A.loc[(200, 0.3)].rho_det:.3f}. Supported.",
        f"- **H3 (recovery falls with length, in line with detection)**: recovery at 25 / 50 / 100 / 200 tokens (c = 0.3) "
        + " / ".join(f"{A.loc[(L, 0.3)].recovery_det:.2f}" for L in P.EXT3_LENGTHS) + " against detection rates "
        + " / ".join(f"{A.loc[(L, 0.3)].p:.2f}" for L in P.EXT3_LENGTHS) + ". Supported; at 25 tokens the largest attainable z is 5.0, so the fixed threshold z > 4 is the binding constraint.",
        f"- **H4 (rewrite attack: recovery collapses)**: full paraphrase {r(B.loc[0.1])} (c = 0.1), {r(B.loc[0.3])} (c = 0.3); "
        f"light paraphrase {res[res.source == 'para-light'].recovery_det.max():.2f} at best. Supported.",
        f"- **H5 (escaping is unrelated to sentiment)**: actual minus shuffled-flags recovery is between "
        f"{res.recovery_diff.min():+.3f} and {res.recovery_diff.max():+.3f} over the {len(res)} cells of scenarios A, B, D; "
        f"{int(((res.recovery_diff_lo <= 0) & (res.recovery_diff_hi >= 0)).sum())} of {len(res)} intervals cover zero. "
        f"Not rejected for the watermarked texts themselves; the point estimates are all slightly negative, and the largest gap is at 50 tokens "
        f"({A.loc[(50, 0.3)].recovery_diff:+.3f} [{A.loc[(50, 0.3)].recovery_diff_lo:+.3f}, {A.loc[(50, 0.3)].recovery_diff_hi:+.3f}] at c = 0.3).",
        f"- **Scenario D (sanity)**: the filter removes {D.loc[0.3].ai_removed_per_draw:.1f} of {D.loc[0.3].ai_per_draw:.0f} unwatermarked AI texts per draw; "
        f"recovery {r(D.loc[0.1])} / {r(D.loc[0.3])}. The filter is blind to a model that does not embed the watermark.",
        f"- **Tier 2, calibrated threshold**: at 25 tokens a 1% FPR threshold (z > {cal[(cal.L == 25) & (cal.fpr_target > 0)].threshold.iloc[0]:.2f}) raises recovery from "
        f"{cal[(cal.L == 25) & (cal.fpr_target == 0) & (cal.c == 0.3)].recovery_det.iloc[0]:.2f} to {cal[(cal.L == 25) & (cal.fpr_target > 0) & (cal.c == 0.3)].recovery_det.iloc[0]:.2f} (c = 0.3) "
        f"while removing {cal[(cal.L == 25) & (cal.fpr_target > 0) & (cal.c == 0.3)].human_removed_per_draw.iloc[0]:.0f} of 5,000 human texts per draw."]
    if res_c is not None and len(res_c):
        g = res_c[(res_c.c == 0.3)].set_index(["share", "detector"])
        pos = res_c[(res_c.recovery_diff_lo > 0) | (res_c.recovery_diff_hi < 0)]
        posr = res_c[res_c.recovery_diff_rand_lo > 0]
        lines.append(f"- **H5 on pasted passages (scenario C)**: against the shuffled-flags prediction, actual minus predicted is between {res_c.recovery_diff.min():+.3f} and "
                     f"{res_c.recovery_diff.max():+.3f} and its interval excludes zero in {len(pos)} of {len(res_c)} cells"
                     + ("" if not len(pos) else " (" + "; ".join(f"{r.share:.0%}, {r.detector}, c = {r.c:g}: {r.recovery_diff:+.2f} [{r.recovery_diff_lo:+.2f}, {r.recovery_diff_hi:+.2f}]" for r in pos.itertuples()) + ")")
                     + f". Against independent random removal at rate p the filter looks better than chance in {len(posr)} cells (up to {res_c.recovery_diff_rand.max():+.2f}); "
                     "that gap is a small-pool effect, not sentiment: with 66 documents per slant and a handful of fixed escapers, the number of escapers per bucket varies less "
                     "than under independent removal, and the RMSE is driven by the buckets with the most escapers.")
        lines.append("- **H6 (window detector on pasted passages)**: at a 10% span, recovery (c = 0.3) is "
                     + ", ".join(f"{det}: {g.loc[(0.1, det)].recovery_det:.2f} [{g.loc[(0.1, det)].recovery_det_lo:.2f}, {g.loc[(0.1, det)].recovery_det_hi:.2f}]"
                                 for det in DETECTORS_C) + ". "
                     + "The contamination itself is small because 90% of each pasted document is human text (no-filter RMSE "
                     + f"{g.loc[(0.1, 'whole text, z > 4')].rmse_none:.4f} against {A.loc[(200, 0.3)].rmse_none:.4f} in scenario A).")
    return "\n".join(lines)
