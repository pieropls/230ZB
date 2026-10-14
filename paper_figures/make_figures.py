"""Title-free vector figures for the written report (report/main.tex), drawn from results/summary and results/raw.

    cd paper_figures && ../.venv/bin/python make_figures.py

Reuses the slide-figure functions of presentation/make_figures.py with every in-axes title removed
(the LaTeX caption carries the title) and writes the PDFs next to this script. The z histogram, the
four-key length figure and the Extension 5 recovery figure are drawn here from the raw and summary files.
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = HERE
sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import importlib.util
_spec = importlib.util.spec_from_file_location("slide_figures", os.path.join(REPO, "presentation", "make_figures.py"))
mf = importlib.util.module_from_spec(_spec)    # the slide-figure module (same file name as this script)
_spec.loader.exec_module(mf)

mf.FIG = OUT                                   # make_figures.save() writes here
plt.Axes.set_title = lambda self, *a, **k: None  # no title inside any image
plt.rcParams["figure.dpi"] = 200


def raw(exp, name):
    with gzip.open(os.path.join(REPO, "results", "raw", exp, f"{name}.jsonl.gz"), "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def z_hist():
    """Headline setting: z of 1,000 human texts, 500 unwatermarked and 497 watermarked generations, line at z = 4."""
    hz = np.array([r["z"] for r in raw("human", "c4") if r["idx"] < 1000])
    kept = lambda rows: [r for r in sorted(rows, key=lambda r: r["idx"]) if 195 <= r["T"] <= 205 and r["T"] > 0]
    d0z = np.array([r["z"] for r in kept(raw("gen", "opt-news-m-d0"))][:500])
    wz = np.array([r["z"] for r in kept(raw("gen", "opt-news-m-d2-g0.5"))][:500])
    fig, ax = plt.subplots(figsize=(mf.COL + 0.15, 2.25), gridspec_kw=dict(left=0.18, right=0.97, top=0.95, bottom=0.17))
    bins = np.arange(-4, 15.5, 0.5)
    ax.hist(hz, bins=bins, color=mf.MUTED, alpha=0.6, lw=0, label=f"human ({len(hz):,})")
    ax.hist(d0z, bins=bins, histtype="step", color=mf.INK, lw=0.8, label=f"no watermark ({len(d0z)})")
    ax.hist(wz, bins=bins, color=mf.BLUE, alpha=0.9, lw=0, label=f"watermarked ({len(wz)})")
    ax.axvline(4, color=mf.INK, ls=(0, (3, 2)), lw=0.9)
    ax.text(4.25, ax.get_ylim()[1] * 0.55, "z = 4", va="center", fontsize=7.5)
    ax.set_xlabel("z-score")
    ax.set_ylabel("texts")
    ax.legend(loc="upper right", fontsize=7, borderaxespad=0.1)
    mf.save(fig, "z_hist.pdf")
    return dict(n_human=len(hz), n_d0=len(d0z), n_wm=len(wz))


def keys_eos():
    """Length side effect under four hash keys (gamma 0.5, delta 2, first 100 prompts): two panels."""
    t = mf.rows("keys_eos")
    labels = [("no watermark" if r["hash_key"].startswith("none") else r["hash_key"]) for r in t]
    fig, axes = plt.subplots(1, 2, figsize=(mf.FULL, 2.1), gridspec_kw=dict(wspace=0.35, left=0.09, right=0.98, top=0.95, bottom=0.26))
    for ax, (col, lo, hi, ylab) in zip(axes, (("kept_share", "kept_lo", "kept_hi", "share reaching 195 tokens"),
                                             ("ended_before_50", "ended_lo", "ended_hi", "share stopping before 50 tokens"))):
        for i, r in enumerate(t):
            v = mf.f(r[col])
            ax.bar(i, v, width=0.6, color=mf.MUTED if labels[i] == "no watermark" else mf.BLUE, lw=0)
            ax.errorbar([i], [v], yerr=mf.err(v, mf.f(r[lo]), mf.f(r[hi])), fmt="none", ecolor=mf.INK, elinewidth=0.8, capsize=2)
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=6.5)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.set_ylim(0, 1.0 if col == "kept_share" else 0.25)
    mf.save(fig, "keys_eos.pdf")


def tokens_to_detect_all():
    """Median tokens to reach z = 4 for every (domain, model) cell of Extension 1, bootstrap intervals."""
    t = [r for r in mf.rows("ext1b_cells") if r["domain"] in ("news", "code", "qa")]
    models = {"opt": "OPT-1.3B", "qwen": "Qwen2.5-1.5B", "qwen-it": "Qwen2.5-1.5B-Instruct"}
    doms = {"news": "news", "code": "code", "qa": "short answers"}
    fig, ax = plt.subplots(figsize=(mf.FULL, 2.4), gridspec_kw=dict(left=0.09, right=0.99, top=0.95, bottom=0.2))
    width, cols = 0.26, {"opt": mf.BLUE, "qwen": "#5E87AD", "qwen-it": "#C3D2E0"}
    for j, (m, mname) in enumerate(models.items()):
        for i, d in enumerate(doms):
            r = next(r for r in t if r["domain"] == d and r["model"] == m)
            x = i + (j - 1) * width
            med, lo, hi, never = mf.f(r["tokens_to_z4_median"]), mf.f(r["tokens_med_lo"]), mf.f(r["tokens_med_hi"]), mf.f(r["never_z4"])
            if np.isfinite(med):
                top = min(hi, 200) if np.isfinite(hi) else 200
                ax.bar(x, med, width=width * 0.9, color=cols[m], lw=0, label=mname if i == 0 else None)
                ax.errorbar([x], [med], yerr=[[med - lo], [top - med]], fmt="none", ecolor=mf.INK, elinewidth=0.8,
                            capsize=0 if not np.isfinite(hi) else 2)
                ax.text(x, top + 6, f"{med:.0f}\nnever {never:.0%}", ha="center", va="bottom", fontsize=6, linespacing=1.0)
            else:
                ax.bar(x, 0, width=width * 0.9, color=cols[m], lw=0, label=mname if i == 0 else None)
                ax.text(x, 6, f"never\n({never:.0%})", ha="center", va="bottom", fontsize=6, color=mf.SIGNAL, linespacing=1.0)
    ax.axhline(200, color=mf.MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(doms.values()))
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 250)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_ylabel("tokens to reach z = 4 (median)", fontsize=7.5)
    ax.legend(loc="upper right", fontsize=7, ncol=3, borderaxespad=0.1)
    mf.save(fig, "tokens_to_detect_all.pdf")


def ext5_recovery():
    """Extension 5, scenario A (results/summary/ext3_filter.csv). Left: recovery of the sentiment signal against text
    length at z > 4 for c = 0.1 and 0.3, with the H5 prediction (same number of AI texts removed, identities shuffled).
    Right: the detection rate, i.e. the share of AI texts the filter removes, with the largest attainable z."""
    t = [r for r in mf.rows("ext3_filter") if r["scenario"] == "A"]
    Ls = sorted({int(float(r["L"])) for r in t})
    x = np.arange(len(Ls))
    fig, axes = plt.subplots(1, 2, figsize=(mf.FULL, 2.3), gridspec_kw=dict(wspace=0.38, left=0.09, right=0.98, top=0.95, bottom=0.2))
    ax = axes[0]
    for c, col, m in ((0.1, "#5E87AD", "o"), (0.3, mf.BLUE, "s")):
        g = sorted([r for r in t if abs(float(r["c"]) - c) < 1e-9], key=lambda r: float(r["L"]))
        v = np.array([mf.f(r["recovery_det"]) for r in g]); lo = np.array([mf.f(r["recovery_det_lo"]) for r in g]); hi = np.array([mf.f(r["recovery_det_hi"]) for r in g])
        ax.errorbar(x, v, yerr=[np.maximum(0, v - lo), np.maximum(0, hi - v)], fmt=m + "-", color=col, ms=4, lw=1.2, elinewidth=0.8, capsize=2,
                    label=f"watermark filter, c = {c:g}", zorder=4)
        ax.plot(x, [mf.f(r["recovery_shuf"]) for r in g], ls=(0, (3, 2)), color=col, lw=0.9, label=f"predicted from p, c = {c:g}", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("text length (tokens)")
    ax.set_ylabel("recovery of the signal")
    ax.legend(loc="lower right", fontsize=6.5, borderaxespad=0.1, handlelength=1.8)
    ax = axes[1]
    g = {int(float(r["L"])): r for r in t}
    v = np.array([mf.f(g[L]["p"]) for L in Ls]); lo = np.array([mf.f(g[L]["p_lo"]) for L in Ls]); hi = np.array([mf.f(g[L]["p_hi"]) for L in Ls])
    ax.errorbar(x, v, yerr=[np.maximum(0, v - lo), np.maximum(0, hi - v)], fmt="o-", color=mf.INK, ms=4, lw=1.2, elinewidth=0.8, capsize=2, zorder=4)
    for i, L in enumerate(Ls):
        last = i == len(Ls) - 1
        ax.annotate(f"max z = {mf.f(g[L]['z_max_possible']):.1f}", (i, v[i]), xytext=(-5 if last else 5, -22 if last else -9), textcoords="offset points",
                    ha="right" if last else "left", fontsize=6.5, color=mf.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("text length (tokens)")
    ax.set_ylabel("share of AI texts removed (z > 4)")
    mf.save(fig, "ext5_recovery.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("z_hist:", z_hist())
    mf.table8_dots_z4()
    mf.t5_attack_z4()
    mf.tokens_to_detect()
    tokens_to_detect_all()
    mf.removal_frontier()
    mf.dilution()
    keys_eos()
    ext5_recovery()
