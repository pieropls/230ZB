"""Slide figures (vector PDF, drawn at the size they occupy on the slide) and the opening-slide token excerpts.

    cd presentation && ../.venv/bin/python make_figures.py

Reads results/summary/*.csv, and results/raw/ through the project's loaders in core.py (read-only); writes figures/*.pdf and
snippets/hook.tex. No statistic is recomputed: every plotted value is a column of a summary file, except the
illustrative push example (computed from the watermark formula) and the hook counts (stored green flags).
"""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUM = os.path.join(ROOT, "results", "summary")
FIG = os.path.join(HERE, "figures")
SNIP = os.path.join(HERE, "snippets")
sys.path.insert(0, ROOT)

# deck palette
BLUE, SIGNAL, GOOD, MUTED, GOLD, INK = "#003262", "#A8322D", "#2E6B4F", "#8A8F98", "#FDB515", "#1C1C1E"
COL = 2.75          # half text column, inches
FULL = 5.7          # text width, inches

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": "#B8BCC2",
    "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "axes.titlelocation": "left", "axes.titlepad": 5,
})


def rows(name):
    with open(os.path.join(SUM, name + ".csv")) as fh:
        return list(csv.DictReader(fh))


def f(x):
    return None if x in (None, "", "nan") else float(x)


def err(v, lo, hi):
    """Asymmetric error bar lengths from a value and its interval."""
    return [[max(0.0, v - lo)], [max(0.0, hi - v)]]


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches=None, metadata={"CreationDate": None})   # deterministic bytes
    plt.close(fig)
    print("wrote figures/" + name)


def paper_marker(ax, x, y, m, label=None):
    ax.plot([x], [y], m, ms=5.5, mfc="none", mec="black", mew=0.9, ls="none", label=label, zorder=5)


# ---------------------------------------------------------------------------------------------- ban and push examples
# Illustrative inputs (not measured): next-word probabilities and colours after two contexts, used by the hard rule
# (Algorithm 1, ban_example) and by the soft watermark (Algorithm 2, push_example).
EXAMPLES = {
    "barack": ("After \"Barack\"", [("Obama", 0.990, False), ("Hussein", 0.004, True), ("and", 0.003, True), ("was", 0.003, False)]),
    "dinner": ("After \"For dinner I had\"", [(w, 1 / 6, True) for w in ("pizza", "rice", "salad")]
               + [(w, 1 / 6, False) for w in ("pasta", "soup", "chicken")]),
}


def ban_example(key):
    """Algorithm 1 on an illustrative context: red words get probability 0, green words share the model's mass."""
    title, words = EXAMPLES[key]
    pg = sum(p for _, p, g in words if g)
    new = [p / pg if g else 0.0 for _, p, g in words]
    n = len(words)
    xmax = 1.0 if key == "barack" else 0.4
    fig, ax = plt.subplots(figsize=(COL, 1.95), gridspec_kw=dict(left=0.21, right=0.97, top=0.75, bottom=0.25))
    y = np.arange(n)[::-1]
    for yi, (w, p, g), q in zip(y, words, new):
        ax.barh(yi + 0.17, p, height=0.22, color=MUTED, alpha=0.55, lw=0)
        if g:
            ax.barh(yi - 0.13, q, height=0.34, color=GOOD, lw=0)
            ax.text(max(p, q), yi, f"  {p:.3f} → {q:.3f}", va="center", fontsize=7.5)
        elif p > xmax / 2:                                             # long bar: label under it, right-aligned
            ax.text(p, yi - 0.17, f"{p:.3f} → 0, banned", ha="right", va="center", fontsize=7.5, color=SIGNAL)
        else:
            ax.text(p, yi, f"  {p:.3f} → 0, banned", va="center", fontsize=7.5, color=SIGNAL)
    ax.set_yticks(y)
    ax.set_yticklabels([w for w, _, _ in words])
    for lab, (_, _, g) in zip(ax.get_yticklabels(), words):
        lab.set_color(GOOD if g else SIGNAL)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax * 1.6)
    ax.set_xticks([0, xmax / 2, xmax])
    ax.set_title(title, pad=15)
    ax.text(0.0, 1.04, f"chance of a green token: {pg:.1%} → 100%".replace(".0%", "%"), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5, color=GOOD)
    h1 = plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.55)
    h2 = plt.Rectangle((0, 0), 1, 1, color=GOOD)
    fig.legend([h1, h2], ["model alone", "with Algorithm 1"], loc="lower center", ncol=2,
               fontsize=7, handlelength=1.0, borderaxespad=0.1, columnspacing=1.2)
    save(fig, f"ban_{key}.pdf")


def push_example():
    """Illustrative: the model's probabilities and the watermarked ones (delta = 2) after the two contexts."""
    a = math.exp(2.0)
    panels = [EXAMPLES["barack"], EXAMPLES["dinner"]]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.95), gridspec_kw=dict(wspace=0.55, left=0.10, right=0.97, top=0.76, bottom=0.21))
    for ax, (title, words) in zip(axes, panels):
        z = sum(p * (a if g else 1) for _, p, g in words)
        new = [p * (a if g else 1) / z for _, p, g in words]
        pg0 = sum(p for _, p, g in words if g)
        pg1 = sum(q for q, (_, _, g) in zip(new, words) if g)
        n = len(words)
        y = np.arange(n)[::-1]
        for yi, (w, p, g), q in zip(y, words, new):
            c = GOOD if g else SIGNAL
            ax.barh(yi + 0.17, p, height=0.22, color=MUTED, alpha=0.55, lw=0)
            ax.barh(yi - 0.13, q, height=0.34, color=c, lw=0)
            ax.text(max(p, q), yi, f"  {p:.3f} → {q:.3f}", va="center", fontsize=7.5)
        ax.set_yticks(y)
        ax.set_yticklabels([w for w, _, _ in words])
        for lab, (_, _, g) in zip(ax.get_yticklabels(), words):
            lab.set_color(GOOD if g else SIGNAL)
        ax.set_title(title, pad=15)
        xmax = 1.0 if n == 4 else 0.36
        ax.set_xlim(0, xmax * 1.55)
        ax.set_xticks([0, xmax / 2, xmax] if n == 6 else [0, 0.5, 1.0])
        ax.tick_params(axis="y", length=0)
        ax.text(0.0, 1.04, f"chance of a green token: {pg0:.1%} → {pg1:.0%}".replace(".0%", "%"),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=GOOD)
    h1 = plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.55)
    h2 = plt.Rectangle((0, 0), 1, 1, color=GOOD)
    h3 = plt.Rectangle((0, 0), 1, 1, color=SIGNAL)
    fig.legend([h1, h2, h3], ["probability, model alone", "green token, with watermark", "red token, with watermark"],
               loc="lower center", ncol=3, fontsize=7, handlelength=1.0, borderaxespad=0.1, columnspacing=1.6)
    save(fig, "push_example.pdf")
    return panels


# ---------------------------------------------------------------------------------------------- z under chance
def z_null():
    """Algorithm 1's test: z of the 1,000 human baselines (gamma 0.5, our key) against the N(0, 1) of chance."""
    import core as C
    import params as P
    hz = np.array([r["z"] for r in C.load_rows(C.raw_path("human", "c4")) if r["idx"] < P.N_GEN_PROMPTS])
    fig, ax = plt.subplots(figsize=(COL, 2.2), gridspec_kw=dict(left=0.16, right=0.97, top=0.88, bottom=0.18))
    ax.hist(hz, bins=np.arange(-4, 8.01, 0.5), density=True, color=MUTED, alpha=0.6, lw=0, label=f"human texts ({len(hz):,})")
    x = np.linspace(-4, 8, 400)
    ax.plot(x, np.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi), color=INK, lw=1.1, label="chance: N(0, 1)")
    ax.axvline(4, color=SIGNAL, lw=1.0, ls=(0, (3, 2)))
    ax.text(4.2, 0.53, "flag: z > 4\np = 3.2 × 10⁻⁵", color=SIGNAL, fontsize=7.5, va="top")
    ax.annotate("attacker,\n200 of 1,000\nedited:\nz = 6.3", xy=(6.3, 0.004), xytext=(6.25, 0.13), ha="center",
                fontsize=7.5, color=BLUE, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9, mutation_scale=8))
    ax.plot([6.3], [0.004], "o", color=BLUE, ms=4, zorder=5)
    ax.set_xlim(-4, 8)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_xlabel("z-score")
    ax.set_ylabel("density")
    ax.set_title("Human texts follow chance")
    ax.legend(loc="upper left", fontsize=7, borderaxespad=0.1)
    save(fig, "z_null.pdf")
    return dict(n=len(hz), above4=int((hz > 4).sum()), mean=float(hz.mean()), sd=float(hz.std()))


# ---------------------------------------------------------------------------------------------- z histogram
def z_hist():
    """Headline config: z of human baselines, delta = 0 generations and watermarked generations (notebook loaders)."""
    import core as C
    import params as P
    human = [r for r in C.load_rows(C.raw_path("human", "c4")) if r["idx"] < P.N_GEN_PROMPTS]
    hz = np.array([r["z"] for r in human])                          # stored z at gamma 0.5 (= notebook's z_ranks)
    d0 = [r for r in C.load_gen("opt-news-m-d0", kept=True) if r["T"] > 0][:500]
    wm = [r for r in C.load_gen("opt-news-m-d2-g0.5", kept=True) if r["T"] > 0][:500]
    d0z, wz = np.array([r["z"] for r in d0]), np.array([r["z"] for r in wm])
    fig, ax = plt.subplots(figsize=(COL + 0.15, 2.25), gridspec_kw=dict(left=0.18, right=0.97, top=0.88, bottom=0.17))
    bins = np.arange(-4, 15.5, 0.5)
    ax.hist(hz, bins=bins, color=MUTED, alpha=0.6, lw=0, label=f"human ({len(hz):,})")
    ax.hist(d0z, bins=bins, histtype="step", color=INK, lw=0.8, label=f"no watermark ({len(d0z)})")
    ax.hist(wz, bins=bins, color=BLUE, alpha=0.9, lw=0, label=f"watermarked ({len(wz)})")
    ax.axvline(4, color=INK, ls=(0, (3, 2)), lw=0.9)
    ax.text(4.25, ax.get_ylim()[1] * 0.55, "z = 4", va="center", fontsize=7.5)
    ax.set_xlabel("z-score")
    ax.set_ylabel("texts")
    ax.set_title("z-scores, 200 tokens")
    ax.legend(loc="upper right", fontsize=7, borderaxespad=0.1)
    save(fig, "z_hist.pdf")
    return dict(n_human=len(hz), n_d0=len(d0z), n_wm=len(wz))


# ---------------------------------------------------------------------------------------------- Table 8 dots
def table8_dots():
    t = rows("table8")
    order = [("m-nom", d, g) for d in (1, 2, 5) for g in (0.5, 0.25)] + [("8-beams", d, g) for d in (1, 2, 5) for g in (0.5, 0.25)]
    by = {(r["sampling"], int(float(r["delta"])), float(r["gamma"])): r for r in t}
    fig, ax = plt.subplots(figsize=(3.05, 2.1), gridspec_kw=dict(left=0.15, right=0.98, top=0.87, bottom=0.295))
    xs = []
    for i, k in enumerate(order):
        x = i + (0.6 if i >= 6 else 0)
        xs.append(x)
        r = by[k]
        for dz, z, m, off in (("z4", "TPR_z4", "o", -0.14), ("z5", "TPR_z5", "s", 0.14)):
            v, lo, hi = f(r[z]), f(r[z + "_lo"]), f(r[z + "_hi"])
            ax.errorbar([x + off], [v], yerr=err(v, lo, hi), fmt=m, ms=3.6, color=BLUE, elinewidth=0.8, capsize=1.5, zorder=4)
            paper_marker(ax, x + off, f(r["paper_TPR_" + dz]), m)
        if r["vs_paper_z5"] == "different":
            ax.axvspan(x - 0.42, x + 0.42, color=GOLD, alpha=0.35, lw=0, zorder=0)
    ax.set_ylim(0.3, 1.04)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([".5" if g == 0.5 else ".25" for _, _, g in order])
    ax.tick_params(axis="x", length=0, pad=2)
    trans = ax.get_xaxis_transform()
    ax.text(xs[0] - 0.95, -0.075, "γ", transform=trans, ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.text(xs[0] - 0.95, -0.17, "δ", transform=trans, ha="center", va="top", fontsize=7.5, color=MUTED)
    for j, d in enumerate((1, 2, 5, 1, 2, 5)):
        xm = (xs[2 * j] + xs[2 * j + 1]) / 2
        ax.text(xm, -0.17, str(d), transform=trans, ha="center", va="top", fontsize=7.5)
    for g0, name in ((0, "multinomial"), (6, "8 beams")):
        xm = (xs[g0] + xs[g0 + 5]) / 2
        ax.text(xm, -0.27, name, transform=trans, ha="center", va="top", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("true positive rate")
    ax.set_title("Table 8: detection rate")
    hz4 = plt.Line2D([], [], marker="o", color=BLUE, ls="none", ms=3.6, label="ours, z = 4")
    hz5 = plt.Line2D([], [], marker="s", color=BLUE, ls="none", ms=3.6, label="ours, z = 5")
    hp = plt.Line2D([], [], marker="o", mfc="none", mec="black", ls="none", ms=5, label="paper")
    ax.legend(handles=[hz4, hz5, hp], loc="lower right", fontsize=7, handletextpad=0.2, borderaxespad=0.1)
    save(fig, "table8_dots.pdf")


# ---------------------------------------------------------------------------------------------- T5 attack
def t5_attack():
    t = [r for r in rows("table9_t5") if r["sampling"] == "m-nom, single-word T5"]
    eps = [f(r["eps"]) for r in t]
    fig, ax = plt.subplots(figsize=(2.45, 2.1), gridspec_kw=dict(left=0.19, right=0.97, top=0.87, bottom=0.19))
    for z, ls, m, lab in (("TPR_z4", "-", "o", "ours, z = 4"), ("TPR_z5", (0, (3, 2)), "s", "ours, z = 5")):
        v = np.array([f(r[z]) for r in t])
        lo = np.array([f(r[z + "_lo"]) for r in t]); hi = np.array([f(r[z + "_hi"]) for r in t])
        ax.errorbar(eps, v, yerr=[np.maximum(0, v - lo), np.maximum(0, hi - v)], fmt=m, ls=ls, ms=3.4, color=BLUE, lw=1.1, elinewidth=0.8, capsize=1.5, label=lab)
    for i, r in enumerate(t):
        paper_marker(ax, eps[i], f(r["paper_TPR_z4"]), "o", "paper" if i == 0 else None)
        paper_marker(ax, eps[i], f(r["paper_TPR_z5"]), "s")
    ax.set_xticks(eps)
    ax.set_xticklabels(["0", "10%", "30%", "50%", "70%"])
    ax.set_xlabel("words swapped (ε)")
    ax.set_ylabel("true positive rate")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("T5 word swaps")
    ax.legend(loc="upper right", fontsize=7, borderaxespad=0.1)
    save(fig, "t5_attack.pdf")


# ---------------------------------------------------------------------------------------------- Ext 1
def tokens_to_detect():
    t = {r["domain"]: r for r in rows("ext1b_cells") if r["model"] == "opt"}
    fig, ax = plt.subplots(figsize=(COL, 2.3), gridspec_kw=dict(left=0.17, right=0.97, top=0.88, bottom=0.15))
    labels = ["news", "code", "short answers"]
    for i, d in enumerate(("news", "code", "qa")):
        r = t[d]
        med, never = f(r["tokens_to_z4_median"]), f(r["never_z4"])
        if math.isinf(med):
            ax.text(i, 12, f"never\n({never:.0%})", ha="center", va="bottom", fontsize=8, color=SIGNAL, fontweight="bold")
            continue
        lo, hi = f(r["tokens_med_lo"]), f(r["tokens_med_hi"])
        top = min(hi, 200) if math.isfinite(hi) else 200
        ax.bar(i, med, width=0.55, color=BLUE, lw=0)
        ax.errorbar([i], [med], yerr=[[med - lo], [top - med]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=0 if not math.isfinite(hi) else 2)
        ax.text(i + 0.3, med, f" {med:.0f}", va="center", ha="left", fontsize=8, fontweight="bold")
        ax.text(i, (top if not math.isfinite(hi) else hi) + 6, f"never: {never:.0%}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(200, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(2.42, 203, "200 tokens", ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 245)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("tokens to reach z = 4 (median)")
    ax.set_title("OPT-1.3B: tokens to detect")
    save(fig, "tokens_to_detect.pdf")


# ---------------------------------------------------------------------------------------------- Ext 2
def removal_frontier():
    t = {(r["attack"], r["level"]): r for r in rows("ext2a_attacks")}
    fig, ax = plt.subplots(figsize=(COL, 2.3), gridspec_kw=dict(left=0.17, right=0.96, top=0.88, bottom=0.17))
    series = [("t5w", ["0.1", "0.3", "0.5", "0.7"], SIGNAL, "D", "T5 word swaps", ["10%", "30%", "50%", "70%"]),
              ("para", ["light", "medium", "full"], BLUE, "s", "Qwen rewrite", ["light", "medium", "full"])]
    offs = {("t5w", "0.1"): (4, -10, "left"), ("t5w", "0.3"): (6, 0, "left"), ("t5w", "0.5"): (-6, 3, "right"),
            ("t5w", "0.7"): (0, 9, "center"), ("para", "light"): (-6, 0, "right"), ("para", "medium"): (-6, 0, "right"),
            ("para", "full"): (-6, -2, "right")}
    for att, levels, c, m, name, labs in series:
        xs, ys = [], []
        for lv, lab in zip(levels, labs):
            r = t[(att, lv)]
            x, y = f(r["meaning_kept"]), f(r["TPR_1pctFPR"])
            xs.append(x); ys.append(y)
            ax.errorbar([x], [y], yerr=err(y, f(r["TPR_1pct_lo"]), f(r["TPR_1pct_hi"])),
                        xerr=err(x, f(r["meaning_lo"]), f(r["meaning_hi"])), fmt=m, ms=4, color=c, elinewidth=0.7, capsize=0, zorder=4)
            dx, dy, ha = offs[(att, lv)]
            ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=7.5, color=c)
        ax.plot(xs, ys, color=c, lw=0.9, zorder=3, label=name, marker=m, ms=4)
    r = t[("none", "0")]
    ax.plot([f(r["meaning_kept"])], [f(r["TPR_1pctFPR"])], "o", color="black", ms=4.5, label="no attack", zorder=5)
    ax.set_xlim(0.69, 1.02)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("meaning kept (MiniLM cosine)")
    ax.set_ylabel("detected at 1% false positives")
    ax.set_title("The price of removal")
    ax.legend(loc="upper left", fontsize=7, borderaxespad=0.1, handlelength=1.6)
    save(fig, "removal_frontier.pdf")


# ---------------------------------------------------------------------------------------------- Ext 3
def dilution():
    t = rows("ext2b_dilution")
    sh = [100 * f(r["share"]) for r in t]
    fig, ax = plt.subplots(figsize=(COL, 2.3), gridspec_kw=dict(left=0.17, right=0.97, top=0.88, bottom=0.17))
    for col, lo, hi, c, name, m in (("TPR_full_z", "full_z_lo", "full_z_hi", MUTED, "whole-text count", "o"),
                                    ("TPR_winmax", "winmax_lo", "winmax_hi", BLUE, "sliding window (WinMax)", "s")):
        v = np.array([f(r[col]) for r in t]); l = np.array([f(r[lo]) for r in t]); h = np.array([f(r[hi]) for r in t])
        ax.errorbar(sh, v, yerr=[np.maximum(0, v - l), np.maximum(0, h - v)], fmt=m + "-", color=c, ms=4, lw=1.2, elinewidth=0.8, capsize=2, label=name)
    v0 = t[0]
    ax.annotate(f"{f(v0['TPR_full_z']):.1%}", (sh[0], f(v0["TPR_full_z"])), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
    ax.annotate(f"{f(v0['TPR_winmax']):.1%}", (sh[0], f(v0["TPR_winmax"])), xytext=(7, -2), textcoords="offset points", va="center", fontsize=7.5, color=BLUE)
    ax.set_xticks(sh)
    ax.set_xticklabels([f"{s:.0f}%" for s in sh])
    ax.set_xlim(5, 55)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("watermarked share of a 600-token text")
    ax.set_ylabel("detected at 1% false positives")
    ax.set_title("A pasted passage")
    ax.legend(loc="lower right", fontsize=7, borderaxespad=0.1)
    save(fig, "dilution.pdf")


# ---------------------------------------------------------------------------------------------- Ext 4
def keys_length():
    t = rows("keys_eos")
    fig, ax = plt.subplots(figsize=(COL, 2.3), gridspec_kw=dict(left=0.40, right=0.95, top=0.86, bottom=0.17))
    labels, colors = [], []
    for r in t:
        if r["hash_key"].startswith("none"):
            labels.append("no watermark"); colors.append(MUTED)
        else:
            ours = r["hash_key"] == "15485863"
            green = r["eos_green_after_dot"] == "True"
            labels.append(f"key {r['hash_key']}{' (ours)' if ours else ''}\nEOS {'green' if green else 'red'} after '.'")
            colors.append(SIGNAL if ours else BLUE)
    y = np.arange(len(t))[::-1]
    for yi, r, c in zip(y, t, colors):
        v, lo, hi = f(r["kept_share"]), f(r["kept_lo"]), f(r["kept_hi"])
        ax.barh(yi, v, height=0.58, color=c, lw=0)
        ax.errorbar([v], [yi], xerr=err(v, lo, hi), fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8)
        ax.text(hi + 0.02, yi, f"{v:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("share reaching 195 tokens")
    fig.text(0.02, 0.965, "Texts that run to full length", ha="left", va="top", fontsize=9, fontweight="bold")
    save(fig, "keys_length.pdf")


# ---------------------------------------------------------------------------------------------- backup: trade-off
def tradeoff():
    t = rows("fig2_tradeoff")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.0), gridspec_kw=dict(wspace=0.32, left=0.08, right=0.98, top=0.865, bottom=0.2))
    ax = axes[0]
    gammas = [0.1, 0.25, 0.5, 0.75, 0.9]
    shades = [BLUE, "#2F5F8A", "#5E87AD", "#93AFC9", "#C3D2E0"]
    for g, c in zip(gammas, shades):
        pts = sorted([(f(r["delta"]), f(r["PPL_mean"]), f(r["z_mean"])) for r in t if r["panel"] == "multinomial" and f(r["gamma"]) == g])
        ax.plot([p[1] for p in pts], [p[2] for p in pts], "o-", color=c, ms=3, lw=1.1, label=f"γ = {g:g}")
        if g == 0.1:
            for d, ppl, z in pts:
                ax.annotate(f"δ={d:g}", (ppl, z), xytext=(4, -3) if d < 5 else (4, 2), textcoords="offset points", fontsize=7, color=INK)
    ax.invert_xaxis()
    ax.set_xlabel("oracle perplexity (lower is better)")
    ax.set_ylabel("mean z")
    ax.set_title("Multinomial sampling")
    ax.legend(loc="lower left", fontsize=7, borderaxespad=0.1, handlelength=1.4)
    ax = axes[1]
    for name, c, m in (("greedy", MUTED, "o"), ("4 beams", GOLD, "D"), ("8 beams", BLUE, "s")):
        pts = sorted([(f(r["delta"]), f(r["PPL_mean"]), f(r["z_mean"])) for r in t if r["panel"] == name])
        lab = name + (" (δ = 2 only)" if len(pts) == 1 else "")                     # 4 beams was run at delta = 2 only
        ax.plot([p[1] for p in pts], [p[2] for p in pts], m + "-", color=c, ms=3.2, lw=1.1, label=lab,
                mec=INK if c == GOLD else c, mew=0.6)
    ax.invert_xaxis()
    ax.set_xlabel("oracle perplexity (lower is better)")
    ax.set_title("Greedy and beams, γ = 0.5")
    ax.legend(loc="lower left", fontsize=7, borderaxespad=0.1)
    save(fig, "tradeoff.pdf")


# ---------------------------------------------------------------------------------------------- backup: Thm 4.2
def theory_bound():
    t = rows("fig7_theory")
    d = [f(r["delta"]) for r in t]
    fig, ax = plt.subplots(figsize=(COL + 0.25, 2.3), gridspec_kw=dict(left=0.16, right=0.97, top=0.88, bottom=0.17))
    ax.fill_between(d, [f(r["obs_q25"]) for r in t], [f(r["obs_q75"]) for r in t], color=BLUE, alpha=0.15, lw=0, label="ours, 25th to 75th pct")
    ax.plot(d, [f(r["obs_mean"]) for r in t], "o-", color=BLUE, ms=3.2, lw=1.2, label="ours, observed")
    ax.plot(d, [f(r["bound_paper"]) for r in t], "^--", color=MUTED, ms=3.2, lw=1.0, label="bound, paper's version")
    ax.plot(d, [f(r["bound_corrected"]) for r in t], "v:", color=SIGNAL, ms=3.2, lw=1.1, label="bound, sampled distribution")
    pd_ = [0, 0.5, 1, 2, 5, 10]
    for x, y in zip(pd_, [0.53, 0.60, 0.68, 0.80, 0.94, 0.99]):
        paper_marker(ax, x, y, "o", "paper, observed (read off Fig. 7)" if x == 0 else None)
    ax.set_xlabel("δ")
    ax.set_ylabel("green fraction")
    ax.set_ylim(0.22, 1.03)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title("Theorem 4.2 at γ = 0.5")
    ax.legend(loc="lower right", fontsize=6.6, borderaxespad=0.1, handlelength=1.6)
    save(fig, "theory_bound.pdf")


# ---------------------------------------------------------------------------------------------- backup: Fig. 3
def z_vs_T():
    t = rows("fig3_z_vs_T")
    panels = list(dict.fromkeys(r["panel"] for r in t))
    shades = [BLUE, "#2F5F8A", "#5E87AD", "#93AFC9", "#C3D2E0"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0), sharey=True,
                             gridspec_kw=dict(wspace=0.10, left=0.075, right=0.99, top=0.865, bottom=0.2))
    for ax, p in zip(axes, panels):
        series = list(dict.fromkeys(r["series"] for r in t if r["panel"] == p))
        for s_, c in zip(series, shades):
            pts = [(f(r["T"]), f(r["z_mean"])) for r in t if r["panel"] == p and r["series"] == s_]
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=c, lw=1.2, label=s_)
        ax.axhline(4, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(p.replace("multinomial", "sampling"), fontsize=8.5)
        ax.set_xlabel("tokens")
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.legend(loc="upper left", fontsize=6.8, borderaxespad=0.1, handlelength=1.2, labelspacing=0.25)
    axes[0].set_ylabel("mean z")
    save(fig, "z_vs_T.pdf")


# ---------------------------------------------------------------------------------------------- backup: Ext 2 at z = 4
def removal_frontier_z4():
    t = {(r["attack"], r["level"]): r for r in rows("ext2a_attacks")}
    fig, ax = plt.subplots(figsize=(COL, 2.3), gridspec_kw=dict(left=0.17, right=0.96, top=0.88, bottom=0.17))
    series = [("t5w", ["0.1", "0.3", "0.5", "0.7"], SIGNAL, "D", "T5 word swaps", ["10%", "30%", "50%", "70%"]),
              ("para", ["light", "medium", "full"], BLUE, "s", "Qwen rewrite", ["light", "medium", "full"])]
    offs = {("t5w", "0.1"): (4, -10, "left"), ("t5w", "0.3"): (6, 2, "left"), ("t5w", "0.5"): (0, 9, "center"),
            ("t5w", "0.7"): (0, 9, "center"), ("para", "light"): (0, 10, "center"), ("para", "medium"): (6, 3, "left"),
            ("para", "full"): (-6, 0, "right")}
    for att, levels, c, m, name, labs in series:
        xs, ys = [], []
        for lv, lab in zip(levels, labs):
            r = t[(att, lv)]
            x, y = f(r["meaning_kept"]), f(r["TPR_z4"])
            xs.append(x); ys.append(y)
            ax.errorbar([x], [y], yerr=err(y, f(r["TPR_z4_lo"]), f(r["TPR_z4_hi"])),
                        xerr=err(x, f(r["meaning_lo"]), f(r["meaning_hi"])), fmt=m, ms=4, color=c, elinewidth=0.7, capsize=0, zorder=4)
            dx, dy, ha = offs[(att, lv)]
            ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=7.5, color=c)
        ax.plot(xs, ys, color=c, lw=0.9, zorder=3, label=name, marker=m, ms=4)
    r = t[("none", "0")]
    ax.plot([f(r["meaning_kept"])], [f(r["TPR_z4"])], "o", color="black", ms=4.5, label="no attack", zorder=5)
    ax.set_xlim(0.69, 1.02)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("meaning kept (MiniLM cosine)")
    ax.set_ylabel("detected at z = 4")
    ax.set_title("At the paper's threshold")
    ax.legend(loc="upper left", fontsize=7, borderaxespad=0.1, handlelength=1.6)
    save(fig, "removal_frontier_z4.pdf")


# ---------------------------------------------------------------------------------------------- backup: Ext 1c entropy
def instruct_entropy():
    """Per-token Shannon entropy, Qwen base vs Instruct, news, delta = 0 (stored per-token H; the mean is ext1b_cells H_nowm)."""
    import core as C
    cells = {r["model"]: r for r in rows("ext1b_cells") if r["domain"] == "news" and f(r["delta"]) == 2}
    fig, ax = plt.subplots(figsize=(COL, 2.3), gridspec_kw=dict(left=0.17, right=0.97, top=0.88, bottom=0.17))
    bins = np.linspace(0, 8, 41)
    out = {}
    for cid, model, c, name, style in (("qwen-news-m-d0", "qwen", MUTED, "base", dict(alpha=0.6, lw=0)),
                                       ("qwen-it-news-m-d0", "qwen-it", BLUE, "Instruct", dict(histtype="step", lw=1.3))):
        h = np.concatenate([r["H"] for r in C.load_gen(cid) if r["T"] > 0])
        out[name] = (len(h), round(float(h.mean()), 3), cells[model]["H_nowm"])
        ax.hist(h, bins=bins, density=True, color=c, label=f"{name}, mean {f(cells[model]['H_nowm']):.2f}", **style)
    ax.set_xlabel("next-token entropy (nats)")
    ax.set_ylabel("share of tokens (density)")
    ax.set_title("Qwen base vs Instruct, news")
    ax.legend(loc="upper right", fontsize=7, borderaxespad=0.1)
    save(fig, "instruct_entropy.pdf")
    return out


# ---------------------------------------------------------------------------------------------- opening emails
EMAIL_PROMPT = ("Write a short email from Piero, an MFE student at UC Berkeley, to his professor, Ali Kakhbod. "
                "Points to cover: Piero and his teammates Romain and Elias chose the LLM watermarking paper for their fall project; "
                "they would like 20 minutes on Zoom this week to go over their plan; Thursday or Friday works for them. "
                "Write it like a real student writing quickly: plain and polite, two or three sentences, no exclamation marks, "
                "no subject line, no 'I hope this email finds you well', no 'looking forward'. "
                "Start with 'Hi Professor Kakhbod,' and end with 'Thanks,' and 'Piero' on separate lines.")
EMAIL_HUMAN = ("Hi Professor Kakhbod,\n\n"
               "Romain, Elias and I settled on the LLM watermarking paper for our fall project. Would you have 20 minutes "
               "on Zoom this week to go over our plan? We're flexible on Thursday and Friday.\n\n"
               "Thanks,\nPiero")                                              # the human email: replace with your own text
EMAIL_N, EMAIL_PICK, EMAIL_SEED, EMAIL_GAMMA, EMAIL_DELTA = 64, 44, 2, 0.25, 2.0   # sample 44: chosen with Piero
EMAIL_CACHE = os.path.join(SNIP, "emails.json")


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", '"': "''", "<": r"\textless{}", ">": r"\textgreater{}"}
    return "".join(rep.get(c, c) for c in s)


def render_tokens(pieces, flags):
    """Coloured token boxes in one paragraph (slide 5): a token with a leading space starts a new word, sub-word
    pieces attach without space; a line break shows as a small coloured arrow, since the detector counts it too."""
    out, gap = [], False
    for p, g in zip(pieces, flags):
        mac = r"\gtok" if g else r"\rtok"
        body = tex_escape(p.replace("\n", " ").strip())
        if body:
            if p.startswith(" ") and body.startswith("''"):
                body = "``" + body[2:]                                 # opening double quote
            out.append((" " if (p.startswith(" ") or gap) and out else "") + mac + "{" + body + "}")
            gap = False
        if "\n" in p:
            out.append(mac + r"{\ensuremath{\hookleftarrow}}")
            gap = True
    return "".join(out)


def plain_text(text):
    """The email as ordinary text (slide 2): escaped, paragraphs separated by a small gap, line breaks kept."""
    def esc(line):
        line = tex_escape(line.strip())
        line = "``" + line[2:] if line.startswith("''") else line
        return line.replace(" ''", " ``")                             # opening double quotes
    paras = [[esc(l) for l in p.split("\n") if l.strip()] for p in text.split("\n\n")]
    return r"\par\vspace{0.35em}".join(r"\newline ".join(p) for p in paras if p)


def generate_emails():
    """Qwen2.5-1.5B-Instruct writes EMAIL_N emails with the watermark (gamma 0.25, delta 2, temperature 0.7, top_k 0).
    Run once; the token ids are cached in snippets/emails.json."""
    import torch
    from transformers import GenerationConfig, LogitsProcessorList
    import core as C
    import params as P
    model, tok = C.load_lm("qwen-it")
    V, eos, pids = C.vocab_size(tok), C.eos_ids(model, tok), C.chat_ids(tok, EMAIL_PROMPT)
    proc = C.KGW(V, EMAIL_GAMMA, EMAIL_DELTA, tau=P.TEMP, log=False)
    cfg = GenerationConfig(max_new_tokens=120, do_sample=True, temperature=P.TEMP, top_k=0, top_p=1.0,
                           eos_token_id=eos, pad_token_id=tok.pad_token_id)
    torch.manual_seed(EMAIL_SEED)
    rows = []
    for _ in range(EMAIL_N // 32):                                     # batches of 32 (memory)
        ids = torch.tensor([pids] * 32)
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(C.DEVICE), attention_mask=torch.ones_like(ids).to(C.DEVICE),
                                 generation_config=cfg, logits_processor=LogitsProcessorList([proc]))
        for seq in out[:, len(pids):].cpu().tolist():
            T = next((t for t, x in enumerate(seq) if x in eos), len(seq))
            rows.append(dict(ids=seq[:T], ended=T < len(seq)))
    data = dict(model=P.MODELS["qwen-it"], prompt=EMAIL_PROMPT, prompt_ids=pids, gamma=EMAIL_GAMMA, delta=EMAIL_DELTA,
                temperature=P.TEMP, top_k=0, seed=EMAIL_SEED, rows=rows)
    with open(EMAIL_CACHE, "w") as fh:
        json.dump(data, fh)
    return data


def odds(z):
    """'1 in N' for the one-sided p-value of z."""
    n = 1 / (0.5 * math.erfc(z / math.sqrt(2)))
    return f"1 in {n / 1e6:.0f} million" if n >= 1e6 else f"1 in {n:,.0f}"


def emails():
    """My email (EMAIL_HUMAN) and watermarked sample EMAIL_PICK, scored by the detector with the Qwen tokenizer and our
    key (gamma 0.25); the first token's context is the last token of the chat prompt, for both."""
    from transformers import AutoTokenizer
    import core as C
    import params as P
    data = json.load(open(EMAIL_CACHE)) if os.path.exists(EMAIL_CACHE) else generate_emails()
    tok = AutoTokenizer.from_pretrained(P.MODELS["qwen-it"])
    V, prev0 = C.vocab_size(tok), data["prompt_ids"][-1]
    zs = np.array([C.detect(r["ids"], prev0, V, EMAIL_GAMMA)["z"] for r in data["rows"]])
    out = {}
    for name, ids in (("Human", tok(EMAIL_HUMAN, add_special_tokens=False).input_ids),
                      ("AI", data["rows"][EMAIL_PICK]["ids"])):
        d = C.detect(ids, prev0, V, EMAIL_GAMMA)
        pieces = [tok.decode([t]) for t in ids]
        text = tok.decode(ids)
        assert text.isascii() and "".join(pieces) == text, text
        T, G = len(ids), int(sum(d["green"]))
        out[name] = dict(plain=plain_text(text), color=render_tokens(pieces, d["green"]),
                         T=T, G=G, E=EMAIL_GAMMA * T, z=d["z"], text=text)
    os.makedirs(SNIP, exist_ok=True)
    with open(os.path.join(SNIP, "emails.tex"), "w") as fh:
        fh.write("%% generated by make_figures.py: AI email = sample %d of %d from %s with the watermark (gamma %.2f, "
                 "delta %g, temperature 0.7), snippets/emails.json; human email = EMAIL_HUMAN in make_figures.py\n"
                 % (EMAIL_PICK, EMAIL_N, data["model"], EMAIL_GAMMA, EMAIL_DELTA))
        for name, d in out.items():
            fh.write("\\newcommand{\\email%sPlain}{%s}\n" % (name, d["plain"]))
            fh.write("\\newcommand{\\email%sColor}{%s}\n" % (name, d["color"]))
            fh.write("\\newcommand{\\email%sTokens}{%d}\n\\newcommand{\\email%sGreen}{%d}\n" % (name, d["T"], name, d["G"]))
            fh.write("\\newcommand{\\email%sExpected}{%.0f}\n\\newcommand{\\email%sZ}{%.1f}\n" % (name, d["E"], name, d["z"]))
        fh.write("\\newcommand{\\emailAIOdds}{%s}\n" % odds(out["AI"]["z"]))
        fh.write("\\newcommand{\\emailSamples}{%d}\n\\newcommand{\\emailSamplesAbove}{%d}\n\\newcommand{\\emailSamplesMedian}{%.1f}\n"
                 % (len(zs), int((zs > 4).sum()), float(np.median(zs))))
    print("wrote snippets/emails.tex:", {n: (d["T"], d["G"], d["E"], round(d["z"], 2)) for n, d in out.items()},
          "samples above 4:", int((zs > 4).sum()), "median z:", round(float(np.median(zs)), 2), "AI odds:", odds(out["AI"]["z"]))
    return out


if __name__ == "__main__":
    ban_example("dinner")
    ban_example("barack")
    print("z_null:", z_null())
    push_example()
    z_hist()
    table8_dots()
    t5_attack()
    tokens_to_detect()
    removal_frontier()
    dilution()
    keys_length()
    tradeoff()
    theory_bound()
    z_vs_T()
    removal_frontier_z4()
    print("instruct entropy (tokens, mean, csv H_nowm):", instruct_entropy())
    emails()
