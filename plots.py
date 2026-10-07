"""One function per figure. matplotlib defaults, bold titles, PNG at dpi 200 in figures/."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]     # fixed order; colour follows the entity


def _save(fig, name):
    """Write figures/<name> at dpi 200 and close the figure."""
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)


def _note(ax, text):
    """Small grey note in the lower right corner (sample sizes)."""
    ax.text(0.99, 0.01, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.4")


def z_hist(groups, name, title):
    """groups: {label: z array}. Histogram of z-scores with the z = 4 threshold."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bins = np.linspace(min(np.min(z) for z in groups.values()) - 0.5, max(np.max(z) for z in groups.values()) + 0.5, 60)
    for i, (label, z) in enumerate(groups.items()):
        ax.hist(z, bins=bins, alpha=0.55, color=COLORS[i], label=f"{label} (n={len(z)})")
    ax.axvline(4, color="k", ls="--", lw=1, label="z = 4")
    ax.set_xlabel("z-score")
    ax.set_ylabel("sequences")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    _save(fig, name)


def roc_curves(curves, name, title):
    """curves: {label: (fpr, tpr, auc)}. Linear and log FPR axes side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, log in zip(axes, (False, True)):
        for i, (label, (fpr, tpr, a)) in enumerate(curves.items()):
            ax.plot(np.maximum(fpr, 1e-4) if log else fpr, tpr, color=COLORS[i % 10], lw=1.5,
                    label=f"{label}: ours {a:.3f}")
        ax.plot([1e-4 if log else 0, 1], [1e-4 if log else 0, 1], color="0.6", lw=1, ls=":")
        if log:
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 1)
        ax.set_xlabel("false positive rate" + (" (log)" if log else ""))
        ax.set_ylabel("true positive rate")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontweight="bold")
    _save(fig, name)


def tradeoff(left, right, name, n_note):
    """Fig. 2. left: [(gamma, delta, mean z, ppl)], right: [(label, delta, mean z, ppl)].
    δ is written once per panel, along the curve that spreads the most (same δ sequence on every curve)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [(axes[0], left, "Multinomial sampling", lambda r: f"γ = {r[0]}"),
              (axes[1], right, "Greedy and beam search, γ = 0.5", lambda r: r[0])]
    for ax, rows, title, lab in panels:
        groups = list(dict.fromkeys(r[0] for r in rows))
        curves = [sorted([r for r in rows if r[0] == k], key=lambda r: r[1]) for k in groups]
        spread = [max(r[2] for r in c) - min(r[2] for r in c) for c in curves]
        for i, pts in enumerate(curves):
            ax.plot([r[3] for r in pts], [r[2] for r in pts], "o-", color=COLORS[i], lw=1.5, ms=5, label=lab(pts[0]))
            if i == int(np.argmax(spread)):
                for r in pts:
                    ax.annotate(f"δ={r[1]:g}", (r[3], r[2]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_title(title, fontweight="bold")
        ax.invert_xaxis()
        ax.set_xlabel("oracle perplexity (OPT-2.7B), better →")
        ax.set_ylabel("mean z-score")
        ax.legend(fontsize=8)
        _note(ax, n_note + "; δ grows along each curve as labelled")
    fig.suptitle("Fig. 2: watermark strength vs text quality (T = 200)", fontweight="bold")
    _save(fig, name)


def z_vs_T(panels, name, n_note):
    """Fig. 3. panels: [(title, {label: (Ts, mean z)})]."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4), squeeze=False)
    for ax, (title, series) in zip(axes[0], panels):
        for i, (label, (Ts, z)) in enumerate(series.items()):
            ax.plot(Ts, z, color=COLORS[i], lw=1.5, label=label)
        ax.axhline(4, color="k", ls="--", lw=1)
        ax.set_xlabel("tokens T")
        ax.set_ylabel("mean z-score")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
        _note(ax, n_note)
    _save(fig, name)


def t5_attack(ours, paper, name, title):
    """Fig. 5 / Tab. 9. ours / paper: {sampling: [(eps, TPR z=4, TPR z=5)]}; paper drawn as hollow markers."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, (s, pts) in enumerate(ours.items()):
        e = [p[0] for p in pts]
        ax.plot(e, [p[1] for p in pts], "o-", color=COLORS[i], lw=1.5, label=f"ours {s}, z = 4")
        ax.plot(e, [p[2] for p in pts], "s--", color=COLORS[i], lw=1, label=f"ours {s}, z = 5")
    for s, pts in paper.items():
        e = [p[0] for p in pts]
        ax.plot(e, [p[1] for p in pts], "o", mfc="none", mec="k", ms=9, ls="none", label=f"{s}, z = 4")
        ax.plot(e, [p[2] for p in pts], "s", mfc="none", mec="k", ms=9, ls="none", label=f"{s}, z = 5")
    ax.set_xlabel("ε (share of the 200 tokens attacked)")
    ax.set_ylabel("true positive rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7)
    _save(fig, name)


def bound_vs_observed(deltas, obs, q25, q75, bounds, name, title, paper_obs=None, n_note=""):
    """Fig. 7 / Ext. 1a. bounds: {label: values per delta}; paper_obs: approximate values read off Fig. 7."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.fill_between(deltas, q25, q75, color=COLORS[0], alpha=0.2, label="ours, 25th-75th pct")
    ax.plot(deltas, obs, "o-", color=COLORS[0], lw=1.5, label="ours, observed mean")
    for i, (label, b) in enumerate(bounds.items(), 1):
        ax.plot(deltas, b, "^--", color=COLORS[i], lw=1.5, label=label)
    if paper_obs:
        for i, (label, (d, v)) in enumerate(paper_obs.items()):
            ax.plot(d, v, "o" if i == 0 else "^", mfc="none", mec="k", ms=8, ls="none", label=label)
    ax.set_xlabel("δ")
    ax.set_ylabel("green fraction |s|_G / T")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7)
    _note(ax, n_note)
    _save(fig, name)


def scatter_cells(cells, x, y, name, title, xlabel, ylabel, diag=False, hline=None):
    """cells: {(domain, model): list of row dicts}; colour = domain, marker = model."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    domains = list(dict.fromkeys(d for d, _ in cells))
    models = list(dict.fromkeys(m for _, m in cells))
    marks = "o^sDv"
    for (d, m), rows in cells.items():
        ax.scatter([r[x] for r in rows], [r[y] for r in rows], s=10, alpha=0.5, color=COLORS[domains.index(d)],
                   marker=marks[models.index(m)], label=f"{d}, {m} (n={len(rows)})")
    if diag:
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.5)
    _save(fig, name)


def tokens_to_detect(labels, medians, never, name, title, lo=None, hi=None):
    """Median tokens to reach z = 4 per cell (bootstrap 95% CI as error bars; an open upper end means the
    interval reaches 'never'), with the share that never reaches it written on the bar."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    m = [v if v is not None else 0 for v in medians]
    ax.bar(x, m, color=COLORS[0], width=0.6)
    top = 210
    if lo is not None:
        for i, (v, a, b) in enumerate(zip(medians, lo, hi)):
            if v is None:
                continue
            b2 = min(b, top) if np.isfinite(b) else top
            ax.errorbar(i, v, yerr=[[v - a], [b2 - v]], color="k", capsize=4, lw=1)
    for i, (v, s) in enumerate(zip(medians, never)):
        y = 3 if v is None else (min(hi[i], top) if lo is not None and np.isfinite(hi[i]) else (top if lo is not None else v))
        ax.text(i, y + 3, f"{'n/a' if v is None else int(v)}\nnever: {s:.0%}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("median tokens to reach z = 4")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 240)
    ax.set_xlabel("error bars: bootstrap 95% CI of the median; n/a: median = never", fontsize=8)
    _save(fig, name)


def entropy_hist(groups, name, title, xlabel):
    """groups: {label: per-token values}."""
    fig, ax = plt.subplots(figsize=(6.2, 4))
    bins = np.linspace(0, 1, 51)
    for i, (label, v) in enumerate(groups.items()):
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=COLORS[i],
                label=f"{label} (mean {np.mean(v):.3f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    _save(fig, name)


def removal_frontier(points, name, title, n_note, ylabel="TPR at 1% FPR (calibrated on attacked human text)"):
    """points: [(attack, level label, meaning kept, TPR, (lo, hi))]; marker per attack family; Wilson 95% CI bars."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fams = list(dict.fromkeys(p[0] for p in points))
    marks = "os^D"
    for i, f in enumerate(fams):
        pts = [p for p in points if p[0] == f]
        ax.plot([p[2] for p in pts], [p[3] for p in pts], marker=marks[i], color=COLORS[i], lw=1, ms=8, label=f)
        if len(pts[0]) > 4:
            ax.errorbar([p[2] for p in pts], [p[3] for p in pts], yerr=[[p[3] - p[4][0] for p in pts], [p[4][1] - p[3] for p in pts]],
                        fmt="none", ecolor=COLORS[i], capsize=3, lw=1)
        for p in pts:
            ax.annotate(str(p[1]), (p[2], p[3]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("meaning kept (MiniLM cosine to the original)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    _note(ax, n_note)
    _save(fig, name)


def dilution(shares, lines, name, title, n_note):
    """lines: {label: TPR at 1% FPR per share}."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, (label, v) in enumerate(lines.items()):
        ax.plot(np.array(shares) * 100, v, "o-", color=COLORS[i], lw=1.5, ms=7, label=label)
    ax.set_xlabel("watermarked share of the 600-token document (%)")
    ax.set_ylabel("TPR at 1% FPR")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    _note(ax, n_note)
    _save(fig, name)


def keys_eos(labels, df, name, title):
    """Kept share (T in 195..205) and share stopping before 50 tokens per hash key, Wilson 95% CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels))
    for ax, col, ylab in ((axes[0], "kept_share", "share with 195-205 tokens"), (axes[1], "ended_before_50", "share stopping before 50 tokens")):
        v, lo, hi = df[col].values, df[col.split("_")[0] + "_lo"].values, df[col.split("_")[0] + "_hi"].values
        ax.bar(x, v, color=[COLORS[0]] * (len(x) - 1) + ["0.6"], width=0.6)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", capsize=4, lw=1)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel(ylab)
    axes[0].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("hash key and estimated P(EOS) multiplier after \".\"; n = 100 per bar, Wilson 95% CI", fontsize=7)
    fig.suptitle(title, fontweight="bold")
    _save(fig, name)
