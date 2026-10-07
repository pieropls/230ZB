"""z-scores, ROC/AUC, TPR at fixed FPR, Theorem 4.2 bounds, confidence intervals."""
import math

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve


# ---- scores ---------------------------------------------------------------------------
def green(ranks, gamma, V):
    """0/1 green flags from stored ranks, for any gamma."""
    return (np.asarray(ranks) < int(gamma * V)).astype(int)


def z_of(g, gamma):
    """One-sided z-score of a 0/1 green sequence (nan if empty)."""
    T = len(g)
    return (np.sum(g) - gamma * T) / math.sqrt(T * gamma * (1 - gamma)) if T else float("nan")


def z_ranks(ranks, gamma, V):
    """z of a text from its stored ranks, for any gamma."""
    return z_of(green(ranks, gamma, V), gamma)


def first_occurrence(ids):
    """Mask of tokens whose (previous token, token) pair appears for the first time (first token always kept)."""
    seen, keep = set(), []
    for pair in zip([None] + list(ids[:-1]), ids):
        keep.append(pair not in seen)
        seen.add(pair)
    return np.array(keep, dtype=bool)


def z_norep(ranks, ids, gamma, V):
    """z ignoring repeated (context, token) pairs."""
    return z_of(green(ranks, gamma, V)[first_occurrence(ids)], gamma)


def prefix_z(g, gamma):
    """z after each prefix length t = 1..T."""
    t = np.arange(1, len(g) + 1)
    return (np.cumsum(g) - gamma * t) / np.sqrt(t * gamma * (1 - gamma))


def tokens_to_z(g, gamma, thr=4.0):
    """First prefix length whose z reaches thr (None if never)."""
    hit = np.flatnonzero(prefix_z(g, gamma) >= thr)
    return int(hit[0]) + 1 if len(hit) else None


def mean_z_vs_T(green_lists, gamma, Ts):
    """Mean z over sequences at each prefix length T (sequences shorter than T are skipped)."""
    return [float(np.mean([z_of(g[:T], gamma) for g in green_lists if len(g) >= T])) for T in Ts]


def winmax(g, gamma):
    """Max z over every window size and position (WinMax, Kirchenbauer et al. 2024)."""
    g = np.asarray(g, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(g)])
    best = -np.inf
    for w in range(1, len(g) + 1):
        s = c[w:] - c[:-w]
        best = max(best, (s.max() - gamma * w) / math.sqrt(w * gamma * (1 - gamma)))
    return best


# ---- detection metrics ----------------------------------------------------------------
def rates(pos, neg, thr):
    """TPR, FNR, FPR, TNR at a z threshold."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    tpr, fpr = float(np.mean(pos > thr)), float(np.mean(neg > thr))
    return dict(TPR=tpr, FNR=1 - tpr, FPR=fpr, TNR=1 - fpr)


def auc(pos, neg):
    """Area under the ROC curve."""
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def roc(pos, neg):
    """ROC curve (fpr, tpr)."""
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    fpr, tpr, _ = roc_curve(y, np.r_[pos, neg])
    return fpr, tpr


def tpr_at_fpr(pos, neg, fpr=0.01):
    """Empirical threshold: the (1 - fpr) quantile of the negatives; TPR = share of positives above it."""
    thr = float(np.quantile(np.asarray(neg), 1 - fpr, method="higher"))
    return float(np.mean(np.asarray(pos) > thr)), thr


# ---- Theorem 4.2 -----------------------------------------------------------------------
def spike_modulus(gamma, delta):
    """Modulus z* = (1 - gamma)(alpha - 1) / (1 + (alpha - 1) gamma) of the spike entropy, alpha = e^delta."""
    a = math.exp(delta)
    return (1 - gamma) * (a - 1) / (1 + (a - 1) * gamma)


def thm42(S, gamma, delta, T):
    """Theorem 4.2: lower bound on E|s|_G and upper bound on its std, from average spike entropy S."""
    a = math.exp(delta)
    c = gamma * a * S / (1 + (a - 1) * gamma)
    return dict(mean=T * c, frac=c, sd=math.sqrt(max(T * c * (1 - c), 0.0)))


def predicted_detection(mean, sd, gamma, T, z=4.0):
    """Gaussian approximation: P(green count > gamma T + z sqrt(T gamma (1 - gamma)))."""
    thr = gamma * T + z * math.sqrt(T * gamma * (1 - gamma))
    return float(norm.sf((thr - mean) / sd))


# ---- misc ---------------------------------------------------------------------------------
def edit_distance(a, b):
    """Token-level Levenshtein distance, one numpy pass per row of the DP table."""
    a, b = list(a), np.asarray(b)
    prev = np.arange(len(b) + 1, dtype=np.int64)
    j = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        tmp = np.empty_like(prev)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b != x))
        prev = np.minimum.accumulate(tmp - j) + j          # insertions from the left
    return int(prev[-1])


# ---- per-run summary ----------------------------------------------------------------------
def run_summary(rows, gamma, V, neg_z, zkey=None):
    """Count, error rates at z = 4 and 5, AUC against neg_z, PPL, mean z and spike entropy for one config."""
    z = np.array([r[zkey] for r in rows]) if zkey else np.array([z_ranks(r["ranks"], gamma, V) for r in rows])
    out = dict(count=len(rows), z_mean=float(np.mean(z)), z_sd=float(np.std(z)))
    for thr in (4, 5):
        for k, v in rates(z, neg_z, thr).items():
            out[f"{k}_z{thr}"] = v
    out["AUC"] = auc(z, neg_z)
    ppl = [r["ppl"] for r in rows if r.get("ppl") is not None]
    out["PPL_mean"] = float(np.mean(ppl)) if ppl else float("nan")
    out["PPL_median"] = float(np.median(ppl)) if ppl else float("nan")
    if rows and "S_paper" in rows[0]:
        out["S_paper"] = float(np.mean([np.mean(r["S_paper"]) for r in rows]))
        out["S_sampled"] = float(np.mean([np.mean(r["S_sampled"]) for r in rows]))
        out["H"] = float(np.mean([np.mean(r["H"]) for r in rows]))
    out["green_frac"] = float(np.mean([green(r["ranks"], gamma, V).mean() for r in rows]))
    return out, z


# ---- 95% confidence intervals ----------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k / n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def wilson_of(flags):
    """Wilson interval of the share of True values."""
    flags = np.asarray(flags, dtype=bool)
    return wilson(int(flags.sum()), len(flags))


def boot_median(x, B=2000, seed=0):
    """Percentile bootstrap 95% interval for the median (values may be inf = never reached)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    meds = np.array([np.quantile(x[rng.integers(0, len(x), len(x))], 0.5, method="inverted_cdf") for _ in range(B)])
    return (float(np.quantile(meds, 0.025, method="inverted_cdf")), float(np.quantile(meds, 0.975, method="inverted_cdf")))


def fmt_ci(p, ci, d=3):
    """'p [lo, hi]' with d decimals."""
    return f"{p:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"
