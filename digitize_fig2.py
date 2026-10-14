"""Digitise the paper's Fig. 2 from the vector PDFs in its arXiv source (needs pymupdf; not part of the pipeline).
Usage: python digitize_fig2.py <unpacked arXiv e-print of 2301.10226> > results/summary/fig2_paper.csv
Marker centres are read from the PDF paths and mapped to data coordinates by a linear fit through the tick labels."""
import sys, csv, pymupdf, numpy as np

SRC = sys.argv[1].rstrip("/") + "/figures/"
# viridis colours used by the paper (rounded) -> group value
COL_LEFT = {(0.197, 0.712, 0.479): 0.1, (0.130, 0.560, 0.552): 0.25, (0.192, 0.403, 0.556): 0.5,
            (0.268, 0.224, 0.512): 0.75, (0.267, 0.005, 0.329): 0.9}
COL_RIGHT = {(0.267, 0.005, 0.329): "greedy", (0.192, 0.403, 0.556): "4 beams", (0.197, 0.712, 0.479): "8 beams"}

def shape(d):
    ops = tuple(i[0] for i in d["items"]); n = len(ops)
    if "c" in ops and n == 8: return 0.0          # circle
    if ops == ("l",) * 12: return 10.0             # plus
    if ops == ("l",) * 10: return 5.0              # star
    if ops == ("l",) * 5:  return 2.0              # pentagon
    if ops == ("l",) * 2:  return 1.0              # x
    if ops == ("re",):     return 0.1              # square (drawn as a rectangle)
    if ops == ("l",) * 4:
        w, h = d["rect"].width, d["rect"].height
        # diamond and square both 4 segments; distinguish by whether a vertex sits on the rect's top edge centre
        pts = [p for it in d["items"] for p in it[1:]]
        xs = [p.x for p in pts]; ys = [p.y for p in pts]
        top = min(ys); cand = [x for x, y in zip(xs, ys) if abs(y - top) < 1e-3]
        mid = (min(xs) + max(xs)) / 2
        return 0.5 if any(abs(x - mid) < 0.2 for x in cand) else 0.1   # diamond apex at centre top; square corner at edge
    return None

def axis_map(words, horizontal):
    """Fit value = a*coord + b from tick labels (numeric words along the axis)."""
    pts = []
    for w in words:
        x0, y0, x1, y1, t = w[:5]
        try: v = float(t)
        except ValueError: continue
        if horizontal and y0 > 250: pts.append(((x0 + x1) / 2, v))
        if not horizontal and x1 < 36: pts.append(((y0 + y1) / 2, v))
    c, v = zip(*pts); a, b = np.polyfit(c, v, 1)
    resid = np.max(np.abs(np.array(c) * a + b - np.array(v)))
    return (lambda q: a * q + b), resid

def extract(name, colmap, label):
    page = pymupdf.open(SRC + name + ".pdf")[0]
    words = page.get_text("words")
    fx, rx = axis_map(words, True); fy, ry = axis_map(words, False)
    print(f"{name}: tick-fit max residual x={rx:.3g} y={ry:.3g}", file=sys.stderr)
    rows = []
    for d in page.get_drawings():
        if d.get("type") != "fs" or d.get("fill") is None: continue
        col = tuple(round(c, 3) for c in d["fill"])
        key = next((k for k in colmap if all(abs(a - b) < 0.01 for a, b in zip(k, col))), None)
        if key is None: continue
        delta = shape(d)
        if delta is None: continue
        r = d["rect"]; cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if cx > 300: continue                        # colourbar region
        rows.append(dict(panel=label(colmap[key]), gamma=colmap[key] if label is lambda_g else 0.5,
                         delta=delta, z_paper=round(float(fy(cy)), 3), PPL_paper=round(float(fx(cx)), 3)))
    return rows

lambda_g = lambda g: "multinomial"
left = extract("pareto_sampling_no_beams", COL_LEFT, lambda_g)
right = extract("pareto_greedy_w_beams", COL_RIGHT, lambda s: s)
rows = sorted(left + right, key=lambda r: (r["panel"] != "multinomial", r["panel"], r["gamma"], r["delta"]))
w = csv.DictWriter(sys.stdout, fieldnames=["panel", "gamma", "delta", "z_paper", "PPL_paper"]); w.writeheader(); w.writerows(rows)
