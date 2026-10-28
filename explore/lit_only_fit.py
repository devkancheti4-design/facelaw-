import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("~/neooos/facelaw"))
import yale as Y
basis = {p: Y.enrol(Y.PEOPLE[p], (1,))[0] for p in Y.IDS}
def res_plain(y, B):
    return float(np.sum((y - B @ np.linalg.lstsq(B, y, rcond=None)[0]) ** 2))
def res_lit(y, B, t):
    lit = y > t * y.mean()
    if lit.sum() < 30: return res_plain(y, B)
    c = np.linalg.lstsq(B[lit], y[lit], rcond=None)[0]
    mu = B @ c
    r = np.where(lit, y - mu, np.maximum(mu - t * y.mean(), 0))     # shadow pixel: only a violation if mu predicts light
    return float(np.sum(r ** 2))
for name, f in [("plain", res_plain)] + [("lit t=%.2f" % t, (lambda y, B, t=t: res_lit(y, B, t))) for t in (0.05, 0.1, 0.2, 0.3)]:
    acc = {}
    for p in Y.IDS:
        for im in Y.PEOPLE[p]:
            sb = Y.subset(im["theta"])
            if sb < 2: continue
            best = min(Y.IDS, key=lambda q: f(im["I"], basis[q]))
            acc.setdefault(sb, []).append(best == p)
    print("%-12s " % name + "  ".join("S%d %6.2f" % (sb, 100 * np.mean(v)) for sb, v in sorted(acc.items())))
