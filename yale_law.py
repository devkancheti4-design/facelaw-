"""The decision law FROM THE DATA, on the user's method: cancel, then judge only the remainder.
For a probe y and a person p: mu_p = P_p y (the 9-term light law cancels light and exposure), the per-pixel remainder
e = (y - mu_p) / mean(y) (exposure cancels), and the predicted level m = mu_p / mean(y).
    code = min(15, floor(8 |e|)) | (e < 0) << 4 | level(m) << 5          7 bits, 128 codes
The law is the data's own log-likelihood ratio  f(code) = round(2 ln(P(code | right person) / P(code | wrong person))),
counted on DESIGN people only; the score of p is sum over pixels f(code); the identity is the best score.
Cross-fitted by people: the law for the odd-numbered people is counted on the even ones and vice versa, so no test
person's images ever touch their own law; a test person contributes only their 7 enrolment images (protocol A)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y

TRAIN, TESTS = (1,), (2, 3, 4, 5)


def level(m):
    return np.where(m <= 0.05, 0, np.where(m <= 0.5, 1, np.where(m <= 1.5, 2, 3)))


def codes(y, B):
    mu = B @ np.linalg.lstsq(B, y, rcond=None)[0]
    s = max(y.mean(), 1e-9)
    e, m = (y - mu) / s, mu / s
    return np.minimum(15, np.floor(8 * np.abs(e))).astype(int) | (e < 0).astype(int) << 4 | level(m) << 5


def remainders(ids, basis):
    """per probe: (true person, subset, [code histogram per candidate])"""
    out = []
    for p in ids:
        for im in Y.PEOPLE[p]:
            sb = Y.subset(im["theta"])
            if sb not in TESTS:
                continue
            out.append((p, sb, {q: np.bincount(codes(im["I"], basis[q]), minlength=128) for q in Y.IDS}))
    return out


def table(rem):
    right = np.ones(128); wrong = np.ones(128)                       # +1: every code is possible
    for p, sb, h in rem:
        for q, c in h.items():
            (right if q == p else wrong).__iadd__(c)
    return np.clip(np.round(2 * np.log((right / right.sum()) / (wrong / wrong.sum()))), -15, 15).astype(int), right, wrong


def decide(rem, f):
    res = {}
    for p, sb, h in rem:
        best = max(h, key=lambda q: int(h[q] @ f))
        res.setdefault(sb, []).append(best == p)
    return res


if __name__ == "__main__":
    t0 = time.time()
    basis = {p: Y.enrol(Y.PEOPLE[p], TRAIN)[0] for p in Y.IDS}
    folds = [[p for i, p in enumerate(Y.IDS) if i % 2 == k] for k in (0, 1)]
    rem = [remainders(f, basis) for f in folds]
    T = [table(rem[1 - k]) for k in (0, 1)]                            # fold k is judged by the law of the OTHER fold
    laws = [t[0] for t in T]
    seen = [((t[1] + t[2]) > 2).tolist() for t in T]                   # codes the design data actually contains
    l2 = {}
    for k in (0, 1):
        for sb, v in decide(rem[k], laws[k]).items():
            l2.setdefault(sb, []).extend(v)
    plain = {}
    for k in (0, 1):
        for p, sb, h in rem[k]:
            pass
    print("data law (cross-fitted), protocol A, 38 people -- recognition %")
    for sb in TESTS:
        print("   subset %d: %6.2f  (%d probes)" % (sb, 100 * np.mean(l2[sb]), len(l2[sb])))
    json.dump(dict(laws=[l.tolist() for l in laws], seen=seen, folds=folds), open("yale_law_tables.json", "w"))
    import pickle
    pickle.dump(dict(rem=rem, folds=folds), open("yale_law_rem.pkl", "wb"))
    for k in (0, 1):
        print("\nlaw counted on fold %d (applies to fold %d), rows = level (shadow, dim, mid, bright), cols = |e| bins 0..15" % (1 - k, k))
        f = laws[k]
        for lv in range(4):
            print("   %-6s +e: %s" % (("shadow", "dim", "mid", "bright")[lv], " ".join("%3d" % f[a | 0 << 4 | lv << 5] for a in range(16))))
            print("   %-6s -e: %s" % ("", " ".join("%3d" % f[a | 1 << 4 | lv << 5] for a in range(16))))
    print("\n%.0f s" % (time.time() - t0))
