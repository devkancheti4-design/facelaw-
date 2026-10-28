"""THE JOINT LIGHT LAW: where the dominant light reaches the skin the 9-term law holds; where a cast shadow blocks it
only ambient light remains:   mu = V_s (B c) + (1 - V_s) b rho,   c in R^9, b >= 0.
For each probe and person, s ranges over the 8 best directions of the cast-shadow search (yale_cast), then the light
cancels in closed form and the identity is the person with the smallest remainder.  Protocol A (train subset 1 only)."""
import os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y
import yale_cast as C
from world import sh9

K = 8


def person(p):
    rho, n, Z = C.shape(Y.PEOPLE[p])
    Rm = C.renders(rho, n, Z)
    B = rho[:, None] * sh9(n)
    return p, rho, n, Z, Rm, B


def score_person(args):
    p, Ymat = args
    _, rho, n, Z, Rm, B = person(p)
    # cone residuals for all directions, keep the K best per probe
    G = Rm @ Ymat; RR = (Rm ** 2).sum(1)[:, None]; Rr = (Rm @ rho)[:, None]; rr = float(rho @ rho)
    rY = (rho @ Ymat)[None, :]; YY = (Ymat ** 2).sum(0)[None, :]
    det = RR * rr - Rr ** 2
    a = (rr * G - Rr * rY) / np.maximum(det, 1e-12); b = (RR * rY - Rr * G) / np.maximum(det, 1e-12)
    res = np.where((a >= 0) & (b >= 0), YY - (a * G + b * rY), np.minimum(np.where(G > 0, YY - G ** 2 / np.maximum(RR, 1e-12), YY), YY))
    top = np.argsort(res, 0)[:K]
    vis = {}
    out = np.empty(Ymat.shape[1])
    for j in range(Ymat.shape[1]):
        y = Ymat[:, j]; best = np.inf
        for k in top[:, j]:
            if k not in vis:
                vis[k] = C.visibility(Z, C.DIRS[k])
            V = vis[k]
            M = np.hstack([V[:, None] * B, ((1 - V) * rho)[:, None]])
            coef = np.linalg.lstsq(M, y, rcond=None)[0]
            if coef[-1] < 0:
                coef = np.append(np.linalg.lstsq(M[:, :-1], y, rcond=None)[0], 0.0)
            best = min(best, float(np.sum((y - M @ coef) ** 2)))
        out[j] = best
    return p, out


if __name__ == "__main__":
    t0 = time.time()
    probes = [(p, Y.subset(im["theta"]), im["I"]) for p in Y.IDS for im in Y.PEOPLE[p] if Y.subset(im["theta"]) >= 2]
    Ymat = np.array([y for _, _, y in probes]).T
    with Pool(12) as pool:
        got = dict(pool.map(score_person, [(p, Ymat) for p in Y.IDS]))
    RES = np.array([got[p] for p in Y.IDS])
    best = RES.argmin(0)
    acc = {}
    for j, (p, sb, y) in enumerate(probes):
        acc.setdefault(sb, []).append(Y.IDS[best[j]] == p)
    print("JOINT LIGHT LAW (9-term where lit, ambient in cast shadow), protocol A, 38 people -- recognition %")
    for sb in sorted(acc):
        print("   subset %d: %6.2f  (%d probes)" % (sb, 100 * np.mean(acc[sb]), len(acc[sb])))
    pickle.dump(dict(res=RES, ids=Y.IDS, probes=[(p, sb) for p, sb, _ in probes]), open("yale_joint_res.pkl", "wb"))
    print("%.0f s" % (time.time() - t0))
