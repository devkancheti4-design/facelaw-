"""v2 SCORE NORMALISATION, NIST-style (one global threshold): a template that scores high against EVERYONE (a flat or
mis-found crop) must not pass.  Each normalisation removes what a template is EXPECTED to score against a stranger:
    RAW      the symmetric v2 score
    E-NORM   minus the expected stranger score of each side, computed from the template alone: the probe codes against
             the population's map-bin frequencies at each point, the map against the population's probe-code
             frequencies at each point (counted on the OTHER half of the people; no face is stored)
    T-NORM   against a cohort of the other half's photos (reference only: it would ship lab faces)
Extended Yale B, automatic eyes, both photos in normal light, N = 1000, 2-fold by person.
    python3 -I v2_norm.py"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import v2ref as R2
import v2_count as VC


def tables(maps, probes):
    """per point: P(map bin) (8) and P(probe code = bin | sig << 3) (32) at the centre placement, +1 smoothed"""
    Pm = np.ones((2652, 8)); Pp = np.ones((2652, 32))
    for m in maps:
        Pm[np.arange(2652), m] += 1
    for pr in probes:
        pb, pc = pr[4]
        Pp[np.arange(2652), pb | (pc << 3)] += 1
    return Pm / Pm.sum(1, keepdims=True), Pp / Pp.sum(1, keepdims=True)


def expected(FW, Pm, Pp, m, pr):
    """the template's expected score against a stranger: as a probe (its centre placement vs Pm), as a map (vs Pp)"""
    k = R2.SCALE_OF
    pb, pc = pr[4]
    allb = np.arange(8)[None]
    e_probe = float((Pm * FW[allb | (pb << 3)[:, None] | (pc << 6)[:, None] | (k << 8)[:, None]]).sum() / 2652)
    allc = np.arange(32)[None]
    e_map = float((Pp * FW[m[:, None] | ((allc & 7) << 3) | ((allc >> 3) << 6) | (k << 8)[:, None]]).sum() / 2652)
    return e_probe, e_map


if __name__ == "__main__":
    import yale as Y, unlock_law as U
    t0 = time.time()
    TR = json.load(open(os.path.join(VC.HERE, "yale_eyes_truth.json")))["eyes"]
    ids = [p for p in Y.IDS if p in TR]
    AUTO = json.load(open(VC.CACHE))
    C = VC.crops_of(ids, AUTO)
    front = {p: next(c for nm, th, c in C[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
    D = json.load(open(os.path.join(VC.HERE, "v2_design.json")))
    R = np.random.default_rng(77)
    halves = ([p for p in Y.IDS[0::2] if p in TR], [p for p in Y.IDS[1::2] if p in TR])
    out = {}
    for fold in (0, 1):
        A, B = halves[fold], halves[1 - fold]
        VC.setlaw(ALPHA=D["ALPHA"])
        FW = VC.count_FW(C, A, front, R)
        VC.setlaw(FW=FW, ALPHA=D["ALPHA"])
        def photos_of(P):
            ph = [(p, U.dark_ae(c, 1000, R)) for p in P for nm, th, c in C[p] if th <= 25.5]
            return ph, [R2.map_bins(x) for _, x in ph], [R2.probe_all(x) for _, x in ph]
        phA, mA, pA = photos_of(A)
        phB, mB, pB = photos_of(B)
        Pm, Pp = tables(mA, pA)
        EB = [expected(FW, Pm, Pp, m, pr) for m, pr in zip(mB, pB)]
        S = np.array([R2.score(pB[i], np.array(mB)) for i in range(len(phB))])             # S[i, j]: i's probe vs j's map
        coh = np.array([R2.score(pB[i], np.array(mA[::4])) for i in range(len(phB))])     # i's probe vs a cohort of A's maps
        cohm = np.array([R2.score(pA[c], np.array(mB)) for c in range(0, len(phA), 4)]).T  # A's probes vs B's maps: [j, c]
        mu_p, sd_p = coh.mean(1), coh.std(1) + 1e-9
        mu_m, sd_m = cohm.mean(1), cohm.std(1) + 1e-9
        n = len(phB)
        for i in range(n):
            for j in range(i + 1, n):
                g = phB[i][0] == phB[j][0]
                raw = (S[i, j] + S[j, i]) / 2.0
                en = ((S[i, j] - EB[i][0] - EB[j][1]) + (S[j, i] - EB[j][0] - EB[i][1])) / 2.0
                tn = ((S[i, j] - mu_p[i]) / sd_p[i] + (S[j, i] - mu_p[j]) / sd_p[j]) / 2.0
                zn = ((S[i, j] - mu_m[j]) / sd_m[j] + (S[j, i] - mu_m[i]) / sd_m[i]) / 2.0
                for k, v in (("RAW (symmetric)", raw), ("E-NORM (no faces stored)", en), ("T-NORM (cohort, reference)", tn),
                             ("Z-NORM (cohort, reference)", zn), ("T+Z (cohort, reference)", (tn + zn) / 2.0)):
                    out.setdefault(k, ([], []))[0 if g else 1].append(v)
        print("fold %d: %d photos of %d people  [%.0fs]" % (fold, n, len(B), time.time() - t0), flush=True)
    print("\nNIST-style 1:1, Yale B, automatic eyes, normal light, one global threshold -- accepted %% at 0 / 1 in 1,000 / 1 in 100 impostor pairs")
    for k, (g, i) in out.items():
        g, i = np.array(g), np.array(i)
        print("   %-28s %5.1f / %5.1f / %5.1f" % (k, 100 * (g > i.max()).mean(), 100 * (g > np.quantile(i, 0.999)).mean(), 100 * (g > np.quantile(i, 0.99)).mean()))
    print("   (%d genuine pairs, %d impostor pairs)" % (len(out["RAW (symmetric)"][0]), len(out["RAW (symmetric)"][1])))
