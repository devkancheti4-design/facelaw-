"""v2 matching, NIST-style: ONE-WAY (the verification photo's probe codes against the enrolment photo's map) or
SYMMETRIC (the mean of both directions)?  Extended Yale B, automatic-eye crops (v2_count.py's cache), both photos in
normal light (<= 25 deg) with mild noise (N = 1000 photons per face pixel), every same-person pair of different photos
as genuine, every cross-person pair as impostor; FW counted on the OTHER half of the people (2-fold); the raw score
with ONE global threshold: accepted %% when 0 / 1 in 1,000 / 1 in 100 impostor pairs pass.
    python3 -I v2_sym.py"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import v2ref as R2
import v2_count as VC

if __name__ == "__main__":
    import yale as Y, unlock_law as U
    t0 = time.time()
    TR = json.load(open(os.path.join(VC.HERE, "yale_eyes_truth.json")))["eyes"]
    ids = [p for p in Y.IDS if p in TR]
    AUTO = json.load(open(VC.CACHE))
    C = VC.crops_of(ids, AUTO)
    front = {p: next(c for nm, th, c in C[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
    D = json.load(open(os.path.join(VC.HERE, "v2_design.json")))
    VC.setlaw(ALPHA=D["ALPHA"])
    R = np.random.default_rng(77)
    halves = ([p for p in Y.IDS[0::2] if p in TR], [p for p in Y.IDS[1::2] if p in TR])
    out = {"one-way": ([], []), "symmetric": ([], [])}
    for fold in (0, 1):
        A, B = halves[fold], halves[1 - fold]
        FW = VC.count_FW(C, A, front, R)
        VC.setlaw(FW=FW, ALPHA=D["ALPHA"])
        photos = [(p, U.dark_ae(c, 1000, R)) for p in B for nm, th, c in C[p] if th <= 25.5]
        maps = [R2.map_bins(x) for _, x in photos]
        probes = [R2.probe_all(x) for _, x in photos]
        n = len(photos)
        S = np.array([R2.score(probes[i], np.array(maps)) for i in range(n)])      # S[i, j]: photo i's probe vs photo j's map
        for i in range(n):
            for j in range(i + 1, n):
                g = photos[i][0] == photos[j][0]
                out["one-way"][0 if g else 1].append(S[i, j])
                out["symmetric"][0 if g else 1].append((S[i, j] + S[j, i]) / 2.0)
        print("fold %d: %d photos of %d people  [%.0fs]" % (fold, n, len(B), time.time() - t0), flush=True)
    print("\nNIST-style 1:1, Yale B, automatic eyes, normal light, one global threshold -- accepted %% at 0 / 1 in 1,000 / 1 in 100 impostor pairs")
    for k, (g, i) in out.items():
        g, i = np.array(g), np.array(i)
        print("   %-10s %5.1f / %5.1f / %5.1f    (%d genuine pairs, %d impostor pairs)" % (
            k, 100 * (g > i.max()).mean(), 100 * (g > np.quantile(i, 0.999)).mean(), 100 * (g > np.quantile(i, 0.99)).mean(), len(g), len(i)))
