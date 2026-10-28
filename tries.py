"""MORE THAN ONE TRY: owner accepted when the owner may try k times, and so may every stranger (k tries each, the same
threshold).  An owner or a stranger is let in when ANY of the k tries passes.
    pairs   (NIST-style, full light) from the P2 score matrices: the k tries are k different photos of the same person
            (other lights within 25 deg), drawn without replacement; exact expectation over all draws (no sampling)
    frames  (light and darkness, P1's people and photos) CONTINUOUS tries: the same photo, frame after frame, each
            frame with fresh photon noise at N photons per face pixel (as a phone camera sees a still face); K = 5
            frames per try; the enrolment is the frontal photo; tables cross-fitted on the other half of the people
    python3 -I tries.py pairs [matrices.npz] | frames"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", os.path.join(HERE, "h2h"))
KS = (1, 2, 3, 5)
K = 5
NS = (5, 20, 100, 1000)
FR_METHODS = ("Full-light law", "FACE LAW v2 (48)", "Tan-Triggs@96")
G = {}


def frames_init(ctx):
    G.clear(); G.update(ctx)
    import v2_count as VC
    VC.setlaw(FW=ctx["FW"], ALPHA=ctx["ALPHA"])


def frames_task(job):
    """one try: K frames of the same photo, fresh photon noise each; every method, no search, against every enrolment"""
    import unlock_law as U, v2ref as R2, h2h_all as H
    c48, c96, N, seed = job
    R = np.random.default_rng(seed)
    out = {m: [] for m in FR_METHODS}
    for _ in range(K):
        x48, x96 = U.dark_ae(c48, N, R), U.dark_ae(c96, N / 4.0, R)
        out["FACE LAW v2 (48)"].append(R2.score([R2.probe_codes(x48)], G["maps"]["v2"]))
        R2.LAW.update(ALPHA=G["ALPHA96"], FW=G["FW96"])
        out["Full-light law"].append(R2.score([R2.probe_codes(x96)], G["maps"]["full"]))
        R2.LAW.update(ALPHA=G["ALPHA"], FW=G["FW"])
        d = H.describe("Tan-Triggs", 96, H.norm_b(x96, G["gmean"]))
        out["Tan-Triggs@96"].append(H.sim_of("Tan-Triggs")(d, G["maps"]["tt"]))
    return {m: np.array(v, np.float64) for m, v in out.items()}


def miss_all(n, a, k):
    """P(none of k distinct draws from n items hits one of the a passing items) = C(n - a, k) / C(n, k)"""
    p = np.ones(np.broadcast(n, a).shape)
    for r in range(k):
        p = p * np.clip((n - a - r) / np.maximum(n - r, 1), 0, None)
    return p


def pools(S, lab):
    """per enrolment j: its genuine tries (sorted), and per stranger person q: q's photos' scores against j (sorted)"""
    ppl = sorted(set(lab)); idx = {p: np.nonzero(lab == p)[0] for p in ppl}
    gen, imp = [], []
    for j in range(len(lab)):
        g = np.sort(S[[i for i in idx[lab[j]] if i != j], j]); gen.append(g)
        imp += [np.sort(S[idx[q], j]) for q in ppl if q != lab[j]]
    return gen, imp


def rate(P, t, k):
    n = np.array([len(x) for x in P], float)
    a = np.array([len(x) - np.searchsorted(x, t, side="right") for x in P], float)
    ok = n >= k
    return float(np.mean(1 - miss_all(n[ok], a[ok], k)))


def threshold(imp, k, far):
    """the lowest threshold whose k-try stranger rate is at most far (0: no stranger let in at all)"""
    top = max(float(x[-1]) for x in imp)
    if far == 0:
        return top
    lo, hi = min(float(x[0]) for x in imp), top
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rate(imp, mid, k) > far else (lo, mid)
    return hi


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "pairs"
    if stage == "pairs":
        path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "p2_full_matrices.npz")
        M = dict(np.load(path, allow_pickle=True))
        methods = ["FACE LAW v2@96 nosearch", "FACE LAW v2@96", "Tan-Triggs@96", "LTP@96", "FACE LAW v2"]
        res = {}
        for m in methods:
            gen, imp = [], []
            for f in (0, 1):
                g, i = pools(M["%s|%d" % (m, f)].astype(np.float64), M["labels_%d" % f]); gen += g; imp += i
            for far in (0, 1e-3):
                for k in KS:
                    t = threshold(imp, k, far)
                    res["%s|%g|%d" % (m, far, k)] = dict(owner=100 * rate(gen, t, k), stranger=100 * rate(imp, t, k))
            print("%-24s 0 strangers: %s   |   1 in 1,000: %s" % (m, "  ".join("%d try %5.1f" % (k, res["%s|0|%d" % (m, k)]["owner"]) for k in KS),
                                                               "  ".join("%d %5.1f" % (k, res["%s|0.001|%d" % (m, k)]["owner"]) for k in KS)), flush=True)
        json.dump(res, open(os.path.join(OUT, "tries_pairs.json"), "w"), indent=1)
    elif stage == "frames":
        import h2h_all as H, v2ref as R2, v2_count as VC
        t0 = time.time()
        Yy, ids, hA, hB = H.load_people()
        C48, C96 = H.crops_table()
        D = json.load(open(os.path.join(HERE, "v2_design.json")))
        front48 = {p: next(c for nm, th, c in C48[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
        front96 = {p: next(c for nm, th, c in C96[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
        gmean96 = float(np.mean([front96[p].mean() for p in ids]))
        S = {}                                                      # (method, N) -> list of (owner index, K x |B| scores)
        for fold, (A, B) in enumerate(((hA, hB), (hB, hA))):
            VC.setlaw(ALPHA=D["ALPHA"])
            FW = VC.count_FW(C48, A, front48, np.random.default_rng(7200 + fold))     # v2's table, its own noise draws
            VC.setlaw(FW=FW, ALPHA=D["ALPHA"])
            ALPHA96 = VC.alpha_of([front96[p] for p in ids])
            FW96 = H.count_FW96(C96, A, front96, np.random.default_rng(7000 + fold), ALPHA96)   # the table P1 used
            maps = {"v2": np.array([R2.map_bins(front48[p]) for p in B])}
            R2.LAW.update(ALPHA=ALPHA96, FW=FW96); maps["full"] = np.array([R2.map_bins(front96[p]) for p in B])
            R2.LAW.update(ALPHA=D["ALPHA"], FW=FW)
            maps["tt"] = np.array([H.describe("Tan-Triggs", 96, H.norm_b(front96[p], gmean96)) for p in B])
            jobs, owner = [], []
            for i, p in enumerate(B):
                for k, ((nm, th, c48), (_, _, c96)) in enumerate(zip(C48[p], C96[p])):
                    if nm.endswith("A+000E+00.pgm"):
                        continue
                    for N in NS:
                        jobs.append((c48, c96, N, 900000 * fold + 1000 * ids.index(p) + 10 * k + NS.index(N))); owner.append((i, N))
            ctx = dict(FW=FW, ALPHA=D["ALPHA"], FW96=FW96, ALPHA96=ALPHA96, maps=maps, gmean=gmean96)
            with Pool(3, frames_init, (ctx,)) as pool:
                out = pool.map(frames_task, jobs, chunksize=8)
            for (i, N), o in zip(owner, out):
                for m, sc in o.items():
                    S.setdefault((m, N), []).append((i, sc))
            print("frames fold %d: %d tries x %d frames  [%.0fs]" % (fold, len(jobs), K, time.time() - t0), flush=True)
        res = {}
        for (m, N), rows in sorted(S.items()):
            for k in KS:
                own = np.array([sc[:k, i].max() for i, sc in rows])
                strg = np.concatenate([np.delete(sc[:k].max(0), i) for i, sc in rows])
                for far in (0, 1e-3):
                    t = strg.max() if far == 0 else np.quantile(strg, 1 - far)
                    res["%s|%d|%g|%d" % (m, N, far, k)] = dict(owner=100 * float((own > t).mean()), stranger=100 * float((strg > t).mean()),
                                                               owners=len(own), strangers=len(strg))
        json.dump(res, open(os.path.join(OUT, "tries_frames.json"), "w"), indent=1)
        for far in (0, 1e-3):
            print("owner accepted with k continuous frames (strangers get the same k frames), %s:" % ("0 strangers" if far == 0 else "1 in 1,000 strangers"))
            for N in NS:
                print("  N=%-5d " % N + "   ".join("%s %s" % (m, " ".join("%5.1f" % res["%s|%d|%g|%d" % (m, N, far, k)]["owner"] for k in KS)) for m in FR_METHODS))
        print("[%.0fs]" % (time.time() - t0))
