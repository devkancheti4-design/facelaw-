"""HEAD-TO-HEAD under the dark-unlock protocol (design data, Extended Yale B; no sealed test set used).
2-fold CROSS-FITTED: people split into halves A / B; OUR score table F is counted on A only and tested on B as owners
and strangers (and swapped), so our law has no home advantage; the cohort (fixed stranger maps) = the other half.
EVERY METHOD gets the same: dark probes (Law 1, auto-exposure; 5 / 20 / 100 / 1000 photons per face pixel), +-1 px
misplacement, +-1 px placement search (its best match), T-norm against the same cohort, the same thresholds
(owner opens at the point where 0 strangers / 1 in 1,000 strangers get in), the same measured light classes.
CONTENDERS  OURS: direction sectors + measured-noise significance + F (dark_law codes, unlock_law scoring)
            Gradientfaces (angle distance), Weberfaces (L1), Tan-Triggs (L1): yale_baselines.py, frozen parameters
    python3 -I unlock_h2h.py"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import shift as nd_shift
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
from yale_baselines import gradientface, weberface, tantriggs, angdist, l1

NS = (5, 20, 100, 1000)
BASE = (("Gradientfaces", gradientface, angdist), ("Weberfaces", weberface, l1), ("Tan-Triggs", tantriggs, l1))
G = {}


def init(F, gs, gf, GB, gmean):
    DL.SHAPE, DL.SCALE_OF = (48, 42), None
    G.update(F=np.asarray(F), gs=gs, gf=gf, GB=GB, gmean=gmean)


def task(x):
    """best of 9 placements: our per-code score, and each baseline's similarity (minus its distance)"""
    out = {"OURS": None}; out.update({nm: None for nm, _, _ in BASE})
    xb = np.maximum(x, 0) * (G["gmean"] / max(x.mean(), 1e-9))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = (lambda a: nd_shift(a, (dy, dx), order=0, mode="nearest")) if (dy or dx) else (lambda a: a)
            ps, pc = U.probe_feats(sh(x))
            s = G["F"][DL.codes(ps, pc, G["gs"], G["gf"])].sum(1) / len(ps)
            out["OURS"] = s if out["OURS"] is None else np.maximum(out["OURS"], s)
            for nm, fn, dist in BASE:
                s = -dist(fn(sh(xb)), G["GB"][nm])
                out[nm] = s if out[nm] is None else np.maximum(out[nm], s)
    return out, U.lbin(U.imbalance(x))


if __name__ == "__main__":
    import yale as Y
    t0 = time.time()
    ids = Y.IDS; n = len(ids)
    front = {p: next(im for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")) for p in ids}
    halves = (ids[0::2], ids[1::2])
    R = np.random.default_rng(31)
    res = {}
    for fold in (0, 1):
        A, B = halves[fold], halves[1 - fold]
        # F counted on half A only
        GA = [DL.gallery_feats(front[p]["I"].reshape(48, 42)) for p in A]
        gsA = np.array([g[0] for g in GA]); gfA = np.array([g[1] for g in GA])
        same, other = np.ones(512), np.ones(512)
        for i, p in enumerate(A):
            for im in Y.PEOPLE[p]:
                if im is front[p] or im["I"].sum() <= 0:
                    continue
                for N in NS:
                    x = U.jitter(U.dark_ae(im["I"].reshape(48, 42), N, R), R)
                    ps, pc = U.probe_feats(x); C = DL.codes(ps, pc, gsA, gfA)
                    same += np.bincount(C[i], minlength=512); other += np.bincount(np.delete(C, i, 0).ravel(), minlength=512)
        F = np.clip(np.round(16 * np.log((same / same.sum()) / (other / other.sum()))), -127, 127).astype(int)
        # maps: B (owners / strangers) then A (the cohort)
        people = list(B) + list(A)
        GL = [DL.gallery_feats(front[p]["I"].reshape(48, 42)) for p in people]
        gs = np.array([g[0] for g in GL]); gf = np.array([g[1] for g in GL])
        GB = {nm: np.array([fn(front[p]["I"].reshape(48, 42)) for p in people]) for nm, fn, _ in BASE}
        gmean = float(np.mean([front[p]["I"].mean() for p in people]))
        probes = [(i, N, U.jitter(U.dark_ae(im["I"].reshape(48, 42), N, R), R)) for i, p in enumerate(B)
                  for im in Y.PEOPLE[p] if im is not front[p] and im["I"].sum() > 0 for N in NS]
        with Pool(3, init, (F, gs, gf, GB, gmean)) as pool:
            out = pool.map(task, [pr[2] for pr in probes], chunksize=16)
        coh = list(range(len(B), len(people)))
        for (i, N, x), (S, l) in zip(probes, out):
            for m, s in S.items():
                for o in range(len(B)):
                    res.setdefault((m, N, l), ([], []))[0 if o == i else 1].append(U.margins(s, o, coh))
        print("fold %d done: F counted on %d people, %d dark probes of %d other people  [%.0fs]" % (fold, len(A), len(probes), len(B), time.time() - t0), flush=True)
    methods = ["OURS"] + [nm for nm, _, _ in BASE]
    for title, lights in (("SCREEN-LIKE light (measured front class)", (0,)), ("ALL light", (0, 1, 2))):
        print("\nHEAD-TO-HEAD, %s -- owner opens %% when 0 strangers get in / when 1 in 1,000 get in" % title)
        print("   %-14s" % "method" + "".join("%22s" % ("N=%d photons" % N) for N in NS))
        for m in methods:
            row = ""
            for N in NS:
                g = np.concatenate([res[(m, N, l)][0] for l in lights if (m, N, l) in res])
                i = np.concatenate([res[(m, N, l)][1] for l in lights if (m, N, l) in res])
                row += "%22s" % ("%5.1f%% / %5.1f%%" % (100 * (g > i.max()).mean(), 100 * (g > np.quantile(i, 0.999)).mean()))
            print("   %-14s%s" % (m, row))
        n_g = sum(len(res[(methods[0], NS[0], l)][0]) for l in lights if (methods[0], NS[0], l) in res)
        n_i = sum(len(res[(methods[0], NS[0], l)][1]) for l in lights if (methods[0], NS[0], l) in res)
        print("   (%d owner tries and %d stranger tries per photon level)" % (n_g, n_i))
    print("[%.0fs]" % (time.time() - t0))
