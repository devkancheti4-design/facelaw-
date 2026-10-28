"""v2 HEAD-TO-HEAD (design data, Extended Yale B; no sealed test set used): does the HYBRID beat v1 and the descriptors?
The unlock_h2h.py protocol exactly: 2-fold CROSS-FITTED by person (F and FW counted on half A only, tested on half B as
owners and strangers, and swapped; the cohort = the other half); dark probes at 5 / 20 / 100 / 1000 photons per face
pixel (Law 1, auto-exposure); +-1 px misplacement and the best of the 9 placements for EVERY method.  Two metrics:
  T-NORM     owner opens when 0 / 1 in 1,000 strangers get in (margins against the cohort: unlock_h2h.py's metric)
  RAW        NIST-style: the raw score, ONE threshold for every pair, no cohort: owner accepted where 0 / 1 in 1,000
             stranger pairs pass
CONTENDERS  v1 (direction codes, F) | WEBER codes alone (FW) | HYBRID W = 1 (the sum of the two log-likelihood ratios,
            the weight fixed BEFORE this run: Bayes if the codes were independent) | HYBRID W = 0.5 (sensitivity) |
            Gradientfaces, Weberfaces, Tan-Triggs (yale_baselines.py, frozen parameters)
    python3 -I hybrid_h2h.py"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import shift as nd_shift
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import hybrid as H
from yale_baselines import gradientface, weberface, tantriggs, angdist, l1

NS = (5, 20, 100, 1000)
BASE = (("Gradientfaces", gradientface, angdist), ("Weberfaces", weberface, l1), ("Tan-Triggs", tantriggs, l1))
WS = (("v1 (direction)", 0.0), ("WEBER codes", None), ("HYBRID W=1", 1.0), ("HYBRID W=0.5", 0.5), ("HYBRID W=2", 2.0), ("HYBRID W=4", 4.0))
G = {}


def init(F, FW, gs, gf, gw, GB, gmean, alpha):
    DL.SHAPE, DL.SCALE_OF = (48, 42), None
    H.ALPHA = list(alpha)
    G.update(F=np.asarray(F), FW=np.asarray(FW), gs=gs, gf=gf, gw=gw, GB=GB, gmean=gmean)


def task(x):
    """per placement: the direction score and the Weber score against every map; each baseline's best similarity"""
    Sd, Sw = [], []
    base = {nm: None for nm, _, _ in BASE}
    xb = np.maximum(x, 0) * (G["gmean"] / max(x.mean(), 1e-9))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = (lambda a: nd_shift(a, (dy, dx), order=0, mode="nearest")) if (dy or dx) else (lambda a: a)
            xs = sh(x)
            ps, pc = U.probe_feats(xs)
            Sd.append(G["F"][DL.codes(ps, pc, G["gs"], G["gf"])].sum(1) / len(ps))
            pb, wc = H.probe_weber(xs, U.measured_var(xs))
            Sw.append(G["FW"][H.wcodes(pb, wc, G["gw"])].sum(1) / len(pb))
            for nm, fn, dist in BASE:
                s = -dist(fn(sh(xb)), G["GB"][nm])
                base[nm] = s if base[nm] is None else np.maximum(base[nm], s)
    return np.array(Sd, np.float32), np.array(Sw, np.float32), base, U.lbin(U.imbalance(x))


def combine(Sd, Sw, w):
    return Sw.max(0) if w is None else (Sd + w * Sw).max(0)


if __name__ == "__main__":
    import yale as Y
    t0 = time.time()
    ids = Y.IDS; n = len(ids)
    front = {p: next(im for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")) for p in ids}
    alpha = H.set_alpha([front[p]["I"].reshape(48, 42) for p in ids])
    print("Weber ALPHA per scale (1 / median |L/G| on the 38 clean front maps): %s" % [round(a, 3) for a in alpha], flush=True)
    halves = (ids[0::2], ids[1::2])
    R = np.random.default_rng(31)
    res = {}
    for fold in (0, 1):
        A, B = halves[fold], halves[1 - fold]
        GA = [DL.gallery_feats(front[p]["I"].reshape(48, 42)) for p in A]
        gsA = np.array([g[0] for g in GA]); gfA = np.array([g[1] for g in GA])
        gwA = np.array([H.gallery_weber(front[p]["I"].reshape(48, 42)) for p in A])
        same, other = np.ones(512), np.ones(512)
        wsame, wother = np.ones(H.NFW), np.ones(H.NFW)
        for i, p in enumerate(A):
            for im in Y.PEOPLE[p]:
                if im is front[p] or im["I"].sum() <= 0:
                    continue
                for N in NS:
                    x = U.jitter(U.dark_ae(im["I"].reshape(48, 42), N, R), R)
                    ps, pc = U.probe_feats(x); C = DL.codes(ps, pc, gsA, gfA)
                    same += np.bincount(C[i], minlength=512); other += np.bincount(np.delete(C, i, 0).ravel(), minlength=512)
                    pb, wc = H.probe_weber(x, U.measured_var(x)); CW = H.wcodes(pb, wc, gwA)
                    wsame += np.bincount(CW[i], minlength=H.NFW); wother += np.bincount(np.delete(CW, i, 0).ravel(), minlength=H.NFW)
        F = np.clip(np.round(16 * np.log((same / same.sum()) / (other / other.sum()))), -127, 127).astype(int)
        FW = np.clip(np.round(16 * np.log((wsame / wsame.sum()) / (wother / wother.sum()))), -127, 127).astype(int)
        print("fold %d: F and FW counted on %d people (FW range %d..%d)  [%.0fs]" % (fold, len(A), FW.min(), FW.max(), time.time() - t0), flush=True)
        people = list(B) + list(A)
        GL = [DL.gallery_feats(front[p]["I"].reshape(48, 42)) for p in people]
        gs = np.array([g[0] for g in GL]); gf = np.array([g[1] for g in GL])
        gw = np.array([H.gallery_weber(front[p]["I"].reshape(48, 42)) for p in people])
        GB = {nm: np.array([fn(front[p]["I"].reshape(48, 42)) for p in people]) for nm, fn, _ in BASE}
        gmean = float(np.mean([front[p]["I"].mean() for p in people]))
        probes = [(i, N, U.jitter(U.dark_ae(im["I"].reshape(48, 42), N, R), R)) for i, p in enumerate(B)
                  for im in Y.PEOPLE[p] if im is not front[p] and im["I"].sum() > 0 for N in NS]
        with Pool(3, init, (F, FW, gs, gf, gw, GB, gmean, alpha)) as pool:
            out = pool.map(task, [pr[2] for pr in probes], chunksize=16)
        coh = list(range(len(B), len(people)))
        for (i, N, x), (Sd, Sw, base, l) in zip(probes, out):
            S = {nm: combine(Sd, Sw, w) for nm, w in WS}; S.update(base)
            for m, s in S.items():
                r = res.setdefault((m, N, l), ([], [], [], []))
                for o in range(len(B)):
                    r[0 if o == i else 1].append(U.margins(s, o, coh))
                    r[2 if o == i else 3].append(float(s[o]))
        print("fold %d done: %d dark probes of %d other people  [%.0fs]" % (fold, len(probes), len(B), time.time() - t0), flush=True)
    methods = [nm for nm, _ in WS] + [nm for nm, _, _ in BASE]
    for title, lights in (("SCREEN-LIKE light (measured front class)", (0,)), ("ALL light", (0, 1, 2))):
        for metric, gi, ii in (("T-NORM margins (unlock metric)", 0, 1), ("RAW score, one global threshold (NIST-style)", 2, 3)):
            print("\n%s, %s -- owner accepted %% when 0 strangers / 1 in 1,000 get in" % (title, metric))
            print("   %-16s" % "method" + "".join("%22s" % ("N=%d photons" % N) for N in NS))
            for m in methods:
                row = ""
                for N in NS:
                    g = np.concatenate([res[(m, N, l)][gi] for l in lights if (m, N, l) in res])
                    i = np.concatenate([res[(m, N, l)][ii] for l in lights if (m, N, l) in res])
                    row += "%22s" % ("%5.1f%% / %5.1f%%" % (100 * (g > i.max()).mean(), 100 * (g > np.quantile(i, 0.999)).mean()))
                print("   %-16s%s" % (m, row))
        n_g = sum(len(res[(methods[0], NS[0], l)][0]) for l in lights if (methods[0], NS[0], l) in res)
        n_i = sum(len(res[(methods[0], NS[0], l)][1]) for l in lights if (methods[0], NS[0], l) in res)
        print("   (%d owner tries and %d stranger tries per photon level)" % (n_g, n_i))
    np.save(os.path.join(os.environ.get("OUTDIR", "."), "hybrid_h2h_res.npy"), {str(k): [np.array(a, np.float32) for a in v] for k, v in res.items()}, allow_pickle=True)
    print("[%.0fs]" % (time.time() - t0))
