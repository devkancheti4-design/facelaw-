"""v2 BUILD: the v2 law's tables counted on Extended Yale B through the DEPLOYED path (automatic eyes -> the app's crop).
    python3 -I v2_build.py check    2-fold cross-fitted head-to-head of the integer reference (v2law.py) against v1 and
                                    Gradientfaces, automatic eyes (auto_e2e.py eyes must have run), T-norm and RAW
    python3 -I v2_build.py final    count T (eye law) and FW (Weber codes) on all 37 people -> v2_design.json"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import shift as nd_shift
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import hybrid as H
import v2law as V2
import auto_e2e as AE
from yale_baselines import gradientface, angdist

HERE = os.path.dirname(os.path.abspath(__file__))
NS = (5, 20, 100, 1000)
G = {}


def init(F, FW, gs, gf, gb, GF, gmean, alpha):
    DL.SHAPE, DL.SCALE_OF = (48, 42), None
    H.ALPHA = list(alpha)
    G.update(F=np.asarray(F), FW=np.asarray(FW), gs=gs, gf=gf, gb=gb, GF=GF, gmean=gmean)


def task(x):
    v2 = V2.score(G["FW"], V2.probe_all(x), G["gb"])
    v1, gfs = None, None
    xb = np.maximum(x, 0) * (G["gmean"] / max(x.mean(), 1e-9))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = (lambda a: nd_shift(a, (dy, dx), order=0, mode="nearest")) if (dy or dx) else (lambda a: a)
            ps, pc = U.probe_feats(sh(x))
            s = G["F"][DL.codes(ps, pc, G["gs"], G["gf"])].sum(1) / len(ps)
            v1 = s if v1 is None else np.maximum(v1, s)
            s = -angdist(gradientface(sh(xb)), G["GF"])
            gfs = s if gfs is None else np.maximum(gfs, s)
    return {"v2 (Weber, integer lanes)": v2, "v1 (direction)": v1, "Gradientfaces": gfs}, U.lbin(U.imbalance(x))


def count_fw(C, people, front, gbA, R):
    same, other = np.ones(H.NFW), np.ones(H.NFW)
    for i, p in enumerate(people):
        for nm, th, c in C[p]:
            if nm.endswith("A+000E+00.pgm"):
                continue
            for N in NS:
                pb, pc = V2.probe_codes(U.dark_ae(c, N, R))
                K = H.wcodes(pb, pc, gbA)
                same += np.bincount(K[i], minlength=H.NFW); other += np.bincount(np.delete(K, i, 0).ravel(), minlength=H.NFW)
    return np.clip(np.round(16 * np.log((same / same.sum()) / (other / other.sum()))), -127, 127).astype(int)


if __name__ == "__main__":
    import yale as Y
    stage = sys.argv[1]
    t0 = time.time()
    TR = json.load(open(os.path.join(HERE, "yale_eyes_truth.json")))["eyes"]
    ids = [p for p in Y.IDS if p in TR]
    AUTO = json.load(open(AE.CACHE))
    C = AE.crops(ids, lambda p, nm: AUTO["%s/%s" % (p, nm)])
    front = {p: next(c for nm, th, c in C[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
    alpha = H.set_alpha([V2.expose(front[p]) for p in ids])
    print("ALPHA (1 / median |L/G| over the 37 frontal crops, automatic eyes): %s" % [round(a, 4) for a in alpha], flush=True)
    R = np.random.default_rng(2028)
    if stage == "check":
        halves = (ids[0::2], ids[1::2])
        res = {}
        for fold in (0, 1):
            A, B = halves[fold], halves[1 - fold]
            gbA = np.array([V2.map_bins(front[p]) for p in A])
            FW = count_fw(C, A, front, gbA, R)
            GA = [DL.gallery_feats(front[p]) for p in A]
            gsA = np.array([g[0] for g in GA]); gfA = np.array([g[1] for g in GA])
            same, other = np.ones(512), np.ones(512)
            for i, p in enumerate(A):
                for nm, th, c in C[p]:
                    if nm.endswith("A+000E+00.pgm"):
                        continue
                    for N in NS:
                        ps, pc = U.probe_feats(U.dark_ae(c, N, R)); K = DL.codes(ps, pc, gsA, gfA)
                        same += np.bincount(K[i], minlength=512); other += np.bincount(np.delete(K, i, 0).ravel(), minlength=512)
            F = np.clip(np.round(16 * np.log((same / same.sum()) / (other / other.sum()))), -127, 127).astype(int)
            people = list(B) + list(A)
            gb = np.array([V2.map_bins(front[p]) for p in people])
            GL = [DL.gallery_feats(front[p]) for p in people]
            gs = np.array([g[0] for g in GL]); gf = np.array([g[1] for g in GL])
            GFm = np.array([gradientface(front[p]) for p in people]); gmean = float(np.mean([front[p].mean() for p in people]))
            probes = [(i, N, U.dark_ae(c, N, R)) for i, p in enumerate(B) for nm, th, c in C[p] if not nm.endswith("A+000E+00.pgm") for N in NS]
            with Pool(3, init, (F, FW, gs, gf, gb, GFm, gmean, alpha)) as pool:
                out = pool.map(task, [pr[2] for pr in probes], chunksize=16)
            coh = list(range(len(B), len(people)))
            for (i, N, x), (S, l) in zip(probes, out):
                for m, s in S.items():
                    r = res.setdefault((m, N, l), ([], [], [], []))
                    for o in range(len(B)):
                        r[0 if o == i else 1].append(U.margins(s, o, coh)); r[2 if o == i else 3].append(float(s[o]))
            print("fold %d: %d probes  [%.0fs]" % (fold, len(probes), time.time() - t0), flush=True)
        methods = ["v2 (Weber, integer lanes)", "v1 (direction)", "Gradientfaces"]
        for title, lights in (("SCREEN-LIKE light (front class)", (0,)), ("ALL light <= 77 deg", (0, 1, 2))):
            for metric, gi, ii in (("T-NORM", 0, 1), ("RAW, one threshold (NIST-style)", 2, 3)):
                print("\n%s, %s, AUTOMATIC eyes -- owner accepted %% at 0 strangers / 1 in 1,000" % (title, metric))
                print("   %-28s" % "method" + "".join("%18s" % ("N=%d" % N) for N in NS))
                for m in methods:
                    row = ""
                    for N in NS:
                        g = np.concatenate([res[(m, N, l)][gi] for l in lights if (m, N, l) in res])
                        i = np.concatenate([res[(m, N, l)][ii] for l in lights if (m, N, l) in res])
                        row += "%18s" % ("%5.1f / %5.1f" % (100 * (g > i.max()).mean(), 100 * (g > np.quantile(i, 0.999)).mean()))
                    print("   %-28s%s" % (m, row))
    elif stage == "final":
        import eyelaw2 as L2
        rng = np.random.default_rng(2610)
        T = L2.count([(img, t) for _, _, img, t in L2.yale_items(ids)], rng)
        gb = np.array([V2.map_bins(front[p]) for p in ids])
        FW = count_fw(C, ids, front, gb, R)
        cL, cR = __import__("phone_eyes").canonical_eyes()
        json.dump(dict(note="v2 law tables counted on Extended Yale B (37 people with hand-checked pupils), deployed path: "
                            "automatic eyes (cross-fitted) -> the app's crop; FW over N = %s photons per face pixel" % (NS,),
                       REF=V2.REF, ALPHA=alpha, E_L=H.E_L, E_G=H.E_G, FW=FW.tolist(), T=T.tolist(), BOXW=L2.BOXW, ED=L2.ED,
                       RW=L2.RW, PAIR=L2.PAIR, canonical=[cL.tolist(), cR.tolist()]),
                  open(os.path.join(HERE, "v2_design.json"), "w"))
        print("v2_design.json: T %s (range %d..%d), FW %d (range %d..%d)  [%.0fs]" % (T.shape, T.min(), T.max(), len(FW), FW.min(), FW.max(), time.time() - t0))
