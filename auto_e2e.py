"""v2 END-TO-END: does recognition survive AUTOMATIC eyes?  Extended Yale B (37 people with hand-checked pupils), every
photo cut to the law's 48 x 42 frame through the app's crop (phone_eyes.crop) by
    TRUE  the person's hand-checked pupils          AUTO  eyelaw2 on the photo (cross-fitted: T counted on the other
                                                          half of the people), from a face box off by ~5% / 8%
The unlock_h2h protocol otherwise: 2-fold cross-fitted by person (F, FW counted on half A through the SAME crop path,
tested on half B, swapped), maps = each person's frontal photo, probes = every other photo at N photons per face pixel,
the best of the 9 placements (+-1 px), T-norm against the other half and the RAW one-threshold score.
    python3 -I auto_e2e.py eyes      find and cache the automatic eyes
    python3 -I auto_e2e.py run       the head-to-head"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import hybrid as H
import hybrid_h2h as HH

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.environ.get("OUTDIR", HERE), "auto_eyes_cache.json")
NS = (100, 1000)
MAXTH = float(os.environ.get("MAXTH", 77.0))


def eyes_job(args):
    p, name, tl, tr, n, th = args
    import yale as Y, eyelaw2 as L2
    T = np.load(os.path.join(HERE, "eyelaw2_T_%s.npy" % th))
    I = Y.read_pgm(os.path.join(Y.ROOT, p, name)).astype(float)
    tl, tr = np.array(tl), np.array(tr); ed = np.linalg.norm(tr - tl); w = ed / L2.BOXW
    rng = np.random.default_rng(n)
    box = np.array([(tl[0] + tr[0]) / 2 - w / 2, (tl[1] + tr[1]) / 2 - 0.36 * w, w, w])
    box = box + np.array([*(rng.normal(0, 0.05, 2) * w), 0, 0]); box[2:] *= np.exp(rng.normal(0, 0.08))
    eL, eR = L2.find(I, box, T)
    return p, name, eL.tolist(), eR.tolist()


def crops(people, eyes_of):
    """{person: [(name, theta, 48x42 crop)]} through the app's crop"""
    import yale as Y, phone_eyes as PE
    cL, cR = PE.canonical_eyes()
    out = {}
    for p in people:
        for im in Y.PEOPLE[p]:
            if im["I"].sum() <= 0 or im["theta"] > MAXTH:
                continue
            e = eyes_of(p, im["name"])
            I = Y.read_pgm(os.path.join(Y.ROOT, p, im["name"])).astype(float)
            out.setdefault(p, []).append((im["name"], im["theta"], PE.crop(I, np.array(e[0]), np.array(e[1]), cL, cR)))
    return out


if __name__ == "__main__":
    import yale as Y
    stage = sys.argv[1]
    TR = json.load(open(os.path.join(HERE, "yale_eyes_truth.json")))["eyes"]
    ids = [p for p in Y.IDS if p in TR]
    halfA = set(Y.IDS[0::2])
    if stage == "eyes":
        t0 = time.time()
        jobs = [(p, im["name"], TR[p][0], TR[p][1], n, "B" if p in halfA else "A")
                for n, (p, im) in enumerate((p, im) for p in ids for im in Y.PEOPLE[p] if im["I"].sum() > 0 and im["theta"] <= MAXTH)]
        with Pool(3) as pool:
            got = pool.map(eyes_job, jobs, chunksize=8)
        json.dump({"%s/%s" % (p, nm): [l, r] for p, nm, l, r in got}, open(CACHE, "w"))
        import eyes2 as E2
        e = [E2.err((np.array(l), np.array(r)), (np.array(TR[p][0]), np.array(TR[p][1]))) for p, nm, l, r in got]
        print("automatic eyes on %d photos (light <= %.0f deg), cross-fitted: %s  [%.0fs]" % (len(got), MAXTH, E2.summary(e), time.time() - t0))
    elif stage == "run":
        t0 = time.time()
        AUTO = json.load(open(CACHE))
        arms = {"TRUE": lambda p, nm: TR[p], "AUTO": lambda p, nm: AUTO["%s/%s" % (p, nm)]}
        R = np.random.default_rng(31)
        halves = (ids[0::2], ids[1::2])
        res = {}
        for arm, eyes_of in arms.items():
            C = crops(ids, eyes_of)
            front = {p: next(c for nm, th, c in C[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
            alpha = H.set_alpha([front[p] for p in ids])
            for fold in (0, 1):
                A, B = halves[fold], halves[1 - fold]
                GA = [DL.gallery_feats(front[p]) for p in A]
                gsA = np.array([g[0] for g in GA]); gfA = np.array([g[1] for g in GA]); gwA = np.array([H.gallery_weber(front[p]) for p in A])
                same, other, wsame, wother = np.ones(512), np.ones(512), np.ones(H.NFW), np.ones(H.NFW)
                for i, p in enumerate(A):
                    for nm, th, c in C[p]:
                        if nm.endswith("A+000E+00.pgm"):
                            continue
                        for N in NS:
                            x = U.dark_ae(c, N, R)
                            ps, pc = U.probe_feats(x); K = DL.codes(ps, pc, gsA, gfA)
                            same += np.bincount(K[i], minlength=512); other += np.bincount(np.delete(K, i, 0).ravel(), minlength=512)
                            pb, wc = H.probe_weber(x, U.measured_var(x)); KW = H.wcodes(pb, wc, gwA)
                            wsame += np.bincount(KW[i], minlength=H.NFW); wother += np.bincount(np.delete(KW, i, 0).ravel(), minlength=H.NFW)
                F = np.clip(np.round(16 * np.log((same / same.sum()) / (other / other.sum()))), -127, 127).astype(int)
                FW = np.clip(np.round(16 * np.log((wsame / wsame.sum()) / (wother / wother.sum()))), -127, 127).astype(int)
                people = list(B) + list(A)
                GL = [DL.gallery_feats(front[p]) for p in people]
                gs = np.array([g[0] for g in GL]); gf = np.array([g[1] for g in GL]); gw = np.array([H.gallery_weber(front[p]) for p in people])
                GB = {nm: np.array([fn(front[p]) for p in people]) for nm, fn, _ in HH.BASE}
                gmean = float(np.mean([front[p].mean() for p in people]))
                probes = [(i, N, U.dark_ae(c, N, R)) for i, p in enumerate(B) for nm, th, c in C[p]
                          if not nm.endswith("A+000E+00.pgm") for N in NS]
                with Pool(3, HH.init, (F, FW, gs, gf, gw, GB, gmean, alpha)) as pool:
                    out = pool.map(HH.task, [pr[2] for pr in probes], chunksize=16)
                coh = list(range(len(B), len(people)))
                for (i, N, x), (Sd, Sw, base, l) in zip(probes, out):
                    S = {nm: HH.combine(Sd, Sw, w) for nm, w in HH.WS}; S.update(base)
                    for m, s in S.items():
                        r = res.setdefault((arm, m, N, l), ([], [], [], []))
                        for o in range(len(B)):
                            r[0 if o == i else 1].append(U.margins(s, o, coh))
                            r[2 if o == i else 3].append(float(s[o]))
                print("%s fold %d: %d probes  [%.0fs]" % (arm, fold, len(probes), time.time() - t0), flush=True)
        methods = [nm for nm, _ in HH.WS] + [nm for nm, _, _ in HH.BASE]
        for title, lights in (("SCREEN-LIKE light (front class)", (0,)), ("ALL light <= %.0f deg" % MAXTH, (0, 1, 2))):
            for metric, gi, ii in (("T-NORM", 0, 1), ("RAW, one threshold (NIST-style)", 2, 3)):
                print("\n%s, %s -- owner accepted %% at 0 strangers / 1 in 1,000:  TRUE eyes  ->  AUTOMATIC eyes" % (title, metric))
                print("   %-16s" % "method" + "".join("%34s" % ("N=%d photons" % N) for N in NS))
                for m in methods:
                    row = ""
                    for N in NS:
                        cell = []
                        for arm in arms:
                            g = np.concatenate([res[(arm, m, N, l)][gi] for l in lights if (arm, m, N, l) in res])
                            i = np.concatenate([res[(arm, m, N, l)][ii] for l in lights if (arm, m, N, l) in res])
                            cell.append("%4.1f/%4.1f" % (100 * (g > i.max()).mean(), 100 * (g > np.quantile(i, 0.999)).mean()))
                        row += "%34s" % ("%s  ->  %s" % tuple(cell))
                    print("   %-16s%s" % (m, row))
        print("[%.0fs]" % (time.time() - t0))
