"""v2 COUNT through the EXACT reference (v2ref.py): the eye law T and the Weber table FW, counted on Extended Yale B (37
people with hand-checked pupils) through the deployed path, with the checks that decide whether the law is kept.
    python3 -I v2_count.py eyes     T on each half; eye error on the unseen half (face box off by ~5% / 8%) and on the
                                    owner's own images (classic face box); automatic eyes for every Yale photo
                                    (cross-fitted) -> the cache
    python3 -I v2_count.py check    cross-fitted head-to-head on the automatic-eye crops: v2 vs v1 vs Gradientfaces
    python3 -I v2_count.py final    T and FW on all 37 people -> v2_design.json (then v2_author.py)"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import v2ops as O
import v2ref as R2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", HERE)
CACHE = os.path.join(OUT, "v2_auto_eyes.json")
NS = (5, 20, 100, 1000)
BOXW, PAIR = 0.45, dict(N=5, LD=3000.0, LT=3000.0)
MAXTH = 77.0


def setlaw(T=None, FW=None, ALPHA=None):
    R2.LAW.clear()
    import v2law as V2
    V2.load_lanes()
    eh, EL_, EG = R2.energies()
    R2.LAW.update(FW=None if FW is None else np.asarray(FW, np.int64), ALPHA=ALPHA, REF=107.7726, EH=eh, EL=EL_, EG=EG,
                  BOXW=BOXW, PAIR=PAIR, CL=np.array([8.0, 11.0]), CR=np.array([33.0, 11.0]))
    if T is not None:
        T = np.asarray(T, np.int64)
        R2.LAW.update(T=T, Text=np.concatenate([T, np.zeros(T.shape[:2] + (1,), np.int64)], 2))


def count_T(items, rng):
    """items: (display image, (left eye, right eye)) -> T (25 x 25 x 51)"""
    pos = np.ones((2 * R2.RW + 1, 2 * R2.RW + 1, R2.NC)); bg = np.ones(R2.NC)
    for img, (eL, eR) in items:
        ed0 = float(np.linalg.norm(eR - eL)); ang0 = np.arctan2(eR[1] - eL[1], eR[0] - eL[0])
        for eye, mir in ((eL, False), (eR, True)):
            js = np.exp(rng.uniform(np.log(0.9), np.log(1.1))); ja = ang0 + np.radians(rng.uniform(-8, 8))
            sub = rng.uniform(-0.5, 0.5, 2) * ed0 / R2.ED
            C = R2.eye_codes(R2.eye_frame(img, eye + sub, ed0 * js, np.cos(ja), np.sin(ja), mir))
            cy, cx = (C.shape[0] - 1) // 2, (C.shape[1] - 1) // 2
            win = C[cy - R2.RW:cy + R2.RW + 1, cx - R2.RW:cx + R2.RW + 1]
            np.add.at(pos, (np.arange(2 * R2.RW + 1)[:, None], np.arange(2 * R2.RW + 1)[None, :], win), 1)
            bg += np.bincount(C[R2.RW:-R2.RW, R2.RW:-R2.RW].ravel(), minlength=R2.NC)
    P = pos / pos.sum(2, keepdims=True); B = bg / bg.sum()
    return np.clip(np.round(16 * np.log(P / B[None, None])), -127, 127).astype(np.int64)


def yale_items(people, maxth=180.0):
    import yale as Y
    TR = json.load(open(os.path.join(HERE, "yale_eyes_truth.json")))["eyes"]
    out = []
    for p in people:
        if p not in TR:
            continue
        for im in Y.PEOPLE[p]:
            if im["I"].sum() <= 0 or im["theta"] > maxth:
                continue
            out.append((p, im["name"], im["theta"], (np.array(TR[p][0], float), np.array(TR[p][1], float))))
    return out


def read(p, name):
    import yale as Y
    return Y.read_pgm(os.path.join(Y.ROOT, p, name)).astype(float)


def noisy_box(tl, tr, n):
    ed = float(np.linalg.norm(tr - tl)); w = ed / BOXW
    rng = np.random.default_rng(n)
    box = np.array([(tl[0] + tr[0]) / 2 - w / 2, (tl[1] + tr[1]) / 2 - 0.36 * w, w, w])
    box = box + np.array([*(rng.normal(0, 0.05, 2) * w), 0, 0]); box[2:] *= np.exp(rng.normal(0, 0.08))
    return box


TH = {}


def eyes_init(TA, TB):
    TH.update(A=TA, B=TB, cur=None)


def eyes_job(args):
    p, name, tl, tr, n, half = args
    if TH["cur"] != half:
        setlaw(T=TH[half]); TH["cur"] = half
    e = R2.find_eyes(read(p, name), noisy_box(np.array(tl), np.array(tr), n))
    return p, name, None if e is None else [e[0].tolist(), e[1].tolist()]


def err(found, truth):
    tl, tr = truth; ed = np.linalg.norm(tr - tl)
    return max(np.linalg.norm(found[0] - tl), np.linalg.norm(found[1] - tr)) / ed


def summary(e):
    e = np.asarray(e, float)
    return "median %.3f; within 0.02 %5.1f%%, 0.04 %5.1f%%, 0.05 %5.1f%%, 0.10 %5.1f%%" % (
        np.median(e), 100 * (e <= .02).mean(), 100 * (e <= .04).mean(), 100 * (e <= .05).mean(), 100 * (e <= .10).mean())


def crops_of(ids, AUTO):
    out = {}
    for p, name, theta, t in yale_items(ids, MAXTH):
        e = AUTO.get("%s/%s" % (p, name))
        if e is None:
            continue
        out.setdefault(p, []).append((name, theta, R2.crop(read(p, name), np.array(e[0]), np.array(e[1]))))
    return out


def alpha_of(fronts):
    al = []
    for k in range(3):
        r = []
        for c in fronts:
            xe = R2.expose(c)
            G = O.gauss(xe, R2.SCALES8[k]); L = R2.lap8(G, R2.STRIDE[k])
            r.append(np.abs(L / np.maximum(G, 1e-3 * R2.LAW["REF"])).ravel())
        al.append(1.0 / float(np.median(np.concatenate(r))))
    return al


def count_FW(C, people, front, R):
    gb = np.array([R2.map_bins(front[p]) for p in people])
    same, other = np.ones(768), np.ones(768)
    import unlock_law as U
    for i, p in enumerate(people):
        for name, th, c in C[p]:
            if name.endswith("A+000E+00.pgm"):
                continue
            for N in NS:
                pb, pc = R2.probe_codes(U.dark_ae(c, N, R))
                K = R2.codes(pb, pc, gb)
                same += np.bincount(K[i], minlength=768); other += np.bincount(np.delete(K, i, 0).ravel(), minlength=768)
    return np.clip(np.round(16 * np.log((same / same.sum()) / (other / other.sum()))), -127, 127).astype(np.int64)


G = {}


def h2h_init(T, FW, ALPHA, gb, F, gs, gf, GF, gmean):
    setlaw(T=T, FW=FW, ALPHA=ALPHA)
    import dark_law as DL
    DL.SHAPE, DL.SCALE_OF = (48, 42), None
    G.update(gb=gb, F=np.asarray(F), gs=gs, gf=gf, GF=GF, gmean=gmean)


def h2h_task(x):
    import dark_law as DL, unlock_law as U
    from yale_baselines import gradientface, angdist
    v2 = R2.score(R2.probe_all(x), G["gb"])
    v1 = gfs = None
    xb = np.maximum(x, 0) * (G["gmean"] / max(x.mean(), 1e-9))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            xs = R2.shifted(x, dy, dx)
            ps, pc = U.probe_feats(xs)
            s = G["F"][DL.codes(ps, pc, G["gs"], G["gf"])].sum(1) / len(ps); v1 = s if v1 is None else np.maximum(v1, s)
            s = -angdist(gradientface(R2.shifted(xb, dy, dx)), G["GF"]); gfs = s if gfs is None else np.maximum(gfs, s)
    return {"v2 (exact reference)": v2, "v1 (direction)": v1, "Gradientfaces": gfs}, U.lbin(U.imbalance(x))


if __name__ == "__main__":
    import yale as Y
    stage = sys.argv[1]
    t0 = time.time()
    TR = json.load(open(os.path.join(HERE, "yale_eyes_truth.json")))["eyes"]
    ids = [p for p in Y.IDS if p in TR]
    halves = {"A": [p for p in Y.IDS[0::2] if p in TR], "B": [p for p in Y.IDS[1::2] if p in TR]}
    setlaw()
    if stage == "eyes":
        rng = np.random.default_rng(2610)
        Th = {}
        for h in ("A", "B"):
            Th[h] = count_T([(read(p, nm), t) for p, nm, th, t in yale_items(halves[h])], rng)
            np.save(os.path.join(OUT, "v2_T_%s.npy" % h), Th[h])
            print("T counted on Yale half %s (exact reference): range %d..%d  [%.0fs]" % (h, Th[h].min(), Th[h].max(), time.time() - t0), flush=True)
        jobs = [(p, nm, t[0].tolist(), t[1].tolist(), n, "B" if p in halves["A"] else "A")
                for n, (p, nm, th, t) in enumerate(yale_items(ids, MAXTH))]
        with Pool(3, eyes_init, (Th["A"], Th["B"])) as pool:
            got = pool.map(eyes_job, jobs, chunksize=8)
        json.dump({"%s/%s" % (p, nm): e for p, nm, e in got}, open(CACHE, "w"))
        th_of = {(p, nm): th for p, nm, th, t in yale_items(ids, MAXTH)}
        for lim in (25.5, MAXTH):
            e = [err((np.array(v[0]), np.array(v[1])), (np.array(TR[p][0]), np.array(TR[p][1]))) if v else np.inf
                 for p, nm, v in got if th_of[(p, nm)] <= lim]
            print("automatic eyes, every Yale photo with light <= %.0f deg (cross-fitted T), %d photos: %s" % (lim, len(e), summary(e)), flush=True)
        import eyes2 as E2
        setlaw(T=Th["A"])
        es = []
        for nm, lin, t in E2.user_items():
            box = E2.face_box(lin)
            disp = O.u8(lin).astype(float)
            e = R2.find_eyes(disp, box) if box is not None else None
            es.append(err(e, t) if e is not None else np.inf)
            print("   %-24s eye distance %4.0f px  error %.3f" % (nm, np.linalg.norm(t[1] - t[0]), es[-1]))
        print("owner's images (T from Yale half A, classic face box): %s  [%.0fs]" % (summary(es), time.time() - t0))
    elif stage in ("check", "final"):
        import dark_law as DL, unlock_law as U
        from yale_baselines import gradientface
        AUTO = json.load(open(CACHE))
        C = crops_of(ids, AUTO)
        front = {p: next(c for nm, th, c in C[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
        ALPHA = alpha_of([front[p] for p in ids])
        setlaw(ALPHA=ALPHA)
        print("ALPHA %s  [%.0fs]" % ([round(a, 4) for a in ALPHA], time.time() - t0), flush=True)
        R = np.random.default_rng(2028)
        if stage == "final":
            Tall = count_T([(read(p, nm), t) for p, nm, th, t in yale_items(ids)], np.random.default_rng(2610))
            FW = count_FW(C, ids, front, R)
            D = dict(note="v2 law tables counted through the EXACT reference (v2ref.py) on Extended Yale B, 37 people with "
                          "hand-checked pupils: T on every photo's true pupils (jittered), FW on the automatic-eye crops "
                          "(cross-fitted eyes) at N = %s photons per face pixel" % (NS,),
                     REF=107.7726, ALPHA=ALPHA, FW=FW.tolist(), T=Tall.tolist(), BOXW=BOXW, ED=R2.ED, RW=R2.RW, PAIR=PAIR,
                     canonical=[[8.0, 11.0], [33.0, 11.0]])
            json.dump(D, open(os.path.join(HERE, "v2_design.json"), "w"))
            print("v2_design.json: T range %d..%d, FW range %d..%d  [%.0fs]" % (Tall.min(), Tall.max(), FW.min(), FW.max(), time.time() - t0))
            sys.exit(0)
        res = {}
        hl = (halves["A"], halves["B"])
        for fold in (0, 1):
            A, B = hl[fold], hl[1 - fold]
            FW = count_FW(C, A, front, R)
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
            gb = np.array([R2.map_bins(front[p]) for p in people])
            GL = [DL.gallery_feats(front[p]) for p in people]
            gs = np.array([g[0] for g in GL]); gf = np.array([g[1] for g in GL])
            GFm = np.array([gradientface(front[p]) for p in people]); gmean = float(np.mean([front[p].mean() for p in people]))
            probes = [(i, N, U.dark_ae(c, N, R)) for i, p in enumerate(B) for nm, th, c in C[p] if not nm.endswith("A+000E+00.pgm") for N in NS]
            with Pool(3, h2h_init, (None, FW, ALPHA, gb, F, gs, gf, GFm, gmean)) as pool:
                out = pool.map(h2h_task, [pr[2] for pr in probes], chunksize=16)
            coh = list(range(len(B), len(people)))
            for (i, N, x), (S, l) in zip(probes, out):
                for m, s in S.items():
                    r = res.setdefault((m, N, l), ([], [], [], []))
                    for o in range(len(B)):
                        r[0 if o == i else 1].append(U.margins(s, o, coh)); r[2 if o == i else 3].append(float(s[o]))
            print("fold %d: %d probes  [%.0fs]" % (fold, len(probes), time.time() - t0), flush=True)
        methods = ["v2 (exact reference)", "v1 (direction)", "Gradientfaces"]
        for title, lights in (("SCREEN-LIKE light (front class)", (0,)), ("ALL light <= 77 deg", (0, 1, 2))):
            for metric, gi, ii in (("T-NORM", 0, 1), ("RAW, one threshold (NIST-style)", 2, 3)):
                print("\n%s, %s, AUTOMATIC eyes -- owner accepted %% at 0 strangers / 1 in 1,000" % (title, metric))
                print("   %-24s" % "method" + "".join("%18s" % ("N=%d" % N) for N in NS))
                for m in methods:
                    row = ""
                    for N in NS:
                        g = np.concatenate([res[(m, N, l)][gi] for l in lights if (m, N, l) in res])
                        i = np.concatenate([res[(m, N, l)][ii] for l in lights if (m, N, l) in res])
                        row += "%18s" % ("%5.1f / %5.1f" % (100 * (g > i.max()).mean(), 100 * (g > np.quantile(i, 0.999)).mean()))
                    print("   %-24s%s" % (m, row))
