"""THE HEAD-TO-HEAD: FACE LAW v2 against the classic face recognisers, every method on the SAME faces.
FACES  Extended Yale B, 37 people with hand-checked pupils (yaleB13 excluded: eyes nearly closed), every photo with the
       light within 77 deg, cut by the v2 pipeline's AUTOMATIC eyes (cross-fitted eye law, face box off by ~5% / 8%):
       48 x 42 (the law's frame) and 96 x 84 (two pixels per law pixel: the size the classic papers worked near).  A probe
       at 96 x 84 gets a quarter of the photons per pixel, so every method sees the same light from the face.
SEARCH every method gets the best of the 9 placements within +-1 law pixel (2 px at 96 x 84).
TRAINING the learned parts (v1's F, v2's FW, Eigenfaces, Fisherfaces) come from the OTHER half of the people (2 folds).
    P1  verification under light and darkness: each person's frontal photo is the enrolment; every other photo is a try at
        N = 5 / 20 / 100 / 1000 photons per law pixel; owner accepted when 0 / 1 in 1,000 strangers get in, with the raw
        score (one threshold for everyone) and with the T-norm margin against the other half's faces
    P2  NIST-style 1:1: every pair of normal-light photos (light within 25 deg, N = 1000), the symmetric score, one
        threshold: accepted at 1 in 100 / 1 in 1,000 / 0 strangers, and the equal error rate
    P3  the literature's protocol: Extended Yale B identification, Yale's own crops (38 people), subset 1 the gallery,
        subsets 2-5 the probes, nearest neighbour (rank 1); with and without the +-1 px search
    P4  the SEALED CMU PIE confirmation (pie_v2_protocol.txt is written and hashed before the first PIE image is read)
    python3 -I h2h_all.py prep | p1 | p2 | p3 | p4protocol | p4"""
import hashlib, json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import v2ops as O
import v2ref as R2
import classics as CL
from yale_baselines import gradientface, weberface, tantriggs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", os.path.join(HERE, "h2h"))
os.makedirs(OUT, exist_ok=True)
CROPS = os.path.join(OUT, "crops.npz")
EYES = os.environ.get("EYECACHE", os.path.join(os.environ.get("OUTDIR", HERE), "v2_auto_eyes.json"))
NS = (5, 20, 100, 1000)
SHIFTS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)] if not os.environ.get("NOSEARCH") else [(0, 0)]


def ang_sim(a, B):
    d = np.abs(a[None] - B)
    return -np.minimum(d, 2 * np.pi - d).reshape(len(B), -1).sum(1)


def l1_sim(a, B):
    return -np.abs(a[None] - B).reshape(len(B), -1).sum(1)


def chi2_u8(a, B):
    return CL.chi2_sim(a.astype(np.float32), B.astype(np.float32))


def inter_u8(a, B):
    return np.minimum(a[None], B).reshape(len(B), -1).sum(1, dtype=np.int64).astype(float)


CLASSIC = [("Gradientfaces", gradientface, ang_sim, np.float32), ("Weberfaces", weberface, l1_sim, np.float32),
           ("Tan-Triggs", tantriggs, l1_sim, np.float32), ("LBP", CL.lbp, chi2_u8, np.uint8), ("LTP", CL.ltp, chi2_u8, np.uint8),
           ("LGBPHS", CL.lgbphs, inter_u8, np.uint8), ("HOG", CL.hog, CL.l2_sim, np.float32)]
LEARNED = [("Eigenfaces", lambda X, y: CL.Eigen(X, 100)), ("Eigenfaces-3", lambda X, y: CL.Eigen(X, 100, 3)), ("Fisherfaces", CL.Fisher)]
RES = (48, 96)
LAWS = ("FACE LAW v2", "FACE LAW v1")
V2X = ("FACE LAW v2 nosearch", "FACE LAW v2@96", "FACE LAW v2@96 nosearch")     # the full-light candidates (2026-10-28)


def method_names():
    return list(LAWS) + list(V2X) + ["%s@%d" % (n, r) for n, *_ in CLASSIC + [(l[0],) for l in LEARNED] for r in RES]


def use96(on):
    """switch the reference between v2's 48 x 42 law and its 96 x 84 full-light law (own ALPHA and FW)"""
    R2.LAW.update(ALPHA=G["ALPHA96"] if on else G["ALPHA"], FW=G["FW96"] if on else G["FW"])


def count_FW96(C96, people, front96, R, alpha96):
    """v2's table counted on the 96 x 84 crops (a quarter of the photons per pixel), as count_FW does at 48 x 42"""
    import unlock_law as U
    keep = R2.LAW["ALPHA"]; R2.LAW.update(ALPHA=alpha96)
    gb = np.array([R2.map_bins(front96[p]) for p in people])
    same, other = np.ones(768), np.ones(768)
    for i, p in enumerate(people):
        for nm, th, c in C96[p]:
            if nm.endswith("A+000E+00.pgm"):
                continue
            for N in NS:
                pb, pc = R2.probe_codes(U.dark_ae(c, N / 4.0, R))
                K = R2.codes(pb, pc, gb)
                same += np.bincount(K[i], minlength=768); other += np.bincount(np.delete(K, i, 0).ravel(), minlength=768)
    R2.LAW.update(ALPHA=keep)
    return np.clip(np.round(16 * np.log((same / same.sum()) / (other / other.sum()))), -127, 127).astype(np.int64)


def crop96(img, eL, eR):
    """the v2 crop at twice the resolution: the same face, 96 x 84, eye distance 50 px"""
    ux, uy, d = O.unit(eR - eL)
    mid = (eL + eR) / 2.0
    yy, xx = np.mgrid[0:96, 0:84].astype(float)
    return O.warp(img, mid[0], mid[1], ux, uy, d / 50.0, xx - 41.5, yy - 22.5)


def shifted(x, dy, dx, res):
    k = 1 if res == 48 else 2
    return R2.shifted(x, k * dy, k * dx)


# ---------------------------------------------------------------- the workers: every method on one try
G = {}


def init(ctx):
    G.clear(); G.update(ctx)
    import dark_law as DL
    DL.SHAPE, DL.SCALE_OF = (48, 42), None
    import v2_count as VC
    VC.setlaw(FW=ctx["FW"], ALPHA=ctx["ALPHA"])


def norm_b(x, gmean):
    return np.maximum(x, 0) * (gmean / max(float(np.mean(x)), 1e-9))


def describe(name, res, x, model=None):
    fn = model if model is not None else dict((n, f) for n, f, *_ in CLASSIC)[name]
    dt = dict((n, t) for n, _, _, t in CLASSIC).get(name, np.float32)
    return np.asarray(fn(x)).astype(dt)


def sim_of(name):
    return dict((n, s) for n, _, s, _ in CLASSIC).get(name, CL.cos_sim)


def all_scores(x48, x96, maps):
    """every method: the best of the 9 placements of this try against every map (maps: method -> matrix / tables)"""
    import dark_law as DL, unlock_law as U
    out = {}
    pa = R2.probe_all(x48) if len(SHIFTS) == 9 else [R2.probe_codes(x48)]
    out["FACE LAW v2"] = R2.score(pa, maps["FACE LAW v2"])
    if "FACE LAW v2@96" in maps:
        use96(True)
        pa96 = R2.probe_all(x96) if len(SHIFTS) == 9 else [R2.probe_codes(x96)]
        out["FACE LAW v2@96"] = R2.score(pa96, maps["FACE LAW v2@96"])
        if len(pa96) == 9:                                  # the centre placement alone: no search
            out["FACE LAW v2@96 nosearch"] = R2.score([pa96[4]], maps["FACE LAW v2@96"])
        use96(False)
        if len(pa) == 9:
            out["FACE LAW v2 nosearch"] = R2.score([pa[4]], maps["FACE LAW v2"])
    gs, gf = maps["FACE LAW v1"]
    best = None
    for dy, dx in SHIFTS:
        ps, pc = U.probe_feats(R2.shifted(x48, dy, dx))
        s = G["F"][DL.codes(ps, pc, gs, gf)].sum(1) / len(ps)
        best = s if best is None else np.maximum(best, s)
    out["FACE LAW v1"] = best
    for res, x in ((48, x48), (96, x96)):
        xb = norm_b(x, G["gmean"][res])
        for name, *_ in CLASSIC + [(l[0],) for l in LEARNED]:
            key = "%s@%d" % (name, res)
            model = G["models"].get(key)
            sim = sim_of(name)
            best = None
            for dy, dx in SHIFTS:
                s = sim(describe(name, res, shifted(xb, dy, dx, res), model), maps[key])
                best = s if best is None else np.maximum(best, s)
            out[key] = best
    return out


def p1_task(job):
    import unlock_law as U
    x48, x96 = job
    return all_scores(x48, x96, G["maps"]), U.lbin(U.imbalance(x48))


def p4_task(x48):
    """every method at 48 x 42 only (PIE): the 96 x 84 entries are skipped"""
    import unlock_law as U
    keep = {k: v for k, v in G["maps"].items()}
    out = {}
    out["FACE LAW v2"] = R2.score(R2.probe_all(x48), keep["FACE LAW v2"])
    import dark_law as DL
    gs, gf = keep["FACE LAW v1"]
    best = None
    for dy, dx in SHIFTS:
        ps, pc = U.probe_feats(R2.shifted(x48, dy, dx))
        s_ = G["F"][DL.codes(ps, pc, gs, gf)].sum(1) / len(ps)
        best = s_ if best is None else np.maximum(best, s_)
    out["FACE LAW v1"] = best
    xb = norm_b(x48, G["gmean"][48])
    for name, *_ in CLASSIC + [(l[0],) for l in LEARNED]:
        key = "%s@48" % name
        sim = sim_of(name); model = G["models"].get(key); best = None
        for dy, dx in SHIFTS:
            s_ = sim(describe(name, 48, shifted(xb, dy, dx, 48), model), keep[key])
            best = s_ if best is None else np.maximum(best, s_)
        out[key] = best
    return out, U.lbin(U.imbalance(x48))


def map_matrices(c48s, c96s, models, gmean, res_list=RES, law96=None):
    """the enrolment side of every method for a list of clean faces"""
    import dark_law as DL
    M = {"FACE LAW v2": np.array([R2.map_bins(c) for c in c48s])}
    if law96 is not None:
        keep = (R2.LAW["ALPHA"], R2.LAW["FW"]); R2.LAW.update(ALPHA=law96[0], FW=law96[1])
        M["FACE LAW v2@96"] = np.array([R2.map_bins(c) for c in c96s])
        R2.LAW.update(ALPHA=keep[0], FW=keep[1])
    GL = [DL.gallery_feats(norm_b(c, gmean[48])) for c in c48s]            # v1's flat rule is absolute: one brightness
    M["FACE LAW v1"] = (np.array([g[0] for g in GL]), np.array([g[1] for g in GL]))
    for res, cs in ((48, c48s), (96, c96s)):
        if res not in res_list:
            continue
        for name, *_ in CLASSIC + [(l[0],) for l in LEARNED]:
            key = "%s@%d" % (name, res)
            M[key] = np.array([describe(name, res, norm_b(c, gmean[res]), models.get(key)) for c in cs])
    return M


def count_F(C48, people, front, R):
    """v1's F counted as unlock_law does, on these people's crops"""
    import dark_law as DL, unlock_law as U
    GA = [DL.gallery_feats(front[p]) for p in people]
    gs = np.array([g[0] for g in GA]); gf = np.array([g[1] for g in GA])
    same, other = np.ones(512), np.ones(512)
    for i, p in enumerate(people):
        for nm, th, c in C48[p]:
            if nm.endswith("A+000E+00.pgm"):
                continue
            for N in NS:
                ps, pc = U.probe_feats(U.dark_ae(c, N, R)); K = DL.codes(ps, pc, gs, gf)
                same += np.bincount(K[i], minlength=512); other += np.bincount(np.delete(K, i, 0).ravel(), minlength=512)
    return np.clip(np.round(16 * np.log((same / same.sum()) / (other / other.sum()))), -127, 127).astype(int)


def train_models(train48, train96, labels):
    models = {}
    for res, X in ((48, train48), (96, train96)):
        for name, ctor in LEARNED:
            models["%s@%d" % (name, res)] = ctor(np.array(X), np.array(labels))
    return models


def tar(g, i, far):
    """owner accepted (%) when the threshold lets in a share `far` of strangers (0: none of them)"""
    g, i = np.asarray(g), np.asarray(i)
    thr = i.max() if far == 0 else np.quantile(i, 1 - far)
    return 100.0 * float((g > thr).mean())


def eer(g, i):
    s = np.sort(np.concatenate([g, i]))
    best = (1.0, 0.0)
    for t in s[:: max(1, len(s) // 4000)]:
        fr, fa = float((g <= t).mean()), float((i > t).mean())
        if abs(fr - fa) < best[0]:
            best = (abs(fr - fa), (fr + fa) / 2)
    return 100.0 * best[1]


def wilson(p, n, z=1.96):
    p = p / 100.0
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


# ---------------------------------------------------------------- the stages
def load_people():
    import yale as Y
    TR = json.load(open(os.path.join(HERE, "yale_eyes_truth.json")))["eyes"]
    ids = [p for p in Y.IDS if p in TR]
    return Y, ids, [p for p in Y.IDS[0::2] if p in TR], [p for p in Y.IDS[1::2] if p in TR]


def crops_table():
    Z = np.load(CROPS, allow_pickle=True)
    meta = json.loads(str(Z["meta"]))
    c48, c96 = Z["c48"], Z["c96"]                     # once: an npz member is decompressed again at every access
    C48, C96 = {}, {}
    for k, (p, nm, th) in enumerate(meta):
        C48.setdefault(p, []).append((nm, th, c48[k].astype(float)))
        C96.setdefault(p, []).append((nm, th, c96[k].astype(float)))
    return C48, C96


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    if stage == "prep":
        import yale as Y
        R2.load()
        Yy, ids, _, _ = load_people()
        AUTO = json.load(open(EYES))
        meta, c48, c96 = [], [], []
        for p in ids:
            for im in Y.PEOPLE[p]:
                if im["I"].sum() <= 0 or im["theta"] > 77.0:
                    continue
                e = AUTO.get("%s/%s" % (p, im["name"]))
                if e is None:
                    continue
                I = Y.read_pgm(os.path.join(Y.ROOT, p, im["name"])).astype(float)
                eL, eR = np.array(e[0]), np.array(e[1])
                meta.append((p, im["name"], float(im["theta"])))
                c48.append(R2.crop(I, eL, eR).astype(np.float32)); c96.append(crop96(I, eL, eR).astype(np.float32))
        np.savez_compressed(CROPS, c48=np.array(c48), c96=np.array(c96), meta=json.dumps(meta))
        print("crops: %d photos of %d people, 48 x 42 and 96 x 84, from the automatic eyes  [%.0fs]" % (len(meta), len(ids), time.time() - t0))
    elif stage in ("p1", "p2"):
        import unlock_law as U
        import v2_count as VC
        Yy, ids, hA, hB = load_people()
        if os.environ.get("LIMIT"):
            k = int(os.environ["LIMIT"]); hA, hB = hA[:k], hB[:k]; ids = hA + hB
        C48, C96 = crops_table()
        D = json.load(open(os.path.join(HERE, "v2_design.json")))
        R = np.random.default_rng(1001 if stage == "p1" else 1002)
        res, raw, MATS = {}, {}, {}
        for fold, (A, B) in enumerate(((hA, hB), (hB, hA))):
            front48 = {p: next(c for nm, th, c in C48[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
            front96 = {p: next(c for nm, th, c in C96[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
            VC.setlaw(ALPHA=D["ALPHA"])
            FW = VC.count_FW(C48, A, front48, R)
            F = count_F(C48, A, front48, R)
            tr48 = [c for p in A for nm, th, c in C48[p]]; tr96 = [c for p in A for nm, th, c in C96[p]]
            lab = [p for p in A for _ in C48[p]]
            gmean = {48: float(np.mean([front48[p].mean() for p in ids])), 96: float(np.mean([front96[p].mean() for p in ids]))}
            models = train_models([norm_b(c, gmean[48]) for c in tr48], [norm_b(c, gmean[96]) for c in tr96], lab)
            VC.setlaw(FW=FW, ALPHA=D["ALPHA"])
            ALPHA96 = VC.alpha_of([front96[p] for p in ids])
            FW96 = count_FW96(C96, A, front96, np.random.default_rng(7000 + fold), ALPHA96)
            law96 = (ALPHA96, FW96)
            if stage == "p1":
                people = list(B) + list(A)
                maps = map_matrices([front48[p] for p in people], [front96[p] for p in people], models, gmean, law96=law96)
                probes = [(i, N, U.dark_ae(c, N, R), U.dark_ae(c96, N / 4.0, R)) for i, p in enumerate(B)
                          for (nm, th, c), (_, _, c96) in zip(C48[p], C96[p]) if not nm.endswith("A+000E+00.pgm") for N in NS]
                ctx = dict(FW=FW, ALPHA=D["ALPHA"], F=F, models=models, gmean=gmean, maps=maps, FW96=FW96, ALPHA96=ALPHA96)
                with Pool(3, init, (ctx,)) as pool:
                    out = pool.map(p1_task, [(x48, x96) for _, _, x48, x96 in probes], chunksize=8)
                coh = list(range(len(B), len(people)))
                for (i, N, _, _), (S, l) in zip(probes, out):
                    for m, s in S.items():
                        r = res.setdefault("%s|%d|%d" % (m, N, l), ([], [], [], []))
                        for o in range(len(B)):
                            r[0 if o == i else 1].append(U.margins(s, o, coh)); r[2 if o == i else 3].append(float(s[o]))
            else:
                photos = [(p, U.dark_ae(c, 1000, R), U.dark_ae(c96, 250.0, R)) for p in B
                          for (nm, th, c), (_, _, c96) in zip(C48[p], C96[p]) if th <= 25.5]
                maps = map_matrices([x for _, x, _ in photos], [x for _, _, x in photos], models, gmean, law96=law96)
                ctx = dict(FW=FW, ALPHA=D["ALPHA"], F=F, models=models, gmean=gmean, maps=maps, FW96=FW96, ALPHA96=ALPHA96)
                with Pool(3, init, (ctx,)) as pool:
                    out = pool.map(p1_task, [(x48, x96) for _, x48, x96 in photos], chunksize=4)
                n = len(photos)
                MATS.setdefault("labels_%d" % fold, np.array([p for p, _, _ in photos]))
                for m in out[0][0]:
                    S = np.array([o[0][m] for o in out])                                  # S[i, j]: photo i's try vs photo j's enrolment
                    Ssym = (S + S.T) / 2.0
                    MATS["%s|%d" % (m, fold)] = Ssym.astype(np.float32)
                    r = res.setdefault(m, ([], []))
                    for i in range(n):
                        for j in range(i + 1, n):
                            r[0 if photos[i][0] == photos[j][0] else 1].append(float(Ssym[i, j]))
            print("%s fold %d done  [%.0fs]" % (stage, fold, time.time() - t0), flush=True)
        TAG = os.environ.get("TAG", "")
        np.savez_compressed(os.path.join(OUT, "%s%s_scores.npz" % (stage, TAG)), **{k: np.array(v, dtype=object) for k, v in res.items()})
        if MATS:
            np.savez_compressed(os.path.join(OUT, "p2%s_matrices.npz" % TAG), **MATS)
        summary = {}
        if stage == "p1":
            for m in method_names():
                for N in NS:
                    for cls, lights in (("front", (0,)), ("all", (0, 1, 2))):
                        rows = [res["%s|%d|%d" % (m, N, l)] for l in lights if "%s|%d|%d" % (m, N, l) in res]
                        if not rows:
                            continue
                        g_t, i_t, g_r, i_r = [np.concatenate([r[k] for r in rows]) for k in range(4)]
                        summary["%s|%d|%s" % (m, N, cls)] = dict(tnorm=[tar(g_t, i_t, 0), tar(g_t, i_t, 1e-3)], raw=[tar(g_r, i_r, 0), tar(g_r, i_r, 1e-3)],
                                                                   owners=len(g_t), strangers=len(i_t))
        else:
            for m, (g, i) in res.items():
                g, i = np.array(g), np.array(i)
                summary[m] = dict(tar=[tar(g, i, 1e-2), tar(g, i, 1e-3), tar(g, i, 0)], eer=eer(g, i), genuine=len(g), impostor=len(i),
                                  ci_1e3=wilson(tar(g, i, 1e-3), len(g)))
        json.dump(summary, open(os.path.join(OUT, "%s%s_summary.json" % (stage, TAG)), "w"), indent=1)
        print("-> %s  [%.0fs]" % (os.path.join(OUT, "%s%s_summary.json" % (stage, TAG)), time.time() - t0))
    elif stage == "p3":
        import yale as Y, unlock_law as U
        import v2_count as VC
        from yale_baselines import load_full
        D = json.load(open(os.path.join(HERE, "v2_design.json")))
        full = load_full()
        box2 = lambda I: I[:192, :168].reshape(96, 2, 84, 2).mean((1, 3))
        C48, C96 = {}, {}
        for p in Y.IDS:
            names = sorted(im["name"] for im in Y.PEOPLE[p])
            th_of = {im["name"]: im["theta"] for im in Y.PEOPLE[p]}
            for nm in names:
                I = Y.read_pgm(os.path.join(Y.ROOT, p, nm)).astype(float)
                if I.shape != (192, 168) or I.sum() <= 0:
                    continue
                C48.setdefault(p, []).append((nm, th_of[nm], Y.box(I))); C96.setdefault(p, []).append((nm, th_of[nm], box2(I)))
        ids = [p for p in Y.IDS if p in C48]
        front48 = {p: next(c for nm, th, c in C48[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
        gal = [(p, c48, c96) for p in ids for (nm, th, c48), (_, _, c96) in zip(C48[p], C96[p]) if Y.subset(th) == 1]
        gmean = {48: float(np.mean([g[1].mean() for g in gal])), 96: float(np.mean([g[2].mean() for g in gal]))}
        models = train_models([norm_b(g[1], gmean[48]) for g in gal], [norm_b(g[2], gmean[96]) for g in gal], [g[0] for g in gal])
        R = np.random.default_rng(1003)
        correct = {}
        hA, hB = ids[0::2], ids[1::2]
        for fold, (A, B) in enumerate(((hA, hB), (hB, hA))):
            VC.setlaw(ALPHA=D["ALPHA"])
            FW = VC.count_FW(C48, A, front48, R); F = count_F(C48, A, front48, R)
            VC.setlaw(FW=FW, ALPHA=D["ALPHA"])
            front96 = {p: next(c for nm, th, c in C96[p] if nm.endswith("A+000E+00.pgm")) for p in ids}
            ALPHA96 = VC.alpha_of([front96[p] for p in ids])
            FW96 = count_FW96(C96, A, front96, np.random.default_rng(7100 + fold), ALPHA96)
            maps = map_matrices([g[1] for g in gal], [g[2] for g in gal], models, gmean, law96=(ALPHA96, FW96))
            owner = np.array([ids.index(g[0]) for g in gal])
            probes = [(p, Y.subset(th), c48, c96) for p in B for (nm, th, c48), (_, _, c96) in zip(C48[p], C96[p]) if Y.subset(th) >= 2]
            ctx = dict(FW=FW, ALPHA=D["ALPHA"], F=F, models=models, gmean=gmean, maps=maps, FW96=FW96, ALPHA96=ALPHA96)
            with Pool(3, init, (ctx,)) as pool:
                out = pool.map(p1_task, [(c48, c96) for _, _, c48, c96 in probes], chunksize=8)
            for (p, sb, _, _), (S, _) in zip(probes, out):
                for m, sc in S.items():
                    best = np.full(len(ids), -np.inf)
                    np.maximum.at(best, owner, sc)
                    correct.setdefault(m, {}).setdefault(sb, []).append(int(np.argmax(best) == ids.index(p)))
            print("p3 fold %d done: %d probes  [%.0fs]" % (fold, len(probes), time.time() - t0), flush=True)
        tag = ("p3_nosearch" if os.environ.get("NOSEARCH") else "p3") + os.environ.get("TAG", "")
        summary = {m: {str(sb): 100.0 * float(np.mean(v)) for sb, v in sorted(d.items())} for m, d in correct.items()}
        json.dump(summary, open(os.path.join(OUT, "%s_summary.json" % tag), "w"), indent=1)
        print("-> %s  [%.0fs]" % (os.path.join(OUT, "%s_summary.json" % tag), time.time() - t0))
    elif stage == "p4protocol":
        files = ["h2h_all.py", "classics.py", "yale_baselines.py", "v2ref.py", "v2ops.py", "v2law.py", "v2_count.py", "personal.py",
                 "phone_eyes.py", "unlock_law.py", "dark_law.py", "v2_design.json", "lanes_v2.json", "v2ops_tables.json", "unlock_lanes.json",
                 "lanes_sector.json", "lane_cast.json", "data/pie/lights=27.mat"]
        txt = ["SEALED CMU PIE HEAD-TO-HEAD  (written %s, before any PIE image is read by these methods)" % time.strftime("%Y-%m-%d %H:%M:%S"),
               "data      CMU PIE lights=27.mat: 68 people x 21 flashes, 32 x 32, warped onto the law's canonical eyes at 48 x 42",
               "          (personal.pie_crops: the dataset's eye alignment, not the eye law: this tests the FACE CODES)",
               "use       FACE LAW v2 was designed on Extended Yale B only and has never been run on PIE; PIE was used before by",
               "          v1's tests (sealed identification, dark identification, the unlock tests) and as v1's fresh-stranger check",
               "methods   FACE LAW v2 (the sealed law: lanes_v2.json), FACE LAW v1 (as shipped: unlock_lanes.json's F), Gradientfaces,",
               "          Weberfaces, Tan-Triggs, LBP, LTP, LGBPHS, HOG, Eigenfaces, Eigenfaces-3, Fisherfaces, all at 48 x 42 (PIE is",
               "          32 x 32: a 96 x 84 copy would only be interpolation); the learned ones trained on Extended Yale B (all 37",
               "          people, automatic-eye crops), never on PIE; every method gets the best of the 9 placements (+-1 px)",
               "enrol     flash 10 (frontal) of each person; tries: the other 20 flashes, each at N = 20 and N = 1000 photons per",
               "          pixel (Law 1 with auto-exposure), one draw, seed 20261028",
               "metric    owner accepted %% when 0 / 1 in 1,000 strangers get in, raw score (one threshold) and T-norm against the",
               "          other half of the PIE people (even / odd), all light and the front-light class; run ONCE; reported as is",
               "prediction (mine, written before the run): at N = 1000 most methods sit near their ceiling (PIE saturated an earlier",
               "          identification test); by the raw score at 1 in 1,000, FACE LAW v2 is in the top three but not certainly",
               "          first; at N = 20 the ratio codes (FACE LAW v2, Weberfaces, LTP) lead the gradient and subspace methods"]
        for f in files:
            txt.append("sha256 %s %s" % (f, hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()))
        open(os.path.join(HERE, "pie_v2_protocol.txt"), "w").write("\n".join(txt) + "\n")
        h = hashlib.sha256(open(os.path.join(HERE, "pie_v2_protocol.txt"), "rb").read()).hexdigest()
        open(os.path.join(HERE, "pie_v2_protocol.sha256"), "w").write(h + "  pie_v2_protocol.txt\n")
        print("\n".join(l for l in txt if not l.startswith("sha256 "))); print("pie_v2_protocol sha256", h)
    elif stage == "p4":
        import unlock_law as U, personal as P
        import v2_count as VC
        for line in open(os.path.join(HERE, "pie_v2_protocol.txt")).read().splitlines():
            if line.startswith("sha256 "):
                _, name, h = line.split()
                assert hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest() == h, "%s changed since the seal" % name
        Z = R2.load(); D = json.load(open(os.path.join(HERE, "v2_design.json")))
        F = np.array(json.load(open(os.path.join(HERE, "unlock_lanes.json")))["F"]["table"])
        Yy, ids, _, _ = load_people()
        C48, C96 = crops_table()
        tr48 = [c for p in ids for nm, th, c in C48[p]]; lab = [p for p in ids for _ in C48[p]]
        gmean = {48: float(np.mean([c.mean() for c in tr48])), 96: 1.0}
        models = {}
        for name, ctor in LEARNED:
            models["%s@48" % name] = ctor(np.array([norm_b(c, gmean[48]) for c in tr48]), np.array(lab))
        X, eyes = P.pie_crops()
        n = X.shape[0]
        R = np.random.default_rng(20261028)
        halves = (list(range(0, n, 2)), list(range(1, n, 2)))
        res = {}
        for fold, (Bh, Ah) in enumerate(((halves[1], halves[0]), (halves[0], halves[1]))):
            people = list(Bh) + list(Ah)
            ctx = dict(FW=Z["FW"], ALPHA=D["ALPHA"], F=F, models=models, gmean=gmean, maps=None)
            VC.setlaw(FW=Z["FW"], ALPHA=D["ALPHA"])
            fronts = [X[p, 10] for p in people]
            ctx["maps"] = map_matrices(fronts, None, models, gmean, res_list=(48,))
            probes = [(i, N, U.dark_ae(X[p, k], N, R)) for i, p in enumerate(Bh) for k in range(21) if k != 10 for N in (20, 1000)]
            with Pool(3, init, (ctx,)) as pool:
                out = pool.map(p4_task, [x for _, _, x in probes], chunksize=8)
            coh = list(range(len(Bh), len(people)))
            for (i, N, _), (S, l) in zip(probes, out):
                for m, sc in S.items():
                    r = res.setdefault("%s|%d|%d" % (m, N, l), ([], [], [], []))
                    for o in range(len(Bh)):
                        r[0 if o == i else 1].append(U.margins(sc, o, coh)); r[2 if o == i else 3].append(float(sc[o]))
            print("p4 fold %d done: %d tries  [%.0fs]" % (fold, len(probes), time.time() - t0), flush=True)
        summary = {}
        for key in sorted({k.rsplit("|", 2)[0] + "|" + k.split("|")[1] for k in res}):
            m, N = key.split("|")
            for cls, lights in (("front", (0,)), ("all", (0, 1, 2))):
                rows = [res["%s|%s|%d" % (m, N, l)] for l in lights if "%s|%s|%d" % (m, N, l) in res]
                if not rows:
                    continue
                g_t, i_t, g_r, i_r = [np.concatenate([r[k] for r in rows]) for k in range(4)]
                summary["%s|%s|%s" % (m, N, cls)] = dict(tnorm=[tar(g_t, i_t, 0), tar(g_t, i_t, 1e-3)], raw=[tar(g_r, i_r, 0), tar(g_r, i_r, 1e-3)],
                                                           owners=len(g_t), strangers=len(i_t))
        json.dump(summary, open(os.path.join(HERE, "pie_v2_results.json"), "w"), indent=1)
        h = hashlib.sha256(open(os.path.join(HERE, "pie_v2_results.json"), "rb").read()).hexdigest()
        open(os.path.join(HERE, "pie_v2_results.sha256"), "w").write(h + "  pie_v2_results.json\n")
        print("pie_v2_results.json sha256 %s  [%.0fs]" % (h, time.time() - t0))
