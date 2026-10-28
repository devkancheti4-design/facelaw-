"""DARK FACE pipeline (laws only).  Stages, each a separate command so the seal sits between design and test:
    split               fixed-seed split of the 6,000 labelled images: 4,700 design, 300 design-validation, 1,000 test
    count FACE          count faces (labels) and random background on the 4,700 design images
    mine FACE CELL R    hard-negative round R: the current law's own false alarms on design images, counted
    val FACE CELL R     the law after round R on the 300 design-validation images (raw peaks saved; AP per prior weight)
    freeze FACE CELL R LAMBDA   the chosen configuration -> darkface/design.json (integer tables)
    test                the frozen, engine-authored law (darkface/lanes.json) on the 1,000 sealed test images, once
    python3 -I darkface_run.py STAGE ARGS"""
import glob, json, os, pickle, sys, time, hashlib
from multiprocessing import Pool
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import darkface_law as L

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "darkface")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "darkface")
os.makedirs(OUT, exist_ok=True)
P = L.T * L.T
NBG_LEVELS, NBG_PER = 3, 40              # random background: 3 levels per image (chosen in proportion to their windows), 40 each
NNEG = 150                               # hard negatives kept per mined image
MINE_IMAGES = 1200                       # design images per mining round (disjoint across rounds)
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)          # the grid the design-validation choice was made on
NPROC = 11


def find_pairs():
    imgs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(ROOT, "**", "image", "*.*"), recursive=True)}
    labs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(ROOT, "**", "label", "*.txt"), recursive=True)}
    keys = sorted(set(imgs) & set(labs), key=lambda k: (len(k), k))
    return [(k, imgs[k], labs[k]) for k in keys]


def read_label(path):
    rows = [l.split() for l in open(path).read().strip().splitlines()]
    n = int(rows[0][0])
    return np.array([[float(v) for v in r[:4]] for r in rows[1:1 + n]]).reshape(-1, 4)


def load_gray(path):
    return L.to_gray(np.asarray(Image.open(path).convert("RGB")))


def items(keys):
    return [(k, os.path.join(ROOT, "image", k + ".png"), os.path.join(ROOT, "label", k + ".txt")) for k in keys]


# ----------------------------------------------------------------------- worker state (macOS spawn: via initializer) ---
W = {}


def init(face, cell, tables=None):
    L.FACE, L.CELL = face, cell
    W.clear(); W["scales"] = L.pyramid_scales()
    if tables:
        W.update(tables)


def level_shape(H, Wd, s):
    return (int(round(H / s)) + 2 * L.PAD, int(round(Wd / s)) + 2 * L.PAD) if abs(s - 1) > 1e-9 else (H + 2 * L.PAD, Wd + 2 * L.PAD)


def add(Cd, Cg, cd, cg, r, c):
    Cd[np.arange(P), cd[r:r + L.T, c:c + L.T].ravel()] += 1
    Cg[np.arange(P), cg[r:r + L.T, c:c + L.T].ravel()] += 1


def count_image(args):
    key, ip, lp, seed = args
    R = np.random.default_rng(seed)
    g = load_gray(ip); gt = read_label(lp); SC = W["scales"]
    nl = len(SC)
    Fd, Fg = np.zeros((P, L.NDIR), np.int64), np.zeros((P, L.NDOG), np.int64)
    Bd, Bg = np.zeros((P, L.NDIR), np.int64), np.zeros((P, L.NDOG), np.int64)
    nf, nw = np.zeros(nl, np.int64), np.zeros(nl, np.int64)
    sigs = []
    cache = {}
    def lev(k):
        if k not in cache:
            cache[k] = L.level(g, SC[k])
        return cache[k]
    for k, s in enumerate(SC):
        nw[k] = L.n_windows(*level_shape(g.shape[0], g.shape[1], s))
    o = (L.T - L.FACE) // 2
    for box in gt:
        k = L.level_of_face(box, SC); r, c = L.window_of_face(box, SC[k]); cd, cg = lev(k)
        if 0 <= r and 0 <= c and r + L.T <= cd.shape[0] and c + L.T <= cd.shape[1]:
            add(Fd, Fg, cd, cg, r, c); nf[k] += 1
            sigs.append(int(((cd > 0) | (cg > 0))[r + o:r + o + L.FACE, c + o:c + o + L.FACE].sum()))
    for k in R.choice(nl, NBG_LEVELS, replace=False, p=nw / nw.sum()):
        cd, cg = lev(int(k)); got = 0
        for _ in range(20 * NBG_PER):
            if got == NBG_PER or cd.shape[0] <= L.T or cd.shape[1] <= L.T:
                break
            r, c = int(R.integers(cd.shape[0] - L.T + 1)), int(R.integers(cd.shape[1] - L.T + 1))
            if len(gt) and L.iou_many(np.array(L.box_of_window(r, c, SC[k])), gt).max() >= 0.3:
                continue
            add(Bd, Bg, cd, cg, r, c); got += 1
    return dict(Fd=Fd, Fg=Fg, Bd=Bd, Bg=Bg, nf=nf, nw=nw, sigs=sigs)


def tables(face, cell, rounds, w_rand=1.0):
    """counted integer law from the counts and the first `rounds` hard-negative rounds; random background counts weighted
    by w_rand (1 = as counted; < 1 = the background is mostly the law's own false alarms)"""
    C = dict(np.load(os.path.join(OUT, "counts_F%d.npz" % face)))
    Bd, Bg = w_rand * C["Bd"].astype(float), w_rand * C["Bg"].astype(float)
    for r in range(1, rounds + 1):
        H = np.load(os.path.join(OUT, "hn_F%d_C%d_r%d.npz" % (face, cell, r)))
        Bd += H["Bd"]; Bg += H["Bg"]
    L.CELL = cell
    ci = L.cell_of_pos(); ncell = (L.T // cell) ** 2
    def llr(F, B):
        Fc, Bc = np.zeros((ncell, F.shape[1])), np.zeros((ncell, B.shape[1]))
        np.add.at(Fc, ci, F); np.add.at(Bc, ci, B)
        Fc += 1; Bc += 1
        return np.rint(L.UNIT * (np.log(Fc / Fc.sum(1, keepdims=True)) - np.log(Bc / Bc.sum(1, keepdims=True)))).astype(np.int64)
    Cd, Cg = llr(C["Fd"], Bd), llr(C["Fg"], Bg)
    prior_f = L.UNIT * np.log((C["nf"] + 1) / (C["nw"] + 1))          # rounded ONCE, after the prior weight is applied
    minsig = int(np.floor(np.percentile(C["sigs"], 1)))
    return dict(Cd=Cd, Cg=Cg, Ld=L.expand(Cd), Lg=L.expand(Cg), prior=np.rint(prior_f).astype(np.int64), prior_f=prior_f, minsig=minsig)


WUNIT = 256                              # shared-covariance law: integer tables = round(256 w), prior = round(256 lambda ln)
OMEGAS = (None, 0.5)                     # hard-negative share of the background statistics: as counted, or one half


def feat_image(args):
    """features phi of every labelled face and of random background windows (3 levels x 60), plus the prior counts"""
    key, ip, lp, seed = args
    R = np.random.default_rng(seed)
    g = load_gray(ip); gt = read_label(lp); SC = W["scales"]; nl = len(SC)
    Fx, Bx = [], []
    nf, nw = np.zeros(nl, np.int64), np.zeros(nl, np.int64)
    sigs = []
    cache = {}
    def lev(k):
        if k not in cache:
            cache[k] = L.level(g, SC[k])
        return cache[k]
    for k, s in enumerate(SC):
        nw[k] = L.n_windows(*level_shape(g.shape[0], g.shape[1], s))
    o = (L.T - L.FACE) // 2
    for box in gt:
        k = L.level_of_face(box, SC); r, c = L.window_of_face(box, SC[k]); cd, cg = lev(k)
        if 0 <= r and 0 <= c and r + L.T <= cd.shape[0] and c + L.T <= cd.shape[1]:
            Fx.append(L.window_features(cd, cg, r, c)); nf[k] += 1
            sigs.append(int(((cd > 0) | (cg > 0))[r + o:r + o + L.FACE, c + o:c + o + L.FACE].sum()))
    for k in R.choice(nl, NBG_LEVELS, replace=False, p=nw / nw.sum()):
        cd, cg = lev(int(k)); got = 0
        for _ in range(20 * 60):
            if got == 60 or cd.shape[0] <= L.T or cd.shape[1] <= L.T:
                break
            r, c = int(R.integers(cd.shape[0] - L.T + 1)), int(R.integers(cd.shape[1] - L.T + 1))
            if len(gt) and L.iou_many(np.array(L.box_of_window(r, c, SC[k])), gt).max() >= 0.3:
                continue
            Bx.append(L.window_features(cd, cg, r, c)); got += 1
    D = (L.T // L.CELL) ** 2 * 10
    return dict(F=np.array(Fx, np.int8).reshape(-1, D), B=np.array(Bx, np.int8).reshape(-1, D), nf=nf, nw=nw, sigs=sigs)


def moments(X, chunk=100000):
    n = len(X); s = np.zeros(X.shape[1]); S2 = np.zeros((X.shape[1], X.shape[1]))
    for i in range(0, n, chunk):
        x = X[i:i + chunk].astype(np.float64); s += x.sum(0); S2 += x.T @ x
    return n, s / n, S2 / n


def lda_tables(face, cell, rounds, omega=None, eps=0.01):
    """the shared-covariance Gaussian-Bayes law from measured moments: background = random windows + the law's own false
    alarms of the first `rounds` rounds (omega = their share of the background; None = as counted)"""
    Z = np.load(os.path.join(OUT, "feats_F%d_C%d.npz" % (face, cell)))
    nr, mr, Er = moments(Z["B"])
    nb, mb, Eb = nr, mr, Er
    if rounds:
        H = np.vstack([np.load(os.path.join(OUT, "hnf_F%d_C%d_r%d.npz" % (face, cell, r)))["H"] for r in range(1, rounds + 1)])
        nh, mh, Eh = moments(H)
        om = nh / (nh + nr) if omega is None else omega
        mb, Eb = (1 - om) * mr + om * mh, (1 - om) * Er + om * Eh
    Sig = Eb - np.outer(mb, mb)
    _, mf, _ = moments(Z["F"])
    w = np.linalg.solve(Sig + eps * np.mean(np.diag(Sig)) * np.eye(len(mf)), mf - mb)
    L.CELL = cell
    Wc = np.rint(WUNIT * w).astype(np.int64).reshape(-1, 10)
    Ld, Lg = L.tables_from_w(Wc)
    prior_f = WUNIT * np.log((Z["nf"] + 1) / (Z["nw"] + 1))
    minsig = int(np.floor(np.percentile(Z["sigs"], 1)))
    return dict(Wc=Wc, Ld=Ld, Lg=Lg, prior=np.rint(prior_f).astype(np.int64), prior_f=prior_f, minsig=minsig,
                sep=float((mf - mb) @ w))


def lmine_image(args):
    key, ip, lp = args
    g = load_gray(ip); gt = read_label(lp)
    cands, maps = peaks(g, W["scales"], W["Ld"], W["Lg"], W["minsig"], W["prior"], 1.0, keep_maps=True)
    H = []
    for d in L.nms(cands, 0.3):
        if len(H) == NNEG:
            break
        if len(gt) and L.iou_many(np.array(d[1:5]), gt).max() >= 0.3:
            continue
        cd, cg = maps[d[5]]; H.append(L.window_features(cd, cg, d[6], d[7]))
    return np.array(H, np.int8).reshape(-1, (L.T // L.CELL) ** 2 * 10)


def peaks(g, SC, Ld, Lg, minsig, prior, lam, keep_maps=False):
    cands, maps = [], {}
    for k, s in enumerate(SC):
        cd, cg = L.level(g, s)
        S = L.score_level(cd, cg, Ld, Lg)
        if S is None:
            continue
        S = np.where(L.face_sig(cd, cg) >= minsig, S, L.NEG)
        from scipy.ndimage import maximum_filter
        pk = (S == maximum_filter(S, size=3)) & (S > L.NEG)
        rr, cc = np.nonzero(pk)
        if len(rr) > 300:
            j = np.argsort(-S[rr, cc], kind="stable")[:300]; rr, cc = rr[j], cc[j]
        if keep_maps:
            maps[k] = (cd, cg)
        for r, c in zip(rr, cc):
            cands.append((int(S[r, c]) + lam * float(prior[k]),) + L.box_of_window(r, c, s) + (k, int(r), int(c), int(S[r, c])))
    return cands, maps


def mine_image(args):
    key, ip, lp = args
    g = load_gray(ip); gt = read_label(lp)
    cands, maps = peaks(g, W["scales"], W["Ld"], W["Lg"], W["minsig"], W["prior"], 1.0, keep_maps=True)
    Bd, Bg = np.zeros((P, L.NDIR), np.int64), np.zeros((P, L.NDOG), np.int64)
    n = 0
    for d in L.nms(cands, 0.3):
        if n == NNEG:
            break
        if len(gt) and L.iou_many(np.array(d[1:5]), gt).max() >= 0.3:
            continue
        cd, cg = maps[d[5]]; add(Bd, Bg, cd, cg, d[6], d[7]); n += 1
    return dict(Bd=Bd, Bg=Bg, n=n)


def raw_image(args):
    key, ip, lp = args
    g = load_gray(ip)
    cands, _ = peaks(g, W["scales"], W["Ld"], W["Lg"], W["minsig"], W["prior"], 0.0)
    return key, [(d[8],) + tuple(d[1:6]) for d in cands], read_label(lp)


def ap_by_lambda(raw, gts, prior_f):
    res = {}
    for lam in LAMBDAS:
        pr = np.rint(lam * prior_f).astype(np.int64)
        dets = {k: L.nms([(d[0] + int(pr[d[5]]),) + tuple(d[1:5]) for d in v], 0.3) for k, v in raw.items()}
        res[lam] = L.average_precision(dets, gts)
    return res


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    if stage == "split":
        pairs = find_pairs()
        R = np.random.default_rng(20261026)
        idx = R.permutation(len(pairs))
        test = sorted(pairs[i][0] for i in idx[:1000]); val = sorted(pairs[i][0] for i in idx[1000:1300])
        design = sorted(pairs[i][0] for i in idx[1300:])
        json.dump(dict(n=len(pairs), seed=20261026, test=test, design_val=val, design=design), open(os.path.join(OUT, "split.json"), "w"))
        print("pairs %d -> test %d, design-validation %d, design %d" % (len(pairs), len(test), len(val), len(design)))
        print("split sha256", hashlib.sha256(open(os.path.join(OUT, "split.json"), "rb").read()).hexdigest())
        raise SystemExit
    SPLIT = json.load(open(os.path.join(OUT, "split.json")))
    assert hashlib.sha256(open(os.path.join(OUT, "split.json"), "rb").read()).hexdigest().startswith("a5881a2e"), "split changed"
    if stage == "count":
        face = int(sys.argv[2])
        jobs = [it + (i,) for i, it in enumerate(items(SPLIT["design"]))]
        tot = None
        with Pool(NPROC, init, (face, 2)) as pool:
            for i, r in enumerate(pool.imap_unordered(count_image, jobs, chunksize=4)):
                if tot is None:
                    tot = {k: (np.array(v) if k != "sigs" else list(v)) for k, v in r.items()}
                else:
                    for k, v in r.items():
                        tot[k] = tot[k] + v if k != "sigs" else tot[k] + v
                if (i + 1) % 500 == 0:
                    print("  counted %d / %d images  (%.0fs)" % (i + 1, len(jobs), time.time() - t0), flush=True)
        np.savez(os.path.join(OUT, "counts_F%d.npz" % face), **{k: np.array(v) for k, v in tot.items()})
        print("FACE %d: %d face windows counted (of %d labelled), %d random background windows, minsig (1%% of faces) %d  [%.0fs]"
              % (face, tot["nf"].sum(), sum(len(read_label(it[2])) for it in items(SPLIT["design"])), tot["Bd"][0].sum(),
                 int(np.floor(np.percentile(tot["sigs"], 1))), time.time() - t0))
    elif stage == "mine":
        face, cell, rnd = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        wr = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        tb = tables(face, cell, rnd - 1, wr)
        keys = SPLIT["design"][(rnd - 1) * MINE_IMAGES:rnd * MINE_IMAGES]
        Bd = Bg = None; n = 0
        with Pool(NPROC, init, (face, cell, tb)) as pool:
            for i, r in enumerate(pool.imap_unordered(mine_image, items(keys), chunksize=2)):
                Bd = r["Bd"] if Bd is None else Bd + r["Bd"]; Bg = r["Bg"] if Bg is None else Bg + r["Bg"]; n += r["n"]
                if (i + 1) % 300 == 0:
                    print("  mined %d / %d images  (%.0fs)" % (i + 1, len(keys), time.time() - t0), flush=True)
        np.savez(os.path.join(OUT, "hn_F%d_C%d_r%d.npz" % (face, cell, rnd)), Bd=Bd, Bg=Bg)
        print("round %d: %d hard negatives counted from %d design images  [%.0fs]" % (rnd, n, len(keys), time.time() - t0))
    elif stage == "val":
        face, cell, rnd = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        wr = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        tb = tables(face, cell, rnd, wr)
        raw, gts = {}, {}
        with Pool(NPROC, init, (face, cell, tb)) as pool:
            for key, dets, gt in pool.imap_unordered(raw_image, items(SPLIT["design_val"]), chunksize=2):
                raw[key] = dets; gts[key] = gt
        pickle.dump(dict(raw=raw, gts=gts, prior_f=tb["prior_f"]), open(os.path.join(OUT, "val_F%d_C%d_r%d_w%g.pkl" % (face, cell, rnd, wr)), "wb"))
        res = ap_by_lambda(raw, gts, tb["prior_f"])
        print("design-validation  FACE %d  CELL %d  hard-negative rounds %d  random-background weight %g  minsig %d  [%.0fs]"
              % (face, cell, rnd, wr, tb["minsig"], time.time() - t0))
        for lam, (ap, rec) in res.items():
            print("   prior weight %.2f:  AP %.2f%%   max recall %.1f%%" % (lam, 100 * ap, 100 * rec))
    elif stage == "lsample":
        face, cell = int(sys.argv[2]), int(sys.argv[3])
        jobs = [it + (i,) for i, it in enumerate(items(SPLIT["design"]))]
        F, B, nf, nw, sigs = [], [], 0, 0, []
        with Pool(NPROC, init, (face, cell)) as pool:
            for i, r in enumerate(pool.imap_unordered(feat_image, jobs, chunksize=4)):
                F.append(r["F"]); B.append(r["B"]); nf = nf + r["nf"]; nw = nw + r["nw"]; sigs += r["sigs"]
                if (i + 1) % 1000 == 0:
                    print("  sampled %d / %d images  (%.0fs)" % (i + 1, len(jobs), time.time() - t0), flush=True)
        F, B = np.vstack(F), np.vstack(B)
        np.savez(os.path.join(OUT, "feats_F%d_C%d.npz" % (face, cell)), F=F, B=B, nf=nf, nw=nw, sigs=np.array(sigs))
        print("FACE %d CELL %d: %d face windows, %d random background windows, %d features each  [%.0fs]" % (face, cell, len(F), len(B), F.shape[1], time.time() - t0))
    elif stage == "lmine":
        face, cell, rnd = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        om = None if sys.argv[5] == "n" else float(sys.argv[5])
        tb = lda_tables(face, cell, rnd - 1, om)
        keys = SPLIT["design"][(rnd - 1) * MINE_IMAGES:rnd * MINE_IMAGES]
        H = []
        with Pool(NPROC, init, (face, cell, tb)) as pool:
            for i, h in enumerate(pool.imap_unordered(lmine_image, items(keys), chunksize=2)):
                H.append(h)
                if (i + 1) % 300 == 0:
                    print("  mined %d / %d images  (%.0fs)" % (i + 1, len(keys), time.time() - t0), flush=True)
        H = np.vstack(H)
        np.savez(os.path.join(OUT, "hnf_F%d_C%d_r%d.npz" % (face, cell, rnd)), H=H)
        print("round %d: %d hard negatives from %d design images  [%.0fs]" % (rnd, len(H), len(keys), time.time() - t0))
    elif stage == "lval":
        face, cell, rnd = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        om = None if sys.argv[5] == "n" else float(sys.argv[5])
        tb = lda_tables(face, cell, rnd, om)
        raw, gts = {}, {}
        with Pool(NPROC, init, (face, cell, tb)) as pool:
            for key, dets, gt in pool.imap_unordered(raw_image, items(SPLIT["design_val"]), chunksize=2):
                raw[key] = dets; gts[key] = gt
        pickle.dump(dict(raw=raw, gts=gts, prior_f=tb["prior_f"]), open(os.path.join(OUT, "lval_F%d_C%d_r%d_o%s.pkl" % (face, cell, rnd, sys.argv[5])), "wb"))
        res = ap_by_lambda(raw, gts, tb["prior_f"])
        print("design-validation  shared-covariance law  FACE %d  CELL %d  rounds %d  hard share %s  separation %.2f  minsig %d  [%.0fs]"
              % (face, cell, rnd, sys.argv[5], tb["sep"], tb["minsig"], time.time() - t0))
        for lam, (ap, rec) in res.items():
            print("   prior weight %.2f:  AP %.2f%%   max recall %.1f%%" % (lam, 100 * ap, 100 * rec))
    elif stage == "lfreeze":
        face, cell, rnd, lam = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[6])
        om = None if sys.argv[5] == "n" else float(sys.argv[5])
        tb = lda_tables(face, cell, rnd, om)
        D = dict(law="shared-covariance", face=face, cell=cell, rounds=rnd, hard_share=sys.argv[5], prior_weight=lam, T=L.T, PAD=L.PAD,
                 K_SIG=L.K_SIG, ASPECT=L.ASPECT, WUNIT=WUNIT, scales=L.pyramid_scales(), Wc=tb["Wc"].tolist(),
                 prior=np.rint(lam * tb["prior_f"]).astype(np.int64).tolist(), minsig=tb["minsig"])
        json.dump(D, open(os.path.join(OUT, "design.json"), "w"))
        print("frozen: shared-covariance law FACE %d CELL %d rounds %d hard share %s prior weight %.2f minsig %d -> design.json sha256 %s" % (
            face, cell, rnd, sys.argv[5], lam, tb["minsig"], hashlib.sha256(open(os.path.join(OUT, "design.json"), "rb").read()).hexdigest()))
    elif stage == "freeze":
        face, cell, rnd, lam, wr = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
        tb = tables(face, cell, rnd, wr)
        D = dict(face=face, cell=cell, rounds=rnd, prior_weight=lam, random_weight=wr, T=L.T, PAD=L.PAD, K_SIG=L.K_SIG, ASPECT=L.ASPECT, UNIT=L.UNIT,
                 scales=L.pyramid_scales(), Cd=tb["Cd"].tolist(), Cg=tb["Cg"].tolist(),
                 prior=np.rint(lam * tb["prior_f"]).astype(np.int64).tolist(), minsig=tb["minsig"])
        json.dump(D, open(os.path.join(OUT, "design.json"), "w"))
        print("frozen: FACE %d CELL %d rounds %d prior weight %.2f minsig %d -> design.json sha256 %s" % (
            face, cell, rnd, lam, tb["minsig"], hashlib.sha256(open(os.path.join(OUT, "design.json"), "rb").read()).hexdigest()))
    elif stage == "protocol":
        Z = json.load(open(os.path.join(OUT, "lanes.json"))); D = Z["design"]
        assert Z["wrong"] == 0 and Z["differ"] == 0
        files = ["darkface_law.py", "darkface_run.py", "the private authoring script", "yale_gate.py", "lanes_sector.json", "lane_cast.json",
                 "darkface/split.json", "darkface/design.json", "darkface/lanes.json", "darkface/prediction.txt", "darkface/user_claim.txt"]
        txt = ["DARK FACE SEALED TEST PROTOCOL  (written %s, before any test image is read)" % time.strftime("%Y-%m-%d %H:%M:%S"),
               "data: DarkFace_Train_2021.zip, 4,378,261,971 bytes, official PKU link (Google Drive id 10W3TDvEAlZfEt88hMxoEuRULr42bIV7s);"
               " 6,000 labelled night images",
               "split: seed 20261026 -> 1,000 test images (8,143 faces), 300 design-validation, 4,700 design; split.json below",
               "design: every count, moment, threshold and choice made on the 4,700 design and 300 design-validation images only",
               "law: %s; FACE %d in a %d window; CELL %d; hard-negative rounds %d (hard share %s); prior weight %s; minsig %d"
               % (D["law"], D["face"], D["T"], D["cell"], D["rounds"], D["hard_share"], D["prior_weight"], D["minsig"]),
               "runtime: the tables rebuilt from the authored lanes ONLY (lanes.json: every lane proven on all its inputs, "
               "Python and compiled C; 0 wrong; 0 entries differing from the design)",
               "metric: VOC all-point average precision at IoU 0.5 over all 1,000 test images and ALL labelled faces (none "
               "ignored); detections = the law's boxes after NMS 0.3; run ONCE; reported whatever it is",
               "reference: fine-tuned DSFD 46.0 mAP (HLA-Face, CVPR 2021, Table 1); NOTE the published protocol differs (their "
               "evaluation split), so the comparison is indicative, not identical",
               "sealed predictions: user_claim.txt (AP > 50%) and prediction.txt (mine, about 10%), hashes below"]
        for f in files:
            txt.append("sha256 %s %s" % (f, hashlib.sha256(open(os.path.join(os.path.dirname(OUT), f), "rb").read()).hexdigest()))
        open(os.path.join(OUT, "protocol.txt"), "w").write("\n".join(txt) + "\n")
        h = hashlib.sha256(open(os.path.join(OUT, "protocol.txt"), "rb").read()).hexdigest()
        open(os.path.join(OUT, "protocol.sha256"), "w").write(h + "  darkface/protocol.txt\n")
        print("\n".join(txt)); print("protocol sha256", h)
    elif stage == "test":
        proto = open(os.path.join(OUT, "protocol.txt")).read()
        for line in proto.splitlines():
            if line.startswith("sha256 "):
                _, name, h = line.split()
                got = hashlib.sha256(open(os.path.join(os.path.dirname(OUT), name), "rb").read()).hexdigest()
                assert got == h, "%s changed since the seal" % name
        Z = json.load(open(os.path.join(OUT, "lanes.json")))
        assert Z["wrong"] == 0 and Z["differ"] == 0
        D = Z["design"]
        L.FACE, L.CELL = D["face"], D["cell"]
        gate = np.array(Z["gate"])
        minsig = int(np.argmax(gate > 0))                       # the gate lane: the first count it lets through
        assert all(gate[n] == int(n >= minsig) for n in range(len(gate)))
        Ld, Lg = L.tables_from_w(np.array(Z["Wc"]))
        tb = dict(Ld=Ld, Lg=Lg, prior=np.array(Z["prior"]), minsig=minsig)
        dets, gts = {}, {}
        with Pool(NPROC, init, (D["face"], D["cell"], tb)) as pool:
            for i, (key, raw, gt) in enumerate(pool.imap_unordered(raw_image, items(SPLIT["test"]), chunksize=2)):
                dets[key] = L.nms([(d[0] + int(tb["prior"][d[5]]),) + tuple(d[1:5]) for d in raw], 0.3); gts[key] = gt
                if (i + 1) % 200 == 0:
                    print("  tested %d / 1000 images  (%.0fs)" % (i + 1, time.time() - t0), flush=True)
        ap, rec = L.average_precision(dets, gts)
        res = dict(AP=ap, max_recall=rec, images=len(gts), faces=int(sum(len(g) for g in gts.values())),
                   detections=int(sum(len(d) for d in dets.values())))
        json.dump(res, open(os.path.join(OUT, "test_result.json"), "w"), indent=1)
        pickle.dump(dict(dets=dets, gts=gts), open(os.path.join(OUT, "test_dets.pkl"), "wb"))
        print("SEALED TEST  %d images, %d faces:  AP@0.5 = %.2f%%   max recall %.1f%%   [%.0fs]" % (
            res["images"], res["faces"], 100 * ap, 100 * rec, time.time() - t0))
