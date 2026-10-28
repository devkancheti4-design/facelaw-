"""DARK FACE v2: LAW 7 -- the verifier.  Stage 1 is the sealed, lane-authored v1 law UNCHANGED (darkface/lanes.json).
Its raw peaks hold a face for 64% of the design-validation faces: ranked perfectly they would give AP 64.1%; v1 ranked
them at 1.7%.  The loss is the ranking, so LAW 7 re-ranks every stage-1 candidate by the log-likelihood ratio of a richer
window description, every number measured on design images only:
    z      the window's pixels with gain AND offset cancelled: (x - mean) / max(std, 2 sigma_pix)        (LAWS 1 + 2)
    hog    per 2x2 cell and gradient sector: summed |grad| in noise units over the significant pixels, 16 log2(1+.)
    dog    per cell: the two centre-surround code counts                                                (LAW 2)
    extra  stage-1 score (+ level prior), level, face significance, window SNR std / sigma_pix, mean SNR mean / sigma_pix
LLR estimated by additive trees (stagewise maximum likelihood of the log-odds).  Every feature is an integer, so every
split is an integer comparison; leaf values are rounded ONCE, to round(4096 v), when the law is frozen.
    python3 -I darkface_v2.py rows N        training rows from design images [0:N] -> darkface/v2_rows.npz
    python3 -I darkface_v2.py fit           additive-tree LLR -> darkface/v2_model.pkl
    python3 -I darkface_v2.py val           design-validation AP of LAW 7 (and of stage 1 on the same candidates)
    python3 -I darkface_v2.py freeze        the trees as integers -> darkface/law7.json (leaves round(256 v), once)
    python3 -I darkface_v2.py ival [FILE]   design-validation AP of the INTEGER law (law7.json, or the synthesis's rebuild)"""
import json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import darkface_law as L
import darkface_run as R

OUT = R.OUT
NNEG_ROWS = 300                          # negatives sampled per design image (uniformly: weight = negatives / sampled)
PER_LEVEL = 300                          # stage-1 peaks per level, as sealed
ZQ = 24                                  # z quantised as round(24 z), clipped to int8
W = {}


def stage1_tables():
    Z = json.load(open(os.path.join(OUT, "lanes.json")))
    assert Z["wrong"] == 0 and Z["differ"] == 0
    D = Z["design"]
    gate = np.array(Z["gate"]); minsig = int(np.argmax(gate > 0))
    assert all(gate[n] == int(n >= minsig) for n in range(len(gate)))
    Ld, Lg = L.tables_from_w(np.array(Z["Wc"]))
    return dict(face=D["face"], cell=D["cell"], Ld=Ld, Lg=Lg, prior=np.array(Z["prior"]), minsig=minsig)


def init(model=None, law=None):
    tb = stage1_tables()
    R.init(tb["face"], tb["cell"])
    W.clear(); W.update(tb); W["scales"] = L.pyramid_scales(); W["model"] = model
    if law is not None:
        W["law"] = law


def level_maps(g, s):
    x = L.resample(g, s)
    G = gaussian_filter(x, 0.75)
    m = np.hypot(np.gradient(G, axis=1), np.gradient(G, axis=0))
    sc = max(float(np.median(m)) / 1.1774, L.Q_SIG * L.GE / max(s, 1.0))
    cd, cg = L.codes(x, s)
    return cd, cg, np.pad(x, L.PAD, mode="reflect"), np.pad(m / sc, L.PAD), sc / L.GE


def box2(a):
    """2x2 box sums, direct (an integral image in float loses the small sums of a large level)"""
    return a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:]


def features(cd, cg, xp, mp, spix, r, c, S1, k, sig):
    """LAW 7 description of the windows at padded (r, c) of one level -> int8 (n, 896), int32 (n, 5)"""
    T, n8 = L.T, L.T // 2
    ar = np.arange(T)
    win = xp[r[:, None, None] + ar[None, :, None], c[:, None, None] + ar[None, None, :]]
    mu = win.mean((1, 2)); sd = win.std((1, 2))
    z = (win - mu[:, None, None]) / np.maximum(sd, 2 * spix)[:, None, None]
    zq = np.clip(np.rint(ZQ * z), -127, 127).reshape(len(r), -1)
    a8 = 2 * np.arange(n8)
    rr, cc = r[:, None, None] + a8[None, :, None], c[:, None, None] + a8[None, None, :]
    hog = np.empty((len(r), 8, n8, n8))
    for sct in range(8):
        hog[:, sct] = box2(np.where(cd == sct + 1, mp, 0.0))[rr, cc]
    dog = np.empty((len(r), 2, n8, n8))
    for code in (1, 2):
        dog[:, code - 1] = box2((cg == code).astype(np.int32))[rr, cc]
    X8 = np.hstack([zq, np.clip(np.rint(16 * np.log2(1 + hog)), 0, 127).reshape(len(r), -1), dog.reshape(len(r), -1)]).astype(np.int8)
    Xe = np.stack([S1, np.full(len(r), k), sig, np.rint(16 * np.log2(1 + sd / spix)),
                   np.rint(16 * np.log2(1 + np.maximum(mu, 0) / spix))], 1).astype(np.int32)
    return X8, Xe


def boxes_of(r, c, s):
    cy, cx = (r - L.PAD + L.T / 2 - 0.5) * s, (c - L.PAD + L.T / 2 - 0.5) * s
    hw, hh = L.FACE * s / np.sqrt(L.ASPECT) / 2, L.FACE * s * np.sqrt(L.ASPECT) / 2
    return np.stack([cx - hw, cy - hh, cx + hw, cy + hh], 1)


def iou_matrix(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    ix = np.maximum(0, np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]))
    iy = np.maximum(0, np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]))
    inter = ix * iy
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def candidates(g, gt=None):
    """stage 1 exactly as sealed (raw peaks, score + level prior), with the LAW 7 description of every candidate;
    with gt also the label-aligned face windows (training only)"""
    SC = W["scales"]
    out = []
    gtk = {}
    if gt is not None:
        for j, box in enumerate(gt):
            k = L.level_of_face(box, SC); gtk.setdefault(k, []).append((j,) + L.window_of_face(box, SC[k]))
    for k, s in enumerate(SC):
        cd, cg, xp, mp, spix = level_maps(g, s)
        S = L.score_level(cd, cg, W["Ld"], W["Lg"])
        if S is None:
            continue
        sig = L.face_sig(cd, cg)
        S = np.where(sig >= W["minsig"], S, L.NEG)
        pk = (S == maximum_filter(S, size=3)) & (S > L.NEG)
        rr, cc = np.nonzero(pk)
        if len(rr) > PER_LEVEL:
            j = np.argsort(-S[rr, cc], kind="stable")[:PER_LEVEL]; rr, cc = rr[j], cc[j]
        src = np.zeros(len(rr), np.int8)
        if k in gtk:                                         # label-aligned face windows (src 1), never at test
            ga = np.array([(r, c) for _, r, c in gtk[k] if 0 <= r < S.shape[0] and 0 <= c < S.shape[1] and S[r, c] > L.NEG],
                          np.int64).reshape(-1, 2)
            rr, cc = np.concatenate([rr, ga[:, 0]]), np.concatenate([cc, ga[:, 1]])
            src = np.concatenate([src, np.ones(len(ga), np.int8)])
        if len(rr) == 0:
            continue
        S1 = S[rr, cc] + int(W["prior"][k])
        X8, Xe = features(cd, cg, xp, mp, spix, rr, cc, S1, k, sig[rr, cc])
        out.append((X8, Xe, boxes_of(rr, cc, s), src))
    if not out:
        return np.zeros((0, 896), np.int8), np.zeros((0, 5), np.int32), np.zeros((0, 4)), np.zeros(0, np.int8)
    return tuple(np.concatenate(a) for a in zip(*out))


def rows_image(args):
    i, key, ip, lp = args
    g = R.load_gray(ip); gt = R.read_label(lp)
    X8, Xe, B, src = candidates(g, gt)
    o = iou_matrix(B, gt).max(1) if len(gt) else np.zeros(len(B))
    pos = (o >= 0.5) | (src == 1)
    neg = np.nonzero((o < 0.3) & (src == 0))[0]
    rng = np.random.default_rng(1000 + i)
    take = rng.choice(neg, min(NNEG_ROWS, len(neg)), replace=False) if len(neg) else neg
    idx = np.concatenate([np.nonzero(pos)[0], take])
    w = np.concatenate([np.ones(pos.sum()), np.full(len(take), len(neg) / max(len(take), 1))])
    y = np.concatenate([np.ones(pos.sum(), np.int8), np.zeros(len(take), np.int8)])
    return X8[idx], Xe[idx], y, w, src[idx], np.full(len(idx), i, np.int32)


VUNIT = 4096                             # integer law: leaf = round(4096 v), base = round(4096 baseline)


def freeze_model(m):
    """the fitted trees as integers.  Features are integers, so 'x <= t' is exactly 'x <= floor(t)'."""
    trees = []
    for (p,) in m._predictors:
        nd = p.nodes
        assert not nd["is_categorical"].any()
        trees.append(dict(feat=nd["feature_idx"].astype(int).tolist(), thr=np.floor(nd["num_threshold"]).astype(int).tolist(),
                          left=nd["left"].astype(int).tolist(), right=nd["right"].astype(int).tolist(),
                          leaf=nd["is_leaf"].astype(int).tolist(),
                          val=np.where(nd["is_leaf"], np.rint(VUNIT * nd["value"]), 0).astype(int).tolist()))
    return dict(base=int(np.rint(VUNIT * float(np.ravel(m._baseline_prediction)[0]))), trees=trees, VUNIT=VUNIT)


def score_int(law, X):
    """integer LAW 7 score of every row of X (integer features)"""
    X = np.asarray(X, np.int64)
    out = np.full(len(X), law["base"], np.int64)
    ar = np.arange(len(X))
    for t in law["trees"]:
        feat, thr, left, right, leaf, val = (np.array(t[k]) for k in ("feat", "thr", "left", "right", "leaf", "val"))
        node = np.zeros(len(X), np.int64)
        while True:
            go = leaf[node] == 0
            if not go.any():
                break
            i = ar[go]; nn = node[i]
            node[i] = np.where(X[i, feat[nn]] <= thr[nn], left[nn], right[nn])
        out += val[node]
    return out


def score_image(args):
    key, ip, lp = args
    g = R.load_gray(ip)
    X8, Xe, B, _ = candidates(g)
    if len(B) == 0:
        return key, [], [], R.read_label(lp)
    X = np.hstack([X8.astype(np.int32), Xe])
    v = score_int(W["law"], X) if "law" in W else W["model"].decision_function(X.astype(np.float32))
    law7 = [(float(v[i]),) + tuple(B[i]) for i in range(len(B))]
    s1 = [(int(Xe[i, 0]),) + tuple(B[i]) for i in range(len(B))]
    return key, law7, s1, R.read_label(lp)


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    SPLIT = json.load(open(os.path.join(OUT, "split.json")))
    if stage == "rows":
        n = int(sys.argv[2])
        jobs = [(i, k, ip, lp) for i, (k, ip, lp) in enumerate(R.items(SPLIT["design"][:n]))]
        parts = []
        with Pool(R.NPROC, init) as pool:
            for i, p in enumerate(pool.imap_unordered(rows_image, jobs, chunksize=2)):
                parts.append(p)
                if (i + 1) % 200 == 0:
                    print("  rows from %d / %d design images  (%.0fs)" % (i + 1, n, time.time() - t0), flush=True)
        X8, Xe, y, w, src, img = (np.concatenate(a) for a in zip(*parts))
        np.savez(os.path.join(OUT, "v2_rows.npz"), X8=X8, Xe=Xe, y=y, w=w, src=src, img=img)
        print("rows: %d (faces %d of which label-aligned %d; negatives %d standing for %.0f)  [%.0fs]"
              % (len(y), y.sum(), (src == 1).sum(), (y == 0).sum(), w[y == 0].sum(), time.time() - t0))
    elif stage == "fit":
        from sklearn.ensemble import HistGradientBoostingClassifier
        Z = np.load(os.path.join(OUT, "v2_rows.npz"))
        X = np.hstack([Z["X8"], Z["Xe"]]).astype(np.float32); y, w = Z["y"], Z["w"]
        wn = w.copy(); wn[y == 0] *= w[y == 1].sum() / w[y == 0].sum()      # classes balanced in total weight
        it = int(sys.argv[2]) if len(sys.argv) > 2 else 400
        m = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=25, random_state=0, verbose=0)
        m.fit(X, y, sample_weight=wn)
        pickle.dump(m, open(os.path.join(OUT, "v2_model.pkl"), "wb"))
        print("fit: %d rows x %d features, %d trees  [%.0fs]" % (X.shape[0], X.shape[1], m.n_iter_, time.time() - t0))
    elif stage == "val":
        m = pickle.load(open(os.path.join(OUT, "v2_model.pkl"), "rb"))
        d7, d1, gts = {}, {}, {}
        with Pool(R.NPROC, init, (m,)) as pool:
            for key, law7, s1, gt in pool.imap_unordered(score_image, R.items(SPLIT["design_val"]), chunksize=2):
                d7[key] = L.nms(law7, 0.3); d1[key] = L.nms(s1, 0.3); gts[key] = gt
        ap7, rec7 = L.average_precision(d7, gts); ap1, rec1 = L.average_precision(d1, gts)
        pickle.dump(dict(d7=d7, d1=d1, gts=gts), open(os.path.join(OUT, "v2_val.pkl"), "wb"))
        print("design-validation (300 images): stage 1 alone AP %.2f%% (recall %.1f%%);  LAW 7 AP %.2f%% (recall %.1f%%)  [%.0fs]"
              % (100 * ap1, 100 * rec1, 100 * ap7, 100 * rec7, time.time() - t0))
    elif stage == "freeze":
        m = pickle.load(open(os.path.join(OUT, "v2_model.pkl"), "rb"))
        law = freeze_model(m)
        Z = np.load(os.path.join(OUT, "v2_rows.npz"))
        X = np.hstack([Z["X8"][:20000].astype(np.int32), Z["Xe"][:20000]])
        f, q = m.decision_function(X.astype(np.float32)), score_int(law, X) / VUNIT
        bound = (len(law["trees"]) + 1) * 0.5 / VUNIT
        print("frozen: %d trees, %d leaves; integer vs float on 20,000 design rows: max |diff| %.5f (rounding bound %.5f)"
              % (len(law["trees"]), sum(sum(t["leaf"]) for t in law["trees"]), np.abs(f - q).max(), bound))
        assert np.abs(f - q).max() <= bound + 1e-9
        json.dump(law, open(os.path.join(OUT, "law7.json"), "w"))
        import hashlib
        print("law7.json sha256", hashlib.sha256(open(os.path.join(OUT, "law7.json"), "rb").read()).hexdigest())
    elif stage == "ival":
        f = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "law7.json")
        law = json.load(open(f))
        law = law.get("law", law)
        d7, gts = {}, {}
        with Pool(R.NPROC, init, (None, law)) as pool:
            for key, law7, s1, gt in pool.imap_unordered(score_image, R.items(SPLIT["design_val"]), chunksize=2):
                d7[key] = L.nms(law7, 0.3); gts[key] = gt
        ap7, rec7 = L.average_precision(d7, gts)
        print("design-validation (300 images): INTEGER LAW 7 (%s) AP %.2f%% (recall %.1f%%)  [%.0fs]"
              % (os.path.basename(f), 100 * ap7, 100 * rec7, time.time() - t0))
