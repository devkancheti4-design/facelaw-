"""DARK FACE v3: LAW 7 with the face's own neighbourhood in scale.  Stage 1 is still the sealed, lane-authored v1 law.
On design-validation LAW 7 (v2) ranked faces of 16 px and up at only 19.5% AP and faces under 8 px at 2.0%: every face
is described at 10 px, so a large face loses its detail, and a tiny face has almost no pixels of its own -- its evidence
is the head, shoulders and body around it.  v3 adds, per candidate, the same measured, gain- and offset-cancelled
descriptions at the two levels one octave away:
    fine     level k-3 (scale s/2, face 20 px): per 2x2 cell and sector, summed |grad| in noise units over the
             central 16 x 16 (eyes, nose, mouth)                                                          512
    context  level k+3 (scale 2s, about 3 face widths): per 4x4 cell and sector, summed |grad| in noise units (12 log2);
             per 4x4 cell the two centre-surround counts; z pooled 2 x 2                            128 + 32 + 64
    flags    whether the finer / coarser level exists                                                         2
    python3 -I darkface_v3.py rows N         training rows from design images [0:N] -> darkface/v3_rows.npz
    python3 -I darkface_v3.py hrows A B      cascade round: on design images [A:B], LAW 7's own surviving false alarms
                                             (the top K_HARD after NMS, weight 1) + a uniform sample of the rest (weight
                                             rest / sampled): the full candidate distribution, resolved where it ranks
    python3 -I darkface_v3.py fit [IT]       -> darkface/v3_model.pkl (all row files present)
    python3 -I darkface_v3.py val            design-validation AP
    python3 -I darkface_v3.py freeze         the trees as integers -> darkface/law7.json
    python3 -I darkface_v3.py ival [FILE]    design-validation AP of the INTEGER law (law7.json, or the synthesis's rebuild)
    python3 -I darkface_v3.py protocol       seal: every file the test reads, hashed, before any test image is read
    python3 -I darkface_v3.py test           the lane-authored LAW 7 (lanes7.json) on the 1,000 sealed test images, ONCE"""
import hashlib, json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import maximum_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import darkface_law as L
import darkface_run as R
import darkface_v2 as V

OUT = R.OUT
NNEG_ROWS = 200
OCT = 3                                  # levels per octave (steps of 2^(1/3))
NB = 896 + 512 + 128 + 32 + 64           # int8 features
W = V.W


def box4(a):
    b = V.box2(a)
    return b[:-2, :-2] + b[2:, :-2] + b[:-2, 2:] + b[2:, 2:]


def at_level(r, c, s, s2, shape):
    """the window of level s2 centred where window (r, c) of level s is centred (clipped to the padded level)"""
    cy, cx = (r - L.PAD + L.T / 2 - 0.5) * s, (c - L.PAD + L.T / 2 - 0.5) * s
    r2 = np.clip(np.rint(cy / s2 - L.T / 2 + 0.5).astype(np.int64) + L.PAD, 0, shape[0] - L.T)
    c2 = np.clip(np.rint(cx / s2 - L.T / 2 + 0.5).astype(np.int64) + L.PAD, 0, shape[1] - L.T)
    return r2, c2


def fine_feats(M, r2, c2):
    cd, cg, xp, mp, spix = M
    a8 = 2 * np.arange(L.T // 2)
    rr, cc = r2[:, None, None] + a8[None, :, None], c2[:, None, None] + a8[None, None, :]
    hog = np.empty((len(r2), 8, 8, 8))
    for sct in range(8):
        hog[:, sct] = V.box2(np.where(cd == sct + 1, mp, 0.0))[rr, cc]
    return np.clip(np.rint(16 * np.log2(1 + hog)), 0, 127).reshape(len(r2), -1)


def context_feats(M, r2, c2):
    cd, cg, xp, mp, spix = M
    n = len(r2)
    a4 = 4 * np.arange(4)
    rr, cc = r2[:, None, None] + a4[None, :, None], c2[:, None, None] + a4[None, None, :]
    hog = np.empty((n, 8, 4, 4))
    for sct in range(8):
        hog[:, sct] = box4(np.where(cd == sct + 1, mp, 0.0))[rr, cc]
    dog = np.empty((n, 2, 4, 4))
    for code in (1, 2):
        dog[:, code - 1] = box4((cg == code).astype(np.int32))[rr, cc]
    ar = np.arange(L.T)
    win = xp[r2[:, None, None] + ar[None, :, None], c2[:, None, None] + ar[None, None, :]]
    mu, sd = win.mean((1, 2)), win.std((1, 2))
    z = (win - mu[:, None, None]) / np.maximum(sd, 2 * spix)[:, None, None]
    zp = z.reshape(n, 8, 2, 8, 2).mean((2, 4))
    return np.hstack([np.clip(np.rint(12 * np.log2(1 + hog)), 0, 127).reshape(n, -1), dog.reshape(n, -1),
                      np.clip(np.rint(V.ZQ * zp), -127, 127).reshape(n, -1)])


def candidates(g, gt=None):
    """stage 1 exactly as sealed, with the v3 description of every candidate (label-aligned windows with gt)"""
    SC = W["scales"]; nl = len(SC)
    cache = {}
    def maps(k):
        if k not in cache:
            cache[k] = V.level_maps(g, SC[k])
        return cache[k]
    gtk = {}
    if gt is not None:
        for j, box in enumerate(gt):
            k = L.level_of_face(box, SC); gtk.setdefault(k, []).append(L.window_of_face(box, SC[k]))
    out = []
    for k, s in enumerate(SC):
        for old in [q for q in cache if q < k - OCT]:
            del cache[old]
        cd, cg, xp, mp, spix = maps(k)
        S = L.score_level(cd, cg, W["Ld"], W["Lg"])
        if S is None:
            continue
        sig = L.face_sig(cd, cg)
        S = np.where(sig >= W["minsig"], S, L.NEG)
        pk = (S == maximum_filter(S, size=3)) & (S > L.NEG)
        rr, cc = np.nonzero(pk)
        if len(rr) > V.PER_LEVEL:
            j = np.argsort(-S[rr, cc], kind="stable")[:V.PER_LEVEL]; rr, cc = rr[j], cc[j]
        src = np.zeros(len(rr), np.int8)
        if k in gtk:
            ga = np.array([(r, c) for r, c in gtk[k] if 0 <= r < S.shape[0] and 0 <= c < S.shape[1] and S[r, c] > L.NEG],
                          np.int64).reshape(-1, 2)
            rr, cc = np.concatenate([rr, ga[:, 0]]), np.concatenate([cc, ga[:, 1]])
            src = np.concatenate([src, np.ones(len(ga), np.int8)])
        if len(rr) == 0:
            continue
        S1 = S[rr, cc] + int(W["prior"][k])
        X8, Xe = V.features(cd, cg, xp, mp, spix, rr, cc, S1, k, sig[rr, cc])
        n = len(rr)
        if k - OCT >= 0:
            M = maps(k - OCT); r2, c2 = at_level(rr, cc, s, SC[k - OCT], M[0].shape)
            F = fine_feats(M, r2, c2); hf = 1
        else:
            F = np.zeros((n, 512)); hf = 0
        if k + OCT < nl and maps(k + OCT)[0].shape[0] >= L.T and maps(k + OCT)[0].shape[1] >= L.T:
            M = maps(k + OCT); r2, c2 = at_level(rr, cc, s, SC[k + OCT], M[0].shape)
            C = context_feats(M, r2, c2); hc = 1
        else:
            C = np.zeros((n, 224)); hc = 0
        X8 = np.hstack([X8, F, C]).astype(np.int8)
        Xe = np.hstack([Xe, np.tile([hf, hc], (n, 1))]).astype(np.int32)
        out.append((X8, Xe, V.boxes_of(rr, cc, s), src))
    if not out:
        return np.zeros((0, NB), np.int8), np.zeros((0, 7), np.int32), np.zeros((0, 4)), np.zeros(0, np.int8)
    return tuple(np.concatenate(a) for a in zip(*out))


def rows_image(args):
    i, key, ip, lp = args
    g = R.load_gray(ip); gt = R.read_label(lp)
    X8, Xe, B, src = candidates(g, gt)
    o = V.iou_matrix(B, gt).max(1) if len(gt) else np.zeros(len(B))
    pos = (o >= 0.5) | (src == 1)
    neg = np.nonzero((o < 0.3) & (src == 0))[0]
    rng = np.random.default_rng(1000 + i)
    take = rng.choice(neg, min(NNEG_ROWS, len(neg)), replace=False) if len(neg) else neg
    idx = np.concatenate([np.nonzero(pos)[0], take])
    w = np.concatenate([np.ones(pos.sum()), np.full(len(take), len(neg) / max(len(take), 1))])
    y = np.concatenate([np.ones(pos.sum(), np.int8), np.zeros(len(take), np.int8)])
    return X8[idx], Xe[idx], y, w, src[idx], np.full(len(idx), i, np.int32)


K_HARD, N_REST = 150, 100


def hrows_image(args):
    i, key, ip, lp = args
    g = R.load_gray(ip); gt = R.read_label(lp)
    X8, Xe, B, src = candidates(g, gt)
    o = V.iou_matrix(B, gt).max(1) if len(gt) else np.zeros(len(B))
    pos = np.nonzero((o >= 0.5) | (src == 1))[0]
    neg = np.nonzero((o < 0.3) & (src == 0))[0]
    v = W["model"].decision_function(np.hstack([X8[neg], Xe[neg]]).astype(np.float32)) if len(neg) else np.zeros(0)
    keep = L.nms([(float(v[j]),) + tuple(B[neg[j]]) + (j,) for j in range(len(neg))], 0.3)[:K_HARD]
    hard = np.array([d[5] for d in keep], np.int64)
    rest = np.setdiff1d(np.arange(len(neg)), hard)
    rng = np.random.default_rng(5000 + i)
    take = rng.choice(rest, min(N_REST, len(rest)), replace=False) if len(rest) else rest
    idx = np.concatenate([pos, neg[hard], neg[take]])
    w = np.concatenate([np.ones(len(pos) + len(hard)), np.full(len(take), len(rest) / max(len(take), 1))])
    y = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(hard) + len(take), np.int8)])
    return X8[idx], Xe[idx], y, w, src[idx], np.full(len(idx), i, np.int32)


def score_image(args):
    key, ip, lp = args
    g = R.load_gray(ip)
    X8, Xe, B, _ = candidates(g)
    if len(B) == 0:
        return key, [], R.read_label(lp)
    X = np.hstack([X8.astype(np.int32), Xe])
    v = V.score_int(W["law"], X) if "law" in W else W["model"].decision_function(X.astype(np.float32))
    return key, [(float(v[i]),) + tuple(B[i]) for i in range(len(B))], R.read_label(lp)


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    SPLIT = json.load(open(os.path.join(OUT, "split.json")))
    if stage == "rows":
        n = int(sys.argv[2])
        jobs = [(i, k, ip, lp) for i, (k, ip, lp) in enumerate(R.items(SPLIT["design"][:n]))]
        parts = []
        with Pool(int(os.environ.get("NPROC", R.NPROC)), V.init) as pool:
            for i, p in enumerate(pool.imap_unordered(rows_image, jobs, chunksize=2)):
                parts.append(p)
                if (i + 1) % 250 == 0:
                    print("  rows from %d / %d design images  (%.0fs)" % (i + 1, n, time.time() - t0), flush=True)
        X8, Xe, y, w, src, img = (np.concatenate(a) for a in zip(*parts))
        np.savez(os.path.join(OUT, "v3_rows.npz"), X8=X8, Xe=Xe, y=y, w=w, src=src, img=img)
        print("rows: %d (faces %d of which label-aligned %d; negatives %d standing for %.0f)  [%.0fs]"
              % (len(y), y.sum(), (src == 1).sum(), (y == 0).sum(), w[y == 0].sum(), time.time() - t0))
    elif stage == "hrows":
        a, b = int(sys.argv[2]), int(sys.argv[3])
        jobs = [(a + i, k, ip, lp) for i, (k, ip, lp) in enumerate(R.items(SPLIT["design"][a:b]))]
        m = pickle.load(open(os.path.join(OUT, "v3_model.pkl"), "rb"))
        parts = []
        with Pool(int(os.environ.get("NPROC", R.NPROC)), V.init, (m,)) as pool:
            for i, p in enumerate(pool.imap_unordered(hrows_image, jobs, chunksize=2)):
                parts.append(p)
                if (i + 1) % 250 == 0:
                    print("  cascade rows from %d / %d design images  (%.0fs)" % (i + 1, len(jobs), time.time() - t0), flush=True)
        X8, Xe, y, w, src, img = (np.concatenate(q) for q in zip(*parts))
        np.savez_compressed(os.path.join(OUT, "v3_hrows.npz"), X8=X8, Xe=Xe, y=y, w=w, src=src, img=img)
        print("cascade rows: %d (faces %d; negatives %d of which hard %d, standing for %.0f)  [%.0fs]"
              % (len(y), y.sum(), (y == 0).sum(), ((y == 0) & (w == 1)).sum(), w[y == 0].sum(), time.time() - t0))
    elif stage == "fit":
        from sklearn.ensemble import HistGradientBoostingClassifier
        fs = [f for f in ("v3_rows.npz", "v3_hrows.npz") if os.path.exists(os.path.join(OUT, f))]
        Zs = [np.load(os.path.join(OUT, f)) for f in fs]
        X = np.vstack([np.hstack([Z["X8"], Z["Xe"]]).astype(np.float32) for Z in Zs])
        y, w = np.concatenate([Z["y"] for Z in Zs]), np.concatenate([Z["w"] for Z in Zs])
        print("fit on", fs, flush=True)
        wn = w.copy(); wn[y == 0] *= w[y == 1].sum() / w[y == 0].sum()
        it = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
        m = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=25, random_state=0, verbose=0)
        m.fit(X, y, sample_weight=wn)
        pickle.dump(m, open(os.path.join(OUT, "v3_model.pkl"), "wb"))
        print("fit: %d rows x %d features, %d trees  [%.0fs]" % (X.shape[0], X.shape[1], m.n_iter_, time.time() - t0))
    elif stage in ("val", "ival"):
        if stage == "val":
            args = (pickle.load(open(os.path.join(OUT, "v3_model.pkl"), "rb")), None); tag = "LAW 7 v3 (float trees)"
        else:
            f = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "law7.json")
            law = json.load(open(f)); law = law.get("law", law); args = (None, law); tag = "INTEGER LAW 7 (%s)" % os.path.basename(f)
        d7, gts = {}, {}
        with Pool(R.NPROC, V.init, args) as pool:
            for key, law7, gt in pool.imap_unordered(score_image, R.items(SPLIT["design_val"]), chunksize=2):
                d7[key] = L.nms(law7, 0.3); gts[key] = gt
        ap7, rec7 = L.average_precision(d7, gts)
        pickle.dump(dict(d7=d7, gts=gts), open(os.path.join(OUT, "v3_%s.pkl" % stage), "wb"))
        print("design-validation (300 images): %s AP %.2f%% (recall %.1f%%)  [%.0fs]" % (tag, 100 * ap7, 100 * rec7, time.time() - t0))
    elif stage == "freeze":
        m = pickle.load(open(os.path.join(OUT, "v3_model.pkl"), "rb"))
        law = V.freeze_model(m)
        Z = np.load(os.path.join(OUT, "v3_rows.npz"))
        X = np.hstack([Z["X8"][:20000].astype(np.int32), Z["Xe"][:20000]])
        f, q = m.decision_function(X.astype(np.float32)), V.score_int(law, X) / V.VUNIT
        bound = (len(law["trees"]) + 1) * 0.5 / V.VUNIT
        print("frozen: %d trees, %d leaves; integer vs float on 20,000 design rows: max |diff| %.5f (rounding bound %.5f)"
              % (len(law["trees"]), sum(sum(t["leaf"]) for t in law["trees"]), np.abs(f - q).max(), bound))
        assert np.abs(f - q).max() <= bound + 1e-9
        json.dump(law, open(os.path.join(OUT, "law7.json"), "w"))
        print("law7.json sha256", hashlib.sha256(open(os.path.join(OUT, "law7.json"), "rb").read()).hexdigest())
    elif stage == "protocol":
        Z = json.load(open(os.path.join(OUT, "lanes7.json")))
        assert Z["wrong"] == 0 and Z["differ"] == 0
        assert Z["law7_sha256"] == hashlib.sha256(open(os.path.join(OUT, "law7.json"), "rb").read()).hexdigest()
        files = ["darkface_law.py", "darkface_run.py", "the private authoring script", "darkface_v2.py", "darkface_v3.py",
                 "the private authoring script", "yale_gate.py", "lanes_sector.json", "lane_cast.json", "darkface/split.json",
                 "darkface/lanes.json", "darkface/law7.json", "darkface/lanes7.json", "darkface/user_claim7.txt",
                 "darkface/prediction7.txt"]
        V0 = sys.argv[2] if len(sys.argv) > 2 else "?"
        txt = ["DARK FACE LAW 7 SEALED TEST PROTOCOL  (written %s, before any test image is read by LAW 7)" % time.strftime("%Y-%m-%d %H:%M:%S"),
               "data: as protocol.txt (DarkFace_Train_2021.zip, 6,000 labelled night images); split.json unchanged",
               "test images: the SAME 1,000 (8,143 faces) as the v1 sealed test -- this is their SECOND use: v1 scored AP 1.14% "
               "on them; LAW 7 was designed after that number was known, with no other test information, on the 4,700 design "
               "and 300 design-validation images only",
               "stage 1: the sealed v1 law exactly (lanes.json, lane-authored), raw peaks, 300 per level",
               "LAW 7: additive-tree log-odds over the v3 description (window z, hog, dog; fine level k-3; context level k+3); "
               "%d trees; integers throughout; design-validation AP of the integer law %s" % (len(Z["law"]["trees"]), V0),
               "runtime: LAW 7 rebuilt from the authored lanes ONLY (lanes7.json: every lane independently, proven on all its rows by Python and compiled C; 0 wrong; 0 entries differing from law7.json)",
               "metric: VOC all-point AP at IoU 0.5 over all 1,000 test images and ALL labelled faces; detections = LAW 7 "
               "boxes after NMS 0.3; run ONCE; reported whatever it is",
               "reference: fine-tuned DSFD 46.0 mAP (HLA-Face, CVPR 2021); their protocol differs (indicative only)",
               "sealed claims: user_claim7.txt (reaches the optimum of its candidates, 64.1% on design-validation; at least "
               "beats 46.0) and prediction7.txt (mine, about 12%, range 5-25%), both written before LAW 7 was fitted"]
        for f in files:
            txt.append("sha256 %s %s" % (f, hashlib.sha256(open(os.path.join(os.path.dirname(OUT), f), "rb").read()).hexdigest()))
        open(os.path.join(OUT, "protocol7.txt"), "w").write("\n".join(txt) + "\n")
        h = hashlib.sha256(open(os.path.join(OUT, "protocol7.txt"), "rb").read()).hexdigest()
        open(os.path.join(OUT, "protocol7.sha256"), "w").write(h + "  darkface/protocol7.txt\n")
        print("\n".join(txt)); print("protocol7 sha256", h)
    elif stage == "test":
        for line in open(os.path.join(OUT, "protocol7.txt")).read().splitlines():
            if line.startswith("sha256 "):
                _, name, h = line.split()
                assert hashlib.sha256(open(os.path.join(os.path.dirname(OUT), name), "rb").read()).hexdigest() == h, \
                    "%s changed since the seal" % name
        Z = json.load(open(os.path.join(OUT, "lanes7.json")))
        assert Z["wrong"] == 0 and Z["differ"] == 0
        dets, gts = {}, {}
        with Pool(R.NPROC, V.init, (None, Z["law"])) as pool:
            for i, (key, law7, gt) in enumerate(pool.imap_unordered(score_image, R.items(SPLIT["test"]), chunksize=2)):
                dets[key] = L.nms(law7, 0.3); gts[key] = gt
                if (i + 1) % 200 == 0:
                    print("  tested %d / 1000 images  (%.0fs)" % (i + 1, time.time() - t0), flush=True)
        ap, rec = L.average_precision(dets, gts)
        res = dict(AP=ap, max_recall=rec, images=len(gts), faces=int(sum(len(g) for g in gts.values())),
                   detections=int(sum(len(d) for d in dets.values())))
        json.dump(res, open(os.path.join(OUT, "test7_result.json"), "w"), indent=1)
        pickle.dump(dict(dets=dets, gts=gts), open(os.path.join(OUT, "test7_dets.pkl"), "wb"))
        h = hashlib.sha256(open(os.path.join(OUT, "test7_result.json"), "rb").read()).hexdigest()
        open(os.path.join(OUT, "test7_result.sha256"), "w").write(h + "  darkface/test7_result.json\n")
        print("SEALED TEST LAW 7  %d images, %d faces:  AP@0.5 = %.2f%%   max recall %.1f%%   [%.0fs]   result sha256 %s" % (
            res["images"], res["faces"], 100 * ap, 100 * rec, time.time() - t0, h))
