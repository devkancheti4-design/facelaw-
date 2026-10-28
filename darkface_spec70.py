"""SPEC 70 -- DARK FACE at 70% AP@0.5 (the user, 2026-10-28: "author spec 70 ... we can test that so we can say it doesn't
exist or is possible or we get it achieved").  The SPEC 50 path is kept (stage 1, the sealed lane-authored v1 law; LAW 3b
cap 3,000; LAW 8 placement; LAW 9 NMS 0.5); the shot is the ranking, the measured wall (SPEC 50: 31.42% on
design-validation where a perfect ranking of the same candidates gives 78.6%).
    protocol  darkface/spec70.txt: target, claims and file hashes, written BEFORE the fit
    fit       LAW 7d: LAW 7c's recipe with four times the leaves (127), at least 20 rows per leaf, on the same design rows
              [0:3800] (every face and non-face row), early stopping on 10%            -> darkface/v70_model.pkl
    val       design-validation (300 images): LAW 7c and LAW 7d on the SAME candidates, LAW 8 placement, NMS 0.5
                                                                                       -> darkface/v70_val.json
    author    (only if LAW 7d beats LAW 7c) LAW 7d frozen to integer leaves and authored as integer lanes, every number,
              as SPEC 100 (the private authoring script)
    test      only if design-validation reaches 70%: the 1,000 sealed test images (their FIFTH use), once
    OMP_NUM_THREADS=3 python3 -I darkface_spec70.py protocol | fit | val"""
import hashlib, json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import darkface_law as L
import darkface_run as R
import darkface_v4 as V4

OUT = R.OUT
TARGET = 0.70
NEG_KEEP = 0.42                         # erratum: the share of non-face rows kept (memory: 24 GB machine, 7 GB disk)
G70 = {}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def val_init(models, place):
    V4.init(model=models["LAW 7c"], place=place)
    G70.clear(); G70.update(models=models, place=place)


def val_image(args):
    """one design-validation image: every ranker on the same candidates, LAW 8 placement, NMS 0.5"""
    key, ip, lp = args
    X8, Xe, B, lev = V4.describe_all(R.load_gray(ip))
    out = {}
    for name, m in G70["models"].items():
        out[name] = []
        if len(B):
            X = np.hstack([X8.astype(np.float32), Xe.astype(np.float32)])
            v = m.decision_function(X)
            o = np.argsort(-v)[:20000]
            vv, BB, XX = v[o], B[o], X[o]
            B8 = V4.apply_place(BB, np.stack([p.predict(XX) for p in G70["place"]], 1))
            out[name] = L.nms([(float(vv[i]),) + tuple(B8[i]) for i in range(len(vv))], 0.5)
    return key, out, R.read_label(lp)


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    if stage == "protocol":
        files = ["darkface_spec70.py", "darkface_v4.py", "darkface_v3.py", "darkface_v2.py", "darkface_law.py", "darkface_run.py",
                 "darkface/split.json", "darkface/lanes.json", "darkface/v4_model.pkl", "darkface/v4_place.pkl", "darkface/law50.json",
                 "darkface/lanes100.json"]
        txt = ["SPEC 70 -- DARK FACE at 70%% AP@0.5  (written %s, before LAW 7d is fitted)" % time.strftime("%Y-%m-%d %H:%M:%S"),
               "target    AP >= 70% on design-validation (300 images, 2,621 faces), the SPEC 50 path with the ranking replaced;",
               "          the 1,000 sealed test images are read (a FIFTH time) ONLY if design-validation reaches 70%",
               "ceiling   today's candidates ranked perfectly: 78.6% on design-validation, so 70% asks for an average precision of",
               "          about 89% over that recall (SPEC 50: 93% at its top 100 detections, 60% at recall 23%, 40% at 30%)",
               "the shot  LAW 7d: LAW 7c's recipe (additive trees, learning rate 0.1, early stopping on 10%) with 127 leaves per tree",
               "          (31 before) and at least 20 rows per leaf, on the same design rows [0:3800]: 625,396 rows, 63,146 faces;",
               "          if it beats LAW 7c on design-validation, program synthesis writes every number of it, (as SPEC 100)",
               "claims    the user: achievable (\"so we can say it doesn't exist or is possible or we get it achieved\")",
               "          mine: NOT achieved; design-validation 31-36% (LAW 7c 31.42%): the wall is the information in this law's",
               "          description of a 4-10 px face in the dark, not the size of the ranker and not its authoring"]
        for f in files:
            txt.append("sha256 %s %s" % (f, sha(os.path.join(os.path.dirname(OUT), f))))
        p = os.path.join(OUT, "spec70.txt")
        open(p, "w").write("\n".join(txt) + "\n")
        open(p[:-4] + ".sha256", "w").write(sha(p) + "  spec70.txt\n")
        print("\n".join(l for l in txt if not l.startswith("sha256")) + "\nspec70.txt sha256 " + sha(p))
    elif stage == "erratum":
        txt = ["SPEC 70 ERRATUM  (written %s, before any tree of LAW 7d was grown)" % time.strftime("%Y-%m-%d %H:%M:%S"),
               "the fit on all 625,396 rows reached 26 GB in scikit-learn (float64 copies of the rows, again for its early-stopping",
               "split) on this 24 GB machine with 7 GB of disk free, and was stopped while binning: no tree grown, nothing saved.",
               "LAW 7d is fitted on every face row (63,146) and a 42% Bernoulli sample of the non-face rows (seed 70); the class",
               "weights are rebalanced exactly as LAW 7c's (non-faces scaled to the faces' total), so the loss is the same in",
               "expectation; everything else in spec70.txt (sha256 %s) is unchanged." % open(os.path.join(OUT, "spec70.sha256")).read().split()[0]]
        p = os.path.join(OUT, "spec70_erratum.txt")
        open(p, "w").write("\n".join(txt) + "\n")
        print("\n".join(txt) + "\nspec70_erratum.txt sha256 " + sha(p))
    elif stage == "fit":
        from sklearn.ensemble import HistGradientBoostingClassifier
        Z = np.load(os.path.join(OUT, "v4_rows_0_3800.npz"))
        zy = Z["y"]
        # ERRATUM (darkface/spec70_erratum.txt): every face row and a 42% Bernoulli sample of the non-face rows (seed 70); the
        # rows are built in float64 at once (scikit-learn copies to float64, and again for its early-stopping split)
        keep_neg = np.random.default_rng(70).random(len(zy)) < NEG_KEEP
        k = np.nonzero((zy == 1) | ((zy == 0) & keep_neg))[0]
        X = np.empty((len(k), V4.NB + 8), np.float64)
        X8, Xe = Z["X8"], Z["Xe"]
        for a in range(0, len(k), 100000):
            kk = k[a:a + 100000]
            X[a:a + len(kk), :V4.NB] = X8[kk]; X[a:a + len(kk), V4.NB:] = Xe[kk]
        del X8, Xe
        y = zy[k]; w = Z["w"][k].copy(); w[y == 0] *= w[y == 1].sum() / w[y == 0].sum()
        print("rows %d x %d features (%d faces)  [%.0fs]" % (X.shape[0], X.shape[1], int((y == 1).sum()), time.time() - t0), flush=True)
        m = HistGradientBoostingClassifier(max_iter=3000, learning_rate=0.1, max_leaf_nodes=127, min_samples_leaf=20,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=25, random_state=0, verbose=2)
        m.fit(X, y, sample_weight=w)
        pickle.dump(m, open(os.path.join(OUT, "v70_model.pkl"), "wb"))
        print("LAW 7d: %d trees of up to 127 leaves  [%.0fs]" % (m.n_iter_, time.time() - t0), flush=True)
    elif stage == "val":
        SPLIT = json.load(open(os.path.join(OUT, "split.json")))
        models = {"LAW 7c": pickle.load(open(os.path.join(OUT, "v4_model.pkl"), "rb")),
                  "LAW 7d": pickle.load(open(os.path.join(OUT, "v70_model.pkl"), "rb"))}
        place = pickle.load(open(os.path.join(OUT, "v4_place.pkl"), "rb"))
        dets, gts = {n: {} for n in models}, {}
        with Pool(int(os.environ.get("NPROC", "3")), val_init, (models, place)) as pool:
            for j, (key, out, gt) in enumerate(pool.imap_unordered(val_image, R.items(SPLIT["design_val"]), chunksize=1)):
                gts[key] = gt
                for n in models:
                    dets[n][key] = out[n]
                if (j + 1) % 50 == 0:
                    print("  %d / %d design-validation images  [%.0fs]" % (j + 1, len(SPLIT["design_val"]), time.time() - t0), flush=True)
        res = {}
        for n in models:
            ap, rec = L.average_precision(dets[n], gts)
            res[n] = dict(AP=100 * ap, recall=100 * rec)
            print("design-validation %s: AP %.2f%%  (recall %.1f%%)" % (n, 100 * ap, 100 * rec), flush=True)
        res["target_reached"] = bool(res["LAW 7d"]["AP"] >= 100 * TARGET)
        json.dump(res, open(os.path.join(OUT, "v70_val.json"), "w"), indent=1)
        print("SPEC 70 on design-validation: %s  [%.0fs]" % ("REACHED -> the sealed test may be run" if res["target_reached"] else
                                                            "NOT reached -> the sealed test images stay untouched", time.time() - t0))
    elif stage == "freeze":
        # LAW 7d as integers (darkface_v2.freeze_model: integer features, so x <= t is x <= floor(t); leaves round(4096 v), once);
        # LAW 8 (dx, dy, ds) unchanged from law50.json -> darkface/law70.json, for program synthesis (LAW=law70.json NAME=70)
        import darkface_v2 as V
        m = pickle.load(open(os.path.join(OUT, "v70_model.pkl"), "rb"))
        law50 = json.load(open(os.path.join(OUT, "law50.json")))
        law = dict(rank=V.freeze_model(m), dx=law50["dx"], dy=law50["dy"], ds=law50["ds"])
        json.dump(law, open(os.path.join(OUT, "law70.json"), "w"))
        print("law70.json: rank %d trees, %d nodes; placement as law50  sha256 %s" % (len(law["rank"]["trees"]),
              sum(len(t["leaf"]) for t in law["rank"]["trees"]), sha(os.path.join(OUT, "law70.json"))))
