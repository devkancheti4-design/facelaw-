"""SPEC 50 on WIDER FACE validation (daylight, 3,226 images), ZERO-SHOT: the frozen night law exactly as sealed in
protocol50 (stage 1 + LAW 3b cap 3000 + compiled LAW 7c + LAW 8 + NMS 0.5), nothing refitted.  One extension of
LAW 3 only: WIDER faces reach ~1000 px, past the pyramid's 320 px top, so a second pass runs the same law on the image
shrunk 4x and keeps its levels whose face is >= 80 px there (>= 322 px in the image).  Up to 750 detections per image.
Scoring: a port of the official WIDER FACE evaluation (eval_tools: Easy / Medium / Hard keep lists, ignored faces,
1000 thresholds on globally min-max normalised scores, IoU 0.5, VOC AP), as in the widely used Python port.
    python3 -I darkface_wider.py run          -> darkface/wider_dets.pkl
    python3 -I darkface_wider.py eval         Easy / Medium / Hard AP"""
import glob, json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import darkface_law as L
import darkface_run as R
import darkface_v2 as V
import darkface_v4 as V4

OUT = R.OUT
WD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wider")
KEEP = 750
SHRINK, BIG_LEVEL = 4, 13                # pass 2: image / 4, levels with face >= 80 px there
W = V.W


def law_pass(g, k_min=0):
    pick = lambda k, s, Bk, S1: np.arange(len(Bk)) if k >= k_min else None
    X8, Xe, B, lev = V4.walk(g, pick)
    if len(B) == 0:
        return np.zeros(0), np.zeros((0, 4))
    C = W["C"]
    X = np.hstack([X8.astype(np.int32), Xe])
    v = C.run("rank", X) / 4096.0
    o = np.argsort(-v, kind="stable")[:V4.TOP_TEST]
    Bp = V4.apply_place(B[o], np.stack([C.run(m, X[o]) for m in ("dx", "dy", "ds")], 1) / 4096.0)
    return v[o], Bp


def detect(path):
    g = L.to_gray(np.asarray(Image.open(path).convert("RGB")))
    v1, B1 = law_pass(g)
    gs = L.resample(g, SHRINK)                           # LAW 3's own anti-aliased resampling
    v2, B2 = law_pass(gs, BIG_LEVEL)
    v = np.concatenate([v1, v2]); B = np.vstack([B1, B2 * SHRINK])
    o = np.argsort(-v, kind="stable")
    d = L.nms([(float(v[i]),) + tuple(B[i]) for i in o], 0.5)[:KEEP]
    return np.array([[x1, y1, x2 - x1, y2 - y1, s] for s, x1, y1, x2, y2 in d]).reshape(-1, 5)


def run_one(path):
    t = time.time()
    return os.path.relpath(path, os.path.join(WD, "WIDER_val", "images")), detect(path), time.time() - t


def init(so):
    import darkface_ctrees as CT
    V.init()
    W["C"] = CT.CTrees(so)


# ------------------------------------------------------------------ the official WIDER FACE evaluation, ported ---
def overlaps(a, b):
    ix = np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1)
    iy = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1); ab = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    return inter / (aa[:, None] + ab[None, :] - inter)


def image_eval(pred, gt, ignore, thr):
    p, g = pred.copy(), gt.copy()
    p[:, 2] += p[:, 0]; p[:, 3] += p[:, 1]; g[:, 2] += g[:, 0]; g[:, 3] += g[:, 1]
    ov = overlaps(p[:, :4], g)
    pred_recall = np.zeros(len(p)); recall_list = np.zeros(len(g)); proposal = np.ones(len(p))
    for h in range(len(p)):
        j = int(ov[h].argmax()); m = ov[h, j]
        if m >= thr:
            if ignore[j] == 0:
                recall_list[j] = -1; proposal[h] = -1
            elif recall_list[j] == 0:
                recall_list[j] = 1
        pred_recall[h] = (recall_list == 1).sum()
    return pred_recall, proposal


def img_pr(n, pred, proposal, pred_recall):
    pr = np.zeros((n, 2))
    for t in range(n):
        r = np.nonzero(pred[:, 4] >= 1 - (t + 1) / n)[0]
        if len(r):
            r = r[-1]; pr[t, 0] = (proposal[:r + 1] == 1).sum(); pr[t, 1] = pred_recall[r]
    return pr


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.])); mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[i + 1] - mrec[i]) * mpre[i + 1]).sum())


def evaluate(dets, n=1000, thr=0.5):
    from scipy.io import loadmat
    gtd = glob.glob(os.path.join(WD, "**", "ground_truth"), recursive=True)[0]
    G = loadmat(os.path.join(gtd, "wider_face_val.mat"))
    boxes, events, files = G["face_bbx_list"], G["event_list"], G["file_list"]
    allv = np.concatenate([d[:, 4] for d in dets.values() if len(d)])
    lo, hi = allv.min(), allv.max()
    nd = {k: (np.hstack([d[:, :4], ((d[:, 4] - lo) / (hi - lo))[:, None]]) if len(d) else d) for k, d in dets.items()}
    out = {}
    for setting in ("easy", "medium", "hard"):
        gl = loadmat(os.path.join(gtd, "wider_%s_val.mat" % setting))["gt_list"]
        count, pr = 0, np.zeros((n, 2))
        for i in range(len(events)):
            ev = str(events[i][0][0]); imgs = files[i][0]
            for j in range(len(imgs)):
                name = str(imgs[j][0][0])
                pred = nd.get(os.path.join(ev, name + ".jpg"), np.zeros((0, 5)))
                gt = boxes[i][0][j][0].astype(float); keep = gl[i][0][j][0]
                count += len(keep)
                if len(gt) == 0 or len(pred) == 0:
                    continue
                ignore = np.zeros(len(gt))
                if len(keep):
                    ignore[keep.ravel() - 1] = 1
                prec_recall, proposal = image_eval(pred, gt, ignore, thr)
                pr += img_pr(n, pred, proposal, prec_recall)
        prec = pr[:, 1] / np.maximum(pr[:, 0], 1e-12); rec = pr[:, 1] / count
        out[setting] = voc_ap(rec, prec)
    return out


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    if stage == "run":
        import darkface_ctrees as CT
        so = CT.build(json.load(open(os.path.join(OUT, "law50.json"))), OUT)
        paths = sorted(glob.glob(os.path.join(WD, "WIDER_val", "images", "*", "*.jpg")))
        dets, secs = {}, []
        with Pool(int(os.environ.get("NPROC", "6")), init, (so,)) as pool:
            for i, (k, d, dt) in enumerate(pool.imap_unordered(run_one, paths, chunksize=1)):
                dets[k] = d; secs.append(dt)
                if (i + 1) % 250 == 0:
                    print("  WIDER val: %d / %d images  (%.0fs)" % (i + 1, len(paths), time.time() - t0), flush=True)
        pickle.dump(dets, open(os.path.join(OUT, "wider_dets.pkl"), "wb"))
        print("WIDER val: %d images, %d detections, %.2f s per image per worker  [%.0fs]" % (len(dets), sum(len(d) for d in dets.values()), np.mean(secs), time.time() - t0))
    elif stage == "eval":
        dets = pickle.load(open(os.path.join(OUT, "wider_dets.pkl"), "rb"))
        ap = evaluate(dets)
        print("WIDER FACE validation, SPEC 50 zero-shot (night-trained):  Easy %.1f   Medium %.1f   Hard %.1f   [%.0fs]"
              % (100 * ap["easy"], 100 * ap["medium"], 100 * ap["hard"], time.time() - t0))
