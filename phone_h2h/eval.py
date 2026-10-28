"""Score the face-detection head-to-head: Android's detector (the phone's boxes, android_dets.jsonl) against Face Law's
dark-face law (SPEC 100 = the SPEC 50 detector, every number authored; its design-validation detections from
darkface/v4_val_numbers.pkl), on the same 300 DARK FACE design-validation images, IoU >= 0.5, one face per box.
Android's detector gives no confidence, so it is one operating point (precision, recall); Face Law is a ranked list, so
it is compared at Android's recall, at Android's precision, and at Android's number of boxes.
    python3 -I phone_h2h/eval.py"""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import darkface_law as L


def match(dets, gt):
    """greedy one-to-one matching by score order (VOC); returns per-detection hit flags and per-face found flags"""
    used = np.zeros(len(gt), bool); hit = []
    for d in dets:
        if len(gt) == 0:
            hit.append(False); continue
        o = L.iou_many(np.array(d[1:5], float), gt); j = int(np.argmax(o))
        ok = o[j] >= 0.5 and not used[j]
        if ok:
            used[j] = True
        hit.append(bool(ok))
    return hit, used


def size_bucket(gt):
    s = np.sqrt(np.maximum(gt[:, 2] - gt[:, 0], 1) * np.maximum(gt[:, 3] - gt[:, 1], 1))
    return np.digitize(s, [10, 20])                       # 0: under 10 px, 1: 10-20 px, 2: 20 px and up


if __name__ == "__main__":
    Z = pickle.load(open(os.path.join(ROOT, "darkface", "v4_val_numbers.pkl"), "rb"))
    gts = Z["gts"]
    ours = {}
    for k, nb in Z["nums"].items():
        if nb is None:
            ours[k] = []; continue
        v, B, B8 = nb[0].astype(float), nb[1], nb[2]
        BB = (B8 if B8 is not None else B).astype(float)
        ours[k] = L.nms([(float(v[i]),) + tuple(BB[i]) for i in range(len(v))], 0.5)
    for k in gts:
        ours.setdefault(k, [])
    ap, rec_max = L.average_precision(ours, gts)
    print("Face Law (SPEC 100) on design-validation: AP %.2f%%, recall at the end of its list %.1f%%" % (100 * ap, 100 * rec_max))
    A = {}
    for line in open(os.path.join(HERE, "android_dets.jsonl")):
        d = json.loads(line); A[d["key"]] = d
    missing = [k for k in gts if k not in A]
    print("Android: %d of %d images answered%s" % (len(A), len(gts), (" (missing %d)" % len(missing)) if missing else ""))
    keys = [k for k in gts if k in A]
    npos = sum(len(gts[k]) for k in keys)
    tp = fp = 0; found_a = {}
    for k in keys:
        dets = [(1.0,) + tuple(b) for b in A[k]["boxes"]]
        hit, used = match(dets, gts[k]); tp += sum(hit); fp += len(hit) - sum(hit); found_a[k] = used
    ra, pa = tp / max(npos, 1), tp / max(tp + fp, 1)
    ms = np.median([A[k]["ms"] for k in keys])
    print("Android's detector: %d boxes, %d on a face: recall %.1f%%, precision %.1f%%; median %.0f ms per photo on the phone"
          % (tp + fp, tp, 100 * ra, 100 * pa, ms))
    # Face Law's ranked list over the same images
    recs = sorted(((d[0], k, d) for k in keys for d in ours[k]), key=lambda t: -t[0])
    used = {k: np.zeros(len(gts[k]), bool) for k in keys}
    ctp = np.zeros(len(recs)); order_hits = []
    for i, (s, k, d) in enumerate(recs):
        g = gts[k]; ok = False
        if len(g):
            o = L.iou_many(np.array(d[1:5], float), g); j = int(np.argmax(o))
            if o[j] >= 0.5 and not used[k][j]:
                used[k][j] = True; ok = True
        ctp[i] = ok; order_hits.append((k, ok))
    ctp = np.cumsum(ctp); n = np.arange(1, len(recs) + 1)
    rec, prec = ctp / npos, ctp / n
    i_r = int(np.searchsorted(rec, ra)) if ra <= rec[-1] else None
    ok_p = np.nonzero(prec >= pa)[0]
    i_n = min(tp + fp, len(recs)) - 1
    print("Face Law at Android's recall (%.1f%%): precision %s" % (100 * ra, ("%.1f%%" % (100 * prec[i_r])) if i_r is not None else "not reached"))
    print("Face Law at Android's precision (%.1f%%): recall %.1f%%" % (100 * pa, 100 * rec[ok_p.max()] if len(ok_p) else 0.0))
    print("Face Law with Android's number of boxes (%d): recall %.1f%%, precision %.1f%%" % (tp + fp, 100 * rec[i_n], 100 * prec[i_n]))
    # by face size: Android's boxes against Face Law's top boxes at the same count
    used_n = {k: np.zeros(len(gts[k]), bool) for k in keys}
    cnt = 0
    for (s, k, d) in recs[:tp + fp]:
        g = gts[k]
        if len(g):
            o = L.iou_many(np.array(d[1:5], float), g); j = int(np.argmax(o))
            if o[j] >= 0.5 and not used_n[k][j]:
                used_n[k][j] = True
    for b, name in enumerate(("under 10 px", "10-20 px", "20 px and up")):
        tot = sum(int((size_bucket(gts[k]) == b).sum()) for k in keys if len(gts[k]))
        fa = sum(int(found_a[k][size_bucket(gts[k]) == b].sum()) for k in keys if len(gts[k]))
        fo = sum(int(used_n[k][size_bucket(gts[k]) == b].sum()) for k in keys if len(gts[k]))
        print("  faces %-12s %5d: Android found %5.1f%%, Face Law (same number of boxes) %5.1f%%" % (name, tot, 100 * fa / max(tot, 1), 100 * fo / max(tot, 1)))
    json.dump(dict(android=dict(recall=ra, precision=pa, boxes=tp + fp, hits=tp, median_ms=ms, images=len(keys)),
                   facelaw=dict(AP=ap, recall_max=rec_max, precision_at_android_recall=None if i_r is None else float(prec[i_r]),
                                recall_at_android_precision=float(rec[ok_p.max()]) if len(ok_p) else 0.0,
                                recall_same_boxes=float(rec[i_n]), precision_same_boxes=float(prec[i_n]))),
              open(os.path.join(HERE, "h2h_result.json"), "w"), indent=1)
