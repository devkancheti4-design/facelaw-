"""DARK FACE by scene light: AP of the saved detections in thirds of the images' measured brightness (median linear-light
luminance, 0..255).  No new detection is run and no test image is scored again: SPEC 50 design-validation numbers
(v4_val_numbers.pkl, LAW 8 placed, NMS 0.5) and the sealed LAW 7 test detections (test7_dets.pkl).
    python3 -I darkface_light.py"""
import json, os, pickle, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import darkface_law as L
import darkface_run as R

OUT = R.OUT


def brightness(keys):
    return {k: float(np.median(L.to_gray(np.asarray(Image.open(ip).convert("RGB"))))) for k, ip, lp in R.items(keys)}


def by_light(dets, gts, br, label):
    keys = sorted(gts, key=lambda k: br[k])
    thirds = np.array_split(np.array(keys), 3)
    ap, rec = L.average_precision(dets, gts)
    print("%s -- all %d images: AP %.2f%% (recall %.1f%%)" % (label, len(keys), 100 * ap, 100 * rec))
    for name, ks in zip(("darkest third", "middle third", "brightest third"), thirds):
        ks = list(ks)
        a, r = L.average_precision({k: dets[k] for k in ks}, {k: gts[k] for k in ks})
        print("   %-16s median luminance %5.2f..%5.2f  faces %5d  AP %.2f%%  recall %.1f%%"
              % (name, br[ks[0]], br[ks[-1]], sum(len(gts[k]) for k in ks), 100 * a, 100 * r))


if __name__ == "__main__":
    SPLIT = json.load(open(os.path.join(OUT, "split.json")))
    Z = pickle.load(open(os.path.join(OUT, "v4_val_numbers.pkl"), "rb"))
    dets = {}
    for k, nb in Z["nums"].items():
        if nb is None:
            dets[k] = []; continue
        v, B, B8, kk, sn = nb
        dets[k] = L.nms([(float(v[i]),) + tuple(B8[i]) for i in range(len(v))], 0.5)
    by_light(dets, Z["gts"], brightness(SPLIT["design_val"]), "SPEC 50, design-validation")
    T = pickle.load(open(os.path.join(OUT, "test7_dets.pkl"), "rb"))
    by_light(T["dets"], T["gts"], brightness(SPLIT["test"]), "LAW 7, sealed test (saved detections, not re-run)")
