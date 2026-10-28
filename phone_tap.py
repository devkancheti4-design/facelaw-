"""TAP MODE -- the first live version: the user taps a point on the face; the laws do the rest.
Among all windows (every scale) that CONTAIN the tap, take the best light-law score that the significant-gradient local
law verifies; align it; recognise against the gallery with per-candidate alignment profiling.
    python3 -I phone_tap.py ENROL.jpg fx,fy  PROBE.jpg fx,fy  [PROBE fx,fy ...]     (fx, fy = tap as fractions of w, h)"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect as D
import yale as Y
import detect_test as T
import phone_test as P
from yale_gate import sectors


def at_tap(x, tap, UF, UB, sigma):
    ty, tx = tap
    best = None
    for s in tuple(2 ** (k / 4) for k in range(2, 12)):
        im = zoom(x, 1 / s, order=1)
        if im.shape[0] < D.WH or im.shape[1] < D.WW:
            continue
        S = D.score_map(im, UF, UB)
        r_lo, r_hi = int(max(0, ty / s - D.WH + 1)), int(min(S.shape[0] - 1, ty / s))
        c_lo, c_hi = int(max(0, tx / s - D.WW + 1)), int(min(S.shape[1] - 1, tx / s))
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sub = S[r_lo:r_hi + 1, c_lo:c_hi + 1]
        for _ in range(4):                                     # a few best windows at this scale that contain the tap
            i = np.unravel_index(np.argmax(sub), sub.shape)
            if not np.isfinite(sub[i]):
                break
            d = (float(sub[i]), (r_lo + i[0]) * s, (c_lo + i[1]) * s, D.WH * s, D.WW * s)
            sub[max(0, i[0] - 3):i[0] + 4, max(0, i[1] - 3):i[1] + 4] = -np.inf
            v = P.verify_sig(x, d, sigma)
            if v is not None and v <= P.VERIFY_MAX and (best is None or d[0] > best[0][0]):
                best = (d, v)
    return best


if __name__ == "__main__":
    UF = D.face_basis([T.harmonic(p) for p in Y.IDS]); UB = D.bg_basis()
    P.SG = T.SG = np.array([sectors(im["I"].reshape(48, 42)) for p in Y.IDS for im in Y.PEOPLE[p] if Y.subset(im["theta"]) <= 4], np.int16)
    args = sys.argv[1:]
    items = [(args[i], tuple(float(v) for v in args[i + 1].split(","))) for i in range(0, len(args), 2)]
    found, pics = {}, {}
    for path, (fx, fy) in items:
        x, pic = P.load(path)
        sigma = P.noise_sigma(x)
        D.NOISE_FLOOR = 48 * 42 * sigma ** 2
        tap = (fy * x.shape[0], fx * x.shape[1])
        got = at_tap(x, tap, UF, UB, sigma)
        name = os.path.basename(path)
        if got is None:
            print("%-22s tap (%.0f, %.0f): no window around the tap passes both laws -> no face, no answer" % (name, tap[1], tap[0]))
            pics[path] = (pic, None, tap); continue
        d, v = got
        a = D.align(x, d, UF, UB)
        found[path] = (x, a[2]); pics[path] = (pic, a[2], tap)
        print("%-22s tap (%.0f, %.0f) -> face box row %.0f col %.0f height %.0f;  light-law %.3f, local-law %.2f, noise sigma %.2f"
              % (name, tap[1], tap[0], a[2][0], a[2][1], a[2][2], d[0], v, sigma))
    enrol = items[0][0]
    if enrol in found:
        x0, b0 = found[enrol]
        me = sectors(D.sample(x0, *b0)).astype(np.int16)
        gal = np.vstack([me[None], np.array([sectors(next(im["I"].reshape(48, 42) for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")))
                                              for p in Y.IDS], np.int16)])
        names = ["YOU (enrolled)"] + Y.IDS
        print("\ngallery: you (from %s) + 38 Yale B people" % os.path.basename(enrol))
        for path, _ in items[1:]:
            if path not in found:
                print("%-22s -> no face, no answer" % os.path.basename(path)); continue
            x, b = found[path]
            dist = P.profiled(x, b, gal); o = np.argsort(dist)
            print("%-22s -> best %-15s %.3f   runner-up %-9s %.3f   margin %.3f   your rank %d of %d"
                  % (os.path.basename(path), names[o[0]], dist[o[0]], names[o[1]], dist[o[1]], dist[o[1]] - dist[o[0]],
                     int(np.where(o == 0)[0][0]) + 1, len(names)))
    W = sum(p[0].width for p in pics.values()) + 10 * len(pics)
    canvas = Image.new("RGB", (W, 400), "white"); xo = 0
    for path, _ in items:
        pic, box, tap = pics[path]; pic = pic.copy(); dr = ImageDraw.Draw(pic)
        dr.ellipse([tap[1] - 5, tap[0] - 5, tap[1] + 5, tap[0] + 5], outline=(40, 160, 255), width=3)
        if box:
            r, c, h, w = box; dr.rectangle([c, r, c + w, r + h], outline=(255, 40, 40), width=3)
        canvas.paste(pic, (xo, 0)); xo += pic.width + 10
    out = os.environ.get("PREVIEW", "phone_tap.png"); canvas.save(out); print("\nboxes drawn in", out)
