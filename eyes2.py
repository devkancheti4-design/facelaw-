"""v2 AUTOMATIC EYES (no taps): the classic face box (OpenCV Viola-Jones) -> an eye region on each side at fixed shares
of the box -> THE IRIS LAW: catchlights removed (grey opening), then for every centre c the share of an iris-sized ring
whose strong gradients point straight away from c (max(0, d . g)^2, Timm & Barth's means of gradients restricted to the
ring), counted per direction so that agreement from MANY directions wins (an iris is ringed on 5-6 sides, an eye corner
on 2-3), times the dark centre -> candidates per side -> the pair rule (both strong, level, ~0.45 box widths apart,
centred on the box).
    python3 -I eyes2.py yale       Extended Yale B, eye-aligned crops placed in a frame (eyes known): absolute error and
                                   CONSISTENCY (the error against the same person's own median: what 1:1 matching feels)
    python3 -I eyes2.py user       the owner's own photos and camera frames against his own taps"""
import json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, grey_opening, maximum_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import finder as FD

HERE = os.path.dirname(os.path.abspath(__file__))
P = dict(TOP=0.22, SIDE=0.10, HGT=0.30, WID=0.38,       # eye regions as shares of the face box
         FAST=60, THR=0.3,                               # region resampled to 60 px wide; strong gradient: mean + 0.3 sd
         EYE=0.45, IRIS=0.095,                           # eye distance ~0.45 box widths; iris radius ~0.095 eye distances
         NC=5, PAIR=1.0)                                 # candidates per side; the pair rule's weight


def u8(lin):
    return FD.to_u8(lin).astype(float)


def face_box(lin):
    """the classic face box at the finest scale where the cascade sees a face -> (x, y, w, h) in image pixels or None"""
    for f in (1, 2, 4, 8):
        if min(lin.shape) // f < 24:
            break
        if max(lin.shape) // f > 1200:
            continue
        im = FD.to_u8(FD.boxdown(lin, f))
        faces = FD.FACE.detectMultiScale(im, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            return np.array([x * f, y * f, w * f, h * f], float)
    return None


def resample(img, x0, y0, w, h, n):
    """the region (x0, y0, w, h) of img on an n-wide grid (anti-aliased), and its scale (image px per grid px)"""
    s = w / n; m = max(2, int(round(h / s)))
    G = gaussian_filter(img, max(0.0, 0.5 * (s - 1)))
    yy, xx = np.mgrid[0:m, 0:n].astype(float)
    return map_coordinates(G, [y0 + (yy + 0.5) * s - 0.5, x0 + (xx + 0.5) * s - 0.5], order=1, mode="nearest"), s


def iris_score(E, r, thr=None):
    thr = P["THR"] if thr is None else thr
    k = max(3, int(round(0.7 * r)) | 1)
    Eo = grey_opening(E, size=(k, k))                    # a catchlight is a bright spot smaller than the iris: gone
    G = gaussian_filter(Eo, max(0.6, r / 6))
    gx, gy = np.gradient(G, axis=1), np.gradient(G, axis=0)
    m = np.hypot(gx, gy)
    strong = m > m.mean() + thr * m.std()
    ux = np.where(strong, gx / np.maximum(m, 1e-12), 0.0); uy = np.where(strong, gy / np.maximum(m, 1e-12), 0.0)
    h, w = E.shape
    R1 = int(np.ceil(1.6 * r)) + 1
    Hs = np.zeros((8, h, w)); Ns = np.zeros(8)
    for dy in range(-R1, R1 + 1):
        for dx in range(-R1, R1 + 1):
            dd = np.hypot(dx, dy)
            if dd < 0.6 * r or dd > 1.6 * r:
                continue
            sec = int(np.floor((np.arctan2(dy, dx) + np.pi) / (np.pi / 4))) % 8
            dot = np.maximum(0.0, (dx * ux + dy * uy) / dd) ** 2       # the pixel at c + (dy, dx): gradient along d
            sh = np.zeros((h, w))
            ys, xs = slice(max(0, -dy), h - max(0, dy)), slice(max(0, -dx), w - max(0, dx))
            yd, xd = slice(max(0, dy), h - max(0, -dy)), slice(max(0, dx), w - max(0, -dx))
            sh[ys, xs] = dot[yd, xd]
            Hs[sec] += sh; Ns[sec] += 1
    frac = Hs / np.maximum(Ns, 1)[:, None, None]
    iso = np.sort(frac, 0)[2:].mean(0)                    # the best 6 of 8 directions: a ring, not an edge
    S = gaussian_filter(Eo, max(0.6, r / 2))
    dark = (S.max() - S) / max(S.max() - S.min(), 1e-9)
    return iso * dark


def peaks(O, n, sep):
    pk = (O == maximum_filter(O, size=max(3, int(sep)))) & (O > 0)
    ys, xs = np.nonzero(pk)
    o = np.argsort(-O[ys, xs])[:n]
    return [(float(O[ys[k], xs[k]]), int(ys[k]), int(xs[k])) for k in o]


def subpix(O, y, x):
    """parabola through the maximum and its neighbours, each axis"""
    def off(a, b, c):
        den = a - 2 * b + c
        return 0.0 if den >= 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    h, w = O.shape
    oy = off(O[y - 1, x], O[y, x], O[y + 1, x]) if 0 < y < h - 1 else 0.0
    ox = off(O[y, x - 1], O[y, x], O[y, x + 1]) if 0 < x < w - 1 else 0.0
    return x + ox, y + oy


def candidates(img, box, p):
    x, y, w, h = box
    cands = []
    for side in (0, 1):
        ex = x + (p["SIDE"] * w if side == 0 else w - p["SIDE"] * w - p["WID"] * w)
        ey = y + p["TOP"] * h
        E, s = resample(img, ex, ey, p["WID"] * w, p["HGT"] * h, p["FAST"])
        r = p["IRIS"] * p["EYE"] * w / s
        O = iris_score(E, r, p["THR"])
        b = int(np.ceil(0.5 * r)); O[:b], O[-b:], O[:, :b], O[:, -b:] = 0, 0, 0, 0
        cs = []
        for v, yy, xx in peaks(O, p["NC"], r):
            fx, fy = subpix(O, yy, xx)
            cs.append((v, np.array([ex + (fx + 0.5) * s - 0.5, ey + (fy + 0.5) * s - 0.5])))
        cands.append(cs or [(0.0, np.array([ex + 0.5 * p["WID"] * w, ey + 0.5 * p["HGT"] * h]))])
    return cands


def pair(cands, box, p):
    x, y, w, h = box
    top = max(c[0] for cs in cands for c in cs) or 1.0
    best, out = -np.inf, None
    for vL, pL in cands[0]:
        for vR, pR in cands[1]:
            v = pR - pL; d = np.hypot(*v)
            if d <= 0:
                continue
            pen = abs(v[1]) / d + abs(d / (p["EYE"] * w) - 1) + abs((pL[0] + pR[0]) / 2 - (x + w / 2)) / w
            sc = (vL + vR) / top - p["PAIR"] * pen
            if sc > best:
                best, out = sc, [pL, pR]
    return out


def pupils(img, box, p=None):
    p = dict(P, **(p or {}))
    return pair(candidates(img, box, p), box, p)


def find(lin, p=None):
    box = face_box(lin)
    if box is None:
        return None
    return dict(box=box, eyes=pupils(u8(lin), box, p))


def err(found, truth):
    tl, tr = truth; ed = np.linalg.norm(tr - tl)
    return max(np.linalg.norm(found[0] - tl), np.linalg.norm(found[1] - tr)) / ed


def summary(e):
    e = np.asarray(e)
    return "median %.3f; within 0.02 %5.1f%%, 0.04 %5.1f%%, 0.05 %5.1f%%, 0.10 %5.1f%%" % (
        np.median(e), 100 * (e <= .02).mean(), 100 * (e <= .04).mean(), 100 * (e <= .05).mean(), 100 * (e <= .10).mean())


def load_upright(path):
    """as the server reads a photo: upright (EXIF), sRGB -> linear light, mean of the channels"""
    from PIL import Image, ImageOps
    a = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), float) / 255
    return 255 * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).mean(2)


def user_items():
    """the owner's own photos and camera frames with his own eye taps (or hand-read pupils): (name, linear image, (L, R))"""
    out = []
    D = os.path.join(HERE, "data", "phone_user")
    for nm, path, l, r in (("IMG_1374", "IMG_1374_up.jpg", (956, 1731), (1589, 1728)), ("photo1", "photo1.jpg", (572, 1016), (828, 1015)),
                           ("photo2", "photo2.jpg", (622, 960), (812, 955))):
        out.append((nm, load_upright(os.path.join(D, path)), (np.array(l, float), np.array(r, float))))
    DB = os.path.join(HERE, "data", "webapp", "debug")
    for d in sorted(os.listdir(DB)):
        r = json.load(open(os.path.join(DB, d, "result.json")))
        if not r.get("eyes") or d.startswith("20261028-072259"):       # 072259 repeats 072254's photo and taps
            continue
        (x1, y1), (x2, y2) = sorted(tuple(map(float, e)) for e in r["eyes"])
        eL, eR = np.array([x1, y1]), np.array([x2, y2])
        f = os.path.join(DB, d, "photo.jpg" if d.endswith("photo") else ("pair0_white.jpg" if d.endswith("scan") else "white0.jpg"))
        if not os.path.exists(f) or d == "20261027-201102-try":            # 201102-try: taps at (134,182),(534,182), a page test
            continue
        out.append((d, load_upright(f), (eL, eR)))
    return out


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "yale":
        import yale as Y
        B = 48
        TR = json.load(open(os.path.join(HERE, "yale_eyes_truth.json")))["eyes"]
        half = os.environ.get("HALF", "A")
        people = [p for p in (Y.IDS[0::2] if half == "A" else Y.IDS[1::2] if half == "B" else Y.IDS) if p in TR]
        maxth = float(os.environ.get("MAXTH", 25.5))
        res = {"classic eye boxes": [], "iris law (detector box)": [], "iris law (ideal box)": []}
        miss = n = 0
        for p in people:
            tl, tr = np.array(TR[p][0]) + B, np.array(TR[p][1]) + B
            for im in Y.PEOPLE[p]:
                if im["I"].sum() <= 0 or im["theta"] > maxth:
                    continue
                I = Y.read_pgm(os.path.join(Y.ROOT, p, im["name"])).astype(float)
                lin = np.pad(255 * (I / 255) ** 2.2, B, mode="edge")
                n += 1
                c = FD.find(lin)
                res["classic eye boxes"].append(err(c["classic"], (tl, tr)) if c else np.inf)
                f = find(lin)
                res["iris law (detector box)"].append(err(f["eyes"], (tl, tr)) if f else np.inf)
                ed = np.linalg.norm(tr - tl); w = ed / P["EYE"]
                box = np.array([(tl[0] + tr[0]) / 2 - w / 2, (tl[1] + tr[1]) / 2 - 0.36 * w, w, w])
                res["iris law (ideal box)"].append(err(pupils(u8(lin), box), (tl, tr)))
        print("Yale B half %s (%d people), light <= %.0f deg: %d faces (errors vs hand-checked pupils; a miss counts as a failure)" % (
            half, len(people), maxth, n))
        for k, v in res.items():
            print("   %-26s found %5.1f%%   %s" % (k, 100 * np.isfinite(v).mean(), summary(v)))
    elif stage == "user":
        es = []
        for nm, lin, t in user_items():
            f = find(lin)
            if f is None:
                print("%-24s no face box" % nm); continue
            e = err(f["eyes"], t); es.append(e)
            print("%-24s eye distance %4.0f px; error %.3f" % (nm, np.linalg.norm(t[1] - t[0]), e))
        print("owner's images: %s" % summary(es))
