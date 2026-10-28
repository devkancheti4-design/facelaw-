"""TWO-TAP MODE: the user taps both eyes; Law 3 with two landmarks fixes position, scale and tilt exactly (a similarity
transform onto the canonical eye positions, which are MEASURED from the mean frontal Yale B face).  Then the local law
recognises with per-candidate profiling.  No detector, no face model.
    python3 -I phone_eyes.py ENROL lx,ly,rx,ry  PROBE lx,ly,rx,ry ...      (pupil positions as fractions of w, h)"""
import os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y
from yale_gate import sectors

WH, WW = 48, 42


def canonical_eyes():
    M = np.mean([next(im["I"] for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")).reshape(WH, WW) for p in Y.IDS], 0)
    S = gaussian_filter(M, 1.0)
    band = S[8:26]                                            # the eye band (upper face), measured on the mean face
    L = np.unravel_index(np.argmin(band[:, :WW // 2]), band[:, :WW // 2].shape)
    R = np.unravel_index(np.argmin(band[:, WW // 2:]), band[:, WW // 2:].shape)
    return np.array([L[1], L[0] + 8], float), np.array([R[1] + WW // 2, R[0] + 8], float)


def load(path):
    a = np.asarray(Image.open(path).convert("RGB"), float) / 255
    return 255 * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).mean(2)


def crop(img, eyeL, eyeR, cL, cR, ds=1.0, dx=0.0, dy=0.0):
    """sample the canonical 48 x 42 frame: canonical point q -> image point A q + t, A a similarity (scale, rotation)"""
    v_img, v_can = eyeR - eyeL, cR - cL
    s = np.linalg.norm(v_img) / np.linalg.norm(v_can) * ds
    th = np.arctan2(v_img[1], v_img[0]) - np.arctan2(v_can[1], v_can[0])
    A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    mid_c, mid_i = (cL + cR) / 2, (eyeL + eyeR) / 2
    yy, xx = np.mgrid[0:WH, 0:WW].astype(float)
    q = np.stack([xx.ravel() - mid_c[0] + dx, yy.ravel() - mid_c[1] + dy])
    p = A @ q + mid_i[:, None]
    # anti-alias: smooth the image to the sampling scale first
    return map_coordinates(gaussian_filter(img, max(0.0, 0.5 * (s - 1))), [p[1], p[0]], order=1, mode="nearest").reshape(WH, WW)


def profiled(img, eL, eR, cL, cR, gal):
    best = np.full(len(gal), np.inf)
    for ds in (0.94, 0.97, 1.0, 1.03, 1.06):
        for dx in (-1.5, -0.75, 0, 0.75, 1.5):
            for dy in (-1.5, -0.75, 0, 0.75, 1.5):
                s = sectors(crop(img, eL, eR, cL, cR, ds, dx, dy)).astype(np.int16)
                dd = np.abs(gal - s[None]); best = np.minimum(best, np.minimum(dd, 8 - dd).sum(1) / s.size)
    return best


if __name__ == "__main__":
    cL, cR = canonical_eyes()
    print("canonical eyes measured on the mean frontal Yale face (x, y in the 42 x 48 frame): L %s  R %s" % (cL, cR))
    args = sys.argv[1:]
    items = []
    for i in range(0, len(args), 2):
        img = load(args[i]); h, w = img.shape
        lx, ly, rx, ry = (float(v) for v in args[i + 1].split(","))
        items.append((os.path.basename(args[i]), img, np.array([lx * w, ly * h]), np.array([rx * w, ry * h])))
    name0, img0, eL0, eR0 = items[0]
    me = sectors(crop(img0, eL0, eR0, cL, cR)).astype(np.int16)
    yale = np.array([sectors(next(im["I"].reshape(WH, WW) for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm"))) for p in Y.IDS], np.int16)
    gal = np.vstack([me[None], yale]); names = ["YOU (enrolled)"] + Y.IDS
    print("gallery: you (from %s, eye-aligned) + 38 Yale B people" % name0)
    for name, img, eL, eR in items[1:]:
        d = profiled(img, eL, eR, cL, cR, gal); o = np.argsort(d)
        print("%-22s -> best %-15s %.3f   runner-up %-9s %.3f   margin %.3f   your rank %d of %d" % (
            name, names[o[0]], d[o[0]], names[o[1]], d[o[1]], d[o[1]] - d[o[0]], int(np.where(o == 0)[0][0]) + 1, len(names)))
    out = os.environ.get("PREVIEW")
    if out:
        tiles = [crop(img, eL, eR, cL, cR) for _, img, eL, eR in items]
        tiles = [255 * (t / max(t.max(), 1e-9)) for t in tiles]
        Image.fromarray(np.hstack([np.kron(t, np.ones((4, 4))) for t in tiles]).astype(np.uint8)).save(out)
        print("aligned crops (each brightness-normalised for viewing) in", out)
