"""the blink signal before it is a law: how open each eye looks in the 96 x 84 eye-law crop (the share of the eye window
darker than half the skin around it), frame to frame on faces that do NOT blink -- rendered live faces at many angles
and photon levels, printed real photographs moved by the hand -- and on a crude closed lid painted over the same frames.
A blink rule must never fire on the first two; the third only says the signal moves the right way."""
import json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, shift as subshift
FL = os.path.expanduser("~/neooos/facelaw")
sys.path.insert(0, FL); sys.path.insert(0, os.path.join(FL, "phone_faceid"))
import pose_law as PL
import server as SV
B = SV.B
TR = json.load(open(os.path.join(FL, "yale_eyes_truth.json")))["eyes"]; ids = sorted(TR)
Zs = np.load(PL.SHAPES); R = np.random.default_rng(4); PAD = 48
EYE96 = ((16.5, 22.5), (66.5, 22.5))


def openness(c):
    out = []
    for cx, cy in EYE96:
        win = c[int(cy - 7):int(cy + 8), int(cx - 11):int(cx + 12)]
        ring = c[int(cy - 10):int(cy + 15), int(cx - 15):int(cx + 16)]
        out.append(float(np.mean(win < 0.5 * np.percentile(ring, 75))))
    return out


def lid(c):
    """a crude closed eye: the upper-lid skin just above each eye window drawn down over it, a thin dark lash line"""
    c = c.copy()
    for cx, cy in EYE96:
        y0, y1, x0, x1 = int(cy - 7), int(cy + 8), int(cx - 11), int(cx + 12)
        skin = c[y0 - 4:y0, x0:x1].mean(0)
        c[y0:y1, x0:x1] = skin[None, :] * (1 + 0.03 * R.standard_normal((y1 - y0, x1 - x0)))
        c[int(cy) + 1:int(cy) + 3, x0 + 3:x1 - 3] *= 0.45
    return c


def frame_of(img, N):
    bgv = 0.25 * img.mean()
    fr = np.full((PL.H + 2 * PAD, PL.W + 2 * PAD), bgv); fr[PAD:PAD + PL.H, PAD:PAD + PL.W] = img
    fr = R.poisson(np.clip(fr / max(np.percentile(fr, 99.5), 1e-9), 0, None) * N) / N
    rgb = np.repeat(PL.srgb8(np.clip(fr, 0, 1))[..., None], 3, -1)
    lin = np.empty(rgb.shape[:2]); B.LIB.fl2_linear_of(np.ascontiguousarray(rgb).ctypes.data_as(B.U8), rgb.shape[1], rgb.shape[0], 3, lin.ctypes.data_as(B.D))
    return lin


ratios = {"live faces, no blink": [], "moving prints, no blink": [], "painted closed lid": []}
for p in ids[:20]:
    hd = PL.Head(Zs[p + "_rho"].astype(float), Zs[p + "_n"].astype(float), Zs[p + "_Z"].astype(float), TR[p])
    for (yaw, pitch) in ((0, 0), (-20, 0), (20, 10), (10, -15)):
        for N in (60.0, 400.0):
            seq = []
            for t in range(8):                                   # 8 frames of one still-ish try: wobble +-2 deg
                Rm, win, _ = hd.turn(yaw + R.uniform(-2, 2), pitch + R.uniform(-2, 2))
                img, v = hd.shade(Rm, win, [(1.0, PL.SCREEN)], 0.2)
                lin = frame_of(np.where(v, img, 0.25 * img[v].mean()), N)
                e, _ = SV.eyes_on(lin)
                if e is None:
                    continue
                c = SV.crop(lin, e, True); seq.append((openness(c), openness(lid(c))))
            if len(seq) >= 4:
                base = np.median([np.mean(o) for o, _ in seq])
                ratios["live faces, no blink"] += [np.mean(o) / base for o, _ in seq]
                ratios["painted closed lid"] += [np.mean(l) / base for _, l in seq]
    for ph in [q for q in PL.photos(p) if q["theta"] <= 25][:3]:  # a print of a real photo, moved 0-3 px between frames
        x = ph["raw"] / max(np.percentile(ph["raw"], 99.5), 1e-9); paper = gaussian_filter(0.06 + 0.86 * np.clip(x, 0, 1), 0.6)
        seq = []
        for t in range(8):
            lin = frame_of(subshift(paper, R.uniform(-3, 3, 2), order=3, mode="nearest"), 400.0)
            e, _ = SV.eyes_on(lin)
            if e is not None:
                seq.append(openness(SV.crop(lin, e, True)))
        if len(seq) >= 4:
            base = np.median([np.mean(o) for o in seq])
            ratios["moving prints, no blink"] += [np.mean(o) / base for o in seq]
for k, v in ratios.items():
    v = np.array(v)
    print("%-26s frames %4d  openness / the try's median: min %.2f  1%% %.2f  5%% %.2f  median %.2f  max %.2f" % (
        k, len(v), v.min(), np.percentile(v, 1), np.percentile(v, 5), np.median(v), v.max()), flush=True)
