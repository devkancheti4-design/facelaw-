"""Real phone photos (JPEG, no RAW): detect -> align -> recognise.  JPEG path of Law 1: the tone curve is assumed sRGB
and inverted to linear light (an assumption -- the phone's real curve is unknown), luminance taken as the mean of R, G, B.
The local law needs no linearity: an increasing tone curve never turns a gradient.
    python3 -I phone_test.py ENROL.jpeg PROBE1.jpeg [PROBE2.jpeg ...]"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect as D
import yale as Y
import detect_test as T
from yale_gate import sectors


def load(path, height=400):
    im = Image.open(path).convert("RGB")
    w = int(round(im.width * height / im.height))
    a = np.asarray(im.resize((w, height), Image.BILINEAR), float) / 255
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)        # sRGB -> linear (assumed)
    return 255 * lin.mean(2), im.resize((w, height))


def noise_sigma(x):
    """Immerkaer (1996): the image's own noise level -- Law 1's measurement when the file carries no noise tag"""
    from scipy.signal import convolve2d
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    return float(np.sqrt(np.pi / 2) * np.abs(convolve2d(x, k, mode="valid")).sum() / (6 * (x.shape[0] - 2) * (x.shape[1] - 2)))


def verify_sig(x, d, sigma):
    """the local law counting only gradients significantly above the noise (>= 2 sigma_g); None if too little structure"""
    from scipy.ndimage import gaussian_filter
    crop = D.sample(x, d[1], d[2], d[3], d[4])
    if crop is None:
        return None
    G = gaussian_filter(crop, 0.75)
    g = np.hypot(np.gradient(G, axis=1), np.gradient(G, axis=0)).ravel()
    sig = g >= 2 * sigma * np.sqrt(2 * DL_EH)
    if sig.mean() < 0.2:
        return None
    s = sectors(crop).astype(np.int16)
    dd = np.abs(SG[:, sig] - s[sig][None]); dd = np.minimum(dd, 8 - dd)
    return float(dd.sum(1).min() / sig.sum())


def best_face(x, UF, UB):
    """noise floor in the light-law score; the best light-law score among windows the local law verifies"""
    sigma = noise_sigma(x)
    D.NOISE_FLOOR = 48 * 42 * sigma ** 2
    dets = D.detect(x, UF, UB, -np.inf, scales=tuple(2 ** (k / 4) for k in range(4, 12)))
    ok = []
    for d in dets[:20]:
        v = verify_sig(x, d, sigma)
        if v is not None and v <= VERIFY_MAX:
            ok.append((d, v))
    return (max(ok, key=lambda t: t[0][0]) if ok else None), sigma


def profiled(x, box, gal):
    rr, cc, hh, ww = box
    best = np.full(len(gal), np.inf)
    for ds in (0.92, 0.96, 1.0, 1.04, 1.08):
        for dr in (-3, -2, -1, 0, 1, 2, 3):
            for dc in (-3, -2, -1, 0, 1, 2, 3):
                h2, w2 = hh * ds, ww * ds
                crop = D.sample(x, rr + dr * hh / 48 - (h2 - hh) / 2, cc + dc * ww / 42 - (w2 - ww) / 2, h2, w2)
                if crop is None:
                    continue
                s = sectors(crop).astype(np.int16); dd = np.abs(gal - s[None])
                best = np.minimum(best, np.minimum(dd, 8 - dd).sum(1) / s.size)
    return best


from dark_law import E_H as _EH
DL_EH = _EH[0]                       # gradient-kernel energy at scale 0.75 (Law 1 significance, as in the dark law)
VERIFY_MAX = 1.6                     # the local-law verification threshold calibrated on DESIGN people (detect_results)

if __name__ == "__main__":
    UF = D.face_basis([T.harmonic(p) for p in Y.IDS]); UB = D.bg_basis()
    SG = np.array([sectors(im["I"].reshape(48, 42)) for p in Y.IDS for im in Y.PEOPLE[p] if Y.subset(im["theta"]) <= 4], np.int16)
    T.SG = SG
    paths = sys.argv[1:]
    boxes, crops, pics = {}, {}, {}
    for pth in paths:
        x, pic = load(pth)
        got, sigma = best_face(x, UF, UB)
        if got is None:
            print("%-12s no face passes both laws (noise sigma %.2f)" % (os.path.basename(pth), sigma)); continue
        d, v = got
        a = D.align(x, d, UF, UB)
        boxes[pth], crops[pth], pics[pth] = a[2], (x, a[2]), pic
        print("%-12s face at row %.0f col %.0f, height %.0f px of %d;  light-law score %.3f, significant-gradient distance %.2f  (noise sigma %.2f)"
              % (os.path.basename(pth), a[2][0], a[2][1], a[2][2], x.shape[0], d[0], v, sigma))
    # enrol: the first photo (aligned crop) joins the 38 Yale B people (their frontal photo) as one gallery of 39
    if paths[0] not in crops:
        print("\nno face found in the enrolment photo: nothing to enrol"); sys.exit(0)
    x0, b0 = crops[paths[0]]
    me = sectors(D.sample(x0, *b0)).astype(np.int16)
    gal = np.vstack([me[None], np.array([sectors(next(im["I"].reshape(48, 42) for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")))
                                          for p in Y.IDS], np.int16)])
    names = ["YOU (enrolled)"] + Y.IDS
    print("\ngallery: you (from %s) + the 38 Yale B people" % os.path.basename(paths[0]))
    for pth in paths[1:]:
        if pth not in crops:
            print("%-12s -> no face found: no answer" % os.path.basename(pth)); continue
        x, b = crops[pth]
        d = profiled(x, b, gal)
        o = np.argsort(d)
        print("%-12s -> best: %-15s distance %.3f   runner-up %-9s %.3f   your rank %d of %d"
              % (os.path.basename(pth), names[o[0]], d[o[0]], names[o[1]], d[o[1]], int(np.where(o == 0)[0][0]) + 1, len(names)))
    # a picture of what was found
    canvas = Image.new("RGB", (sum(p.width for p in pics.values()) + 10 * len(pics), 400), (255, 255, 255)); xo = 0
    for pth in paths:
        if pth not in pics:
            continue
        pic = pics[pth].copy(); dr = ImageDraw.Draw(pic); r, c, h, w = boxes[pth]
        dr.rectangle([c, r, c + w, r + h], outline=(255, 40, 40), width=3)
        canvas.paste(pic, (xo, 0)); xo += pic.width + 10
    out = os.environ.get("PREVIEW", "phone_found.png")
    canvas.save(out); print("\nboxes drawn in", out)
