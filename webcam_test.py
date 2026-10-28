"""The webcam darkness test -- laws only, nothing tuned on the room.  For every frame:
  * noise sigma measured from the frame itself (Immerkaer); the light-law score gets the floor P sigma^2
  * TOO DARK TO TELL iff even the brightest face-sized window carries less signal energy than its noise energy
    (yy - P sigma^2 < P sigma^2): no model can then tell a face from the background
  * otherwise every window that passes BOTH laws (light-law score >= 0 against the background model, local-law
    distance on significant gradients <= the DESIGN threshold 1.6), after non-maximum suppression, is a face
    python3 -I webcam_test.py FRAME_DIR      -> webcam_dets.json and contact sheets in FRAME_DIR/sheets"""
import glob, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect as D
import yale as Y
import detect_test as T
import phone_test as P
from yale_gate import sectors

H = 360


def load(path):
    im = Image.open(path).convert("RGB")
    w = int(round(im.width * H / im.height))
    a = np.asarray(im.resize((w, H), Image.BILINEAR), float) / 255
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return 255 * lin.mean(2), np.asarray(im.resize((w, H)), float)


def frame_law(x, UF, UB):
    sigma = P.noise_sigma(x)
    Pn = 48 * 42 * sigma ** 2
    D.NOISE_FLOOR = Pn
    best_sig = 0.0
    dets = []
    for s in tuple(2 ** (k / 4) for k in range(2, 12)):
        im = zoom(x, 1 / s, order=1)
        if im.shape[0] < D.WH or im.shape[1] < D.WW:
            continue
        yy = D.window_energy(im)
        best_sig = max(best_sig, float(((yy - Pn) / max(Pn, 1e-12)).max()))
        S = D.score_map(im, UF, UB)
        for _ in range(6):
            i = np.unravel_index(np.argmax(S), S.shape)
            if not np.isfinite(S[i]) or S[i] < 0:
                break
            d = (float(S[i]), i[0] * s, i[1] * s, D.WH * s, D.WW * s)
            S[max(0, i[0] - D.WH // 2):i[0] + D.WH // 2, max(0, i[1] - D.WW // 2):i[1] + D.WW // 2] = -np.inf
            v = P.verify_sig(x, d, sigma)
            if v is not None and v <= P.VERIFY_MAX:
                dets.append(d + (v,))
    dets.sort(reverse=True)
    keep = []
    for d in dets:
        if all(D.iou(d[:5], k[:5]) < 0.3 for k in keep):
            keep.append(d)
    too_dark = best_sig < 1.0
    return dict(sigma=sigma, snr=best_sig, too_dark=too_dark, faces=[] if too_dark else
                [dict(score=d[0], row=d[1], col=d[2], h=d[3], w=d[4], local=d[5]) for d in keep])


def sheets(frames, out, res, boost):
    os.makedirs(out, exist_ok=True)
    per = 20
    for k in range(0, len(frames), per):
        tiles = []
        for f in frames[k:k + per]:
            x, rgb = load(f)
            a = rgb
            if boost:
                a = np.clip(a * (200 / max(np.percentile(a, 99.5), 1)), 0, 255)
            im = Image.fromarray(a.astype(np.uint8)).resize((320, 180)); dr = ImageDraw.Draw(im)
            sx = 320 / x.shape[1]; sy = 180 / x.shape[0]
            if boost:
                for t in range(1, 10):
                    dr.line([(32 * t, 0), (32 * t, 180)], fill=(0, 120, 120)); dr.line([(0, 18 * t), (320, 18 * t)], fill=(0, 120, 120))
            else:
                r = res[os.path.basename(f)]
                for fc in r["faces"]:
                    dr.rectangle([fc["col"] * sx, fc["row"] * sy, (fc["col"] + fc["w"]) * sx, (fc["row"] + fc["h"]) * sy], outline=(255, 40, 40), width=2)
                if r["too_dark"]:
                    dr.text((4, 4), "TOO DARK", fill=(255, 255, 0))
            dr.text((4, 166), os.path.basename(f)[6:10], fill=(255, 255, 255))
            tiles.append(im)
        sheet = Image.new("RGB", (5 * 324, 4 * 184), "black")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 5) * 324, (i // 5) * 184))
        sheet.save(os.path.join(out, "%s_%03d.png" % ("boost" if boost else "dets", k)))


if __name__ == "__main__":
    fd = sys.argv[1]
    frames = sorted(glob.glob(os.path.join(fd, "frame_*.png")))
    UF = D.face_basis([T.harmonic(p) for p in Y.IDS]); UB = D.bg_basis()
    P.SG = T.SG = np.array([sectors(im["I"].reshape(48, 42)) for p in Y.IDS for im in Y.PEOPLE[p] if Y.subset(im["theta"]) <= 4], np.int16)
    res = {}
    for f in frames:
        x, rgb = load(f)
        r = frame_law(x, UF, UB)
        r["brightness"] = float(rgb.mean())
        res[os.path.basename(f)] = r
    json.dump(res, open(os.path.join(fd, "webcam_dets.json"), "w"), indent=1)
    sheets(frames, os.path.join(fd, "sheets"), res, boost=True)
    sheets(frames, os.path.join(fd, "sheets"), res, boost=False)
    print("%d frames analysed; too dark: %d; frames with >= 1 face: %d" % (
        len(frames), sum(r["too_dark"] for r in res.values()), sum(bool(r["faces"]) for r in res.values())))
