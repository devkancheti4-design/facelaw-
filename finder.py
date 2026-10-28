"""AUTOMATIC FACE AND EYE FINDER = classic + ours.
CLASSIC  OpenCV's Viola-Jones cascades (frontal face, then eyes inside the face's upper part): face box and eye boxes
OURS     inside each eye box, the eye law's pupil score (the synthesis's AGREE on the synthesis's sectors, significance against the
         box's own measured noise, + a dark centre) moves the eye-box centre to the pupil
    python3 -I finder.py yale      alignment error on Extended Yale B (known eye positions), clean and dark
    python3 -I finder.py phone     the user's two photos against the hand-read pupils"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/neooos/ext/pylib4"))   # OpenCV 4.x: the classic cascades
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import cv2
import eye_law as EL

HERE = os.path.dirname(os.path.abspath(__file__))
FACE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
EYE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml"))
IRIS = 0.13                                     # iris radius as a share of the classic eye box's width (measured on Yale:
                                                # 0.13 with the central 30% window beat 0.06 / 0.09 and the 60% window)


def to_u8(lin):
    """linear light -> display-encoded 8-bit, as the classic cascades were trained on"""
    a = np.clip(lin / max(np.percentile(lin, 99.5), 1e-9), 0, 1)
    return np.round(255 * np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)).astype(np.uint8)


def boxdown(lin, f):
    """LAW 3's pixel: integrate f x f blocks -- f^2 times the photons per pixel, the noise down f times"""
    if f == 1:
        return lin
    h, w = lin.shape[0] // f * f, lin.shape[1] // f * f
    return lin[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))


def find(lin, T=None, iris=None, win=None):
    """classic face + eyes on the box-integrated image (coarser until the cascade sees a face), then the synthesis's pupil
    rule inside each eye box (iris radius iris x box width, searched over the central win share of the box)
    -> dict(face, classic=(eL, eR), refined=(eL, eR), f) or None"""
    iris = IRIS if iris is None else iris; win = 0.3 if win is None else win
    got = None
    for f in (1, 2, 4, 8):                                  # the WHOLE chain (face + both eyes) at a scale, else coarser
        faces = FACE.detectMultiScale(to_u8(boxdown(lin, f)), scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        for fb in sorted(faces, key=lambda b: -b[2] * b[3])[:3]:
            fx, fy, fw, fh = [int(v * f) for v in fb]
            fe = max(f, int(round(fw / 160.0)), 1)          # eyes: the face about 160 px wide, never finer than the face pass
            sm = boxdown(lin, fe); u8 = to_u8(sm)
            X0, Y0, Wd, Ht = fx // fe, fy // fe, max(fw // fe, 1), max(fh // fe, 1)
            top = u8[Y0:Y0 + int(0.6 * Ht), X0:X0 + Wd]
            eyes = EYE.detectMultiScale(top, scaleFactor=1.05, minNeighbors=3, minSize=(max(6, Wd // 10), max(6, Wd // 10)), maxSize=(Wd // 2, Wd // 2))
            left = [e for e in eyes if e[0] + e[2] / 2 < Wd / 2]; right = [e for e in eyes if e[0] + e[2] / 2 >= Wd / 2]
            if left and right:
                got = (f, fx, fy, fw, fh, fe, sm, X0, Y0, left, right); break
        if got:
            break
    if got is None:
        return None
    f, fx, fy, fw, fh, fe, sm, X0, Y0, left, right = got
    pick = lambda es: max(es, key=lambda e: e[2] * e[3])
    out = dict(face=(fx, fy, fw, fh), classic=[], refined=[], f=(f, fe))
    for ex, ey, ew, eh in (pick(left), pick(right)):
        bx, by = X0 + ex, Y0 + ey                            # in the eye pass's pixels
        out["classic"].append(np.array([(bx + ew / 2) * fe, (by + eh / 2) * fe], float))
        if T is not None:
            R = sm[by:by + eh, bx:bx + ew].astype(float)
            S = EL.pupil_score(R, max(1.0, iris * ew), T)
            lo, hi = (1 - win) / 2, (1 + win) / 2
            m = np.full(S.shape, -np.inf)
            m[int(lo * eh):int(np.ceil(hi * eh)), int(lo * ew):int(np.ceil(hi * ew))] = S[int(lo * eh):int(np.ceil(hi * eh)), int(lo * ew):int(np.ceil(hi * ew))]
            yy, xx = np.unravel_index(np.argmax(m), m.shape)
            out["refined"].append(np.array([(bx + xx + 0.5) * fe, (by + yy + 0.5) * fe], float))
    return out


def err(found, truth):
    tl, tr = truth; ed = np.linalg.norm(tr - tl)
    return max(np.linalg.norm(found[0] - tl), np.linalg.norm(found[1] - tr)) / ed


if __name__ == "__main__":
    stage = sys.argv[1]
    T = EL.load_agree()
    if stage == "yale":
        import yale as Y, unlock_law as U
        B = 48                                                    # border so the cascade sees the face with context
        tl, tr = np.array([33.5 + B, 45.5 + B]), np.array([133.5 + B, 45.5 + B])
        rng = np.random.default_rng(3)
        GRID = [(0.13, 0.3)]
        for N in (None, 100, 20):
            ec, er, miss, n = [], {g: [] for g in GRID}, 0, 0
            for p in Y.IDS[: int(os.environ.get("NP", len(Y.IDS)))]:
                for im in Y.PEOPLE[p]:
                    if im["I"].sum() <= 0 or im["theta"] > 25.5:
                        continue
                    I = Y.read_pgm(os.path.join(Y.ROOT, p, im["name"])).astype(float)
                    lin = 255 * (I / 255) ** 2.2
                    if N is not None:
                        lin = np.maximum(U.dark_ae(lin, N / 16.0, rng), 0)
                    lin = np.pad(lin, B, mode="edge")
                    n += 1
                    f = find(lin)
                    if f is None:
                        miss += 1; continue
                    ec.append(err(f["classic"], (tl, tr)))
                    for g in GRID:
                        er[g].append(err(find(lin, T, *g)["refined"], (tl, tr)))
            tag = "clean" if N is None else "dark, %d photons per face pixel" % N
            print("Yale B (light <= 25 deg), %s: %d faces, classic found face + both eyes in %.1f%%" % (tag, n, 100 * (1 - miss / n)))
            rows = [("classic eye boxes", np.array(ec))] + [("+ synthesis pupil, iris %.2f, window %.1f" % g, np.array(er[g])) for g in GRID]
            for nm, e in rows:
                if len(e):
                    print("    %-34s median %.3f; within 0.05 %5.1f%%, 0.10 %5.1f%%, 0.25 %5.1f%%"
                          % (nm, np.median(e), 100 * (e <= .05).mean(), 100 * (e <= .10).mean(), 100 * (e <= .25).mean()), flush=True)
    elif stage == "phone":
        import phone_eyes as PE
        truth = {"photo1": ((572, 1016), (828, 1015)), "photo2": ((622, 960), (812, 955))}
        for nm in ("photo1", "photo2"):
            path = os.path.join(HERE, "data", "phone_user", nm + ".jpg")
            lin = PE.load(path)
            f = find(lin, T, float(os.environ.get("IRIS", IRIS)), float(os.environ.get("WIN", 0.6)))
            t = (np.array(truth[nm][0], float), np.array(truth[nm][1], float))
            if f is None:
                print("%s: classic detector found no face or not both eyes" % nm); continue
            print("%s: face box %s; classic eyes (%d, %d) (%d, %d), error %.3f; + synthesis pupil (%d, %d) (%d, %d), error %.3f (hand-read (%d, %d) (%d, %d))" % (
                nm, tuple(int(v) for v in f["face"]), *f["classic"][0], *f["classic"][1], err(f["classic"], t),
                *f["refined"][0], *f["refined"][1], err(f["refined"], t), *t[0], *t[1]))
