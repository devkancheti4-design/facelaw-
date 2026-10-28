"""v2 EYE LAW (counted, as the face law's F is): the pupil = the centre whose window of the law's codes is most likely
an eye.
FRAME   the eye region is resampled so the eye distance is ED = 50 px (from the face box at run time: ED ~ 0.45 of its
        width; from the true eyes, with +-10% scale and +-8 deg tilt jitter, when counting)
CODES   per pixel: the synthesis's direction sector where the gradient is significant against the noise measured from the
        frame (2 levels), else flat (17 codes) x the Weber band of the smoothed image (dark / even / bright: 3, the synthesis's WBAND lanes)  -> 51
LAW     T[j, code] = round(16 ln P(code at offset j | pupil at the centre) / P(code | anywhere around the eye)), counted
        on Extended Yale B people the test never sees (every light, both eyes: the right eye mirrored onto the left)
SCORE   S(c) = sum over the 25 x 25 window of T[j, code(c + j)]; the pupil = the best centre in the eye's search box,
        then the pair rule (level, one eye distance apart)
    python3 -I eyelaw2.py count A        count T on Yale half A -> eyelaw2_T_A.npy
    python3 -I eyelaw2.py yale B A       test on half B with T from half A (and the owner's images)"""
import json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import eye_law as EL
import hybrid as H
import v2law as V2

HERE = os.path.dirname(os.path.abspath(__file__))
ED, RW = 50.0, 12                                 # normalised eye distance; window radius (25 x 25)
NDIR, NWB = 17, 3
NC = NDIR * NWB
SG, WSG = 1.0, 1.5                                # gradient scale; Weber scale (band edge |L/G| > 1/8: v2law)
BOXW = 0.45                                       # eye distance / face box width (measured: Yale 0.45, owner 0.41-0.46)


def frame(img, centre, scale, angle, shape):
    """sample img on a grid of `shape` around `centre` (image px), `scale` image px per frame px, rotated by angle"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    u, v = xx - (w - 1) / 2, yy - (h - 1) / 2
    c, s = np.cos(angle), np.sin(angle)
    X = centre[0] + scale * (c * u - s * v); Y = centre[1] + scale * (s * u + c * v)
    G = gaussian_filter(img, max(0.0, 0.5 * (scale - 1)))
    return map_coordinates(G, [Y, X], order=1, mode="nearest")


def codes(F):
    """the code map of a normalised frame (display-encoded brightness)"""
    G = gaussian_filter(F, SG)
    gx, gy = np.gradient(G, axis=1), -np.gradient(G, axis=0)
    m = np.hypot(gx, gy)
    sg = EL.noise_sigma(F) * np.sqrt(2 * EL.kernel_energy(SG))
    lev = np.where(m >= 4 * sg, 2, np.where(m >= 2 * sg, 1, 0))
    sec = EL.sector_of(gx, gy).reshape(F.shape)
    d = np.where(lev == 0, 0, 1 + sec + 8 * (lev - 1))
    Gw = gaussian_filter(F, WSG); L = H.lap8(Gw, 2)
    wb = V2.wband(V2.pack(4 * L, 4 * np.maximum(Gw, 1e-3 * max(F.mean(), 1e-9))))   # the synthesis's band lanes: |L/G| > 1/8
    return d * NWB + wb


def score_map(C, T):
    """S(c) = sum over the window of T[j, C(c + j)] at every centre c (correlation per code)"""
    S = np.zeros(C.shape)
    for k in range(NC):
        m = (C == k)
        if m.any():
            S += fftconvolve(m.astype(float), T[::-1, ::-1, k], mode="same")
    return S


def subpix(S, y, x):
    def off(a, b, c):
        den = a - 2 * b + c
        return 0.0 if den >= 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    h, w = S.shape
    return (x + (off(S[y, x - 1], S[y, x], S[y, x + 1]) if 0 < x < w - 1 else 0.0),
            y + (off(S[y - 1, x], S[y, x], S[y + 1, x]) if 0 < y < h - 1 else 0.0))


def eye_frame(img, eye, ed, angle=0.0, mirror=False):
    """a 2.2 ED x 1.6 ED frame centred on `eye` at ED = 50 px; mirrored for a right eye"""
    shape = (int(1.6 * ED), int(2.2 * ED))
    F = frame(img, eye, ed / ED, angle, shape)
    return F[:, ::-1] if mirror else F


def count(items, rng):
    """items: (display image, (left eye, right eye)) -> T"""
    pos = np.ones((2 * RW + 1, 2 * RW + 1, NC)); bg = np.ones(NC)
    for img, (eL, eR) in items:
        ed0 = np.linalg.norm(eR - eL); ang0 = np.arctan2(eR[1] - eL[1], eR[0] - eL[0])
        for eye, mir in ((eL, False), (eR, True)):
            js = np.exp(rng.uniform(np.log(0.9), np.log(1.1))); ja = ang0 + np.radians(rng.uniform(-8, 8))
            sub = rng.uniform(-0.5, 0.5, 2) * ed0 / ED
            F = eye_frame(img, eye + sub, ed0 * js, ja if not mir else ja, mir)
            C = codes(F)
            cy, cx = (F.shape[0] - 1) // 2, (F.shape[1] - 1) // 2
            win = C[cy - RW:cy + RW + 1, cx - RW:cx + RW + 1]
            np.add.at(pos, (np.arange(2 * RW + 1)[:, None], np.arange(2 * RW + 1)[None, :], win), 1)
            bg += np.bincount(C[RW:-RW, RW:-RW].ravel(), minlength=NC)
    P = pos / pos.sum(2, keepdims=True); B = bg / bg.sum()
    return np.clip(np.round(16 * np.log(P / B[None, None])), -127, 127).astype(np.int32)


def peaks_in(S, ok, n, sep):
    from scipy.ndimage import maximum_filter
    S2 = np.where(ok, S, -np.inf)
    pk = (S2 == maximum_filter(S2, size=sep)) & ok
    ys, xs = np.nonzero(pk)
    o = np.argsort(-S2[ys, xs])[:n]
    return [(int(ys[k]), int(xs[k])) for k in o]


def locate(img, guess, ed, T, angle=0.0, mirror=False, reach=0.30, n=1):
    """the best n centres within reach x ED of `guess` -> [(image point, score)]"""
    F = eye_frame(img, guess, ed, angle, mirror)
    S = score_map(codes(F), T)
    cy, cx = (F.shape[0] - 1) / 2, (F.shape[1] - 1) / 2
    yy, xx = np.mgrid[0:F.shape[0], 0:F.shape[1]]
    ok = (np.abs(xx - cx) <= reach * ED) & (np.abs(yy - cy) <= reach * ED)
    out = []
    c, s = np.cos(angle), np.sin(angle); k = ed / ED
    for y, x in peaks_in(S, ok, n, int(0.1 * ED) | 1):
        fx, fy = subpix(S, y, x)
        u, v = (fx - cx) * (-1 if mirror else 1), fy - cy
        out.append((guess + k * np.array([c * u - s * v, s * u + c * v]), float(S[y, x])))
    return out if n > 1 else out[0]


PAIR = dict(N=5, LD=3000.0, LT=3000.0)          # candidates per eye; penalty per unit |ln(d / box ED)| and per unit tilt


def find(img, box, T):
    """face box -> (left pupil, right pupil): the best few centres per eye, the pair rule (eye distance near the box's,
    level), then each eye once more around the pair's own geometry"""
    x, y, w, h = box
    ed = BOXW * w
    gL = np.array([x + w / 2 - ed / 2, y + 0.36 * w]); gR = np.array([x + w / 2 + ed / 2, y + 0.36 * w])
    cL = locate(img, gL, ed, T, 0.0, False, 0.35, PAIR["N"]); cR = locate(img, gR, ed, T, 0.0, True, 0.35, PAIR["N"])
    best, eL, eR = -np.inf, cL[0][0], cR[0][0]
    for pL, sL in cL:
        for pR, sR in cR:
            v = pR - pL; d = np.hypot(*v)
            if d <= 0.5 * ed:
                continue
            sc = sL + sR - PAIR["LD"] * abs(np.log(d / ed)) - PAIR["LT"] * abs(v[1]) / d
            if sc > best:
                best, eL, eR = sc, pL, pR
    for _ in range(2):
        ed2 = float(np.clip(np.linalg.norm(eR - eL), 0.8 * ed, 1.25 * ed)); ang = np.arctan2(eR[1] - eL[1], eR[0] - eL[0])
        eL, _ = locate(img, eL, ed2, T, ang, False, 0.10); eR, _ = locate(img, eR, ed2, T, ang, True, 0.10)
    return eL, eR


def yale_items(people, maxth=180.0):
    import yale as Y
    TR = json.load(open(os.path.join(HERE, "yale_eyes_truth.json")))["eyes"]
    out = []
    for p in people:
        if p not in TR:
            continue
        tl, tr = np.array(TR[p][0], float), np.array(TR[p][1], float)
        for im in Y.PEOPLE[p]:
            if im["I"].sum() <= 0 or im["theta"] > maxth:
                continue
            I = Y.read_pgm(os.path.join(Y.ROOT, p, im["name"])).astype(float)
            out.append((p, im["theta"], 255.0 * (I / 255.0), (tl, tr)))
    return out


if __name__ == "__main__":
    import yale as Y
    import eyes2 as E2
    stage = sys.argv[1]
    halves = {"A": Y.IDS[0::2], "B": Y.IDS[1::2]}
    if stage == "count":
        h = sys.argv[2]
        rng = np.random.default_rng(2610)
        items = [(img, t) for _, _, img, t in yale_items(halves[h])]
        T = count(items, rng)
        np.save(os.path.join(HERE, "eyelaw2_T_%s.npy" % h), T)
        print("T counted on Yale half %s: %d faces x 2 eyes; range %d..%d" % (h, len(items), T.min(), T.max()))
    elif stage == "yale":
        h, th = sys.argv[2], sys.argv[3]
        T = np.load(os.path.join(HERE, "eyelaw2_T_%s.npy" % th))
        for maxth in (25.5, 77.0):
            e_ideal, e_box, n = [], [], 0
            for p, theta, img, (tl, tr) in yale_items(halves[h], maxth):
                n += 1
                ed = np.linalg.norm(tr - tl); w = ed / BOXW
                rng = np.random.default_rng(n)
                box = np.array([(tl[0] + tr[0]) / 2 - w / 2, (tl[1] + tr[1]) / 2 - 0.36 * w, w, w])
                box = box + np.array([*(rng.normal(0, 0.05, 2) * w), 0, 0]); box[2:] *= np.exp(rng.normal(0, 0.08))
                e_box.append(E2.err(find(img, box, T), (tl, tr)))
            print("Yale half %s (T from half %s), light <= %.0f deg, %d faces, face box off by ~5%% position / 8%% size:  %s" % (
                h, th, maxth, n, E2.summary(e_box)), flush=True)
        es = []
        for nm, lin, t in E2.user_items():
            box = E2.face_box(lin)
            if box is None:
                print("   %-24s no face box" % nm); es.append(np.inf); continue
            e = E2.err(find(E2.u8(lin), box, T), t); es.append(e)
            print("   %-24s eye distance %4.0f px  error %.3f" % (nm, np.linalg.norm(t[1] - t[0]), e))
        print("owner's images: %s" % E2.summary(es))
