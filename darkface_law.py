"""DARK FACE with the face laws only (no network, no learned black box).  Every number is counted or measured.
LAW 1  noise measured at every pyramid level from the image itself: the per-component gradient noise sigma from the
       Rayleigh median of |grad| (most of a night scene is noise only), floored at the 8-bit quantisation noise.
       A pixel is SIGNIFICANT iff |grad| >= K_SIG sigma  (noise-only rate exp(-K_SIG^2 / 2)).
LAW 2  local cancellation, two codes per pixel:
       direction  0 = insignificant, else 1 + gradient sector (the engine's proven sector lanes): any monotone tone
                  curve, gain and smooth light cancel exactly
       centre-surround  sign of (G1 - G2.5) x when significant against its own measured noise: gain and offset cancel
LAW 3  position and scale: an anti-aliased image pyramid (x2.5 up for 4 px faces, steps of 2^(1/3)); a face fills the
       central FACE x FACE of a T x T window -- the ring around it is context
LAW 4  descriptor: both codes at every template position (positions tied in CELL x CELL cells)
LAW 5  decision (v2, used): the Gaussian-Bayes log-likelihood ratio with a MEASURED shared covariance:
       score = w . phi + PRIOR[level],  w = Sigma_bg^-1 (mu_face - mu_bg),  phi = per-cell histograms of the non-zero codes;
       correlated pixels then cancel exactly instead of being counted again (the independent-pixel sum below over-counts).
       Every parameter is a counted mean or covariance; the runtime is the same integer table per position.
LAW 5  decision (v1, superseded on design-validation): score = sum over positions of LLR[pos, code] + PRIOR[level], 16 ln,
       LLR COUNTED on design images (faces from the labels; background = random windows + the law's own false alarms
       on design images: hard-negative counting), PRIOR = ln(faces / windows) at that level, counted.
       Peaks, then non-maximum suppression.
LAW 6  a window whose face region holds fewer than MINSIG significant pixels has no evidence: never reported."""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, maximum_filter
from scipy import fft as sfft
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yale_gate import sectors

T = 16                                   # window side at every level
FACE = 10                                # face side inside the window (design-validation choice: 10 or 16)
CELL = 2                                 # template positions tied in CELL x CELL cells (design-validation choice: 1 or 2)
PAD = T // 2                             # zero-evidence padding: faces at the image border are scored too
K_SIG = 2.5
ASPECT = 1.10                            # measured on the design labels: median h / w
NDIR, NDOG = 9, 3
UNIT = 16                                # integer tables: round(16 ln ratio)
NEG = -10 ** 9
Q_SIG = (1 / 12.92) / np.sqrt(12)        # quantisation noise of the darkest 8-bit sRGB codes, linear 0..255 units


def _energy(f):
    d = np.zeros((41, 41)); d[20, 20] = 1
    return float(np.sqrt((f(d) ** 2).sum()))


GE = _energy(lambda d: np.gradient(gaussian_filter(d, 0.75), axis=1))
DE = _energy(lambda d: gaussian_filter(d, 1.0) - gaussian_filter(d, 2.5))


def to_gray(rgb):
    """linear-light luminance (sRGB assumed: the JPEG/PNG path of Law 1)"""
    a = rgb.astype(float) / 255
    return 255 * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).mean(2)


def pyramid_scales(min_face=4, max_face=320):
    """scale s: the image is resized by 1/s; a face of FACE * s pixels fills the face region"""
    s = min_face / FACE
    out = []
    while FACE * s <= max_face:
        out.append(s); s *= 2 ** (1 / 3)
    return out


def resample(gray, s):
    if s > 1:
        return zoom(gaussian_filter(gray, 0.5 * np.sqrt(s * s - 1)), 1 / s, order=1)
    if s < 1:
        return zoom(gray, 1 / s, order=1)
    return gray


def codes(x, s):
    """LAW 1 + LAW 2 at one level -> padded direction-code map and centre-surround-code map"""
    G = gaussian_filter(x, 0.75)
    m = np.hypot(np.gradient(G, axis=1), np.gradient(G, axis=0))
    sc = max(float(np.median(m)) / 1.1774, Q_SIG * GE / max(s, 1.0))
    sec = sectors(x * (2.0 / sc)).reshape(x.shape)          # gain cancels exactly; the factor only keeps the lanes' fields fine
    cd = np.where(m >= K_SIG * sc, 1 + sec, 0).astype(np.int8)
    d = gaussian_filter(x, 1.0) - gaussian_filter(x, 2.5)
    sd = max(float(np.median(np.abs(d))) / 0.6745, Q_SIG * DE / max(s, 1.0))
    cg = np.where(d >= K_SIG * sd, 1, np.where(d <= -K_SIG * sd, 2, 0)).astype(np.int8)
    return np.pad(cd, PAD), np.pad(cg, PAD)


def level(gray, s):
    return codes(resample(gray, s), s)


def n_windows(H, W):
    return max(0, H - T + 1) * max(0, W - T + 1)


def window_of_face(box, s):
    """the window (padded row, col) that centres a face box at scale s -- the exact inverse of box_of_window"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return int(round(cy / s - T / 2 + 0.5)) + PAD, int(round(cx / s - T / 2 + 0.5)) + PAD


def box_of_window(r, c, s):
    cy, cx = (r - PAD + T / 2 - 0.5) * s, (c - PAD + T / 2 - 0.5) * s
    hw, hh = FACE * s / np.sqrt(ASPECT) / 2, FACE * s * np.sqrt(ASPECT) / 2
    return cx - hw, cy - hh, cx + hw, cy + hh


def level_of_face(box, scales):
    f = np.sqrt(max(box[2] - box[0], 1e-3) * max(box[3] - box[1], 1e-3))
    return int(np.argmin([abs(np.log(f / (FACE * s))) for s in scales]))


def expand(Lcell):
    """cell table (cells x codes) -> position table (T*T x codes)"""
    n = T // CELL
    idx = (np.arange(T)[:, None] // CELL) * n + np.arange(T)[None, :] // CELL
    return Lcell[idx.ravel()]


def window_features(cd, cg, r, c):
    """phi: per-cell counts of each non-zero code (8 direction + 2 centre-surround), int8"""
    ci = cell_of_pos(); n = (T // CELL) ** 2
    wd = cd[r:r + T, c:c + T].ravel().astype(np.int64); wg = cg[r:r + T, c:c + T].ravel().astype(np.int64)
    f = np.zeros((n, 10), np.int64)
    np.add.at(f, (ci[wd > 0], wd[wd > 0] - 1), 1)
    np.add.at(f, (ci[wg > 0], 8 + wg[wg > 0] - 1), 1)
    return f.ravel().astype(np.int8)


def tables_from_w(Wc):
    """cell weights (cells x 10, integer) -> position tables; code 0 carries no weight"""
    ci = cell_of_pos()
    Ld = np.zeros((T * T, NDIR), np.int64); Lg = np.zeros((T * T, NDOG), np.int64)
    Ld[:, 1:] = Wc[ci, :8]; Lg[:, 1:] = Wc[ci, 8:]
    return Ld, Lg


def cell_of_pos():
    n = T // CELL
    return ((np.arange(T)[:, None] // CELL) * n + np.arange(T)[None, :] // CELL).ravel()


def score_level(cd, cg, Ldir, Ldog):
    """integer score of every window = sum_p L[p, code].  Computed as one FFT correlation per code; the result is an
    integer sum, so rounding is exact (checked against the direct sum in the design stage)."""
    H, W = cd.shape
    if H < T or W < T:
        return None
    shape = (H, W)
    acc = None
    base = int(Ldir[:, 0].sum() + Ldog[:, 0].sum())         # every pixel has some code: code 0 folded into a constant
    for C, L, n in ((cd, Ldir, NDIR), (cg, Ldog, NDOG)):
        for c in range(1, n):
            k = (L[:, c] - L[:, 0]).reshape(T, T)
            if not k.any():
                continue
            K = np.zeros(shape); K[:T, :T] = k[::-1, ::-1]
            t = sfft.rfft2((C == c).astype(np.float64)) * sfft.rfft2(K)
            acc = t if acc is None else acc + t
    if acc is None:
        return np.full((H - T + 1, W - T + 1), base, np.int64)
    return np.rint(sfft.irfft2(acc, s=shape)[T - 1:, T - 1:]).astype(np.int64) + base


def score_level_f(cd, cg, Ldir, Ldog):
    """real-valued tables (design-stage experiments only): the same FFT correlation without the integer rounding"""
    H, W = cd.shape
    acc = None
    base = float(Ldir[:, 0].sum() + Ldog[:, 0].sum())
    for C, L, n in ((cd, Ldir, NDIR), (cg, Ldog, NDOG)):
        for c in range(1, n):
            K = np.zeros((H, W)); K[:T, :T] = (L[:, c] - L[:, 0]).reshape(T, T)[::-1, ::-1]
            t = sfft.rfft2((C == c).astype(np.float64)) * sfft.rfft2(K)
            acc = t if acc is None else acc + t
    return sfft.irfft2(acc, s=(H, W))[T - 1:, T - 1:] + base


def score_direct(cd, cg, Ldir, Ldog):
    H, W = cd.shape
    out = np.zeros((H - T + 1, W - T + 1), np.int64)
    for dy in range(T):
        for dx in range(T):
            p = dy * T + dx
            out += Ldir[p][cd[dy:dy + H - T + 1, dx:dx + W - T + 1]] + Ldog[p][cg[dy:dy + H - T + 1, dx:dx + W - T + 1]]
    return out


def face_sig(cd, cg):
    """significant pixels inside the central FACE x FACE of every window (LAW 6)"""
    s = ((cd > 0) | (cg > 0)).astype(np.int64)
    I = np.pad(s.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    o = (T - FACE) // 2
    H, W = s.shape
    h, w = H - T + 1, W - T + 1
    return I[o + FACE:o + FACE + h, o + FACE:o + FACE + w] - I[o:o + h, o + FACE:o + FACE + w] \
        - I[o + FACE:o + FACE + h, o:o + w] + I[o:o + h, o:o + w]


def raw_detect(gray, Ldir, Ldog, minsig, scales, per_level=300):
    """peaks of every level BEFORE the level prior and NMS: (score, x1, y1, x2, y2, level)"""
    out = []
    for k, s in enumerate(scales):
        cd, cg = level(gray, s)
        S = score_level(cd, cg, Ldir, Ldog)
        if S is None:
            continue
        S = np.where(face_sig(cd, cg) >= minsig, S, NEG)
        peak = (S == maximum_filter(S, size=3)) & (S > NEG)
        rr, cc = np.nonzero(peak)
        if len(rr) > per_level:
            j = np.argsort(-S[rr, cc], kind="stable")[:per_level]; rr, cc = rr[j], cc[j]
        for r, c in zip(rr, cc):
            out.append((int(S[r, c]),) + box_of_window(r, c, s) + (k,))
    return out


def finish(raw, prior, nms_thr=0.3):
    return nms([(d[0] + int(prior[d[5]]),) + tuple(d[1:5]) for d in raw], nms_thr)


def nms(dets, thr):
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []
    boxes = np.array([d[1:5] for d in dets]) if dets else np.zeros((0, 4))
    alive = np.ones(len(dets), bool)
    for i in range(len(dets)):
        if not alive[i]:
            continue
        keep.append(dets[i])
        if i + 1 < len(dets):
            alive[i + 1:] &= iou_many(boxes[i], boxes[i + 1:]) < thr
    return keep


def iou_many(b, B):
    ix = np.maximum(0, np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]))
    iy = np.maximum(0, np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]))
    inter = ix * iy
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-12)


def average_precision(all_dets, all_gt, iou_thr=0.5):
    """VOC all-point AP at IoU 0.5 over a set of images.  all_dets: {img: [(score, x1, y1, x2, y2)]}, all_gt: {img: N x 4}"""
    recs = []
    npos = sum(len(g) for g in all_gt.values())
    for k, dets in all_dets.items():
        for d in dets:
            recs.append((d[0], k, np.array(d[1:5], float)))
    recs.sort(key=lambda t: -t[0])
    used = {k: np.zeros(len(g), bool) for k, g in all_gt.items()}
    tp = np.zeros(len(recs)); fp = np.zeros(len(recs))
    for i, (s, k, b) in enumerate(recs):
        g = all_gt.get(k, np.zeros((0, 4)))
        if len(g) == 0:
            fp[i] = 1; continue
        o = iou_many(b, g); j = int(np.argmax(o))
        if o[j] >= iou_thr and not used[k][j]:
            tp[i] = 1; used[k][j] = True
        else:
            fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / max(npos, 1); prec = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum()), float(rec[-1] if len(rec) else 0.0)
