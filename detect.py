"""FACE FINDING BY LAW (no learned detector): a face is where the face light subspace explains the pixels and a smooth
background subspace of the same dimension does not.
FACE MODEL  the light subspace of enrolled faces: the 9 harmonic images rho Y_j(n) of each design person (photometric
            stereo, Law 2), stacked, orthonormal top-K directions                       (48 x 42 window)
BACKGROUND  the K lowest 2-D cosines of the same window (smooth walls, gradients, vignetting)
SCORE       s = ln( r_bg^2 / r_face^2 ),  r^2 = |y|^2 - |P y|^2 per window: light and exposure scale both remainders
            alike and cancel in the ratio; a flat wall is explained by both (s ~ 0); only face structure scores high.
            All windows at once by FFT correlation; scales by a pyramid; peaks by non-maximum suppression.
ALIGN       the detected window refined on a finer grid of shifts and scales, then resampled to the canonical 48 x 42."""
import os, sys
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import fftconvolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WH, WW = 48, 42
K = 12


def orth(M, k):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :k]


def face_basis(bases, k=K):
    """bases: list of (P x 9) harmonic bases of design people -> orthonormal P x k face light subspace"""
    return orth(np.hstack(bases), k)


def bg_basis(k=K):
    r, c = np.mgrid[0:WH, 0:WW]
    cols = []
    for s in range(0, 12):
        for u in range(s + 1):
            v = s - u
            cols.append((np.cos(np.pi * (r + 0.5) * u / WH) * np.cos(np.pi * (c + 0.5) * v / WW)).ravel())
            if len(cols) == k:
                return orth(np.array(cols).T, k)


def window_energy(img):
    ii = np.pad(np.cumsum(np.cumsum(img ** 2, 0), 1), ((1, 0), (1, 0)))
    return ii[WH:, WW:] - ii[:-WH, WW:] - ii[WH:, :-WW] + ii[:-WH, :-WW]


def proj_energy(img, U):
    """|U^T y|^2 for every window position (U orthonormal): one correlation per column"""
    e = 0
    for j in range(U.shape[1]):
        t = U[:, j].reshape(WH, WW)[::-1, ::-1]
        e = e + fftconvolve(img, t, mode="valid") ** 2
    return e


NOISE_FLOOR = 0.0                    # P sigma^2: the remainder no model can remove (set from the measured noise)


def score_map(img, Uf, Ub):
    yy = window_energy(img)
    rf = np.maximum(yy - proj_energy(img, Uf), 1e-9 * (yy + 1e-12)) + NOISE_FLOOR
    rb = np.maximum(yy - proj_energy(img, Ub), 1e-9 * (yy + 1e-12)) + NOISE_FLOOR
    return np.log(rb / rf)


SCALES = tuple(2 ** (k / 4) for k in range(-2, 7))      # face heights 34 .. 136 px in the frame


def detect(img, Uf, Ub, tau, scales=SCALES, nms=0.3):
    """-> list of (score, row, col, height, width) in frame pixels, best first"""
    cands = []
    for s in scales:
        im = zoom(img, 1 / s, order=1) if s != 1 else img
        if im.shape[0] < WH or im.shape[1] < WW:
            continue
        S = score_map(im, Uf, Ub)
        for _ in range(8):                                 # the strongest peaks at this scale
            i = np.unravel_index(np.argmax(S), S.shape)
            v = S[i]
            if v < tau:
                break
            cands.append((float(v), i[0] * s, i[1] * s, WH * s, WW * s))
            r0, c0 = max(0, i[0] - WH // 2), max(0, i[1] - WW // 2)
            S[r0:i[0] + WH // 2, c0:i[1] + WW // 2] = -np.inf
    cands.sort(reverse=True)
    keep = []
    for c in cands:
        if all(iou(c, k) < nms for k in keep):
            keep.append(c)
    return keep


def iou(a, b):
    _, r1, c1, h1, w1 = a; _, r2, c2, h2, w2 = b
    ih = max(0, min(r1 + h1, r2 + h2) - max(r1, r2)); iw = max(0, min(c1 + w1, c2 + w2) - max(c1, c2))
    inter = ih * iw
    return inter / (h1 * w1 + h2 * w2 - inter)


def align(img, det, Uf, Ub):
    """refine position and scale on a finer grid; return the canonical 48 x 42 crop (resampled) and the box"""
    _, r, c, h, w = det
    best = None
    for ds in (0.92, 0.96, 1.0, 1.04, 1.08):
        hh, ww = h * ds, w * ds
        for dr in np.linspace(-0.08, 0.08, 5) * hh:
            for dc in np.linspace(-0.08, 0.08, 5) * ww:
                rr, cc = r + dr - (hh - h) / 2, c + dc - (ww - w) / 2
                crop = sample(img, rr, cc, hh, ww)
                if crop is None:
                    continue
                y = crop.ravel()
                yy = float(y @ y)
                sf = yy - float(((Uf.T @ y) ** 2).sum()); sb = yy - float(((Ub.T @ y) ** 2).sum())
                sc = np.log(max(sb, 1e-12) / max(sf, 1e-12))
                if best is None or sc > best[0]:
                    best = (sc, crop, (rr, cc, hh, ww))
    return best


def sample(img, r, c, h, w):
    """bilinear resample of the box (r, c, h, w) to 48 x 42"""
    if r < 0 or c < 0 or r + h > img.shape[0] - 1 or c + w > img.shape[1] - 1:
        return None
    rr = r + (np.arange(WH) + 0.5) * h / WH - 0.5; cc = c + (np.arange(WW) + 0.5) * w / WW - 0.5
    R, C = np.meshgrid(rr, cc, indexing="ij")
    r0, c0 = np.floor(R).astype(int), np.floor(C).astype(int)
    fr, fc = R - r0, C - c0
    r0, c0 = np.clip(r0, 0, img.shape[0] - 2), np.clip(c0, 0, img.shape[1] - 2)
    return ((1 - fr) * (1 - fc) * img[r0, c0] + (1 - fr) * fc * img[r0, c0 + 1] + fr * (1 - fc) * img[r0 + 1, c0]
            + fr * fc * img[r0 + 1, c0 + 1])
