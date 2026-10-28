"""FACE LAW v2 -- the reference law (Python) for the NIST FRTE 1:1 submission and the face lock.  Nothing here is tapped.
IMAGE     8-bit sRGB (or grey) -> linear light, mean of the channels
FACE      OpenCV's Viola-Jones frontal-face cascade at the finest scale where it fires (finder.py), the largest box
EYES      the v2 eye law (eyelaw2.py): T[offset, code] counted over the law's codes in a 25 x 25 window around the pupil,
          eye distance 50 px; the best few centres per eye; the pair rule; refined twice around the pair's own geometry
CROP      a similarity onto the canonical eyes (8, 11) and (33, 11) of the 48 x 42 frame (phone_eyes.crop)
CODES     WEBER at 3 scales (sigma 0.75 / 1.5 / 3, strides 1 / 2 / 4; 2,652 points): L = 8 G(c) - the 8 neighbours of G at
          the scale's stride; the bin of atan(ALPHA_k L / G) in 8 equal angles -- an INTEGER law: a = rint(ALPHA_k L),
          g = rint(G) on the crop exposed to the lab mean REF, x = (a + 2048) | g << 12, bin = WBIN(x), the synthesis's
          lanes (|ALPHA L / G| against 1/2, 1, 2: shifts and compares; atan 26.6, 45, 63.4 deg); the probe's significance of L
          against the noise measured from the frame (Law 1), 4 levels
TEMPLATE  enrolment = the map's bins (2,652 x 3 bits); verification = the probe's bins and significance at the 9
          placements within +-1 px
SCORE     the best placement's mean FW[map bin | probe bin << 3 | significance << 6 | scale << 8]   (higher = same face)
Program synthesis writes the lanes (WBIN, WBAND, FW, T); the law used is rebuilt from the lanes only."""
import json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import hybrid as H

HERE = os.path.dirname(os.path.abspath(__file__))
REF = 107.7726                                      # the lab frontal mean: every crop is exposed to it before the codes


def wbin_claw(x):
    """the integer Weber bin (the lanes' rows): a = (x & 4095) - 2048, g = x >> 12 >= 1"""
    a = (x & 4095) - 2048; g = x >> 12; m = np.abs(a)
    k = (2 * m > g).astype(np.int64) + (m > g) + (m > 2 * g)
    return np.where(a >= 0, 4 + k, 3 - k)


def wband_claw(x):
    a = (x & 4095) - 2048; g = x >> 12
    e = 8 * np.abs(a) > g                            # the eye law's Weber band: |L/G| > 1/8
    return np.where(e, np.where(a >= 0, 2, 0), 1)


LANES = {}


def load_lanes(path=os.path.join(HERE, "lanes_v2.json")):
    if os.path.exists(path) and not LANES:
        LANES.update(json.load(open(path)))


def ev(expr, x):
    return eval(expr, {"x": x})


def wbin(x):
    if "WBIN" in LANES:
        L = LANES["WBIN"]
        s, m, g = ev(L["W_s"], x), ev(L["W_m"], x), x >> 12
        y = m | (g << 12)
        k = ev(L["W_c1"], y) + ev(L["W_c2"], y) + ev(L["W_c2"], ((m + 1) >> 1) | (g << 12))   # [m > 2g] = [(m+1)>>1 > g]
        return ev(L["W_join"], s | (k << 1))
    return wbin_claw(x)


def wband(x):
    if "WBAND" in LANES:
        L = LANES["WBAND"]
        s, m, g = ev(L["W_s"], x), ev(L["W_m"], x), x >> 12
        e = ev(L["W_e"], m | (g << 12)) if "W_e" in L else ev(L["W_c2"], m | ((g >> 3) << 12))       # [8m > g] = [m > g>>3]
        return ev(L["B_join"], s | (e << 1))
    return wband_claw(x)


def pack(a, g):
    return (np.clip(np.rint(a), -2048, 2047).astype(np.int64) + 2048) | (np.clip(np.rint(g), 1, 4095).astype(np.int64) << 12)


def weber_x(xe, k, V=None):
    """the packed (a, g) of scale k on an exposed crop; V (measured noise) only floors G for the probe"""
    G = gaussian_filter(xe, DL.SCALES[k])
    L = H.lap8(G, DL.STRIDE[k])
    fl = 1e-3 * REF + (np.sqrt(V.reshape(xe.shape) * H.E_G[k]) if V is not None else 0.0)
    return L, pack(H.ALPHA[k] * L, np.maximum(G, fl))


def expose(x):
    return x * (REF / max(float(np.abs(x).mean()), 1e-9))


def map_bins(crop):
    xe = expose(crop)
    out = []
    for k in range(len(DL.SCALES)):
        ix = DL.grid_idx(*xe.shape)[k]
        out.append(wbin(weber_x(xe, k)[1].ravel()[ix]))
    return np.concatenate(out).astype(np.int64)


def probe_codes(x):
    """bins and significance of one placement"""
    xe = expose(x)
    V = U.measured_var(xe)
    B, C = [], []
    for k in range(len(DL.SCALES)):
        ix = DL.grid_idx(*xe.shape)[k]
        L, X = weber_x(xe, k, V)
        B.append(wbin(X.ravel()[ix]))
        C.append(np.minimum(H.NWC - 1, np.floor(np.abs(L.ravel()[ix]) / np.sqrt(H.E_L[k] * V[ix]))))
    return np.concatenate(B).astype(np.int64), np.concatenate(C).astype(np.int64)


def probe_all(x):
    """the verification template: (bins, significance) at the 9 placements within +-1 px"""
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out.append(probe_codes(nd_shift(x, (dy, dx), order=0, mode="nearest") if (dy or dx) else x))
    return out


def score(FW, probes, gb):
    """the best placement's mean FW over the codes, against one or more maps gb (maps x points)"""
    gb = np.atleast_2d(gb)
    return np.max([FW[H.wcodes(pb, pc, gb)].sum(1) / len(pb) for pb, pc in probes], 0)
