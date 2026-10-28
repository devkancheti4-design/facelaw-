"""v2 HYBRID descriptor (design on Extended Yale B only): the v1 law's direction codes PLUS Weber (brightness-ratio) codes.
WEBER  at each of the law's three scales (sigma 0.75 / 1.5 / 3, strides 1 / 2 / 4): L = 8 G(c) - the 8 neighbours of G at
       the scale's own stride, over G(c) -- a RATIO, so the light's gain cancels where the light is smooth (Weberfaces'
       idea, Wang et al. 2011) -- quantised to 8 bins of atan(ALPHA_k L / G); the probe's SIGNIFICANCE of L against the
       noise measured from the frame itself (Law 1, unlock_law.measured_var), 4 levels
CODE   map bin | probe bin << 3 | significance << 6 | scale << 8   (768 codes); FW = round(16 ln P(code|same)/P(code|other))
SCORE  hybrid = S_dir + W * S_weber at each of the 9 placements (+-1 px), the best placement"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_law as DL

NWB, NWC = 8, 4                                   # Weber bins, significance levels
NFW = NWB * NWB * NWC * len(DL.SCALES)            # 768
OFFS = DL.STRIDE                                  # the neighbour offset at each scale = that scale's grid stride
EDGES = np.linspace(-np.pi / 2, np.pi / 2, NWB + 1)[1:-1]


def lap8(G, d):
    P = np.pad(G, d, mode="edge"); h, w = G.shape
    s = sum(P[d + a:d + a + h, d + b:d + b + w] for a in (-d, 0, d) for b in (-d, 0, d) if a or b)
    return 8 * G - s


def _energies():
    z = np.zeros((41, 41)); z[20, 20] = 1
    EL = [float((lap8(gaussian_filter(z, s), d) ** 2).sum()) for s, d in zip(DL.SCALES, OFFS)]
    EG = [float((gaussian_filter(z, s) ** 2).sum()) for s in DL.SCALES]
    return EL, EG


E_L, E_G = _energies()
ALPHA = None                                      # per scale: 1 / median |L / G| over the clean Yale front maps (set once)


def ratio(x, k, V=None):
    G = gaussian_filter(x, DL.SCALES[k])
    L = lap8(G, OFFS[k])
    fl = 1e-3 * max(float(np.abs(x).mean()), 1e-9) + (np.sqrt(V.reshape(x.shape) * E_G[k]) if V is not None else 0.0)
    return L, L / np.maximum(G, fl)


def set_alpha(maps):
    global ALPHA
    ALPHA = [1.0 / float(np.median(np.concatenate([np.abs(ratio(m, k)[1]).ravel() for m in maps]))) for k in range(len(DL.SCALES))]
    return ALPHA


def gallery_weber(I):
    out = []
    for k in range(len(DL.SCALES)):
        ix = DL.grid_idx(*I.shape)[k]
        out.append(np.digitize(np.arctan(ALPHA[k] * ratio(I, k)[1].ravel()[ix]), EDGES))
    return np.concatenate(out).astype(np.int64)


def probe_weber(x, V):
    B, C = [], []
    for k in range(len(DL.SCALES)):
        ix = DL.grid_idx(*x.shape)[k]
        L, r = ratio(x, k, V)
        B.append(np.digitize(np.arctan(ALPHA[k] * r.ravel()[ix]), EDGES))
        C.append(np.minimum(NWC - 1, np.floor(np.abs(L.ravel()[ix]) / np.sqrt(E_L[k] * V[ix]))))
    return np.concatenate(B).astype(np.int64), np.concatenate(C).astype(np.int64)


def wcodes(pb, pc, gb):
    """gallery x points: map bin | probe bin << 3 | significance << 6 | scale << 8"""
    if DL.SCALE_OF is None or len(DL.SCALE_OF) != len(pb):
        sizes = [len(ix) for ix in DL.grid_idx(*DL.SHAPE)]
        DL.SCALE_OF = np.repeat(np.arange(len(DL.SCALES)), sizes)
    return gb | (pb[None] << 3) | (pc[None] << 6) | (DL.SCALE_OF[None] << 8)
