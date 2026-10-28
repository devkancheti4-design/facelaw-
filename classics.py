"""CLASSIC face recognisers for the head-to-head, written from their papers (reimplementations, not the authors' code),
with the parameters the papers state.  Every function takes an aligned face (linear light) and returns a descriptor;
each method has its own similarity (higher = more alike).
  LBP        Ahonen, Hadid, Pietikainen 2006: LBP u2 (8 neighbours, radius 2, bilinear), 7 x 7 regions, 59-bin
             histograms, chi-square
  LTP        Tan & Triggs 2010: their preprocessing chain, then local ternary patterns (t = 0.2) split into upper and
             lower u2 codes, 7 x 7 regions, chi-square (they also give a distance-transform match; not used here)
  LGBPHS     Zhang et al. 2005: Gabor magnitudes, 5 scales x 8 orientations (kmax pi/2, f sqrt 2, sigma 2 pi), LBP u2
             (8, 1) on each, 7 x 7 regions, histogram intersection
  HOG        Dalal & Triggs 2005 as used for faces (Deniz et al. 2011): 6 x 6 cells, 9 unsigned bins, 2 x 2 blocks,
             L2-Hys, Euclidean
  EIGEN      Turk & Pentland 1991: PCA (100 components) learned on the training people, cosine; EIGEN-3 drops the first
             three components (Belhumeur et al. 1997: they carry most of the light)
  FISHER     Belhumeur et al. 1997: PCA to N - c, then LDA to c - 1, learned on the training people, cosine
Gradientfaces, Weberfaces and Tan-Triggs (pixels, L1) are in yale_baselines.py."""
import os, sys
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yale_baselines import tantriggs


# ---------------------------------------------------------------- local binary patterns
def _u2_table(P):
    t = np.full(1 << P, -1, np.int64); k = 0
    for c in range(1 << P):
        b = [(c >> i) & 1 for i in range(P)]
        if sum(b[i] != b[(i + 1) % P] for i in range(P)) <= 2:
            t[c] = k; k += 1
    t[t < 0] = k
    return t, k + 1


U2_8, NU2 = _u2_table(8)                                # 59 labels


def neighbours(I, R, P=8):
    """bilinear samples of the P points on a circle of radius R around every pixel (edges clamped)"""
    h, w = I.shape
    pad = int(np.ceil(R)) + 1
    E = np.pad(I, pad, mode="edge")
    out = []
    for p in range(P):
        a = 2 * np.pi * p / P
        dy, dx = -R * np.sin(a), R * np.cos(a)
        if abs(dy - round(dy)) < 1e-9: dy = float(round(dy))
        if abs(dx - round(dx)) < 1e-9: dx = float(round(dx))
        y0, x0 = int(np.floor(dy)), int(np.floor(dx)); fy, fx = dy - y0, dx - x0
        S = lambda a_, b_: E[pad + a_:pad + a_ + h, pad + b_:pad + b_ + w]
        out.append(S(y0, x0) * (1 - fy) * (1 - fx) + S(y0, x0 + 1) * (1 - fy) * fx + S(y0 + 1, x0) * fy * (1 - fx) + S(y0 + 1, x0 + 1) * fy * fx)
    return np.array(out)


def regions_hist(labels, nb, grid=(7, 7)):
    h, w = labels.shape
    ys = np.linspace(0, h, grid[0] + 1).astype(int); xs = np.linspace(0, w, grid[1] + 1).astype(int)
    out = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            out.append(np.bincount(labels[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].ravel(), minlength=nb))
    return np.concatenate(out).astype(np.float32)


def lbp_u2(I, R):
    N = neighbours(I, R)
    code = sum(((N[p] >= I).astype(np.int64) << p) for p in range(8))
    return U2_8[code]


def lbp(I):
    return regions_hist(lbp_u2(I.astype(float), 2.0), NU2)


def ltp(I, t=0.2):
    X = tantriggs(I.astype(float))
    N = neighbours(X, 2.0)
    up = sum(((N[p] >= X + t).astype(np.int64) << p) for p in range(8))
    lo = sum(((N[p] <= X - t).astype(np.int64) << p) for p in range(8))
    return np.concatenate([regions_hist(U2_8[up], NU2), regions_hist(U2_8[lo], NU2)])


def chi2_sim(a, B):
    d = (a[None] - B) ** 2 / (a[None] + B + 1e-9)
    return -d.reshape(len(B), -1).sum(1)


# ---------------------------------------------------------------- Gabor + LBP (LGBPHS)
def gabor_bank(size=33):
    kmax, f, sg = np.pi / 2, np.sqrt(2), 2 * np.pi
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    bank = []
    for v in range(5):
        for u in range(8):
            k = kmax / f ** v; phi = np.pi * u / 8
            kx, ky = k * np.cos(phi), k * np.sin(phi)
            env = (k * k / sg ** 2) * np.exp(-(k * k) * (x * x + y * y) / (2 * sg ** 2))
            bank.append(env * (np.exp(1j * (kx * x + ky * y)) - np.exp(-sg ** 2 / 2)))
    return bank


GABOR = gabor_bank()


_GF = {}


def lgbphs(I):
    """the 40 Gabor magnitudes by one FFT of the face and 40 products (kernel transforms cached per size)"""
    I = I.astype(float)
    h, w = I.shape; r = GABOR[0].shape[0] // 2
    H, W = h + 2 * r, w + 2 * r
    if (H, W) not in _GF:
        _GF[(H, W)] = [np.fft.fft2(g, (H, W)) for g in GABOR]
    F = np.fft.fft2(np.pad(I, r, mode="edge"), (H, W))
    hs = []
    for G in _GF[(H, W)]:
        M = np.abs(np.fft.ifft2(F * G))[2 * r:2 * r + h, 2 * r:2 * r + w]
        hs.append(regions_hist(lbp_u2(M, 1.0), NU2))
    return np.concatenate(hs)


def intersect_sim(a, B):
    return np.minimum(a[None], B).reshape(len(B), -1).sum(1)


# ---------------------------------------------------------------- HOG
def hog(I, cell=6, nb=9):
    I = I.astype(float)
    gx = np.zeros_like(I); gy = np.zeros_like(I)
    gx[:, 1:-1] = I[:, 2:] - I[:, :-2]; gy[1:-1] = I[2:] - I[:-2]
    mag = np.hypot(gx, gy); ang = np.mod(np.arctan2(gy, gx), np.pi)
    ch, cw = I.shape[0] // cell, I.shape[1] // cell
    pos = ang / np.pi * nb - 0.5
    b0 = np.floor(pos).astype(int); fr = pos - b0
    H = np.zeros((ch, cw, nb))
    yy, xx = np.mgrid[0:ch * cell, 0:cw * cell]
    cy, cx = yy // cell, xx // cell
    m, b, f_ = mag[:ch * cell, :cw * cell], b0[:ch * cell, :cw * cell], fr[:ch * cell, :cw * cell]
    np.add.at(H, (cy, cx, b % nb), m * (1 - f_))
    np.add.at(H, (cy, cx, (b + 1) % nb), m * f_)
    out = []
    for i in range(ch - 1):
        for j in range(cw - 1):
            v = H[i:i + 2, j:j + 2].ravel()
            v = v / np.sqrt((v * v).sum() + 1e-6); v = np.minimum(v, 0.2); v = v / np.sqrt((v * v).sum() + 1e-6)
            out.append(v)
    return np.concatenate(out).astype(np.float32)


def l2_sim(a, B):
    return -np.sqrt(((a[None] - B) ** 2).reshape(len(B), -1).sum(1))


# ---------------------------------------------------------------- subspaces (learned on the training people)
class Eigen:
    def __init__(self, X, k=100, drop=0):
        X = np.asarray(X, float).reshape(len(X), -1)
        self.mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.W = Vt[drop:drop + k]

    def __call__(self, I):
        return (self.W @ (np.asarray(I, float).ravel() - self.mu)).astype(np.float32)


class Fisher:
    def __init__(self, X, y):
        X = np.asarray(X, float).reshape(len(X), -1); y = np.asarray(y)
        cls = np.unique(y); n, c = len(X), len(cls)
        self.mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        Wp = Vt[:n - c]
        Z = (X - self.mu) @ Wp.T
        m = Z.mean(0)
        Sw = np.zeros((Z.shape[1],) * 2); Sb = np.zeros_like(Sw)
        for k in cls:
            Zk = Z[y == k]; mk = Zk.mean(0)
            Sw += (Zk - mk).T @ (Zk - mk); Sb += len(Zk) * np.outer(mk - m, mk - m)
        ev, V = np.linalg.eig(np.linalg.solve(Sw + 1e-6 * np.eye(len(Sw)) * np.trace(Sw) / len(Sw), Sb))
        o = np.argsort(-ev.real)[:c - 1]
        self.W = (V[:, o].real.T @ Wp)

    def __call__(self, I):
        return (self.W @ (np.asarray(I, float).ravel() - self.mu)).astype(np.float32)


def cos_sim(a, B):
    return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-12)
