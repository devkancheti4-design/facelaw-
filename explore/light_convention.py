import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("~/neooos/facelaw"))
import yale as Y
H, Wd = 48, 42
def ps(imgs, fx, fy):
    E = [im for im in imgs if Y.subset(im["theta"]) == 1]
    S = np.array([im["s"] * [fx, fy, 1] for im in E]); I = np.array([im["I"] for im in E])
    b = np.linalg.lstsq(S, I, rcond=None)[0].T
    return b
res = {}
for fx in (1, -1):
    for fy in (1, -1):
        c = []
        for p in Y.IDS[:12]:
            b = ps(Y.PEOPLE[p], fx, fy)
            nz = np.maximum(b[:, 2], 1e-3)
            P = (-b[:, 0] / nz).reshape(H, Wd); Q = (-b[:, 1] / nz).reshape(H, Wd)   # dz/dx (x = column), dz/dy (y = up)
            dPdy = -(np.roll(P, -1, 0) - np.roll(P, 1, 0)) / 2     # y up = row index decreasing
            dQdx = (np.roll(Q, -1, 1) - np.roll(Q, 1, 1)) / 2
            curl = (dPdy - dQdx)[3:-3, 3:-3]
            c.append(np.median(np.abs(curl)))
        res[(fx, fy)] = np.mean(c)
        print("s_x sign %+d, s_y sign %+d: median |curl| %.4f" % (fx, fy, res[(fx, fy)]))

def fc(P, Q):
    """Frankot-Chellappa: the integrable surface closest to the gradients (x = column, y = up)"""
    h, w = P.shape
    wx = 2 * np.pi * np.fft.fftfreq(w)[None, :]; wy = 2 * np.pi * np.fft.fftfreq(h)[:, None]
    Fp, Fq = np.fft.fft2(P), np.fft.fft2(Q)
    # y up = -row: d/dy = -d/drow
    den = wx ** 2 + wy ** 2; den[0, 0] = 1
    Z = (-1j * wx * Fp + 1j * wy * Fq) / den
    Z[0, 0] = 0
    return np.real(np.fft.ifft2(Z))
for fx, fy in ((1, -1), (-1, 1)):
    d = []
    for p in Y.IDS[:12]:
        b = ps(Y.PEOPLE[p], fx, fy); nz = np.maximum(b[:, 2], 1e-3)
        Z = fc((-b[:, 0] / nz).reshape(H, Wd), (-b[:, 1] / nz).reshape(H, Wd))
        nose = Z[22:30, 17:25].mean(); rim = np.concatenate([Z[20:30, 3:7].ravel(), Z[20:30, 35:39].ravel()]).mean()
        d.append(nose - rim)
    print("convention (%+d, %+d): nose minus cheeks depth  median %+.2f px  (%d of 12 positive)" % (fx, fy, np.median(d), sum(x > 0 for x in d)))
