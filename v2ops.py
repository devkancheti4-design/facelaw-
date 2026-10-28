"""v2 EXACT OPERATIONS: every float step of the v2 law written so the C++ port matches it BIT FOR BIT on any machine.
Only IEEE add / subtract / multiply / divide / sqrt / floor / rint and comparisons run at run time, each in a fixed
order; nothing transcendental: the Gaussian kernels (scipy's own, sigma in steps of 1/8) and the sRGB tables are FROZEN
numbers (v2ops_tables.json), read by this file and by the C++ alike.
    GAUSS   scipy.ndimage's separable filter (symmetric correlate1d, 'reflect'), axis 0 then axis 1, written out with every
            product and sum rounded on its own (scipy's own build fuses multiply-adds on some machines)
    BOX     f x f block means, the f*f terms added in row-major order
    WARP    output pixel (r, c) -> image point  O + s R (u, v),  R the rotation by the unit vector (ux, uy); the image is
            first box-integrated by f = max(1, s // 2) and smoothed by sigma = rint(8 max(0, (s/f - 1) / 2)) / 8, then
            sampled bilinearly (scipy's order-1 sum, edges clamped)
    MEAN    a sequential sum in row-major order over n
    U8      linear light -> the display byte: a = clip(lin / P99.5, 0, 1), byte = the count of frozen thresholds <= a
    python3 -I v2ops.py freeze      write v2ops_tables.json (once)"""
import json, os
import numpy as np
from scipy.ndimage import correlate1d

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "v2ops_tables.json")
K8MAX = 24                                           # sigma up to 3.0 in steps of 1/8
TB = {}


def freeze():
    from scipy.ndimage._filters import _gaussian_kernel1d
    ker = {}
    for k in range(1, K8MAX + 1):
        s = k / 8.0
        w = _gaussian_kernel1d(s, 0, int(4.0 * s + 0.5))[::-1]          # exactly what gaussian_filter1d correlates with
        ker[k] = [float(v).hex() for v in w]
    a = np.arange(256) / 255.0
    dec = 255.0 * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    e = (np.arange(1, 256) - 0.5) / 255.0
    thr = np.where(e <= 0.04045, e / 12.92, ((e + 0.055) / 1.055) ** 2.4)    # byte k <=> a >= thr[k - 1]
    json.dump(dict(note="frozen numbers of the v2 law (exact hex doubles): Gaussian kernels by sigma*8, the sRGB decode of "
                        "each byte (x 255), the display-byte thresholds", kernels=ker,
                   srgb_decode=[float(v).hex() for v in dec], u8_thresholds=[float(v).hex() for v in thr]),
              open(TABLES, "w"), indent=0)


def load():
    if not TB:
        Z = json.load(open(TABLES))
        TB["K"] = {int(k): np.array([float.fromhex(v) for v in w]) for k, w in Z["kernels"].items()}
        TB["DEC"] = np.array([float.fromhex(v) for v in Z["srgb_decode"]])
        TB["THR"] = np.array([float.fromhex(v) for v in Z["u8_thresholds"]])
    return TB


def reflect_index(n, r):
    """scipy's 'reflect' extension (half-sample symmetric), any distance: the source index of -r .. n + r - 1"""
    i = np.arange(-r, n + r) % (2 * n)
    return np.where(i < n, i, 2 * n - 1 - i)


def corr_axis(x, w, axis):
    """the symmetric correlation along one axis, outermost tap pair first, each product and sum rounded on its own (no
    fused multiply-add: scipy's own build fuses them on some machines, so its last bits differ between machines)"""
    r = len(w) // 2
    e = np.take(x, reflect_index(x.shape[axis], r), axis=axis)
    n = x.shape[axis]
    sl = lambda a, b: np.take(e, np.arange(a, b), axis=axis)
    acc = sl(r, r + n) * w[r]
    for jj in range(-r, 0):
        acc = acc + (sl(r + jj, r + jj + n) + sl(r - jj, r - jj + n)) * w[r + jj]
    return acc


def gauss(x, k8):
    """the Gaussian at sigma = k8 / 8 ('reflect', truncate 4, the frozen kernel), axis 0 then axis 1"""
    if k8 == 0:
        return np.array(x, float)
    w = load()["K"][k8]
    return corr_axis(corr_axis(np.asarray(x, float), w, 0), w, 1)


def k8_of(sigma):
    k = int(np.rint(8.0 * sigma))
    assert 0 <= k <= K8MAX and k == 8.0 * sigma, sigma
    return k


def box(x, f):
    if f == 1:
        return np.array(x, float)
    h, w = x.shape[0] // f, x.shape[1] // f
    acc = np.zeros((h, w))
    for i in range(f):
        for j in range(f):
            acc = acc + x[i:h * f:f, j:w * f:f]
    return acc / float(f * f)


def mean(x):
    return float(np.cumsum(np.ravel(x))[-1]) / x.size


def bilinear(G, Y, X):
    """scipy map_coordinates(order=1, mode='nearest'), its own sum order"""
    H, W = G.shape
    y0 = np.floor(Y); x0 = np.floor(X); fy = Y - y0; fx = X - x0
    y0 = y0.astype(np.int64); x0 = x0.astype(np.int64)
    ya, yb = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1); xa, xb = np.clip(x0, 0, W - 1), np.clip(x0 + 1, 0, W - 1)
    return G[ya, xa] * (1 - fy) * (1 - fx) + G[ya, xb] * (1 - fy) * fx + G[yb, xa] * fy * (1 - fx) + G[yb, xb] * fy * fx


def warp(img, ox, oy, ux, uy, s, U, V):
    """sample img at  (ox, oy) + s R (U, V)  (R: rotation by the unit vector (ux, uy)), anti-aliased (BOX, then GAUSS)"""
    f = max(1, int(s // 2))
    D = box(img, f)
    sp = s / f
    k8 = int(np.rint(8.0 * max(0.0, 0.5 * (sp - 1.0))))
    G = gauss(D, min(k8, K8MAX))
    X = ox + s * (ux * U - uy * V); Y = oy + s * (uy * U + ux * V)
    return bilinear(G, (Y - (f - 1) / 2.0) / f, (X - (f - 1) / 2.0) / f)


def unit(v):
    d = float(np.sqrt(v[0] * v[0] + v[1] * v[1]))
    return v[0] / d, v[1] / d, d


def linear_of(rgb_u8):
    """8-bit sRGB (H x W x 3, or H x W grey read as R = G = B) -> linear light, the mean of the channels"""
    D = load()["DEC"]
    a = np.asarray(rgb_u8)
    if a.ndim == 2:
        L = D[a]
        return ((L + L) + L) / 3.0
    return ((D[a[..., 0]] + D[a[..., 1]]) + D[a[..., 2]]) / 3.0


def quantile(sorted_vals, q):
    """numpy's 'linear' quantile on an already sorted 1-D array (its own lerp)"""
    n = len(sorted_vals)
    vi = (n - 1) * q
    lo = int(np.floor(vi)); hi = min(lo + 1, n - 1); t = vi - lo
    a, b = sorted_vals[lo], sorted_vals[hi]
    d = b - a
    return float(b - d * (1 - t)) if t >= 0.5 else float(a + d * t)


def median(v):
    s = np.sort(np.ravel(v)); n = len(s)
    return float(s[n // 2]) if n % 2 else float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def u8(lin):
    """linear light -> display bytes (as finder.to_u8, but the sRGB curve read from frozen thresholds)"""
    p = quantile(np.sort(np.ravel(lin)), 0.995)
    a = np.clip(lin / max(p, 1e-9), 0.0, 1.0)
    return np.searchsorted(load()["THR"], a, side="right").astype(np.uint8)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
        T = load()
        print("frozen: %d kernels, sRGB decode %d, thresholds %d -> %s" % (len(T["K"]), len(T["DEC"]), len(T["THR"]), TABLES))
        x = np.random.default_rng(0).random((40, 33)) * 255
        from scipy.ndimage import gaussian_filter
        print("gauss vs scipy gaussian_filter: max |difference| %.2e (scipy fuses multiply-adds on this machine)" % max(np.abs(gauss(x, k) - gaussian_filter(x, k / 8.0)).max() for k in range(1, K8MAX + 1)))
        sys.path.insert(0, HERE); sys.path.insert(0, os.path.expanduser('~/the private toolkit'))
        import finder as FD
        lin = np.random.default_rng(1).random((50, 61)) * 300
        print("u8 == finder.to_u8 on a random image: %d of %d bytes differ" % (int((u8(lin) != FD.to_u8(lin)).sum()), lin.size))
