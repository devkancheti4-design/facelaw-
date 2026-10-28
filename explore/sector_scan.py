import os, sys, numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.expanduser("~/neooos/facelaw"))
import yale as Y
from yale_baselines import load_full
from yale_local import resize
full = load_full()
def grad(I):
    G = gaussian_filter(I, 0.75); return np.gradient(G, axis=1), -np.gradient(G, axis=0)   # gx right, gy up
def run(f, K):
    gal, lab = [], []
    for p in Y.IDS:
        for sb, I in full[p]:
            if sb == 1:
                gx, gy = grad(resize(I, f)); gal.append(np.arctan2(gy, gx).ravel()); lab.append(p)
    gal = np.array(gal); lab = np.array(lab)
    if K: galq = np.floor((gal + np.pi) / (2 * np.pi) * K).astype(int) % K
    acc = {}
    for p in Y.IDS:
        for sb, I in full[p]:
            if sb < 2: continue
            gx, gy = grad(resize(I, f)); a = np.arctan2(gy, gx).ravel()
            if K:
                q = np.floor((a + np.pi) / (2 * np.pi) * K).astype(int) % K
                d = np.abs(q[None] - galq); d = np.minimum(d, K - d).sum(1)
            else:
                d = np.abs(a[None] - gal); d = np.minimum(d, 2 * np.pi - d).sum(1)
            acc.setdefault(sb, []).append(lab[int(np.argmin(d))] == p)
    return [100 * np.mean(acc[s]) for s in (2, 3, 4, 5)]
for f in (2, 1):
    for K in (0, 8, 16, 32):
        r = run(f, K)
        print("%3dx%-3d %s: " % (192 // f, 168 // f, "continuous" if K == 0 else "%2d sectors" % K) + "  ".join("%6.2f" % v for v in r), flush=True)
