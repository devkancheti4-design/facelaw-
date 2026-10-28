"""AM as additive cancellation over LAW 7c: score = v + w . f(state), f = one-hot of perspective agreement (4), crowd
support (1), darkness band (3), size band (3); w measured on held-out design [3800:4250] by maximum likelihood WITH v
held fixed as the offset -- AM adds only the evidence LAW 7c does not already carry."""
import os, pickle, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import darkface_law as L
import darkface_v4 as V4


def feats(persp, crowd, snr, k, cn):
    n = len(persp)
    F = np.zeros((n, 10))
    F[np.arange(n), persp] = 1                       # 0..3 (3 = no line), column 0 = on the line
    F[:, 4] = crowd
    bd = (snr[:, None] >= np.array(cn)[None, :]).sum(1); F[np.arange(n), 5 + bd] = 1
    bk = (k[:, None] >= np.array(V4.KCUT)[None, :]).sum(1); F[:, 8] = bk == 0; F[:, 9] = bk == 1
    return F[:, 1:]                                  # drop one perspective column (the intercept is absorbed by v)


def fit(v, F, y, iters=25, ridge=1.0):
    w = np.zeros(F.shape[1])
    for _ in range(iters):
        z = v + F @ w; p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = F.T @ (y - p) - ridge * w
        H = (F * (p * (1 - p))[:, None]).T @ F + ridge * np.eye(len(w))
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-6:
            break
    return w


def val_image(args):
    key, nb, cn, w, nms = args
    if nb is None:
        return key, []
    vv, B, B8, kk, sn = nb
    BB = B8.astype(np.float64); vv = vv.astype(np.float64)
    pp, cc = V4.scene(vv, BB)
    s = vv + feats(pp, cc, sn.astype(int), kk.astype(int), cn) @ w
    o = np.argsort(-s, kind="stable")
    return key, L.nms([(float(s[i]),) + tuple(BB[i]) for i in o], nms)


if __name__ == "__main__":
    OUT = V4.OUT
    M = np.load(os.path.join(OUT, "v4_am_measure.npz"))
    v, y = M["v"].astype(np.float64), M["y"].astype(np.float64)
    cn = np.quantile(M["snr"][M["y"] == 1], [1 / 3, 2 / 3]).tolist()
    F = feats(M["persp"].astype(int), M["crowd"].astype(int), M["snr"].astype(int), M["k"].astype(int), cn)
    w = fit(v, F, y)
    names = ["persp near", "persp far", "persp none", "crowd", "dark mid", "dark bright", "dark low?", "size<8", "size8-16"]
    print("AM additive weights (log-odds added to LAW 7c), measured on %d held-out candidates:" % len(y))
    print("  " + ", ".join("%s %+.3f" % (n, x) for n, x in zip(["persp 0.25-0.6", "persp > 0.6", "no line", "crowd", "dark band 2", "dark band 3", "size < 8 px", "size 8-16 px"], [w[0], w[1], w[2], w[3], w[5], w[6], w[7], w[8]])))
    Z = pickle.load(open(os.path.join(OUT, "v4_val_numbers.pkl"), "rb"))
    jobs = [(k, nb, cn, w, 0.5) for k, nb in Z["nums"].items()]
    with Pool(8) as pool:
        d = dict(pool.imap_unordered(val_image, jobs, chunksize=4))
    ap, rec = L.average_precision(d, Z["gts"])
    print("design-validation (300 images, NMS 0.5, LAW 8 on): AM additive over LAW 7c  AP %.2f%% (recall %.1f%%)   [LAW 7c alone 31.42%%]" % (100 * ap, 100 * rec))
    pickle.dump(dict(w=w.tolist(), cn=cn), open(os.path.join(OUT, "v4_amadd.pkl"), "wb"))
