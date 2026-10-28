"""LAW 2 WITH CAST SHADOWS on Extended Yale B, protocol A (train subset 1 only: 7 images; test subsets 2-5; 38 people).
ENROL   photometric stereo on the 7 images -> albedo rho, normal n; Frankot-Chellappa integration -> height field z
        (convention measured from the data: s = (-sin A cos E, sin E, cos A cos E) makes the normals integrable AND
        the nose stand out, 12 of 12 faces)
LIGHT   image(s) = a rho max(0, n.s) V_s + b rho,  a, b >= 0;  V_s = 1 iff the ray from the pixel toward s clears the
        surface (cast shadow), found by marching the height field.  s is NOT read from the probe: the law searches
        1,484 directions (azimuth -130..130, elevation -45..90, 5 deg) and everything but the remainder cancels.
DECIDE  the person whose cone leaves the smallest remainder."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y

H, Wd = 48, 42
AZ, EL = np.radians(np.arange(-130, 131, 5)), np.radians(np.arange(-45, 91, 5))
DIRS = np.array([[-np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)] for a in AZ for e in EL])


def frankot(P, Q):
    wx = 2 * np.pi * np.fft.fftfreq(Wd)[None, :]; wy = 2 * np.pi * np.fft.fftfreq(H)[:, None]
    den = wx ** 2 + wy ** 2; den[0, 0] = 1
    Z = (-1j * wx * np.fft.fft2(P) + 1j * wy * np.fft.fft2(Q)) / den
    Z[0, 0] = 0
    return np.real(np.fft.ifft2(Z))


def shape(imgs):
    E = [im for im in imgs if Y.subset(im["theta"]) == 1]
    S = np.array([im["s"] * [-1, 1, 1] for im in E]); I = np.array([im["I"] for im in E])
    b = np.linalg.lstsq(S, I, rcond=None)[0].T                                   # P x 3: rho n
    rho = np.linalg.norm(b, axis=1); n = b / np.maximum(rho[:, None], 1e-9)
    nz = np.maximum(n[:, 2], 0.05)
    Z = frankot((-n[:, 0] / nz).reshape(H, Wd), (-n[:, 1] / nz).reshape(H, Wd))
    return rho, n, Z


import json as _json
_lc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lane_cast.json")
LANE = _json.load(open(_lc))["expr"] if os.path.exists(_lc) and os.environ.get("CAST_LANE", "1") == "1" else None


def visibility(Z, s, T=60, eps=0.5):
    """V_s per pixel: 1 iff marching toward the light never meets the surface (heights in pixel units)"""
    rxy = np.hypot(s[0], s[1])
    if rxy < 1e-6:
        return np.ones(H * Wd)
    dc, dr, dz = s[0] / rxy, -s[1] / rxy, s[2] / rxy
    rr, cc = np.mgrid[0:H, 0:Wd]
    z0 = Z.copy(); vis = np.ones((H, Wd), bool); alive = np.ones((H, Wd), bool)
    for t in range(1, T + 1):
        r = np.rint(rr + t * dr).astype(int); c = np.rint(cc + t * dc).astype(int)
        inside = (r >= 0) & (r < H) & (c >= 0) & (c < Wd)
        alive &= inside
        if not alive.any():
            break
        if LANE:   # the lane-authored blocking test on 12-bit heights (1/16 px, offset 2048)
            zs = np.clip(np.rint(16 * Z[np.clip(r, 0, H - 1), np.clip(c, 0, Wd - 1)]) + 2048, 0, 4095).astype(np.int64)
            zr = np.clip(np.rint(16 * (z0 + t * dz + eps)) + 2048, 0, 4095).astype(np.int64)
            blocked = eval(LANE, {"x": zs | (zr << 12)}).astype(bool)
        else:
            zs = np.where(alive, Z[np.clip(r, 0, H - 1), np.clip(c, 0, Wd - 1)], -np.inf)
            blocked = zs > z0 + t * dz + eps
        vis &= ~(alive & blocked)
    return vis.ravel().astype(float)


def renders(rho, n, Z):
    shade = np.maximum(0, n @ DIRS.T).T                                           # dirs x P (attached shadows)
    V = np.array([visibility(Z, s) for s in DIRS])                                # dirs x P (cast shadows)
    return (rho[None, :] * shade * V).astype(np.float64)


def cone_fit(Rm, rho, Y):
    """for every direction and probe: min over a, b >= 0 of |y - a R_s - b rho|^2 -> (best residual, best s, a, b)"""
    G = Rm @ Y; RR = (Rm ** 2).sum(1)[:, None]; Rr = (Rm @ rho)[:, None]; rr = float(rho @ rho)
    rY = (rho @ Y)[None, :]; YY = (Y ** 2).sum(0)[None, :]
    det = RR * rr - Rr ** 2
    a = (rr * G - Rr * rY) / np.maximum(det, 1e-12); b = (RR * rY - Rr * G) / np.maximum(det, 1e-12)
    both = YY - (a * G + b * rY)
    onlyR = np.where(G > 0, YY - G ** 2 / np.maximum(RR, 1e-12), YY)
    onlyr = np.broadcast_to(np.where(rY > 0, YY - rY ** 2 / rr, YY), G.shape)
    res = np.where((a >= 0) & (b >= 0), both, np.minimum(onlyR, onlyr))
    k = res.argmin(0)
    cols = np.arange(Y.shape[1])
    aa = np.where((a[k, cols] >= 0) & (b[k, cols] >= 0), a[k, cols], np.where(onlyR[k, cols] <= onlyr[k, cols], np.maximum(G[k, cols], 0) / np.maximum(RR[k, 0], 1e-12), 0))
    bb = np.where((a[k, cols] >= 0) & (b[k, cols] >= 0), b[k, cols], np.where(onlyR[k, cols] <= onlyr[k, cols], 0, np.maximum(rY[0, cols], 0) / rr))
    return res[k, cols], k, aa, bb


if __name__ == "__main__":
    t0 = time.time()
    probes = [(p, Y.subset(im["theta"]), im["I"]) for p in Y.IDS for im in Y.PEOPLE[p] if Y.subset(im["theta"]) >= 2]
    Ymat = np.array([y for _, _, y in probes]).T
    RES = np.empty((len(Y.IDS), len(probes))); FIT = {}
    for i, p in enumerate(Y.IDS):
        rho, n, Z = shape(Y.PEOPLE[p])
        Rm = renders(rho, n, Z)
        res, k, a, b = cone_fit(Rm, rho, Ymat)
        RES[i] = res
        FIT[p] = (Rm.astype(np.float32), rho, k, a, b)
        if i % 10 == 0:
            print("   person %d of %d  (%.0f s)" % (i + 1, len(Y.IDS), time.time() - t0), flush=True)
    best = RES.argmin(0)
    acc = {}
    for j, (p, sb, y) in enumerate(probes):
        acc.setdefault(sb, []).append(Y.IDS[best[j]] == p)
    print("\nCAST-SHADOW LAW, protocol A, 38 people -- recognition %")
    for sb in sorted(acc):
        print("   subset %d: %6.2f  (%d probes)" % (sb, 100 * np.mean(acc[sb]), len(acc[sb])))
    # remainders of the best cast-shadow fit, for the data law
    import pickle
    rem = []
    for j, (p, sb, y) in enumerate(probes):
        h = {}
        for i, q in enumerate(Y.IDS):
            Rm, rho, k, a, b = FIT[q]
            mu = a[j] * Rm[k[j]] + b[j] * rho
            s = max(y.mean(), 1e-9); e, m = (y - mu) / s, mu / s
            lv = np.where(m <= 0.05, 0, np.where(m <= 0.5, 1, np.where(m <= 1.5, 2, 3)))
            code = np.minimum(15, np.floor(8 * np.abs(e))).astype(int) | (e < 0).astype(int) << 4 | lv << 5
            h[q] = np.bincount(code, minlength=128)
        rem.append((p, sb, h))
    pickle.dump(dict(rem=rem, res=RES, ids=Y.IDS, probes=[(p, sb) for p, sb, _ in probes]), open("yale_cast_rem.pkl", "wb"))
    print("%.0f s" % (time.time() - t0))
