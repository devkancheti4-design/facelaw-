"""Laws 1-4, checked on synthetic images before any noise-dependent claim.  Prints measured vs predicted."""
import json, sys
import numpy as np
from world import (World, render, photons, sensor, light, shade, sh9, du, dv, pu, pv, IMG, Z0, F_PX, G_DN, B_DN,
                   SIG_R, rot)
from laws import inverse, basis, fit, SIG2

R = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
W = World(0)
OUT = {}


def pose_draw():
    return (R.uniform(-0.44, 0.44), R.uniform(-0.26, 0.26), R.uniform(-0.17, 0.17),
            R.uniform(-0.3, 0.3), R.uniform(-0.3, 0.3), R.uniform(0.9, 1.1))


def cam_normals(P, pose):
    A, n, rho, info = render(W, P, pose, S=(pu, pv))
    return n[np.unique(A.indices)]


# ---------------------------------------------------------------- LAW 1 --
print("LAW 1  sensor inverse   (predicted: bias 0, variance ratio 1.000)")
OUT["L1"] = []
for Phi in (1.0, 3.0, 10.0):
    x = sensor(np.full(400_000, Phi), R)
    Nh, var, ok = inverse(x)
    bias, ratio = Nh.mean() - Phi, Nh.var() / (Phi + SIG2)
    se_b, se_r = np.sqrt((Phi + SIG2) / len(x)), np.sqrt(2 / len(x))
    print("   Phi = %4.1f   bias %+.4f (+- %.4f)   Var ratio %.4f (+- %.4f)" % (Phi, bias, 2 * se_b, ratio, 2 * se_r))
    OUT["L1"].append(dict(Phi=Phi, bias=bias, ratio=ratio))

# ---------------------------------------------------------------- LAW 2 --
print("\nLAW 2  light and exposure cancel: relative leftover |(I-P)y| / |y| on the rigid region, noiseless")
rows = {"sh, same pixelation": [], "sh, independent pixelation": [], "point, independent pixelation": [],
        "sh, exposure x1000": []}
for trial in range(60):
    P = W.person(R); pose = pose_draw()
    pix, B = basis(W, P, pose)
    for kind, probe in (("sh", False), ("sh", True), ("point", True)):
        A, n, rho, info = render(W, P, pose, S=(pu, pv) if probe else None)
        L = light(kind, R, n[np.unique(A.indices)])
        Phi, info2, k = photons(W, P, pose, L, 5.0, probe=probe)
        y = Phi[pix]
        mu, w, c = fit(y, B)
        rel = np.linalg.norm(y - mu) / np.linalg.norm(y)
        rows["%s, %s pixelation" % (kind, "independent" if probe else "same")].append(rel)
        if kind == "sh" and not probe:
            mu2, _, _ = fit(1000 * y, B)
            rows["sh, exposure x1000"].append(np.linalg.norm(1000 * y - mu2) / np.linalg.norm(1000 * y))
for k, v in rows.items():
    v = np.array(v)
    print("   %-32s median %.2e   95th %.2e   max %.2e" % (k, np.median(v), np.percentile(v, 95), v.max()))
OUT["L2"] = {k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(np.max(v))) for k, v in rows.items()}

# ---------------------------------------------------------------- LAW 3 --
print("\nLAW 3  depth ambiguity (generalized bas-relief): z' = lam z + mu u + nu v, rho' = rho |m'| / (lam |m|), "
      "s' = lam A^-T s")
OUT["L3"] = []
for lam_, mu_, nu_ in ((0.6, 0.0, 0.0), (1.4, 0.1, -0.05), (0.8, -0.15, 0.1)):
    P = W.person(R)
    z = W.z(P, du, dv); h = 1e-4
    zu = (W.z(P, du + h, dv) - W.z(P, du - h, dv)) / (2 * h); zv = (W.z(P, du, dv + h) - W.z(P, du, dv - h)) / (2 * h)
    m = np.stack([zu, zv, np.ones_like(zu)], -1)
    A = np.array([[lam_, 0, mu_], [0, lam_, nu_], [0, 0, 1.0]])
    m2 = m @ A.T
    z2 = lam_ * z + mu_ * du + nu_ * dv
    rho = W.rho(P, du, dv); rho2 = rho * np.linalg.norm(m2, axis=1) / (lam_ * np.linalg.norm(m, axis=1))
    n1, n2 = -m / np.linalg.norm(m, axis=1, keepdims=True), -m2 / np.linalg.norm(m2, axis=1, keepdims=True)
    s = np.array([0.4, 0.5, -0.77]); s /= np.linalg.norm(s); s2 = lam_ * np.linalg.inv(A).T @ s
    res = {}
    for ortho in (True, False):
        imgs = []
        for zz, nn, rr, ss in ((z, n1, rho, s), (z2, n2, rho2, s2)):
            A, nc, rc, info = render(W, P, (0, 0, 0, 0, 0, 1), ortho=ortho, z_override=zz, n_override=nn, rho_override=rr)
            img = A @ (rc * np.maximum(0, nc @ ss))
            imgs.append(img)
        res["ortho" if ortho else "perspective"] = np.linalg.norm(imgs[0] - imgs[1]) / np.linalg.norm(imgs[0])
    print("   lam %.1f mu %+.2f nu %+.2f   orthographic image difference %.1e   perspective (Z0 = %.0f) %.2e"
          % (lam_, mu_, nu_, res["ortho"], Z0, res["perspective"]))
    OUT["L3"].append(dict(lam=lam_, mu=mu_, nu=nu_, **res))

# ---------------------------------------------------------------- LAW 4 --
print("\nLAW 4  expression cancels on the rigid region (v >= -0.15): probe with E != 0 vs E = 0, rigid pixels")
d = []
for trial in range(40):
    P = W.person(R); pose = pose_draw(); E = R.uniform(-1.5, 1.5, 3)
    pix, B = basis(W, P, pose)
    A, n, rho, info = render(W, P, pose, S=(pu, pv)); L = light("point", R, n[np.unique(A.indices)])
    y0 = photons(W, P, pose, L, 5.0)[0][pix]; y1 = photons(W, P, pose, L, 5.0, E=E)[0][pix]
    d.append(np.linalg.norm(y0 / np.linalg.norm(y0) - y1 / np.linalg.norm(y1)))   # exposure is renormalised
d = np.array(d)
print("   | y(E)/|y(E)| - y(0)/|y(0)| | over rigid pixels:  median %.1e  max %.1e  (%d of %d exactly 0)"
      % (np.median(d), d.max(), int((d == 0).sum()), len(d)))
OUT["L4"] = dict(median=float(np.median(d)), max=float(d.max()), exact=int((d == 0).sum()), n=len(d))
json.dump(OUT, open("static_results.json", "w"), indent=1)
