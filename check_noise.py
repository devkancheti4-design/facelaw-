"""Laws 5 and 6 on synthetic dark images: predicted (derived, nothing fitted) vs measured.
Per trial: a gallery person A, a random pose / light / expression, the probe rendered from the WORLD's samples; the
nearest impostor B (smallest lambda at A's pose, then pose re-fitted to the probe without noise: its best case).
For each Nbar: 40 sensor-noise draws -> z for A (genuine) and for B (impostor); predicted accept probabilities from
Law 5's exact moments.  Law 6: exact Poisson-Gaussian Bhattacharyya bound vs the oracle likelihood-ratio test.
    python3 check_noise.py TRIALS"""
import json, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from world import World, render, photons, sensor, light, pu, pv
from laws import inverse, basis, fit, score, rates, bhattacharyya, bayes_bounds, pmf_dn, SIG2
from world import B_DN, G_DN, SIG_R

TAU, EPS, REPS = 2.326, 0.01, 40
NBARS = (1, 2, 5, 10, 20, 50)
W = World(0)
GAL = [W.person(np.random.default_rng(100 + i)) for i in range(10)]


def lam_at(muA_img, P, pose):
    pix, B = basis(W, P, pose)
    mu, w, c = fit(muA_img[pix], B)
    return float((w * (muA_img[pix] - mu) ** 2).sum()), pix, B


def trial(t):
    R = np.random.default_rng(1000 + t)
    a = int(R.integers(len(GAL))); A = GAL[a]
    pose = (R.uniform(-0.35, 0.35), R.uniform(-0.2, 0.2), R.uniform(-0.12, 0.12), R.uniform(-0.2, 0.2),
            R.uniform(-0.2, 0.2), R.uniform(0.92, 1.08))
    E = R.uniform(-1, 1, 3)
    Am, n, rho, info = render(W, A, pose, E, S=(pu, pv))
    L = light("point", R, n[np.unique(Am.indices)])
    unit, _, _ = photons(W, A, pose, L, 1.0, E=E)               # expected photons at Nbar = 1; scales linearly
    # nearest impostor at A's pose, then its pose re-fitted (noiseless, at Nbar = 5): the impostor's best case
    lams = [(lam_at(5 * unit, GAL[j], pose)[0], j) for j in range(len(GAL)) if j != a]
    b = min(lams)[1]
    n0 = len(basis(W, GAL[b], pose)[0])

    def obj(p):                       # a pose is valid only if the whole face stays in frame
        lam, pix, B = lam_at(5 * unit, GAL[b], tuple(p))
        return lam if basis.cut == 0 and len(pix) >= 50 else 1e12
    res = minimize(obj, np.array(pose), method="Nelder-Mead",
                   options=dict(initial_simplex=np.array(pose) + np.vstack([np.zeros(6), np.diag([.04, .04, .04, .05, .05, .02])]),
                                maxiter=160, xatol=1e-3, fatol=1e-3))
    poseB = tuple(res.x)
    pixA, BA = basis(W, A, pose)
    pixB, BB = basis(W, GAL[b], poseB)
    out = dict(t=t, a=a, b=b, rows=[])
    for Nbar in NBARS:
        Phi = Nbar * unit
        pa, lamA = rates(Phi[pixA], BA, TAU)                   # P(accept A | truth A)  = 1 - FRR
        pb, lamB = rates(Phi[pixB], BB, TAU)                   # P(accept B | truth A)  = FAR
        za, zb = [], []
        for r in range(REPS):
            x = sensor(Phi, R)
            Nh, var, ok = inverse(x)
            za.append(score(Nh, ok, pixA, BA)[0]); zb.append(score(Nh, ok, pixB, BB)[0])
        out["rows"].append(dict(Nbar=Nbar, pred_accA=pa, obs_accA=float(np.mean(np.array(za) <= TAU)),
                                pred_accB=pb, obs_accB=float(np.mean(np.array(zb) <= TAU)), lamA=lamA, lamB=lamB))
    # LAW 6 -- the two hypotheses an ideal observer faces: A at its nuisance, B at B's best nuisance (light refit)
    pix = np.intersect1d(pixA, pixB)
    out["law6"] = []
    for Nbar in NBARS[:4]:
        Phi = Nbar * unit
        mB_full = np.zeros_like(Phi)
        mu, w, c = fit(Phi[pixB], BB); mB_full[pixB] = mu
        mA_, mB_ = Phi[pix], np.maximum(mB_full[pix], 1e-9)
        DB = bhattacharyya(mA_, mB_)
        lo, hi = bayes_bounds(DB)
        top = max(mA_.max(), mB_.max())
        xs = np.arange(int(B_DN - 8 * SIG_R * G_DN), int(B_DN + G_DN * (top + 10 * np.sqrt(top + 1) + 8 * SIG_R)) + 1)
        lpa = np.vstack([np.log(pmf_dn(mA_[i:i + 64], xs) + 1e-300) for i in range(0, len(pix), 64)])
        lpb = np.vstack([np.log(pmf_dn(mB_[i:i + 64], xs) + 1e-300) for i in range(0, len(pix), 64)])
        err = 0
        for r in range(200):                                    # oracle LRT, equal priors, 100 draws per hypothesis
            mu_true = mA_ if r % 2 == 0 else mB_
            xx = np.clip(np.rint(B_DN + G_DN * (R.poisson(mu_true) + SIG_R * R.standard_normal(len(pix)))), xs[0], xs[-1]).astype(int) - xs[0]
            llr = lpa[np.arange(len(pix)), xx].sum() - lpb[np.arange(len(pix)), xx].sum()
            err += (llr < 0) if r % 2 == 0 else (llr >= 0)
        out["law6"].append(dict(Nbar=Nbar, DB=DB, lower=lo, upper=hi, oracle_err=err / 200))
    return out


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    t0 = time.time()
    with Pool(12) as p:
        res = p.map(trial, range(T))
    json.dump(res, open("noise_results.json", "w"), indent=1)
    print("%d trials in %.0f s" % (T, time.time() - t0))
    print("\nLAW 5  tau = %.3f   (accept iff z <= tau);  predicted vs measured, mean over trials (+- 2 s.e.)" % TAU)
    print("  Nbar   FRR pred   FRR meas            FAR pred   FAR meas            lambda_A   lambda_B")
    for i, Nbar in enumerate(NBARS):
        rr = [r["rows"][i] for r in res]
        fp, fm = 1 - np.mean([r["pred_accA"] for r in rr]), 1 - np.mean([r["obs_accA"] for r in rr])
        ap, am = np.mean([r["pred_accB"] for r in rr]), np.mean([r["obs_accB"] for r in rr])
        n = len(rr) * REPS
        print("  %4d   %7.3f   %7.3f +- %.3f    %7.3f   %7.3f +- %.3f   %8.1f   %8.1f" % (
            Nbar, fp, fm, 2 * np.sqrt(max(fm * (1 - fm), 1e-4) / n), ap, am, 2 * np.sqrt(max(am * (1 - am), 1e-4) / n),
            np.median([r["lamA"] for r in rr]), np.median([r["lamB"] for r in rr])))
    print("\nLAW 6  exact Bhattacharyya bounds on ANY method's error vs the oracle likelihood-ratio test (200 draws)")
    print("  Nbar   D_B (median)   lower bound   oracle error   upper bound   violations of lower <= err")
    for i, Nbar in enumerate(NBARS[:4]):
        ll = [r["law6"][i] for r in res]
        lo, er, hi = np.mean([l["lower"] for l in ll]), np.mean([l["oracle_err"] for l in ll]), np.mean([l["upper"] for l in ll])
        viol = sum(l["oracle_err"] < l["lower"] - 2 * np.sqrt(l["lower"] * (1 - l["lower"]) / 200) for l in ll)
        print("  %4d   %10.2f   %10.4f   %10.4f   %10.4f      %d of %d" % (Nbar, np.median([l["DB"] for l in ll]), lo, er, hi, viol, len(ll)))
