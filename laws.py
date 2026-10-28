"""THE SIX LAWS, as computation.  Every quantity is closed form or exact enumeration; nothing is fitted to outcomes.
L1  N^ = (x - b) / g,   Var N^ = Phi + s_r^2 + 1/(12 g^2)          (b, s_r: masked pixels;  g: photon-transfer slope)
L2  Phi(p) = k rho(q) (l . Y(n(q)))  ->  y lies in span B,  B_j(p) = rho(q(p)) Y_j(n(q(p)));   T(y) = (I - P_B) y
L3  p = pi(R S(q) + t);   compare in canonical q  (render the candidate at the pose, never warp the noisy image)
L4  D = the 9-dim subspace span{rho Y_j(n)} on the rigid region (v >= -0.15), in canonical coordinates
L5  z = (R - m) / sd0,  R = sum w (y - P_B y)^2,  m = |F| - 9;   FRR = 1 - Phi(tau),  FAR from lambda(N)
L6  D_B = -ln BC  (exact Poisson-Gaussian);   Bayes error >= (1 - sqrt(1 - BC^2)) / 2;   "too dark" below N_min"""
import numpy as np
from scipy.stats import norm
from world import (render, sh9, du, dv, EXPR_V, SIG_R, G_DN, B_DN, W_DN, IMG)

SIG2 = SIG_R ** 2 + 1 / (12 * G_DN ** 2)        # read noise + quantisation, in e-^2


# ------------------------------------------------------------------ LAW 1 --
def inverse(x, g=G_DN, b=B_DN, white=W_DN):
    """-> photon estimate, its variance, and the valid mask (saturated pixels carry no number)"""
    Nh = (x - b) / g
    return Nh, np.maximum(Nh, 0) + SIG2, x < white


# ------------------------------------------------------------- LAWS 2-4 ----
def basis(W, P, pose):
    """the candidate's 9 basis images on its rigid, visible pixels (Law 2 + Law 3 + Law 4)"""
    A, n, rho, info = render(W, P, pose)
    pix = np.nonzero(info["full"] & (info["minv"] > EXPR_V + 0.03))[0]    # whole footprint on the rigid face
    basis.cut = info["cut"]                                                # > 0: the pose puts the face out of frame
    return pix, (A @ (rho[:, None] * sh9(n)))[pix]


def fit(y, B, iters=4, alpha=0.0):
    """weighted projection onto span B; weights 1/(mu + SIG2 + (alpha mu)^2) from the fit itself (IRLS).
    alpha = the model's relative leftover (Law 2's measured bound), entered as variance so a likelihood stays TRUE
    when the 9-term light law is not exact.  -> mu, w, c"""
    m0 = max(y.mean(), 0)
    w = np.full(len(y), 1 / (m0 + SIG2 + (alpha * m0) ** 2))
    for _ in range(iters):
        sw = np.sqrt(w)
        c = np.linalg.lstsq(B * sw[:, None], y * sw, rcond=None)[0]
        mu = B @ c
        mp = np.maximum(mu, 0)
        w = 1 / (mp + SIG2 + (alpha * mp) ** 2)
    return mu, w, c


def score(Nh, valid, pix, B):
    """Law 5's statistic for one candidate: z = (R - m) / sd0"""
    keep = valid[pix]
    y, Bk = Nh[pix][keep], B[keep]
    mu, w, c = fit(y, Bk)
    R = float((w * (y - mu) ** 2).sum())
    m = len(y) - 9
    sd0 = float(np.sqrt(2 * m + (w ** 2 * np.maximum(mu, 0)).sum()))
    return (R - m) / sd0, dict(R=R, m=m, sd0=sd0, c=c)


# ------------------------------------------------------------------ LAW 5 --
def predict_z(muA, B):
    """the distribution of z when the truth is muA (expected photons on the candidate's pixels) and the candidate's
    subspace is span B: exact first two moments of R under Poisson + read noise + quantisation."""
    mu, w, c = fit(muA, B)                               # the noiseless fit = the fixed point the noisy fit centres on
    sw = np.sqrt(w)
    A = B * sw[:, None]
    Hm = A @ np.linalg.pinv(A)                           # weighted hat matrix
    r = sw * (muA - mu)                                  # the deterministic leftover (impostor / model error)
    lam = float(r @ r)
    S = (np.maximum(muA, 0) + SIG2) * w                  # whitened noise variances
    Mres = np.eye(len(muA)) - Hm
    ER = lam + float(np.trace(Mres * S[None, :]))      # E[R] = lambda + tr((I-H) S)
    k4 = np.maximum(muA, 0) * w ** 2                     # excess fourth cumulant of the whitened noise
    VR = 2 * float(np.sum((Mres * S[None, :]) * (Mres * S[None, :]).T)) + float((np.diag(Mres) ** 2 * k4).sum()) \
        + 4 * float(((Mres @ r) ** 2 * S).sum())
    m = len(muA) - 9
    sd0 = float(np.sqrt(2 * m + (w ** 2 * np.maximum(mu, 0)).sum()))
    return (ER - m) / sd0, np.sqrt(VR) / sd0, lam


def rates(muA, B, tau):
    """P(z <= tau) -- the accept probability: FAR for an impostor subspace, 1 - FRR for the true one"""
    mz, sz, lam = predict_z(muA, B)
    return float(norm.cdf((tau - mz) / sz)), lam


# ------------------------------------------------------------------ LAW 6 --
def pmf_dn(mu, xs, g=G_DN, b=B_DN, sr=SIG_R):
    """exact P(x = xs | Phi = mu) per pixel: Poisson photons, Gaussian read noise, rounding to DN"""
    kmax = int(np.ceil(mu.max() + 8 * np.sqrt(mu.max() + 1) + 10))
    k = np.arange(kmax + 1)
    from scipy.stats import poisson
    pk = poisson.pmf(k[None, :], mu[:, None])                        # pixels x k
    lo = (xs[None, None, :] - 0.5 - b) / g - k[None, :, None]       # e-bin edges for each DN value, per k
    hi = lo + 1 / g
    pb = norm.cdf(hi / sr) - norm.cdf(lo / sr)                       # 1 x k x DN
    return np.einsum("pk,pkx->px", pk, pb)


def bhattacharyya(muA, muB):
    """exact -ln BC between the two images' DN distributions (independent pixels)"""
    top = max(muA.max(), muB.max())
    xs = np.arange(int(B_DN - 8 * SIG_R * G_DN), int(B_DN + G_DN * (top + 10 * np.sqrt(top + 1) + 8 * SIG_R)) + 1)
    DB = 0.0
    for i in range(0, len(muA), 64):
        pa, pb = pmf_dn(muA[i:i + 64], xs), pmf_dn(muB[i:i + 64], xs)
        DB += -np.log(np.sqrt(pa * pb).sum(1)).sum()
    return DB


def bayes_bounds(DB):
    BC = np.exp(-DB)
    return 0.5 * (1 - np.sqrt(1 - BC ** 2)), 0.5 * BC          # lower, upper bound on the error of ANY method
