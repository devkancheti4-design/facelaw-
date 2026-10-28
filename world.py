"""THE DARK-FACE WORLD -- exact by construction; every law is checked against it.
FACE    canonical points q on a 32 x 32 grid over (u, v) in [-1, 1]^2 inside the ellipse (u/0.82)^2 + v^2 <= 1.
        shape  z(u, v) = mean face + identity bumps (12) + expression bumps (3, support EXACTLY v < -0.15)
        albedo rho(u, v) = mean skin + brows/lips/eyes + identity bumps (16) + tone shift, clipped to [0.08, 0.95]
        head frame P = (u, v, -z): the camera looks along +Z, so the face looks along -Z
POSE    R (yaw, pitch, roll), t = (tx, ty, Z0 = 8), f = 160 px; perspective onto a 64 x 64 sensor
LIGHT   distant; 'sh' = an exact 9-term spherical-harmonic field, 'point' = ambient + max(0, n.l) (attached shadows)
SENSOR  N ~ Poisson(Phi), e = N + s_r Z, x = clip(rint(b + g e), 0, W); b = 64, g = 2.7 DN/e-, s_r = 1.5 e-, W = 1023"""
import numpy as np

G, UP, IMG, F_PX, Z0 = 32, 3, 64, 160.0, 8.0
B_DN, G_DN, SIG_R, W_DN = 64, 2.7, 1.5, 1023
EXPR_V = -0.15                       # expression support is v < EXPR_V, exactly
RIGID_ROWS = 18                      # canonical rows r = 0..17 have v >= -0.097 > EXPR_V


def grid(n):
    lin = np.linspace(-1, 1, n)
    u, v = np.meshgrid(lin, lin[::-1])           # row 0 is the top of the face (v = +1)
    return u, v


def in_face(u, v):
    return (u / 0.82) ** 2 + v ** 2 <= 1


def _g(u, v, cu, cv, w2):
    return np.exp(-((u - cu) ** 2 + (v - cv) ** 2) / w2)


class World:
    def __init__(self, seed=0):
        R = np.random.default_rng(seed)
        self.sc = R.uniform(-0.75, 0.75, (12, 2))
        self.ac = R.uniform(-0.75, 0.75, (16, 2))

    def person(self, R):
        return dict(a=R.normal(0, 0.05, 12), b=R.normal(0, 0.07, 16), t=float(R.normal(0, 0.08)))

    # ---------------------------------------------------------------- the face
    def z(self, P, u, v, E=(0, 0, 0)):
        z = 0.55 * np.sqrt(np.clip(1 - (u / 0.9) ** 2 - (v / 1.1) ** 2, 0, 1))
        z = z + 0.22 * np.exp(-(u ** 2 / 0.015 + (v + 0.05) ** 2 / 0.06))
        z = z - 0.07 * (_g(u, v, 0.36, 0.22, 0.012) + _g(u, v, -0.36, 0.22, 0.012))
        z = z + 0.05 * np.exp(-(u ** 2 / 0.05 + (v + 0.5) ** 2 / 0.01))
        for k in range(12):
            z = z + P["a"][k] * _g(u, v, self.sc[k, 0], self.sc[k, 1], 2 * 0.28 ** 2)
        H = np.clip((EXPR_V - v) / 0.25, 0, 1) ** 2   # exactly 0 for v >= EXPR_V
        chi = (-0.08 * np.exp(-(u ** 2 / 0.03 + (v + 0.55) ** 2 / 0.01)),
               0.04 * (_g(u, v, 0.35, -0.35, 0.02) + _g(u, v, -0.35, -0.35, 0.02)),
               -0.05 * np.exp(-(u ** 2 / 0.3 + (v + 0.85) ** 2 / 0.03)))
        for m in range(3):
            z = z + E[m] * H * chi[m]
        return z

    def rho(self, P, u, v):
        r = 0.55 + P["t"] + 0 * u
        r = r - 0.28 * (np.exp(-((u - 0.33) ** 2 / 0.02 + (v - 0.45) ** 2 / 0.002))
                        + np.exp(-((u + 0.33) ** 2 / 0.02 + (v - 0.45) ** 2 / 0.002)))
        r = r - 0.15 * (_g(u, v, 0.36, 0.25, 0.004) + _g(u, v, -0.36, 0.25, 0.004))
        r = r - 0.12 * np.exp(-(u ** 2 / 0.04 + (v + 0.52) ** 2 / 0.004))
        for k in range(16):
            r = r + P["b"][k] * _g(u, v, self.ac[k, 0], self.ac[k, 1], 2 * 0.18 ** 2)
        return np.clip(r, 0.08, 0.95)

    def normal(self, P, u, v, E=(0, 0, 0), h=1e-4):
        zu = (self.z(P, u + h, v, E) - self.z(P, u - h, v, E)) / (2 * h)
        zv = (self.z(P, u, v + h, E) - self.z(P, u, v - h, E)) / (2 * h)
        n = np.stack([-zu, -zv, -np.ones_like(zu)], -1)   # toward the camera (-Z)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)


def rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def sh9(n):
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1], -1)


def sh18(n):
    """the light law one order further: the clamped-cosine (attached-shadow) kernel is non-zero at l = 0, 1, 2, 4, 6...
    and exactly zero at l = 3, so the next physical terms are the 9 of l = 4"""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    l4 = np.stack([x * y * (x * x - y * y), y * z * (3 * x * x - y * y), x * y * (7 * z * z - 1), y * z * (7 * z * z - 3),
                   35 * z ** 4 - 30 * z * z + 3, x * z * (7 * z * z - 3), (x * x - y * y) * (7 * z * z - 1),
                   x * z * (x * x - 3 * y * y), x ** 4 - 6 * x * x * y * y + y ** 4], -1)
    return np.concatenate([sh9(n), l4], -1)


def samples(n, jitter=0.0, seed=0):
    """dense surface samples used to render; the PROBE uses an independent, jittered set, so the recogniser never
    shares the world's rasterisation (no inverse crime)"""
    U, V = grid(n)
    if jitter:
        R = np.random.default_rng(seed)
        U = U + R.uniform(-jitter, jitter, U.shape); V = V + R.uniform(-jitter, jitter, V.shape)
    M = in_face(U, V)
    return U[M], V[M]


du, dv = samples((G - 1) * UP + 1)                          # the recogniser's samples
pu, pv = samples((G - 1) * 8 + 1, jitter=0.4 * 2 / ((G - 1) * 8), seed=7)   # the world's (probe) samples: ~4x denser


def render(W, P, pose, E=(0, 0, 0), ortho=False, z_override=None, n_override=None, rho_override=None, S=None):
    """A pixel INTEGRATES the radiance over its area: every front-most sample that lands in a pixel is averaged.
    -> A (pixels x samples, the averaging operator), camera-frame normals n, albedo rho, and per pixel:
       'full' -- covered and all 8 neighbours covered (no face/background mixing), 'minv' -- the lowest v it sees.
    pose = (yaw, pitch, roll, tx, ty, s) with s scaling f (the scale nuisance)."""
    from scipy.sparse import csr_matrix
    from scipy.ndimage import binary_erosion
    du, dv = (globals()["du"], globals()["dv"]) if S is None else S
    yaw, pitch, roll, tx, ty, s = pose
    Rm = rot(yaw, pitch, roll)
    z = W.z(P, du, dv, E) if z_override is None else z_override
    Ph = np.stack([du, dv, -z], -1)
    Pc = Ph @ Rm.T + np.array([tx, ty, Z0])
    n = (W.normal(P, du, dv, E) if n_override is None else n_override) @ Rm.T
    rho = W.rho(P, du, dv) if rho_override is None else rho_override
    f = F_PX * s
    if ortho:
        px, py = f * Pc[:, 0] / Z0 + IMG / 2, -f * Pc[:, 1] / Z0 + IMG / 2
        front = n[:, 2] < 0
    else:
        px, py = f * Pc[:, 0] / Pc[:, 2] + IMG / 2, -f * Pc[:, 1] / Pc[:, 2] + IMG / 2
        front = np.einsum("ij,ij->i", n, -Pc) > 0
    ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
    ok = front & (ix >= 1) & (ix < IMG - 1) & (iy >= 1) & (iy < IMG - 1)
    idx = np.nonzero(ok)[0]
    pix = iy[idx] * IMG + ix[idx]
    zmin = np.full(IMG * IMG, np.inf); np.minimum.at(zmin, pix, Pc[idx, 2])
    vis = Pc[idx, 2] <= zmin[pix] + 0.1                    # the front-most surface layer of each pixel
    idx = idx[vis]
    # bilinear footprint: each sample spreads over the 4 pixels around it, so the image is CONTINUOUS in the pose
    fx, fy = px[idx] - 0.5, py[idx] - 0.5
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    ax, ay = fx - x0, fy - y0
    rows, cols, wts = [], [], []
    for dx, dy, wt in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)), (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        xx, yy = np.clip(x0 + dx, 0, IMG - 1), np.clip(y0 + dy, 0, IMG - 1)
        rows.append(yy * IMG + xx); cols.append(idx); wts.append(wt)
    rows, cols, wts = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    tot = np.bincount(rows, weights=wts, minlength=IMG * IMG)
    A = csr_matrix((wts / np.maximum(tot[rows], 1e-12), (rows, cols)), shape=(IMG * IMG, len(du)))
    cnt = (tot > 1e-9).astype(int) * np.bincount(rows, minlength=IMG * IMG)
    minv = np.full(IMG * IMG, np.inf); np.minimum.at(minv, rows, dv[cols])
    pix = rows
    covered = (tot > 0.5 * np.median(tot[tot > 0])).reshape(IMG, IMG)
    full = binary_erosion(covered, np.ones((3, 3), bool)).ravel()
    cut = int((front & ~ok).sum())                           # face samples outside the frame: the face is cut off
    return A, n, rho, dict(full=full, minv=minv, cnt=cnt, cut=cut)


def light(kind, R, normals):
    """normals: the camera-frame normals the light must be physical on (an 'sh' field must be >= 0 there)"""
    if kind == "sh":                                  # exact 9-term field
        while True:
            l = np.concatenate([[1.0], R.normal(0, 0.45, 3), R.normal(0, 0.15, 5)])
            if (sh9(normals) @ l).min() > 0.05:
                return ("sh", l)
    d = R.normal(0, 1, 3); d[2] = -abs(d[2]) - 0.3; d /= np.linalg.norm(d)
    return ("point", (float(R.uniform(0.05, 0.3)), d))


def shade(L, n):
    if L[0] == "sh":
        return sh9(n) @ L[1]
    a, d = L[1]
    return a + (1 - a) * np.maximum(0, n @ d)


def photons(W, P, pose, L, Nbar, E=(0, 0, 0), probe=True):
    """expected photons Phi(p) per pixel, scaled so the face pixels average Nbar (rendered from the world's samples)"""
    A, n, rho, info = render(W, P, pose, E, S=(pu, pv) if probe else None)
    rad = A @ (rho * shade(L, n))
    face = info["cnt"] > 0
    k = Nbar / rad[face].mean()
    return rad * k, info, k


def sensor(Phi, R, g=G_DN, b=B_DN, sr=SIG_R, white=W_DN):
    N = R.poisson(Phi)
    e = N + sr * R.standard_normal(Phi.shape)
    return np.clip(np.rint(b + g * e), 0, white).astype(np.int64)

