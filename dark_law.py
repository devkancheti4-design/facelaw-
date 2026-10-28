"""THE DARK LAW -- designed on Extended Yale B only (PIE stays sealed).
Enrolment: one clean frontal image per person.  Probe: a real image thinned to Nbar photons/pixel through Law 1
(Poisson photons, read noise, gain, black level, rounding), then N^ = (x - b) / g.
Per pixel: the probe's gradient sector (the authored lanes), its SIGNIFICANCE c = min(7, floor(|g| / sigma_g)) where
sigma_g^2 = 2 E_h V(p), V = local photon mean + s_r^2 + 1/(12 g^2) (Law 1) and E_h the gradient kernel's energy --
nothing tuned -- and the gallery's flatness bit.  code = delta | c << 3 | flat << 6  (delta = circular sector gap 0..4)
LAW:  score(person) = sum over pixels of f(code),  f = round(16 ln P(code | same) / P(code | other))  counted here.
GATE: answer only if the margin (best - runner-up) bin is >= 99% right on this design data (Chow); else 'too dark'."""
import json, os, sys, time
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y
from world import G_DN, B_DN, SIG_R, sensor
from laws import SIG2
from yale_gate import sectors

NBARS = (1, 2, 5, 10)
SCALES = (0.75, 1.5, 3.0)                                     # scale space: direction measured at three supports
STRIDE = (1, 2, 4)                                            # each scale votes on a grid matching its support


def grid_idx(h, w):
    return [np.nonzero(((np.arange(h)[:, None] % st) == 0) & ((np.arange(w)[None, :] % st) == 0))[0] if False else
            (np.arange(h)[:, None] * w + np.arange(w)[None, :])[::st, ::st].ravel() for st in STRIDE]
_d = np.zeros((31, 31)); _d[15, 15] = 1
E_H = [float((np.gradient(gaussian_filter(_d, s), axis=1) ** 2).sum()) for s in SCALES]   # gradient kernel energies
MBINS = (0, 32, 64, 128, 256, 512, 1024, 2048)                 # margin bins (score units, 1/16 nat), fixed before counts


def dark(I, Nbar, R):
    Phi = Nbar * I / max(I.mean(), 1e-9)
    Nh = (sensor(Phi, R) - B_DN) / G_DN                        # LAW 1 inverse
    return Nh


def probe_feats(Nh):
    """per scale: sectors of the noisy probe and their significance from Law 1's exact noise (scale k stacked)"""
    V = np.maximum(gaussian_filter(Nh, 2.0), 0).ravel() + SIG2
    S, C = [], []
    for k, sg in enumerate(SCALES):
        G = gaussian_filter(Nh, sg)
        g = np.hypot(np.gradient(G, axis=1), np.gradient(G, axis=0)).ravel()
        ix = grid_idx(*Nh.shape)[k]
        S.append(sectors(Nh, sg)[ix]); C.append(np.minimum(7, np.floor(g / np.sqrt(2 * E_H[k] * V)))[ix])
    return np.concatenate(S).astype(np.int64), np.concatenate(C).astype(np.int64)


def gallery_feats(I):
    S, F = [], []
    for k, sg in enumerate(SCALES):
        G = gaussian_filter(I, sg)
        ix = grid_idx(*I.shape)[k]
        S.append(sectors(I, sg)[ix]); F.append((np.hypot(np.gradient(G, axis=1), np.gradient(G, axis=0)).ravel() < 1.0)[ix])
    return np.concatenate(S).astype(np.int64), np.concatenate(F).astype(np.int64)


SCALE_OF = None
SHAPE = (48, 42)


def codes(ps, pc, gs, gflat):
    global SCALE_OF
    if SCALE_OF is None or len(SCALE_OF) != len(ps):
        sizes = [len(ix) for ix in grid_idx(*SHAPE)]
        SCALE_OF = np.repeat(np.arange(len(SCALES)), sizes)
    d = np.abs(gs - ps[None]); d = np.minimum(d, 8 - d)
    return d | (pc[None] << 3) | (gflat << 6) | (SCALE_OF[None] << 7)       # gallery x (pixels x scales)


def mbin(m):
    return int(np.searchsorted(MBINS, m, side="right") - 1)


if __name__ == "__main__":
    t0 = time.time()
    R = np.random.default_rng(2026)
    ids = Y.IDS
    front = {p: next(im for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")) for p in ids}
    GAL = [gallery_feats(front[p]["I"].reshape(48, 42)) for p in ids]
    gs = np.array([g[0] for g in GAL]); gf = np.array([g[1] for g in GAL])
    same, other = np.ones(512), np.ones(512)
    probes = []
    for p_i, p in enumerate(ids):
        for im in Y.PEOPLE[p]:
            if im is front[p] or im["I"].sum() <= 0:
                continue
            for Nbar in NBARS:
                ps, pc = probe_feats(dark(im["I"].reshape(48, 42), Nbar, R))
                C = codes(ps, pc, gs, gf)
                same += np.bincount(C[p_i], minlength=512)
                other += np.bincount(np.delete(C, p_i, 0).ravel(), minlength=512)
                probes.append((p_i, Nbar, C))
    f = np.clip(np.round(16 * np.log((same / same.sum()) / (other / other.sum()))), -127, 127).astype(int)
    seen = (same + other > 2)
    # the gate: margin bin -> right fraction on the design probes (scored with this same law)
    right = np.zeros(len(MBINS)); tot = np.zeros(len(MBINS)); acc = {N: [] for N in NBARS}
    for p_i, Nbar, C in probes:
        sc = f[C].sum(1)
        o = np.argsort(-sc); b = mbin(sc[o[0]] - sc[o[1]])
        tot[b] += 1; right[b] += o[0] == p_i; acc[Nbar].append(o[0] == p_i)
    gate = [int(tot[b] > 0 and right[b] / tot[b] >= 0.99) for b in range(len(MBINS))]
    print("DESIGN (Yale B, 38 people, 1 clean frontal enrolment, %d dark probes)" % len(probes))
    print("   %-20s " % "DARK LAW (this)" + "  ".join("N=%d %5.1f" % (N, 100 * np.mean(acc[N])) for N in NBARS))
    print("   margin bins %s\n   right fraction %s\n   gate (answer?) %s" % (
        list(MBINS), [round(r / t, 3) if t else None for r, t in zip(right, tot)], gate))
    # the same design probes through the frozen local law and the descriptors (design data: looking is allowed)
    from yale_baselines import gradientface, weberface, tantriggs, angdist, l1
    R2 = np.random.default_rng(2026)
    cmp = {k: {N: [] for N in NBARS} for k in ("local law (frozen)", "Gradientfaces", "Weberfaces", "Tan-Triggs")}
    G0 = {k: np.array([fn(front[p]["I"].reshape(48, 42)) for p in ids]) for k, fn in
          (("Gradientfaces", gradientface), ("Weberfaces", weberface), ("Tan-Triggs", tantriggs))}
    for p_i, p in enumerate(ids):
        for im in Y.PEOPLE[p]:
            if im is front[p] or im["I"].sum() <= 0:
                continue
            for Nbar in NBARS:
                Nh = dark(im["I"].reshape(48, 42), Nbar, R2)
                x = np.maximum(Nh, 0) * (front[p]["I"].mean() / Nbar)
                ps = sectors(Nh).astype(np.int64)
                dd = np.abs(gs[:, :ps.size] - ps[None]); dd = np.minimum(dd, 8 - dd).sum(1)
                cmp["local law (frozen)"][Nbar].append(int(np.argmin(dd)) == p_i)
                for k, fn, dist in (("Gradientfaces", gradientface, angdist), ("Weberfaces", weberface, l1), ("Tan-Triggs", tantriggs, l1)):
                    cmp[k][Nbar].append(int(np.argmin(dist(fn(x), G0[k]))) == p_i)
    for k, v in cmp.items():
        print("   %-20s " % k + "  ".join("N=%d %5.1f" % (N, 100 * np.mean(v[N])) for N in NBARS))
    for k, sg in enumerate(SCALES):
        print("   law table at scale %.2f, rows = significance 0..7, cols = sector gap 0..4 (gallery not flat):" % sg)
        for c in range(8):
            print("     c=%d  %s" % (c, " ".join("%4d" % f[d | c << 3 | k << 7] for d in range(5))))
    json.dump(dict(f=f.tolist(), seen=seen.tolist(), gate=gate, mbins=MBINS, E_H=E_H, nbars=NBARS),
              open("dark_design.json", "w"), indent=1)
    print("%.0f s" % (time.time() - t0))
