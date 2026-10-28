"""Detection + alignment + recognition on composite scenes.  Real Yale B faces (full-resolution crops, real lighting)
pasted at random places and sizes into 240 x 320 frames of clutter; 1 in 3 frames has no face; the frame is thinned to
Nbar photons/pixel through Law 1.  The face subspace is built from the DESIGN people (even index); the threshold tau is
the 99th percentile of the best background score in face-free DESIGN frames (<= 1% of empty frames fire); everything
is then measured on the OTHER people (odd index).  Recognition: the synthesis's local law on the 48 x 42 crop against one
clean frontal photo per test person, on the detected crop and on the true crop."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y
import yale_cast as C
import detect as D
from world import sh9, sensor, B_DN, G_DN
from yale_baselines import load_full
from yale_gate import sectors

FULL = {p: [(sb, I.astype(np.uint8)) for sb, I in v] for p, v in load_full().items()}   # 8-bit, as stored
DESIGN = [p for i, p in enumerate(Y.IDS) if i % 2 == 0]
TEST = [p for i, p in enumerate(Y.IDS) if i % 2 == 1]
NBARS = (2, 5, 10, 50)
H, W = 240, 320


def harmonic(p):
    rho, n, Z = C.shape(Y.PEOPLE[p])
    return rho[:, None] * sh9(n)


def background(R):
    kind = R.integers(3)
    rr, cc = np.mgrid[0:H, 0:W] / max(H, W)
    if kind == 0:                                             # smooth wall: gradient + low-frequency shading
        b = 60 + 40 * R.normal() * rr + 40 * R.normal() * cc + gaussian_filter(R.normal(0, 30, (H, W)), 25) * 6
    elif kind == 1:                                           # 1/f texture
        f = np.fft.fftfreq(H)[:, None] ** 2 + np.fft.fftfreq(W)[None, :] ** 2
        b = np.real(np.fft.ifft2(np.fft.fft2(R.normal(0, 1, (H, W))) / np.maximum(np.sqrt(f), 1 / max(H, W))))
        b = 70 + 30 * b / b.std()
    else:                                                     # clutter: random boxes and discs
        b = np.full((H, W), R.uniform(30, 90))
        for _ in range(25):
            r0, c0 = R.integers(0, H), R.integers(0, W); hh, ww = R.integers(10, 80), R.integers(10, 80)
            if R.random() < 0.5:
                b[r0:r0 + hh, c0:c0 + ww] = R.uniform(10, 160)
            else:
                m = (np.mgrid[0:H, 0:W][0] - r0) ** 2 / hh ** 2 + (np.mgrid[0:H, 0:W][1] - c0) ** 2 / ww ** 2 < 1
                b[m] = R.uniform(10, 160)
        b = gaussian_filter(b, 1.5)
    return np.clip(b, 1, 250)


def scene(people, R):
    img = background(R)
    if R.random() < 1 / 3:
        return img, None
    p = people[int(R.integers(len(people)))]
    ims = [I for sb, I in FULL[p] if sb <= 4]
    I = ims[int(R.integers(len(ims)))]
    h = int(R.uniform(40, 130)); w = int(round(h * 168 / 192))
    face = zoom(I.astype(float), (h / 192, w / 168), order=1)
    r0, c0 = int(R.integers(0, H - face.shape[0])), int(R.integers(0, W - face.shape[1]))
    img[r0:r0 + face.shape[0], c0:c0 + face.shape[1]] = face
    return img, (p, r0, c0, face.shape[0], face.shape[1])


def verify(x, d, SG):
    """stage 2: the local-law distance of the proposal to the nearest DESIGN face pattern (per pixel, 0..4)"""
    crop = D.sample(x, d[1], d[2], d[3], d[4])
    if crop is None:
        return 4.0
    s = sectors(crop).astype(np.int16); dd = np.abs(SG - s[None])
    return float(np.minimum(dd, 8 - dd).sum(1).min() / s.size)


def darken(img, Nbar, R):
    Phi = Nbar * img / img.mean()
    return (sensor(Phi, R) - B_DN) / G_DN


def run(args):
    which, t, Nbar = args
    R = np.random.default_rng(1000 * Nbar + t + {"design": 0, "design_empty": 250000, "test": 500000}[which])
    img, truth = scene(DESIGN if which.startswith("design") else TEST, R)
    if which == "design_empty":
        img, truth = background(R), None
    x = darken(img, Nbar, R)
    if which.startswith("design"):
        dets = D.detect(x, UF, UB, -np.inf)
        cand = []
        for d in dets[:12]:
            is_face = truth is not None and D.iou(d, (0,) + truth[1:]) >= 0.5
            cand.append((d[0], verify(x, d, SG), is_face))
        return dict(face=truth is not None, cand=cand, Nbar=Nbar)
    nb = float(np.mean(x))                                              # LAW 1: photons per pixel, measured
    lvl = min(TAU, key=lambda N: abs(np.log(max(nb, 0.1) / N)))
    a1, b2 = TAU[lvl]
    dets = [d for d in D.detect(x, UF, UB, a1) if verify(x, d, SG) <= b2]
    out = dict(face=truth is not None, n_det=len(dets), Nbar=Nbar)
    if truth is None:
        out["fp"] = len(dets)
        return out
    p, r0, c0, h, w = truth
    hit = [d for d in dets if D.iou(d, (0, r0, c0, h, w)) >= 0.5]
    out["fp"] = len(dets) - len(hit)
    out["found"] = bool(hit)
    gal = GAL
    def who(crop):
        s = sectors(crop).astype(np.int16); dd = np.abs(gal - s[None]); return TEST[int(np.minimum(dd, 8 - dd).sum(1).argmin())]
    def who_profiled(box):
        """each candidate takes its best small shift / scale (alignment profiled per person, like pose)"""
        rr, cc, hh, ww = box
        best = np.full(len(TEST), np.inf)
        for ds in (0.96, 1.0, 1.04):
            for dr in (-2, -1, 0, 1, 2):
                for dc in (-2, -1, 0, 1, 2):
                    h2, w2 = hh * ds, ww * ds
                    crop = D.sample(x, rr + dr * hh / 48 - (h2 - hh) / 2, cc + dc * ww / 42 - (w2 - ww) / 2, h2, w2)
                    if crop is None:
                        continue
                    s = sectors(crop).astype(np.int16); dd = np.abs(gal - s[None])
                    best = np.minimum(best, np.minimum(dd, 8 - dd).sum(1))
        return TEST[int(best.argmin())]
    true_crop = D.sample(x, r0, c0, h, w)
    out["right_true_crop"] = true_crop is not None and who(true_crop) == p
    if hit:
        a = D.align(x, hit[0], UF, UB)
        out["right_detected"] = a is not None and who_profiled(a[2]) == p
        out["box_err"] = float(np.hypot(a[2][0] - r0, a[2][1] - c0) / h) if a else None
    else:
        out["right_detected"] = False
    return out


UF = UB = TAU = GAL = SG = None


def _init(uf, ub, tau, gal, sg):
    global UF, UB, TAU, GAL, SG
    UF, UB, TAU, GAL, SG = uf, ub, tau, gal, sg


if __name__ == "__main__":
    t0 = time.time()
    UF = D.face_basis([harmonic(p) for p in DESIGN]); UB = D.bg_basis()
    SG = np.array([sectors(im["I"].reshape(48, 42)) for p in DESIGN for im in Y.PEOPLE[p] if Y.subset(im["theta"]) <= 4], np.int16)
    with Pool(12, initializer=_init, initargs=(UF, UB, None, None, SG)) as pool:
        des = pool.map(run, [("design", t, N) for N in NBARS for t in range(60)])
        des += pool.map(run, [("design_empty", t, N) for N in NBARS for t in range(200)])
    TAU = {}
    for N in NBARS:                      # the two thresholds: most DESIGN faces found with <= 1% of empty frames firing
        dd = [d for d in des if d["Nbar"] == N]
        empty = [d for d in dd if not d["face"]]; faces = [d for d in dd if d["face"]]
        best = None
        for a1 in np.quantile([c[0] for d in dd for c in d["cand"]], np.linspace(0, 0.95, 20)):
            for b2 in np.linspace(0.6, 2.0, 29):
                fire = np.mean([any(c[0] >= a1 and c[1] <= b2 for c in d["cand"]) for d in empty])
                if fire > 0.01:
                    continue
                rec = np.mean([any(c[0] >= a1 and c[1] <= b2 and c[2] for c in d["cand"]) for d in faces])
                if best is None or rec > best[0]:
                    best = (rec, float(a1), float(b2))
        TAU[N] = best[1:]
        print("   %3d photons: light-subspace score >= %.3f and local-law distance <= %.2f  (design recall %.1f%%, empty firing <= 1%%)"
              % (N, best[1], best[2], 100 * best[0]))
    front = {p: next(I for sb, I in FULL[p] if sb == 1) for p in TEST}      # one clean near-frontal photo each
    GAL = np.array([sectors(zoom(next(im["I"].reshape(48, 42) for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")), 1)) for p in TEST], np.int16)
    with Pool(12, initializer=_init, initargs=(UF, UB, TAU, GAL, SG)) as pool:
        res = pool.map(run, [("test", t, N) for N in NBARS for t in range(90)])
    print("\nTEST people (never used to build the detector) -- %d scenes per photon level, 1 in 3 without a face" % 90)
    print("  photons/px   faces found   empty frames that fire   extra boxes in face frames   recognised: detected crop   true crop   box error")
    for N in NBARS:
        rr = [r for r in res if r["Nbar"] == N]
        f = [r for r in rr if r["face"]]
        e = [r for r in rr if not r["face"]]
        print("     %4d         %5.1f%%            %5.1f%%                   %5.2f                      %5.1f%%         %5.1f%%      %5.3f" % (
            N, 100 * np.mean([r["found"] for r in f]), 100 * np.mean([r["fp"] > 0 for r in e]), np.mean([r["fp"] for r in f]),
            100 * np.mean([r["right_detected"] for r in f]), 100 * np.mean([r["right_true_crop"] for r in f]),
            np.median([r["box_err"] for r in f if r.get("box_err") is not None] or [np.nan])))
    json.dump(dict(tau={str(k): v for k, v in TAU.items()}, res=res), open("detect_results.json", "w"), indent=1, default=float)
    print("%.0f s" % (time.time() - t0))
