"""Phone file -> answer, end to end.  A held-out Yale B face in clutter is photographed by a simulated 12-bit RGGB phone
sensor (black 256, white 4095, gain 2.2 DN/e-, read noise 1.8 e-, channel response R 0.5 / G 1.0 / B 0.4), written as a
real DNG (lossless-JPEG tiles, NoiseProfile tag as phone makers write it), read back by dng.py, turned into a photon map
by Law 1 using ONLY the file's tags, then detected (detect.py), aligned and recognised (the local law)."""
import json, os, sys, tempfile, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import zoom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dng
import detect as D
import detect_test as T
import yale as Y
from yale_gate import sectors

BLACK, WHITE, GAIN, SR = 256, 4095, 2.2, 1.8
RESP = np.array([[0.5, 1.0], [1.0, 0.4]])                      # RGGB response of each CFA site
S = [GAIN / (WHITE - BLACK)] * 3; O = [s * s * SR * SR for s in S]


def shoot(scene, Nbar, R):
    """scene (H x W, any units) -> 12-bit CFA image (2H x 2W) whose 2x2 blocks collect Nbar photons on average"""
    up = np.kron(scene, np.ones((2, 2)))
    resp = np.tile(RESP, (scene.shape[0], scene.shape[1]))
    Phi = up * resp
    Phi *= Nbar / (Phi.reshape(scene.shape[0], 2, scene.shape[1], 2).sum((1, 3)).mean())
    e = R.poisson(Phi) + SR * R.standard_normal(Phi.shape)
    return np.clip(np.rint(BLACK + GAIN * e), 0, WHITE).astype(np.int64), Phi


def run(args):
    t, Nbar, comp = args
    R = np.random.default_rng(700000 + 1000 * Nbar + t)
    img, truth = T.scene(T.TEST, R)
    cfa, Phi = shoot(img, Nbar, R)
    path = os.path.join(tempfile.mkdtemp(), "shot.dng")
    dng.write(path, cfa, BLACK, WHITE, S, O, compression=comp, tile=(256, 256) if comp == 7 else None)
    back, meta = dng.read(path)
    L = dng.law1(back, meta)
    x = L["photon"]
    out = dict(Nbar=Nbar, comp=comp, identical=bool(np.array_equal(back, cfa)), g=L["g"], sr=L["sr"],
               bias=float((x - Phi.reshape(x.shape[0], 2, x.shape[1], 2).sum((1, 3))).mean()), face=truth is not None)
    nb = float(np.mean(x))
    lvl = min(TAU, key=lambda N: abs(np.log(max(nb, 0.1) / N)))
    a1, b2 = TAU[lvl]
    dets = [d for d in D.detect(x, UF, UB, a1) if T.verify(x, d, SG) <= b2]
    out["fp"] = len(dets) if truth is None else sum(D.iou(d, (0,) + truth[1:]) < 0.5 for d in dets)
    if truth is not None:
        p, r0, c0, h, w = truth
        hit = [d for d in dets if D.iou(d, (0, r0, c0, h, w)) >= 0.5]
        out["found"] = bool(hit)
        out["right"] = False
        if hit:
            a = D.align(x, hit[0], UF, UB)
            if a is not None:
                rr, cc, hh, ww = a[2]; best = np.full(len(T.TEST), np.inf)
                for ds in (0.96, 1.0, 1.04):
                    for dr in (-2, -1, 0, 1, 2):
                        for dc in (-2, -1, 0, 1, 2):
                            h2, w2 = hh * ds, ww * ds
                            crop = D.sample(x, rr + dr * hh / 48 - (h2 - hh) / 2, cc + dc * ww / 42 - (w2 - ww) / 2, h2, w2)
                            if crop is None:
                                continue
                            s = sectors(crop).astype(np.int16); dd = np.abs(GAL - s[None])
                            best = np.minimum(best, np.minimum(dd, 8 - dd).sum(1))
                out["right"] = T.TEST[int(best.argmin())] == p
    return out


UF = UB = TAU = GAL = SG = None


def _init(uf, ub, tau, gal, sg):
    global UF, UB, TAU, GAL, SG
    UF, UB, TAU, GAL, SG = uf, ub, tau, gal, sg


if __name__ == "__main__":
    t0 = time.time()
    uf = D.face_basis([T.harmonic(p) for p in T.DESIGN]); ub = D.bg_basis()
    sg = np.array([sectors(im["I"].reshape(48, 42)) for p in T.DESIGN for im in Y.PEOPLE[p] if Y.subset(im["theta"]) <= 4], np.int16)
    tau = {int(k): tuple(v) for k, v in json.load(open("detect_results.json"))["tau"].items()}
    gal = np.array([sectors(next(im["I"].reshape(48, 42) for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm"))) for p in T.TEST], np.int16)
    jobs = [(t, N, 7) for N in (2, 5, 10, 50) for t in range(45)] + [(t, 10, 1) for t in range(12)]
    with Pool(12, initializer=_init, initargs=(uf, ub, tau, gal, sg)) as pool:
        res = pool.map(run, jobs)
    print("DNG round trip: %d files, every pixel identical: %s" % (len(res), all(r["identical"] for r in res)))
    print("Law 1 from the tags: gain %.4f DN/e- (true %.1f), read noise %.4f e- (true %.1f), photon-map bias %+.4f"
          % (np.mean([r["g"] for r in res]), GAIN, np.mean([r["sr"] for r in res]), SR, np.mean([r["bias"] for r in res])))
    print("\nphone DNG (lossless-JPEG tiles) -> photon map -> detect -> align -> recognise   (held-out people)")
    print("  photons per 2x2 block   faces found   empty frames that fire   recognised")
    for N in (2, 5, 10, 50):
        rr = [r for r in res if r["Nbar"] == N and r["comp"] == 7]
        f = [r for r in rr if r["face"]]; e = [r for r in rr if not r["face"]]
        print("        %4d               %5.1f%%            %5.1f%%             %5.1f%%" % (
            N, 100 * np.mean([r["found"] for r in f]), 100 * np.mean([r["fp"] > 0 for r in e]), 100 * np.mean([r["right"] for r in f])))
    json.dump(res, open("dng_results.json", "w"), indent=1, default=float)
    print("%.0f s" % (time.time() - t0))
