"""THE GENERIC ALIGNER (identity-blind): the eyes are found by sliding the law's map of the MEAN frontal lab face over a
window around a guide (where the face should be), so the owner's score at the chosen placement is not inflated by a search
for the owner.  Coarse grid (1 canonical px, eye distance in 6% steps, tilt), then two refinement rounds.
    python3 -I align.py      error on the user's photos (guides deliberately off) and on Extended Yale B (clean, dark)"""
import json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import phone_eyes as PE
import finder as FD

HERE = os.path.dirname(os.path.abspath(__file__))
WIN, SRANGE, TILTS = 0.20, 1.25, (-8.0, -4.0, 0.0, 4.0, 8.0)   # guide tolerance: +-20% eye distance, size x/÷1.25, tilt


def generic_map():
    import yale as Y
    front = np.array([next(im["I"] for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")).reshape(48, 42) for p in Y.IDS])
    M = front.mean(0)
    return [np.array([v]) for v in DL.gallery_feats(M * (front.mean() / M.mean()))]


def align(lin, mid0, ed0, F, gmap, B=None):
    """the placement (eye midpoint, eye distance, tilt) whose law score against the mean-face map is highest -> (eL, eR)"""
    cL, cR = PE.canonical_eyes(); ced = float(np.linalg.norm(cR - cL)); midc = (cL + cR) / 2
    B = B or max(1, int(ed0 / ced / 4))                   # box-integrate so the crop samples ~4 px apart
    img = FD.boxdown(lin, B); cache = {}
    yy, xx = np.mgrid[0:48, 0:42].astype(float); qx, qy = xx.ravel() - midc[0], yy.ravel() - midc[1]
    def score(mx, my, s, th):
        sg = round(4 * np.sqrt(max((0.5 * (B * s - 1)) ** 2 - (B * B - 1) / 12.0, 0.0)) / B) / 4
        if sg not in cache:
            cache[sg] = gaussian_filter(img, sg)
        c, n = np.cos(th), np.sin(th)
        x = map_coordinates(cache[sg], [s * (n * qx + c * qy) + my, s * (c * qx - n * qy) + mx], order=1, mode="nearest").reshape(48, 42)
        ps, pc = U.probe_feats(x)
        return float(F[DL.codes(ps, pc, *gmap)].sum() / len(ps))
    u = lambda X: (X - (B - 1) / 2.0) / B
    m0, s0 = np.array([u(mid0[0]), u(mid0[1])]), ed0 / B / ced
    seen = {}
    def visit(pl):
        if pl not in seen:
            seen[pl] = score(*pl)
    for k in np.arange(-np.log(SRANGE), np.log(SRANGE) + 1e-9, np.log(1.06)):
        s = s0 * np.exp(k)
        for dy in np.arange(-WIN * 25, WIN * 25 + 1e-9, 1.0):
            for dx in np.arange(-WIN * 25, WIN * 25 + 1e-9, 1.0):
                for t in TILTS:
                    visit((m0[0] + dx * s, m0[1] + dy * s, s, np.radians(t)))
    for d, f, a in ((0.5, 1.03, 2.0), (0.25, 1.015, 1.0)):
        for mx, my, s, th in sorted(seen, key=seen.get)[-5:]:
            for dy in (-d, 0, d):
                for dx in (-d, 0, d):
                    for ff in (1 / f, 1.0, f):
                        for dt in (-a, 0.0, a):
                            visit((mx + dx * s, my + dy * s, s * ff, th + np.radians(dt)))
    mx, my, s, th = max(seen, key=seen.get)
    v = 0.5 * B * s * ced * np.array([np.cos(th), np.sin(th)]); mid = np.array([B * mx + (B - 1) / 2.0, B * my + (B - 1) / 2.0])
    return mid - v, mid + v, seen[(mx, my, s, th)], len(seen)


if __name__ == "__main__":
    import yale as Y
    F = np.array(json.load(open(os.path.join(HERE, "unlock_lanes.json")))["F"]["table"])
    gmap = generic_map()
    D = os.path.join(HERE, "data", "phone_user")
    HAND = {"IMG_1374": ("IMG_1374_up.jpg", (956, 1731), (1589, 1728)), "photo1": ("photo1.jpg", (572, 1016), (828, 1015)),
            "photo2": ("photo2.jpg", (622, 960), (812, 955))}
    rng = np.random.default_rng(5)
    for nm, (path, l, r) in HAND.items():
        lin = PE.load(os.path.join(D, path)); tl, tr = np.array(l, float), np.array(r, float); ed = np.linalg.norm(tr - tl)
        for off in ((0, 0, 1.0), (0.12, -0.10, 1.15), (-0.10, 0.12, 0.87)):
            mid0 = (tl + tr) / 2 + ed * np.array(off[:2])
            eL, eR, sc, n = align(lin, mid0, ed * off[2], F, gmap)
            print("%-9s guide off by (%+.2f, %+.2f) eye distances, size x%.2f -> error %.3f eye distances  (score %.2f, %d placements)"
                  % (nm, off[0], off[1], off[2], FD.err((eL, eR), (tl, tr)), sc, n), flush=True)
    Bp = 48; tl, tr = np.array([33.5 + Bp, 45.5 + Bp]), np.array([133.5 + Bp, 45.5 + Bp])
    for N in (None, 100, 20):
        errs = []
        for p in Y.IDS[19:27]:
            for im in Y.PEOPLE[p]:
                if im["I"].sum() <= 0 or im["theta"] > 25.5 or rng.random() > 0.35:
                    continue
                I = Y.read_pgm(os.path.join(Y.ROOT, p, im["name"])).astype(float)
                lin = 255 * (I / 255) ** 2.2
                if N is not None:
                    lin = np.maximum(U.dark_ae(lin, N / 16.0, rng), 0)
                lin = np.pad(lin, Bp, mode="edge")
                off = rng.uniform(-0.12, 0.12, 2); sz = np.exp(rng.uniform(-0.15, 0.15))
                eL, eR, sc, n = align(lin, (tl + tr) / 2 + 100 * off, 100 * sz, F, gmap)
                errs.append(FD.err((eL, eR), (tl, tr)))
        e = np.array(errs)
        print("Yale B (8 people never in calibration, light <= 25 deg, guide off by up to 12%%, size up to 15%%), %s: %d faces; error median %.3f, within 0.04 %.0f%%, 0.08 %.0f%%"
              % ("clean" if N is None else "dark %d photons/face px" % N, len(e), np.median(e), 100 * (e <= .04).mean(), 100 * (e <= .08).mean()), flush=True)
