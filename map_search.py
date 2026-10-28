"""THE MAP FINDS YOU: no face finder.  The owner's map is slid over the whole photo -- eye midpoint, eye distance, tilt --
and the law scores every placement.  EVERY map in the gallery gets exactly the same search on the same photo, so the
luck of looking in many places is the same for all:
    the owner's map (IMG_1374, the dark scan, hand-read pupils)        -- the one under test
    decoys from the owner's own phone and night: the owner's map MIRRORED; the owner's room-light maps (photo2, photo1)
    106 strangers: Extended Yale B (38, frontal) and CMU PIE (68, the most evenly lit flash)
The law recognises the owner in a photo if the owner's map beats all 106 strangers' maps (chance: 1 in 107).
Search: coarse grid (2 canonical px steps, eye distance in 9.6% steps, tilt -4/0/+4 deg), then two refinement rounds
around every map's best placement (all maps scored at every placement).  Local only; no map is ever shown.
    python3 -I map_search.py           (MODE=null: 3 dark photos as taken, pixel-shuffled and upside down -- the artifact check)"""
import heapq, json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import phone_eyes as PE
import finder as FD

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "phone_user")
B = 4                                                    # box-integrate 4 x 4 first: the crop samples >= 16 px apart
WIN = (0.25, 0.80, 0.30, 0.75)                           # eye-midpoint window, shares of width / height
ANG = np.radians([-4.0, 0.0, 4.0])
STEP = 1.096                                             # eye-distance ratio between searched sizes
TESTS = [("IMG_%d" % n, "IMG_%d_up.jpg" % n, "dark, 1/15 s, ISO 2000, no flash", (0.16, 0.34), "asis") for n in range(1378, 1389)] + [
    ("IMG_1374", "IMG_1374_up.jpg", "CONTROL: the scan photo itself", (0.16, 0.34), "asis"),
    ("IMG_1376", "IMG_1376_up.jpg", "CONTROL: dark Night mode, head turned", (0.16, 0.34), "asis"),
    ("photo2", "photo2.jpg", "CONTROL: room light, other session", (0.11, 0.24), "asis"),
    ("photo1", "photo1.jpg", "CONTROL: room light, other session", (0.11, 0.24), "asis")]
NULLS = [("IMG_%d/%s" % (n, xf), "IMG_%d_up.jpg" % n, "dark photo, %s" % {"asis": "as taken", "scramble": "NULL: pixels shuffled (no face left)",
          "flipud": "NULL: upside down"}[xf], (0.16, 0.34), xf) for n in (1378, 1382, 1385) for xf in ("asis", "scramble", "flipud")]
HAND = {"IMG_1374": ((956, 1731), (1589, 1728)), "photo1": ((572, 1016), (828, 1015)), "photo2": ((622, 960), (812, 955))}
G = {}


def init(F, gs, gf, cL, cR):
    DL.SHAPE, DL.SCALE_OF = (48, 42), None
    G.update(F=np.asarray(F), gs=gs, gf=gf, cL=cL, cR=cR)


def sampler(img):
    cache = {}
    def at(s):                                           # the crop's anti-alias smoothing at s (B-px per canonical px)
        sg = np.sqrt(max((0.5 * (B * s - 1)) ** 2 - (B * B - 1) / 12.0, 0.0)) / B
        k = round(4 * sg) / 4
        if k not in cache:
            cache[k] = gaussian_filter(img, k)
        return cache[k]
    return at


def crop(Gs, mx, my, s, th):
    mid = (G["cL"] + G["cR"]) / 2
    yy, xx = np.mgrid[0:48, 0:42].astype(float)
    qx, qy = xx.ravel() - mid[0], yy.ravel() - mid[1]
    c, n = np.cos(th), np.sin(th)
    return map_coordinates(Gs, [s * (n * qx + c * qy) + my, s * (c * qx - n * qy) + mx], order=1, mode="nearest").reshape(48, 42)


def search(task):
    name, path, _, (elo, ehi), xf = task
    lin = PE.load(os.path.join(D, path)); H, W = lin.shape
    if xf == "scramble":
        lin = np.random.default_rng(1).permutation(lin.ravel()).reshape(H, W)
    elif xf == "flipud":
        lin = lin[::-1].copy()
    img = FD.boxdown(lin, B); del lin
    at = sampler(img)
    ced = float(np.linalg.norm(G["cR"] - G["cL"]))
    nmap = len(G["gs"]); best = np.full(nmap, -np.inf); arg = [None] * nmap; top0 = []; n = 0
    def visit(pl):
        nonlocal n
        mx, my, s, th = pl
        ps, pc = U.probe_feats(crop(at(s), mx, my, s, th))
        S = G["F"][DL.codes(ps, pc, G["gs"], G["gf"])].sum(1) / len(ps)
        n += 1
        up = S > best
        best[up] = S[up]
        for m in np.nonzero(up)[0]:
            arg[m] = pl
        heapq.heappush(top0, (float(S[0]), pl))
        if len(top0) > 5:
            heapq.heappop(top0)
    u = lambda X: (X - (B - 1) / 2.0) / B                 # full-res px -> B-px coordinates
    eds = [elo * W * STEP ** k for k in range(64) if elo * W * STEP ** k <= ehi * W * 1.0001]
    for ed in eds:
        s = ed / B / ced; st = 2 * s
        for my in np.arange(u(WIN[2] * H), u(WIN[3] * H) + 1e-9, st):
            for mx in np.arange(u(WIN[0] * W), u(WIN[1] * W) + 1e-9, st):
                for th in ANG:
                    visit((mx, my, s, th))
    for d, f, a in ((1.0, STEP ** 0.5, np.radians(2.0)), (0.5, STEP ** 0.25, np.radians(1.0))):
        seeds = set(arg) | set(pl for _, pl in top0)
        for mx, my, s, th in seeds:
            for dy in (-d, 0, d):
                for dx in (-d, 0, d):
                    for ff in (1 / f, 1.0, f):
                        for dth in (-a, 0.0, a):
                            visit((mx + dx * s, my + dy * s, s * ff, th + dth))
    full = lambda pl: (B * pl[0] + (B - 1) / 2.0, B * pl[1] + (B - 1) / 2.0, B * pl[2] * ced, float(np.degrees(pl[3])))
    return name, best, [full(pl) for pl in arg], n


def eyes_of(mx, my, ed, deg):
    v = 0.5 * ed * np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    return np.array([mx, my]) - v, np.array([mx, my]) + v


if __name__ == "__main__":
    t0 = time.time()
    import yale as Y, personal as P
    F = np.array(json.load(open(os.path.join(HERE, "unlock_lanes.json")))["F"]["table"])
    front = [next(im["I"] for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")).reshape(48, 42) for p in Y.IDS]
    REF = float(np.mean(front))
    expose = lambda x: x * (REF / max(float(x.mean()), 1e-9))
    cL, cR = PE.canonical_eyes()
    own = {nm: PE.crop(PE.load(os.path.join(D, p)), *[np.array(e, float) for e in HAND[nm]], cL, cR)
           for nm, p in (("IMG_1374", "IMG_1374_up.jpg"), ("photo2", "photo2.jpg"), ("photo1", "photo1.jpg"))}
    pc_, _ = P.pie_crops()
    pie = [pc_[p][int(np.argmin([U.imbalance(x) for x in pc_[p]]))] for p in range(68)]
    maps = [own["IMG_1374"], np.fliplr(own["IMG_1374"]).copy(), own["photo2"], own["photo1"]] + front + pie
    names = ["YOU (IMG_1374 map)", "YOU mirrored (decoy)", "YOU photo2 map", "YOU photo1 map"] + ["Yale " + p for p in Y.IDS] + ["PIE %d" % p for p in range(68)]
    feats = [DL.gallery_feats(expose(x)) for x in maps]
    gs = np.array([a for a, _ in feats]); gf = np.array([b for _, b in feats])
    NS = 4                                               # strangers start here
    print("gallery: %d maps (%d strangers); search on each photo identical for every map  [%.0fs]" % (len(maps), len(maps) - NS, time.time() - t0), flush=True)
    tests = NULLS if os.environ.get("MODE") == "null" else TESTS
    with Pool(3, init, (F, gs, gf, cL, cR)) as pool:
        out = {nm: (b, a, n) for nm, b, a, n in pool.imap_unordered(search, tests, chunksize=1)}
    json.dump(dict(names=names, out={nm: dict(best=b.tolist(), arg=a, n=n) for nm, (b, a, n) in out.items()}),
              open(os.path.join(D, "map_search_%s.json" % os.environ.get("MODE", "main")), "w"))
    wins = {i: 0 for i in range(NS)}
    for nm, _, what, _, _ in tests:
        b, a, n = out[nm]
        st = b[NS:]; k = int(np.argmax(st))
        print("%-18s (%s) %d placements | best stranger %.2f (%s)" % (nm, what, n, st[k], names[NS + k]))
        for i in (0, 3, 2, 1):
            eL, eR = eyes_of(*a[i]); rank = 1 + int((st > b[i]).sum())
            hand = ""
            if nm in HAND and i == 0:
                hand = "  (vs hand-read pupils %.3f eye distances)" % FD.err((eL, eR), tuple(np.array(e, float) for e in HAND[nm]))
            print("    %-21s %5.2f  rank %3d of %d  T-norm %+5.1f  eyes found (%4d, %4d) (%4d, %4d) tilt %+.0f%s" % (
                names[i], b[i], rank, len(st) + 1, (b[i] - st.mean()) / (st.std() + 1e-9), *eL, *eR, a[i][3], hand))
            wins[i] += ("1378" <= nm[4:8] <= "1388") and rank == 1
    print("\nDARK PHOTOS (as tested here): rank 1 of 107 -- %s  [%.0fs]" % ("; ".join("%s %d" % (names[i], wins[i]) for i in (0, 3, 2, 1)), time.time() - t0))
