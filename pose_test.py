"""IS IT THE LAW OR THE HARNESS?  The owner's maps (data/webapp/vault.json) against the owner's photos from other sessions,
cameras and head angles, and against lab strangers -- under two harnesses:
    TAPS   the eyes as given, the law's own +-1 px search (what the app does today)
    POSE   the law searches the placement around the eyes: shift +-2 canonical px, width x0.92..1.08, height x0.90..1.10
           (a face seen from below or above is squashed vertically), tilt -5..+5 deg -- 675 placements
Strangers get exactly the same search on full photos (Extended Yale B, 19 people never used for the bars, light <= 25
deg, clean and dark).  OPEN = any map's best score above that map's bar (the bars are today's, set without POSE).
    python3 -I pose_test.py"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import dark_law as DL
import unlock_law as U
import phone_eyes as PE
import finder as FD

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "phone_user")
SHIFT, SX, SY, TILT = (-2, -1, 0, 1, 2), (0.92, 1.0, 1.08), (0.90, 1.0, 1.10), (-5.0, 0.0, 5.0)
G = {}


def init(F, gs, gf, bars, EB, cL, cR):
    DL.SHAPE, DL.SCALE_OF = (48, 42), None
    G.update(F=np.asarray(F), gs=gs, gf=gf, bars=bars, EB=EB, cL=cL, cR=cR)


def cell(pc, x):
    l = U.lbin(U.imbalance(x)); return l * U.NE + U.ebin(U.eshare(pc), G["EB"][l])


def judge(S, c):
    """best margin over maps: score minus that map's bar in this cell (None where a map has no bar)"""
    m = [S[j] - G["bars"][j][c] for j in range(len(S)) if G["bars"][j][c] is not None]
    return max(m) if m else None


def run(task):
    name, lin, eL, eR = task
    cL, cR = G["cL"], G["cR"]; ced = float(np.linalg.norm(cR - cL)); midc = (cL + cR) / 2
    x = PE.crop(lin, eL, eR, cL, cR)                       # TAPS: exactly the app's crop and the law's +-1 px
    S, pc = U.scores(G["F"], x, G["gs"], G["gf"])
    taps = judge(S, cell(pc, x))
    ed = float(np.linalg.norm(eR - eL)); B = max(1, int(ed / ced / 4))
    img = FD.boxdown(lin, B); s0 = ed / B / ced; th0 = np.arctan2(*(eR - eL)[::-1])
    mid0 = ((eL + eR) / 2 - (B - 1) / 2.0) / B
    Gs = gaussian_filter(img, max(0.0, 0.5 * (B * s0 - 1)) / B)
    yy, xx = np.mgrid[0:48, 0:42].astype(float); qx, qy = xx.ravel() - midc[0], yy.ravel() - midc[1]
    best = None
    for th in TILT:
        c, n = np.cos(th0 + np.radians(th)), np.sin(th0 + np.radians(th))
        for sx in SX:
            for sy in SY:
                ax, ay = s0 * sx * qx, s0 * sy * qy
                for dy in SHIFT:
                    for dx in SHIFT:
                        mx, my = mid0[0] + s0 * dx, mid0[1] + s0 * dy
                        xp = map_coordinates(Gs, [n * ax + c * ay + my, c * ax - n * ay + mx], order=1, mode="nearest").reshape(48, 42)
                        ps, pcp = U.probe_feats(xp)
                        Sp = G["F"][DL.codes(ps, pcp, G["gs"], G["gf"])].sum(1) / len(ps)
                        m = judge(Sp, cell(pcp, xp))
                        if m is not None and (best is None or m > best):
                            best = m
    return name, taps, best


if __name__ == "__main__":
    t0 = time.time()
    import yale as Y
    Z3 = json.load(open("unlock_lanes.json"))
    v = json.load(open(os.path.join(HERE, "data", "webapp", "vault.json"))); maps = v["maps"]
    gs = np.array([m["gs"] for m in maps]); gf = np.array([m["gf"] for m in maps]); bars = [m["bars"] for m in maps]
    cL, cR = PE.canonical_eyes()
    owner = [("IMG_1374 (iPhone, Night mode, front)", "IMG_1374_up.jpg", (956, 1731), (1589, 1728)),
             ("IMG_1376 (iPhone, Night mode, HEAD TURNED)", "IMG_1376_up.jpg", (1001, 1710), (1586, 1689)),
             ("photo2 (iPhone, room light, other day)", "photo2.jpg", (622, 960), (812, 955)),
             ("photo1 (iPhone, side light, other day)", "photo1.jpg", (572, 1016), (828, 1015))]
    tasks = [("OWNER " + nm, PE.load(os.path.join(D, p)), np.array(l, float), np.array(r, float)) for nm, p, l, r in owner]
    rng = np.random.default_rng(11); Bp = 48
    tl, tr = np.array([33.5 + Bp, 45.5 + Bp]), np.array([133.5 + Bp, 45.5 + Bp])
    for p in Y.IDS[19:38]:
        ims = [im for im in Y.PEOPLE[p] if im["I"].sum() > 0 and im["theta"] <= 25.5]
        for im in [ims[k] for k in rng.choice(len(ims), min(4, len(ims)), replace=False)]:
            lin = 255 * (Y.read_pgm(os.path.join(Y.ROOT, p, im["name"])).astype(float) / 255) ** 2.2
            for N in (None, 1000, 100):
                x = lin if N is None else np.maximum(U.dark_ae(lin, N / 16.0, rng), 0)
                tasks.append(("stranger", np.pad(x, Bp, mode="edge"), tl, tr))
    with Pool(2, init, (Z3["F"]["table"], gs, gf, bars, Z3["EBINS"], cL, cR)) as pool:
        out = pool.map(run, tasks, chunksize=4)
    print("owner's vault: %d maps from the Mac webcam scan (screen-lit), bars as set today\n" % len(maps))
    print("%-46s %-22s %s" % ("", "TAPS (today)", "POSE search"))
    f = lambda m: "no bar" if m is None else ("OPEN  %+.2f" % m if m > 0 else "below %+.2f" % m)
    for name, a, b in out:
        if name.startswith("OWNER"):
            print("%-46s %-22s %s" % (name[6:], f(a), f(b)))
    st = [(a, b) for name, a, b in out if name == "stranger"]
    A = np.array([a for a, _ in st if a is not None]); Bm = np.array([b for _, b in st if b is not None])
    print("\nSTRANGERS (%d tries, same harness): TAPS opens %d (closest %+.2f) | POSE opens %d (closest %+.2f)"
          % (len(st), int((A > 0).sum()), A.max(), int((Bm > 0).sum()), Bm.max()))
    print("[%.0fs]" % (time.time() - t0))
