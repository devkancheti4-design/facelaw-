"""The user's real photos, each in turn as the MAP (eyes found automatically, the scan's exposure set to the lab reference
the F table was authored on -- a camera's auto-exposure; without it a dark scan is 80% 'flat' and maps nothing); the
personal threshold (B: raw score, bars per cell = best calibration lab stranger + margin, authored as integer lanes)
calibrated for THAT map; tries: the user's OTHER photos, as taken (different sessions, different light); a fresh-stranger
check on CMU PIE.  Local only; the map is never shown.
    python3 -I scan_dark.py [margin]          (EYES=hand: the hand-read pupils instead of the finder -- the alignment control)"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import unlock_v4 as V4
import personal as P
import finder as FD
import phone_eyes as PE

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "phone_user")
NAME = {0: "NOT YOU", 1: "OPEN", 2: "NOT NOW"}
PHOTOS = (("IMG_1374", "IMG_1374_up.jpg", "dark room, Night mode 0.5 s, no flash"), ("IMG_1376", "IMG_1376_up.jpg", "dark room, head turned"),
          ("photo1", "photo1.jpg", "room light"), ("photo2", "photo2.jpg", "room light"))
MAPS = ("IMG_1374", "photo2", "photo1")
HAND = {"IMG_1374": ((956, 1731), (1589, 1728)), "photo1": ((572, 1016), (828, 1015)), "photo2": ((622, 960), (812, 955))}   # hand-read pupils


def expose(x, ref):
    """the scan at the lab reference exposure (sectors and probe significance cancel gain; the map's flat rule does not)"""
    return x * (ref / max(float(x.mean()), 1e-9))


if __name__ == "__main__":
    t0 = time.time()
    import yale as Y
    margin = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
    Z3 = json.load(open(os.path.join(HERE, "unlock_lanes.json"))); Z4 = json.load(open(os.path.join(HERE, "unlock_v4_lanes.json")))
    F, EB, D1 = np.array(Z3["F"]["table"]), Z3["EBINS"], Z4["D1"]["value"]
    REF = float(np.mean([next(im["I"] for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")) for p in Y.IDS]))
    cL, cR = PE.canonical_eyes()
    crops, notes = {}, {}
    hand = os.environ.get("EYES") == "hand"
    for nm, path, light in PHOTOS:
        lin = PE.load(os.path.join(D, path))
        f = FD.find(lin)
        if nm in HAND and f is not None:
            print("   %-9s finder vs hand-read pupils: worse eye off by %.3f eye distances" % (nm, FD.err(f["classic"], [np.array(e, float) for e in HAND[nm]])))
        if hand:
            if nm not in HAND:
                notes[nm] = "%s: no hand-read pupils (one eye unseen)" % light; continue
            eyes = [np.array(e, float) for e in HAND[nm]]
        elif f is None:
            notes[nm] = "%s: the finder saw no face with both eyes" % light; continue
        else:
            eyes = f["classic"]
        crops[nm] = PE.crop(lin, *eyes, cL, cR)
        x = crops[nm]; V = float(np.mean(U.measured_var(x)))
        notes[nm] = "%s, ~%d photons/px by its noise" % (light, int(round(x.mean() ** 2 / max(V, 1e-12))))
    R = np.random.default_rng(1374)
    cal = Y.IDS[0:19]
    cal_tries = [U.jitter(U.dark_ae(im["I"].reshape(48, 42), N, R), R) for p in cal for im in Y.PEOPLE[p] if im["I"].sum() > 0
                 for N in U.NBARS for _ in range(2)]
    pie, _ = P.pie_strangers(np.random.default_rng(7))
    pie = [t[2] for t in pie]
    ncell = U.NL * U.NE
    cell_of = lambda sh, l: l * U.NE + U.ebin(sh, EB[l])
    print("EYES: %s" % ("HAND-READ pupils (the alignment control)" if hand else "the automatic finder"))
    print("exposure reference: the Yale B frontal scans, mean %.1f; calibration: %d lab-stranger tries (19 Yale B people, every light, 8 darkness levels); margin %.2f"
          % (REF, len(cal_tries), margin))
    for nm in notes:
        if nm not in crops:
            print("   %-9s %s -> NOT NOW (face the screen), never scored" % (nm, notes[nm]))
    for m in MAPS:
        scan = expose(crops[m], REF)
        gs, gf = (np.array([v]) for v in DL.gallery_feats(scan))
        owner = [nm for nm in crops]
        with Pool(3, V4.init, (F, gs, gf, (48, 42))) as pool:
            oc = pool.map(V4.score_task, cal_tries, chunksize=64)
            op = pool.map(V4.score_task, pie, chunksize=64)
            oo = pool.map(V4.score_task, [crops[nm] for nm in owner], chunksize=1)
        bars = []
        for c in range(ncell):
            v = [float(S[0]) for (S, sh, l) in oc if cell_of(sh, l) == c]
            bars.append(max(v) + margin if len(v) >= P.MIN_CAL else 1e9)
        RB, nl = P.author_bars([b if b < 1e8 else 999999.0 for b in bars])
        RB = [b if b < 999999.0 - 1e-6 else np.inf for b in RB]
        print("\nMAP = %s (%s); flat %.3f; bars per cell authored as integer lanes: %d lanes, 0 wrong, alive" % (m, notes[m], gf.mean(), nl))
        for nm, (S, sh, l) in zip(owner, oo):
            c = cell_of(sh, l); mi = int(V4.mismatch(crops[nm], scan) >= D1)
            a = 1 if float(S[0]) > RB[c] else (2 if (mi or not np.isfinite(RB[c])) else 0)
            print("   try %-9s %-52s light %-7s score %5.2f vs bar %s -> %s" % (nm, "(" + notes[nm] + ")", ("front", "oblique", "side")[l], float(S[0]),
                  ("%.2f" % RB[c]) if np.isfinite(RB[c]) else "none", NAME[a] + ("   [the scan itself: sanity only]" if nm == m else "")))
        o = sum(float(S[0]) > RB[cell_of(sh, l)] for (S, sh, l) in op)
        print("   FRESH CHECK: CMU PIE strangers (68 people, never used) who OPEN this map: %d of %d tries" % (o, len(op)))
    print("[%.0fs]" % (time.time() - t0))
