"""UNIVERSAL BARS (deployable: anyone scans, the page carries only numbers): for every lab map (38 Yale B frontal + 68 CMU
PIE most-even flash) the best impostor score per cell (light x evidence) over the Yale calibration tries of OTHER people
(every light, 8 darkness levels, 2 draws, the law's +-1 px search); the universal bar per cell = the worst map's best
impostor + margin, authored as integer lanes.  Checks: fresh PIE impostors on every Yale map and on the user's own maps
(never used to set the bars), and the user's own tries on the user's maps.
    python3 -I universal.py"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import unlock_v4 as V4
import personal as P
import phone_eyes as PE

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "phone_user")
MARGIN = 0.20
NAME = {0: "NOT YOU", 1: "OPEN", 2: "NOT NOW"}

if __name__ == "__main__":
    t0 = time.time()
    import yale as Y
    Z3 = json.load(open(os.path.join(HERE, "unlock_lanes.json"))); Z4 = json.load(open(os.path.join(HERE, "unlock_v4_lanes.json")))
    F, EB, D1 = np.array(Z3["F"]["table"]), Z3["EBINS"], Z4["D1"]["value"]
    front = [next(im["I"] for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")).reshape(48, 42) for p in Y.IDS]
    REF = float(np.mean(front)); expose = lambda x: x * (REF / max(float(x.mean()), 1e-9))
    cL, cR = PE.canonical_eyes()
    pc_, _ = P.pie_crops()
    pie_maps = [pc_[p][int(np.argmin([U.imbalance(x) for x in pc_[p]]))] for p in range(68)]
    HAND = {"IMG_1374": ("IMG_1374_up.jpg", (956, 1731), (1589, 1728)), "photo2": ("photo2.jpg", (622, 960), (812, 955)),
            "photo1": ("photo1.jpg", (572, 1016), (828, 1015)), "IMG_1376": ("IMG_1376_up.jpg", (1001, 1710), (1586, 1689))}
    own = {nm: PE.crop(PE.load(os.path.join(D, p)), np.array(l, float), np.array(r, float), cL, cR) for nm, (p, l, r) in HAND.items()}
    users = ["IMG_1374", "photo2", "photo1"]
    maps = front + pie_maps + [own[u] for u in users]
    who = [("yale", p) for p in Y.IDS] + [("pie", r) for r in range(68)] + [("user", u) for u in users]
    feats = [DL.gallery_feats(expose(x)) for x in maps]
    gs = np.array([a for a, _ in feats]); gf = np.array([b for _, b in feats])
    R = np.random.default_rng(2611)
    cal = [(p, U.jitter(U.dark_ae(im["I"].reshape(48, 42), N, R), R)) for p in Y.IDS[0:19] for im in Y.PEOPLE[p] if im["I"].sum() > 0
           for N in U.NBARS for _ in range(2)]
    pie, _ = P.pie_strangers(np.random.default_rng(7))
    tries = [x for _, x in cal] + [t[2] for t in pie] + [own[u] for u in HAND]
    with Pool(3, V4.init, (F, gs, gf, (48, 42))) as pool:
        out = pool.map(V4.score_task, tries, chunksize=64)
    print("scored %d tries against %d maps  [%.0fs]" % (len(tries), len(maps), time.time() - t0), flush=True)
    ncell = U.NL * U.NE
    cell = np.array([l * U.NE + U.ebin(sh, EB[l]) for (_, sh, l) in out])
    S = np.array([o[0] for o in out])                            # tries x maps
    nc = len(cal); ci = np.arange(nc); pi = np.arange(nc, nc + len(pie)); oi = np.arange(nc + len(pie), len(tries))
    calp = np.array([p for p, _ in cal])
    lab = [j for j, (k, _) in enumerate(who) if k != "user"]
    worst = np.full((len(maps), ncell), -np.inf); cnt = np.zeros(ncell, int)
    for c in range(ncell):
        m = ci[cell[ci] == c]; cnt[c] = len(m)
        for j in lab:
            keep = m if who[j][0] != "yale" else m[calp[m] != who[j][1]]
            if len(keep):
                worst[j, c] = S[keep, j].max()
    q = lambda c: np.percentile(worst[lab, c], [50, 90, 99, 100])
    for c in range(ncell):
        print("cell %d (light %s, evidence %s): %5d calibration tries; best impostor per lab map: median %.2f, 90%% %.2f, 99%% %.2f, worst %.2f"
              % (c, ("front", "oblique", "side")[c // 2], ("low", "high")[c % 2], cnt[c], *q(c)))
    bars = [worst[lab, c].max() + MARGIN if cnt[c] >= P.MIN_CAL else 1e9 for c in range(ncell)]
    RB, nl = P.author_bars([b if b < 1e8 else 999999.0 for b in bars])
    RB = np.array([b if b < 999999.0 - 1e-6 else np.inf for b in RB])
    print("UNIVERSAL BARS (worst lab map + %.2f), authored as integer lanes: %d lanes, 0 wrong, alive:" % (MARGIN, nl), ["%.3f" % b for b in RB])
    yale_maps = [j for j, (k, _) in enumerate(who) if k == "yale"]
    o = sum(int((S[pi, j] > RB[cell[pi]]).sum()) for j in yale_maps)
    print("FRESH CHECK: PIE impostors (68 people, never used) opening any of the 38 Yale maps: %d of %d tries" % (o, len(pi) * len(yale_maps)))
    for u in users:
        j = len(maps) - len(users) + users.index(u)
        print("   your %-8s map: PIE impostors who OPEN it: %d of %d" % (u, int((S[pi, j] > RB[cell[pi]]).sum()), len(pi)))
        for k, t in zip(oi, HAND):
            if t == u:
                continue
            mi = int(V4.mismatch(own[t], expose(own[u])) >= D1)
            a = 1 if S[k, j] > RB[cell[k]] else (2 if (mi or not np.isfinite(RB[cell[k]])) else 0)
            print("      try %-9s score %5.2f vs bar %.2f -> %s" % (t, S[k, j], RB[cell[k]], NAME[a]))
    json.dump(dict(bars=[float(b) if np.isfinite(b) else None for b in RB], margin=MARGIN, lanes=nl),
              open(os.path.join(HERE, "data", "webapp", "bars.json"), "w"))
    print("[%.0fs]" % (time.time() - t0))
