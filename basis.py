"""PER-MAP BARS WITHOUT A DATASET IN THE PAGE: the hardest impostor tries (those that are the best impostor for some lab map
in some cell) are kept as CODES only -- sectors + significance at the law's 9 placements, no image -- and the page sets each
new map's bar per cell = its best score over these codes + a per-cell margin.  The margin covers, for every lab map, the gap
between this compressed best and the best over ALL calibration impostors, + 0.20.  Checks: fresh PIE impostors on every Yale
map and on the user's maps, and the user's own tries.
    python3 -I basis.py [K]          K = how many impostor tries the page carries (default 160)"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import shift as nd_shift
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import unlock_v4 as V4
import personal as P
import phone_eyes as PE
import universal as UV

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "phone_user")
NAME = {0: "NOT YOU", 1: "OPEN", 2: "NOT NOW"}


def codes9(x):
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ps, pc = U.probe_feats(nd_shift(x, (dy, dx), order=0, mode="nearest") if (dy or dx) else x)
            out.append((ps | (pc << 3)).astype(np.uint8))
    return np.array(out)


if __name__ == "__main__":
    t0 = time.time()
    import yale as Y
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 160
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
    ncell = U.NL * U.NE
    cell = np.array([l * U.NE + U.ebin(sh, EB[l]) for (_, sh, l) in out]); S = np.array([o[0] for o in out])
    nc = len(cal); ci = np.arange(nc); pi = np.arange(nc, nc + len(pie)); oi = np.arange(nc + len(pie), len(tries))
    calp = np.array([p for p, _ in cal])
    lab = [j for j, (k, _) in enumerate(who) if k != "user"]
    def allowed(j, m):
        return m if who[j][0] != "yale" else m[calp[m] != who[j][1]]
    full = np.full((len(maps), ncell), -np.inf); arg = {}
    for c in range(ncell):
        m = ci[cell[ci] == c]
        for j in range(len(maps)):
            k = allowed(j, m)
            full[j, c] = S[k, j].max(); arg[(j, c)] = k[np.argmax(S[k, j])]
    votes = {}
    for j in lab:
        for c in range(ncell):
            votes[arg[(j, c)]] = votes.get(arg[(j, c)], 0) + 1
    # greedy: the tries that are the best impostor for the most (lab map, cell) pairs; then fill by each map's runner-ups
    basis = [t for t, _ in sorted(votes.items(), key=lambda kv: -kv[1])][:K]
    if len(basis) < K:
        extra = []
        for c in range(ncell):
            m = ci[cell[ci] == c]
            top = m[np.argsort(-S[m][:, lab].max(1))]
            extra += [t for t in top if t not in basis][: (K - len(basis)) // ncell + 1]
        basis = (basis + extra)[:K]
    basis = np.array(sorted(set(basis)))
    comp = np.full((len(maps), ncell), -np.inf)
    for c in range(ncell):
        m = basis[cell[basis] == c]
        for j in range(len(maps)):
            k = allowed(j, m)
            if len(k):
                comp[j, c] = S[k, j].max()
    gap = full - comp
    MARG = [float(np.max(gap[lab, c])) + UV.MARGIN for c in range(ncell)]
    MA, nl = P.author_bars(MARG)
    print("basis: %d impostor tries kept as codes (%.1f MB); per-cell margin = worst lab-map gap + %.2f, authored as integer lanes (%d lanes, 0 wrong, alive): %s"
          % (len(basis), len(basis) * 9 * 2652 / 1e6, UV.MARGIN, nl, ["%.3f" % v for v in MA]), flush=True)
    for c in range(ncell):
        print("   cell %d: tries in basis %3d; lab-map gap median %.2f, worst %.2f" % (c, int((cell[basis] == c).sum()), np.median(gap[lab, c]), gap[lab, c].max()))
    bar = lambda j: np.array([np.round(1000 * (comp[j, c] + MA[c])) / 1000 if np.isfinite(comp[j, c]) else np.inf for c in range(ncell)])
    yale_maps = [j for j, (k, _) in enumerate(who) if k == "yale"]
    o = sum(int((S[pi, j] > bar(j)[cell[pi]]).sum()) for j in yale_maps)
    po = sum(int((S[pi, j] > bar(j)[cell[pi]]).sum()) for j, (k, _) in enumerate(who) if k == "pie")
    print("FRESH CHECK: PIE impostors opening any of the 38 Yale maps (per-map bars): %d of %d tries" % (o, len(pi) * len(yale_maps)))
    print("   (PIE maps vs PIE impostors incl. the same person's other flashes -- not a stranger test, shown for size only: %d)" % po)
    for u in users:
        j = len(maps) - len(users) + users.index(u); b = bar(j)
        print("   your %-8s map: bars %s; PIE impostors who OPEN it: %d of %d (full-calibration best impostor per cell %s)" % (
            u, ["%.2f" % v for v in b], int((S[pi, j] > b[cell[pi]]).sum()), len(pi), ["%.2f" % v for v in full[j]]))
        for k, t in zip(oi, HAND):
            if t == u:
                continue
            mi = int(V4.mismatch(own[t], expose(own[u])) >= D1)
            a = 1 if S[k, j] > b[cell[k]] else (2 if (mi or not np.isfinite(b[cell[k]])) else 0)
            print("      try %-9s score %5.2f vs bar %.2f -> %s" % (t, S[k, j], b[cell[k]], NAME[a]))
    C = np.array([codes9(tries[t]) for t in basis])
    with open(os.path.join(HERE, "data", "webapp", "site", "impostors.bin"), "wb") as fh:
        fh.write(np.array([len(basis), 9, 2652], np.uint32).tobytes()); fh.write(cell[basis].astype(np.uint8).tobytes()); fh.write(C.tobytes())
    json.dump(dict(margins=[float(v) for v in MA], K=int(len(basis)), lanes=nl), open(os.path.join(HERE, "data", "webapp", "basis.json"), "w"))
    print("[%.0fs]" % (time.time() - t0))
