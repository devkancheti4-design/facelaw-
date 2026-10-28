"""THE UNION OF K MAPS (phone v4: several enrolled angles x the screen-lit and the bright template): a stranger gets K
chances, one per map, so the margin of each law must hold for the best of K.  Proxy on Extended Yale B (no poses in the
crops): each owner's K maps come from K different photos of that owner (the K smallest light angles), each map the
per-point majority of 8 noise draws at N = 1,000.  Leave-one-person-out with the strangers split in two halves BY PERSON:
the bar of every map is set on one half, the other half are new people; the margin is the smallest at which no new
person's try clears ANY of the K bars; then the halves swap.
    python3 -I phone_faceid/union.py K [K ...]                 -> bars.json gains margin_unionK"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bars as B

if __name__ == "__main__":
    import h2h_all as H, unlock_law as U
    t0 = time.time()
    Ks = [int(k) for k in sys.argv[1:]] or [10]
    Yy, ids, hA, hB = H.load_people()
    C48, C96 = H.crops_table()
    Z = np.load(os.path.join(HERE, "bundle.npz"))
    who, Ns, TH, P96, P48, H96, H48 = (Z[k] for k in ("who", "N", "th", "P96", "P48", "H96", "H48"))
    T = B.tables()
    res = json.load(open(os.path.join(HERE, "bars.json")))
    for K in Ks:
        gap = {"96": [], "48": []}; own = {"96": [], "48": []}
        for i, p in enumerate(ids):
            ph = sorted((th, k) for k, (nm, th, c) in enumerate(C48[p]))[:K]          # the K most frontal lights
            R = np.random.default_rng([5, K, i])
            M = [(B.mode_map([B.codes96(U.dark_ae(C96[p][k][2], 250.0, R))[0] for _ in range(8)]),
                  B.mode_map([B.codes48(U.dark_ae(C48[p][k][2], 1000.0, R))[0] for _ in range(8)])) for _, k in ph]
            half = [np.array([j for j in range(len(ids)) if j != i and j % 2 == h]) for h in (0, 1)]
            mine = who == i
            for li, (law, fw, PB, HB) in enumerate((("96", T["fw96"], P96, H96), ("48", T["fw48"], P48, H48))):
                for h in (0, 1):
                    g, o = None, None
                    for m in M:
                        bmax = B.scores(fw, m[li], PB[np.isin(who, half[h])]).max()
                        gi = B.scores(fw, m[li], PB[np.isin(who, half[1 - h])]) - bmax
                        oi = B.scores(fw, m[li], HB[mine]) - bmax
                        g = gi if g is None else np.maximum(g, gi); o = oi if o is None else np.maximum(o, oi)
                    gap[law].append(float(g.max())); own[law].append((o, Ns[mine], TH[mine]))
        for law in ("96", "48"):
            m = max(0.0, max(gap[law])) + 1e-9
            res[law]["margin_union%d" % K] = m
            for N in B.NS:
                of = np.concatenate([d[(Nn == N) & (t <= 25.5)] for d, Nn, t in own[law]])
                res[law]["owner_open_front_union%d_N%d" % (K, N)] = float((of > m).mean())
            print("K = %2d maps, law %s: margin %.3f; owner front-light tries open: %s  [%.0fs]" % (
                K, law, m, ", ".join("N=%d %.1f%%" % (N, 100 * res[law]["owner_open_front_union%d_N%d" % (K, N)]) for N in B.NS), time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(HERE, "bars.json"), "w"), indent=1)
    print("-> bars.json")
