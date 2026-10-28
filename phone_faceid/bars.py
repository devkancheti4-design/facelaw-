"""THE BARS for the face ID test: an owner's lock opens when either law's score clears that law's bar, and each bar is
set for that owner's own map: the best score any stranger in the BUNDLE reaches against the map, plus a margin.
    bundle   Extended Yale B tries of the 37 people (the automatic-eye crops of h2h_all.py), every photo, at N = 5, 20,
             100 and 1,000 photons per face pixel through the sensor model; the probe codes of both laws with the
             SHIPPED tables (full-light law: v2_full; dark law: v2_design), one placement each       -> bundle.npz
    margin   each person in turn is the owner: the map is the per-point majority of 8 frontal tries at N = 1,000.  The
             other people are split in two halves BY PERSON: the bar is set on one half, and the other half -- people the
             bar has never seen -- are the held-out strangers; then the halves swap.  The margin of each law is the
             smallest that lets no held-out stranger try clear any owner's bar; the owner's own tries (every photo, every
             N; and front light alone) show what it costs                                         -> bars.json
    python3 -I phone_faceid/bars.py"""
import ctypes, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
os.chdir(ROOT)
NS = (5, 20, 100, 1000)
U8, D = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double)
LIB = ctypes.CDLL(os.path.join(ROOT, "v2cpp", "libfacelaw2.dylib"))


def dp(a):
    return np.ascontiguousarray(a, np.float64).ctypes.data_as(D)


def codes96(c):
    """the full-light law on a 96 x 84 crop: (map bins, probe code = bin | sig << 3)"""
    m = np.empty(10584, np.uint8); b = np.empty(10584, np.uint8); s = np.empty(10584, np.uint8)
    LIB.fl2_map_bins96(dp(c), m.ctypes.data_as(U8)); LIB.fl2_probe_codes96(dp(c), b.ctypes.data_as(U8), s.ctypes.data_as(U8))
    return m, (b | (s << 3)).astype(np.uint8)


def codes48(c):
    """the dark law on a 48 x 42 crop, one placement (the centre of the 9)"""
    m = np.empty(2652, np.uint8); b = np.empty((9, 2652), np.uint8); s = np.empty((9, 2652), np.uint8)
    LIB.fl2_map_bins(dp(c), m.ctypes.data_as(U8)); LIB.fl2_probe_all(dp(c), b.ctypes.data_as(U8), s.ctypes.data_as(U8))
    return m, (b[4] | (s[4] << 3)).astype(np.uint8)


def tables():
    import v2ref as R2, v2_full as VF
    a96, fw96 = VF.load()
    return dict(fw96=np.asarray(fw96, np.int64), fw48=np.asarray(R2.load()["FW"], np.int64))


SC = {10584: np.repeat(np.arange(3), [8064, 2016, 504]), 2652: np.repeat(np.arange(3), [2016, 504, 132])}


def scores(fw, mp, P, chunk=500):
    """mean table entry of each probe (rows of P: bin | sig << 3) against one map"""
    base = mp.astype(np.int64) | (SC[len(mp)] << 8)
    out = []
    for a in range(0, len(P), chunk):
        Q = P[a:a + chunk].astype(np.int64)
        out.append(fw[base[None, :] | ((Q & 7) << 3) | ((Q >> 3) << 6)].mean(1))
    return np.concatenate(out) if out else np.zeros(0)


def mode_map(maps):
    """per point, the bin most of the frames agree on (ties: the lowest bin)"""
    M = np.stack(maps)
    counts = np.stack([(M == b).sum(0) for b in range(8)])
    return counts.argmax(0).astype(np.uint8)


if __name__ == "__main__":
    import h2h_all as H, unlock_law as U
    t0 = time.time()
    Yy, ids, hA, hB = H.load_people()
    C48, C96 = H.crops_table()
    bpath = os.path.join(HERE, "bundle.npz")
    if not os.path.exists(bpath):
        who, n, th_, P96, P48, H96, H48 = [], [], [], [], [], [], []
        for p in ids:
            for k, ((nm, th, c48), (_, _, c96)) in enumerate(zip(C48[p], C96[p])):
                for N in NS:
                    for out96, out48, seed in ((P96, P48, 1), (H96, H48, 2)):          # the bundle, and held-out draws
                        R = np.random.default_rng([seed, ids.index(p), k, N])
                        out96.append(codes96(U.dark_ae(c96, N / 4.0, R))[1]); out48.append(codes48(U.dark_ae(c48, N, R))[1])
                    who.append(ids.index(p)); n.append(N); th_.append(th)
        np.savez_compressed(bpath, who=np.array(who), N=np.array(n), th=np.array(th_), P96=np.array(P96), P48=np.array(P48),
                            H96=np.array(H96), H48=np.array(H48))
        print("bundle: %d tries of %d people x 2 laws (+ the same number held out)  [%.0fs]" % (len(who), len(ids), time.time() - t0), flush=True)
    Z = np.load(bpath)
    who, Ns, TH, P96, P48, H96, H48 = (Z[k] for k in ("who", "N", "th", "P96", "P48", "H96", "H48"))
    T = tables()
    # every person as the owner: the map from 8 frontal tries at N = 1,000
    best_h = {"96": [], "48": []}; own = {"96": [], "48": []}; bars0 = {"96": [], "48": []}
    for i, p in enumerate(ids):
        front96 = next(c for nm, th, c in C96[p] if nm.endswith("A+000E+00.pgm")); front48 = next(c for nm, th, c in C48[p] if nm.endswith("A+000E+00.pgm"))
        R = np.random.default_rng([3, i])
        m96 = mode_map([codes96(U.dark_ae(front96, 250.0, R))[0] for _ in range(8)])
        m48 = mode_map([codes48(U.dark_ae(front48, 1000.0, R))[0] for _ in range(8)])
        half = [np.array([j for j in range(len(ids)) if j != i and j % 2 == h]) for h in (0, 1)]
        for law, mp, fw, PB, HB in (("96", m96, T["fw96"], P96, H96), ("48", m48, T["fw48"], P48, H48)):
            mine = who == i
            so = scores(fw, mp, HB[mine])
            for h in (0, 1):                                     # bar on one half of the strangers, held out: the other
                sb = scores(fw, mp, PB[np.isin(who, half[h])]); sh = scores(fw, mp, PB[np.isin(who, half[1 - h])])
                best_h[law].append(float(sh.max() - sb.max()))
                own[law].append((so - sb.max(), Ns[mine], TH[mine]))
            bars0[law].append(float(scores(fw, mp, PB[~mine]).max()))
        if (i + 1) % 10 == 0:
            print("  owners %d / %d  [%.0fs]" % (i + 1, len(ids), time.time() - t0), flush=True)
    res = {}
    for law in ("96", "48"):
        m = max(0.0, max(best_h[law])) + 1e-9                    # the smallest margin no held-out stranger try clears
        res[law] = dict(margin=m, worst_held_out_over_bundle_max=max(best_h[law]))
        for N in NS:
            o = np.concatenate([d[Nn == N] for d, Nn, t in own[law]]); of = np.concatenate([d[(Nn == N) & (t <= 25.5)] for d, Nn, t in own[law]])
            res[law]["owner_open_N%d" % N] = float((o > m).mean()); res[law]["owner_open_front_N%d" % N] = float((of > m).mean())
        print("law %s: margin %.3f (held-out strangers' best over the bar half's best, worst owner: %.3f); owner tries open, all light: %s | front light: %s"
              % (law, m, max(best_h[law]), ", ".join("N=%d %.1f%%" % (N, 100 * res[law]["owner_open_N%d" % N]) for N in NS),
                 ", ".join("N=%d %.1f%%" % (N, 100 * res[law]["owner_open_front_N%d" % N]) for N in NS)), flush=True)
    both = {}
    for N in NS:
        for cls, sel in (("all", lambda Nn, t: Nn == N), ("front", lambda Nn, t: (Nn == N) & (t <= 25.5))):
            a = np.concatenate([d[sel(Nn, t)] for d, Nn, t in own["96"]]) > res["96"]["margin"]
            b = np.concatenate([d[sel(Nn, t)] for d, Nn, t in own["48"]]) > res["48"]["margin"]
            both["%s_N%d" % (cls, N)] = float((a | b).mean())
    res["both_owner_open"] = both
    print("both laws, either says yes: owner tries open %s" % ", ".join("%s %.1f%%" % (k, 100 * v) for k, v in both.items()))
    # THE UNION (phone v3): two maps per owner and law (the screen-lit face and the bright frame) -> a stranger gets two
    # chances.  Map B is the owner's photo with the next-smallest light angle; the margin of each law is re-set so that no
    # held-out new person's try clears EITHER map's bar.
    gap_u = {"96": [], "48": []}; own_u = {"96": [], "48": []}
    for i, p in enumerate(ids):
        ph = sorted(((th, k) for k, (nm, th, c) in enumerate(C48[p]) if not nm.endswith("A+000E+00.pgm")))
        kB = ph[0][1]
        front96 = next(c for nm, th, c in C96[p] if nm.endswith("A+000E+00.pgm")); front48 = next(c for nm, th, c in C48[p] if nm.endswith("A+000E+00.pgm"))
        R = np.random.default_rng([4, i])
        MA = (mode_map([codes96(U.dark_ae(front96, 250.0, R))[0] for _ in range(8)]), mode_map([codes48(U.dark_ae(front48, 1000.0, R))[0] for _ in range(8)]))
        MB = (mode_map([codes96(U.dark_ae(C96[p][kB][2], 250.0, R))[0] for _ in range(8)]), mode_map([codes48(U.dark_ae(C48[p][kB][2], 1000.0, R))[0] for _ in range(8)]))
        half = [np.array([j for j in range(len(ids)) if j != i and j % 2 == h]) for h in (0, 1)]
        for li, (law, fw, PB, HB) in enumerate((("96", T["fw96"], P96, H96), ("48", T["fw48"], P48, H48))):
            mine = who == i
            for h in (0, 1):
                g, o = [], []
                for mp in (MA[li], MB[li]):
                    bmax = scores(fw, mp, PB[np.isin(who, half[h])]).max()
                    g.append(scores(fw, mp, PB[np.isin(who, half[1 - h])]) - bmax); o.append(scores(fw, mp, HB[mine]) - bmax)
                gap_u[law].append(float(np.maximum(g[0], g[1]).max())); own_u[law].append((np.maximum(o[0], o[1]), Ns[mine], TH[mine]))
    for law in ("96", "48"):
        m = max(0.0, max(gap_u[law])) + 1e-9
        res[law]["margin_union2"] = m
        for N in NS:
            of = np.concatenate([d[(Nn == N) & (t <= 25.5)] for d, Nn, t in own_u[law]])
            res[law]["owner_open_front_union2_N%d" % N] = float((of > m).mean())
        print("UNION of two maps, law %s: margin %.3f (was %.3f); owner front-light tries open: %s" % (
            law, m, res[law]["margin"], ", ".join("N=%d %.1f%%" % (N, 100 * res[law]["owner_open_front_union2_N%d" % N]) for N in NS)), flush=True)
    json.dump(dict(res, note="margins: leave-one-person-out on Extended Yale B; held-out = new people (the other half); bundle = %s" % bpath),
              open(os.path.join(HERE, "bars.json"), "w"), indent=1)
    print("-> bars.json  [%.0fs]" % (time.time() - t0))
