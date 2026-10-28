"""THE BARS, MEASURED THE WAY THE APP SEES A FACE.  The vault's maps against lab strangers made exactly like a live try:
a full photo (Extended Yale B, every light, 8 darkness levels) -> two eye taps with tap error -> the app's crop -> the law.
    CALIBRATION  28 people (bars = their best score per map per cell + 0.20; no light below front light's bar)
    CHECK        10 other people the bars never saw
    OWNER        the owner's live screen-lit tries (kept on this Mac) and the iPhone photos
Compares today's bars, today's bars + the front-light floor, and the new bars.  Writes the new bars (lane-authored)
into the vault only with WRITE=1, after backing the vault up.
    python3 -I calibrate_pipeline.py"""
import json, os, shutil, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import dark_law as DL
import unlock_law as U
import unlock_v4 as V4
import personal as P
import phone_eyes as PE

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "data", "webapp"); VAULT = os.path.join(WEB, "vault.json")
MARGIN, TAP = 0.20, 0.02                                   # tap error: each eye off by up to 2% of the eye distance
G = {}


def init(F, gs, gf, EB, cL, cR):
    V4.init(F, gs, gf, (48, 42)); G.update(EB=EB, cL=cL, cR=cR)


def stranger(task):
    import yale as Y
    p, name, N, seed = task
    rng = np.random.default_rng(seed); Bp = 48
    lin = 255 * (Y.read_pgm(os.path.join(Y.ROOT, p, name)).astype(float) / 255) ** 2.2
    if N is not None:
        lin = np.maximum(U.dark_ae(lin, N / 16.0, rng), 0)
    eL, eR = np.array([33.5 + Bp, 45.5 + Bp]), np.array([133.5 + Bp, 45.5 + Bp])
    eL, eR = eL + rng.uniform(-TAP, TAP, 2) * 100, eR + rng.uniform(-TAP, TAP, 2) * 100
    x = PE.crop(np.pad(lin, Bp, mode="edge"), eL, eR, G["cL"], G["cR"])
    S, sh, l = V4.score_task(x)
    return p, l * U.NE + U.ebin(sh, G["EB"][l]), S


def decide(S, c, bars, floor):
    def eff(b):
        if b[c] is None:
            return None
        fr = [v for v in b[:U.NE] if v is not None]
        return max(b[c], max(fr)) if (floor and fr) else b[c]
    m = [S[j] - eff(b) for j, b in enumerate(bars) if eff(b) is not None]
    return max(m) if m else None


if __name__ == "__main__":
    t0 = time.time()
    import yale as Y
    Z3 = json.load(open("unlock_lanes.json")); F, EB = np.array(Z3["F"]["table"]), Z3["EBINS"]
    v = json.load(open(VAULT)); maps = v["maps"]; K = len(maps); nc = U.NL * U.NE
    gs = np.array([m["gs"] for m in maps]); gf = np.array([m["gf"] for m in maps]); old = [m["bars"] for m in maps]
    cL, cR = PE.canonical_eyes()
    cal_ids, chk_ids = Y.IDS[:28], Y.IDS[28:]
    tasks = [(p, im["name"], N, hash((p, im["name"], N)) % 2**32) for p in Y.IDS for im in Y.PEOPLE[p] if im["I"].sum() > 0 for N in U.NBARS]
    # the owner: live screen-lit tries kept on this Mac, and the iPhone photos
    D = os.path.join(WEB, "debug"); owner = []
    for d in sorted(os.listdir(D)):
        if d.endswith("-try") and d > "20261027-2014":
            r = json.load(open(os.path.join(D, d, "result.json")))
            ld = lambda k: np.mean([PE.load(os.path.join(D, d, f)) for f in sorted(os.listdir(os.path.join(D, d))) if f.startswith(k)], 0)
            lit = np.maximum(ld("white") - ld("black"), 0)
            (x1, y1), (x2, y2) = sorted(map(tuple, r["eyes"]))
            owner.append(("live try " + d[9:15] + " (Mac, screen-lit)", PE.crop(lit, np.array([x1, y1]), np.array([x2, y2]), cL, cR)))
    PU = os.path.join(HERE, "data", "phone_user")
    for nm, p, l, rr in (("IMG_1374 (iPhone, front)", "IMG_1374_up.jpg", (956, 1731), (1589, 1728)), ("IMG_1376 (iPhone, head turned)", "IMG_1376_up.jpg", (1001, 1710), (1586, 1689)),
                         ("photo2 (iPhone, room light)", "photo2.jpg", (622, 960), (812, 955))):
        owner.append((nm, PE.crop(PE.load(os.path.join(PU, p)), np.array(l, float), np.array(rr, float), cL, cR)))
    with Pool(2, init, (F, gs, gf, EB, cL, cR)) as pool:
        out = pool.map(stranger, tasks, chunksize=32)
    init(F, gs, gf, EB, cL, cR)
    own = []
    for nm, x in owner:
        S, sh, l = V4.score_task(x); own.append((nm, l * U.NE + U.ebin(sh, EB[l]), S))
    cal = [(c, S) for p, c, S in out if p in cal_ids]; chk = [(c, S) for p, c, S in out if p in chk_ids]
    new = []
    for j in range(K):
        row = []
        for c in range(nc):
            s = [S[j] for cc, S in cal if cc == c]
            row.append(max(s) + MARGIN if len(s) >= P.MIN_CAL else None)
        new.append(row)
    flat = [b if b is not None else 999999.0 for row in new for b in row]
    with Pool(1) as pp:
        RB, nl = pp.apply(P.author_bars, (flat,))
    new = [[b if b < 999999.0 - 1e-6 else None for b in RB[j * nc:(j + 1) * nc]] for j in range(K)]
    print("tries made like a live try: %d calibration (28 people), %d check (10 other people)  [%.0fs]" % (len(cal), len(chk), time.time() - t0))
    print("new bars authored as integer lanes: %d lanes, 0 wrong, alive" % nl)
    for tag, bars, floor in (("today's bars", old, False), ("today's bars + front-light floor (live now)", old, True), ("NEW bars + front-light floor", new, True)):
        o = [decide(S, c, bars, floor) for c, S in chk]; o = [m for m in o if m is not None]
        ow = [(nm, decide(S, c, bars, floor)) for nm, c, S in own]
        print("\n%s:\n   check strangers who open: %d of %d (closest %+.2f)" % (tag, sum(m > 0 for m in o), len(o), max(o)))
        for nm, m in ow:
            print("   you, %-36s %s" % (nm, "no bar" if m is None else ("OPEN %+.2f" % m if m > 0 else "below %+.2f" % m)))
    if os.environ.get("WRITE"):
        shutil.copy2(VAULT, VAULT.replace(".json", "_before_recal.json"))
        for j, m in enumerate(maps):
            m["bars"] = new[j]
        v["bars_note"] = "measured through the app's crop path on 28 Yale people, checked on 10 others (calibrate_pipeline.py)"
        tmp = VAULT + ".tmp"; json.dump(v, open(tmp, "w")); os.replace(tmp, VAULT)
        print("\nvault updated (backup: vault_before_recal.json)")
