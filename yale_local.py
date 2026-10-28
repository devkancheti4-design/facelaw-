"""LAW L -- local cancellation (the other side): every pixel against its 8 neighbours, in three states
    gt = census(a, b + d), lt = census(b, a + d)      (the synthesis's proven lane L2_census, used twice; d = dead zone)
so a smooth light field, any exposure and any increasing tone curve cancel; a pair inside the dead zone (shadow, flat
skin) carries no evidence.  Distance to a gallery image = disagreeing decisive pairs / pairs decisive in both.
Identity = the nearest of the 7 subset-1 images of each person (protocol A)."""
import glob, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y
from yale_baselines import load_full

CEN = next(l["expr"] for l in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lanes_face.json")))
           if l["name"] == "L2_census")
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def census(a, b):
    """the synthesis's lane, vectorised: [a > b] for 8-bit codes"""
    return eval(CEN, {"x": (a.astype(np.int64) | (b.astype(np.int64) << 8))}).astype(bool)


def ternary(I, d):
    I = np.clip(np.rint(I), 0, 255).astype(np.int64)
    h, w = I.shape
    c = I[1:-1, 1:-1]
    gt, lt = [], []
    for dr, dc in NB:
        n = I[1 + dr:h - 1 + dr, 1 + dc:w - 1 + dc]
        gt.append(census(c, np.minimum(255, n + d)).ravel()); lt.append(census(n, np.minimum(255, c + d)).ravel())
    return np.concatenate(gt), np.concatenate(lt)


def resize(I, f):
    if f == 1:
        return I
    h, w = I.shape[0] // f * f, I.shape[1] // f * f
    return I[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))


def distances(full, f, d):
    gal, lab = [], []
    for p in Y.IDS:
        for sb, I in full[p]:
            if sb == 1:
                gal.append(ternary(resize(I, f), d)); lab.append(p)
    Ggt = np.array([g[0] for g in gal], np.float32); Glt = np.array([g[1] for g in gal], np.float32)
    Gd = Ggt + Glt
    probes, out = [], []
    for p in Y.IDS:
        for sb, I in full[p]:
            if sb >= 2:
                probes.append((p, sb))
                gt, lt = ternary(resize(I, f), d)
                gt, lt = gt.astype(np.float32), lt.astype(np.float32)
                flips = Ggt @ lt + Glt @ gt
                both = Gd @ (gt + lt)
                out.append(flips / np.maximum(both, 1))
    return probes, np.array(out), lab


def person_dist(D, lab):
    """distance to each person = the nearest of their gallery images"""
    lab = np.array(lab)
    return np.array([D[:, lab == p].min(1) for p in Y.IDS]).T          # probes x 38


if __name__ == "__main__":
    t0 = time.time()
    full = load_full()
    res = {}
    print("LAW L alone, protocol A -- recognition %   (S2 S3 S4 S5)")
    for f in (4, 2, 1):
        for d in (1, 2, 4, 8):
            probes, D, lab = distances(full, f, d)
            PD = person_dist(D, lab)
            acc = {}
            for j, (p, sb) in enumerate(probes):
                acc.setdefault(sb, []).append(Y.IDS[int(PD[j].argmin())] == p)
            res["%d_%d" % (f, d)] = {sb: 100 * np.mean(v) for sb, v in acc.items()}
            print("   %3dx%-3d  dead zone %d:  " % (192 // f, 168 // f, d) + "  ".join("%6.2f" % res["%d_%d" % (f, d)][sb] for sb in (2, 3, 4, 5)), flush=True)
    json.dump(res, open("yale_local_scan.json", "w"), indent=1)
    print("%.0f s" % (time.time() - t0))
