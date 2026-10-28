"""The user's own phone photos (data/phone_user, local only) through the sealed unlock law v4.
ALIGN   two-eye mode (phone_eyes.py): pupil estimates refined to the darkest point of a smoothed window, then the
        similarity crop onto the canonical eyes measured on the mean Yale face -> the law's 48 x 42 grid
SCAN    one photo is the enrolled face; the other photo, and both photos darkened through Law 1 (auto-exposure) to
        5..1000 photons per face pixel, are unlock tries
LAW     v4 exactly: F and ACT4 rebuilt from the authored lanes (unlock_lanes.json, unlock_v4_lanes.json), measured noise,
        placement search +-1, T-norm against a fixed cohort (18 Yale B frontal faces), light class, evidence, mismatch
STRANGER every Yale B and PIE face (frontal) tries to unlock the user's phone, darkened the same way (lab images: an
        indicative stranger check, not a population test)
    python3 -I phone_unlock.py align        -> data/phone_user/crops.npz + enlarged crops to look at
    python3 -I phone_unlock.py test"""
import json, os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import phone_eyes as PE
import dark_law as DL
import unlock_law as U
import unlock_v4 as V4

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "phone_user")
PHOTOS = {"photo1": ((572, 1016), (828, 1015)), "photo2": ((622, 960), (812, 955))}  # pupils read off a coordinate grid
RREF = 0                                        # no refinement: a wide window walked onto hair and shadow, a +-10 px one onto
                                                # eyelashes and lid shadow (both hit the window edge); the grid reading stands


def refine(img, guess, r=RREF, sigma=3.0):
    x0, y0 = guess
    if r == 0:
        return np.array([x0, y0], float)
    win = gaussian_filter(img[y0 - r:y0 + r, x0 - r:x0 + r], sigma)
    yy, xx = np.unravel_index(np.argmin(win), win.shape)
    return np.array([x0 - r + xx, y0 - r + yy], float)


if __name__ == "__main__":
    stage = sys.argv[1]
    cL, cR = PE.canonical_eyes()
    if stage == "align":
        crops = {}
        for name, (gl, gr) in PHOTOS.items():
            img = PE.load(os.path.join(DIR, name + ".jpg"))
            eL, eR = refine(img, gl), refine(img, gr)
            c = PE.crop(img, eL, eR, cL, cR)
            crops[name] = c
            print("%s: image %dx%d, pupils refined to (%d, %d) and (%d, %d), eye distance %.0f px; crop mean %.1f" % (
                name, img.shape[1], img.shape[0], eL[0], eL[1], eR[0], eR[1], np.linalg.norm(eR - eL), c.mean()))
            big = np.kron(c / max(c.max(), 1e-9) * 255, np.ones((8, 8))).astype(np.uint8)
            Image.fromarray(big).save(os.path.join(DIR, name + "_crop_x8.png"))
        mean_yale = np.mean([next(im["I"] for im in PE.Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")).reshape(48, 42) for p in PE.Y.IDS], 0)
        Image.fromarray(np.kron(mean_yale / mean_yale.max() * 255, np.ones((8, 8))).astype(np.uint8)).save(os.path.join(DIR, "mean_yale_x8.png"))
        np.savez(os.path.join(DIR, "crops.npz"), **crops)
        print("canonical eyes (x, y) in the 42 x 48 frame:", cL, cR)
    elif stage == "auto":
        import finder as FD
        crops = {}
        for name in PHOTOS:
            img = PE.load(os.path.join(DIR, name + ".jpg"))
            f = FD.find(img)
            assert f is not None, "%s: the classic finder found no face + both eyes" % name
            eL, eR = f["classic"]
            crops[name] = PE.crop(img, eL, eR, cL, cR)
            print("%s: automatic eyes (classic, box pass x%d / eye pass x%d) (%d, %d) and (%d, %d)" % (name, f["f"][0], f["f"][1], *eL, *eR))
        np.savez(os.path.join(DIR, "crops_auto.npz"), **crops)
    elif stage == "test":
        from multiprocessing import Pool
        import yale as Y
        Z3 = json.load(open(os.path.join(HERE, "unlock_lanes.json"))); Z4 = json.load(open(os.path.join(HERE, "unlock_v4_lanes.json")))
        F, EB, A4, D1 = np.array(Z3["F"]["table"]), Z3["EBINS"], Z4["ACT4"]["table"], Z4["D1"]["value"]
        C = np.load(os.path.join(DIR, os.environ.get("CROPS", "crops.npz"))); scan_name = sys.argv[2] if len(sys.argv) > 2 else "photo2"
        print("crops: %s" % os.environ.get("CROPS", "crops.npz (hand-read pupils)"))
        other = "photo1" if scan_name == "photo2" else "photo2"
        scan = C[scan_name]
        ids = Y.IDS
        front = {p: next(im for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")) for p in ids}
        cohort = ids[:18]; strangers = ids[18:]
        gal = [DL.gallery_feats(scan)] + [DL.gallery_feats(front[p]["I"].reshape(48, 42)) for p in cohort]
        gs = np.array([g[0] for g in gal]); gf = np.array([g[1] for g in gal])
        R = np.random.default_rng(424242)
        tries = [("%s as taken (no darkening)" % other, None, C[other])]
        for nm in (scan_name, other):
            for N in U.NBARS:
                for d in range(20):
                    tries.append(("%s darkened" % nm, N, U.jitter(U.dark_ae(C[nm], N, R), R)))
        for p in strangers:
            for im in Y.PEOPLE[p]:
                if im["I"].sum() > 0:
                    for N in U.NBARS:
                        tries.append(("stranger", N, U.jitter(U.dark_ae(im["I"].reshape(48, 42), N, R), R)))
        with Pool(3, V4.init, (F, gs, gf, (48, 42))) as pool:
            out = pool.map(V4.score_task, [t[2] for t in tries], chunksize=32)
        coh = list(range(1, len(gal)))
        res = {}
        for (kind, N, x), (S, sh, l) in zip(tries, out):
            e = U.ebin(sh, EB[l]); mi = int(V4.mismatch(x, scan) >= D1)
            act = A4[V4.state4(l, e, mi, U.mbin(U.margins(S, 0, coh)))]
            res.setdefault((kind, N), []).append((act, l, U.margins(S, 0, coh)))
        name = {0: "NOT YOU", 1: "OPEN", 2: "NOT NOW"}
        print("SCAN = %s.  The law: sealed v4 (synthesis lanes).  Cohort: 18 Yale B faces.  Strangers: 20 other Yale B people, all lights." % scan_name)
        r0 = res[("%s as taken (no darkening)" % other, None)][0]
        print("  %s as taken: %s  (light class %s, T-norm margin %.2f)" % (other, name[r0[0]], ("front", "oblique", "side")[r0[1]], r0[2]))
        for nm in (scan_name, other):
            print("  %s darkened -- per photon level: OPEN / NOT NOW / NOT YOU of 20 tries (light class of the tries)" % nm)
            for N in U.NBARS:
                a = [t[0] for t in res[("%s darkened" % nm, N)]]; lc = [t[1] for t in res[("%s darkened" % nm, N)]]
                print("     N=%-5d %2d / %2d / %2d   (front %d, oblique %d, side %d)" % (N, a.count(1), a.count(2), a.count(0), lc.count(0), lc.count(1), lc.count(2)))
        st = [(N, t[0]) for (k, N), v in res.items() if k == "stranger" for t in v]
        if os.environ.get("WHO"):
            k_ = 0
            who = [(p, im["name"]) for p in strangers for im in Y.PEOPLE[p] if im["I"].sum() > 0 for N in U.NBARS]
            si = [j for j, t in enumerate(tries) if t[0] == "stranger"]
            ops = [(who[n_], out[j][2], tries[j][1]) for n_, j in enumerate(si)
                   if A4[V4.state4(out[j][2], U.ebin(out[j][1], EB[out[j][2]]), int(V4.mismatch(tries[j][2], scan) >= D1), U.mbin(U.margins(out[j][0], 0, coh)))] == 1]
            print("  stranger OPENs: person, light class, photons:", [(w[0], ("front", "oblique", "side")[l], N) for w, l, N in ops])
            sm = np.array([U.margins(out[j][0], 0, coh) for j in si])
            print("  stranger T-norm margins vs YOUR scan: median %.2f, 99%% %.2f, max %.2f   (lab-to-lab strangers in the sealed design: median about -0.2 per code scale differs)" % (np.median(sm), np.quantile(sm, 0.99), sm.max()))
        print("  strangers who OPEN your phone: %d of %d tries (%s)" % (sum(a == 1 for _, a in st), len(st),
              ", ".join("N=%d: %d/%d" % (N, sum(a == 1 for n, a in st if n == N), sum(1 for n, a in st if n == N)) for N in U.NBARS)))
