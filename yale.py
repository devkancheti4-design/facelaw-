"""The laws on REAL faces: the Extended Yale Face Database B, cropped (38 people x 64 point-light directions, frontal,
aligned; Georghiades, Belhumeur, Kriegman 2001; Lee, Ho, Kriegman 2005).  Images are read as data only.
ENROL   photometric stereo per person from the lights within 25 deg of the camera only (subsets 1-2): b(p) = rho n
        by least squares over the lit images (shadowed samples excluded) -> the 9 harmonic basis images rho Y_j(n)
LAW 2   energy of each person's 64 images in 9 dimensions (bound for a convex Lambertian object: >= 97.96%)
LAW 2+4 identification on unseen lights 26-130 deg off-axis (subsets 3, 4, 5) by the subspace residual
LAW 5+6 each real image thinned to Nbar photons per pixel (Poisson) through the sensor law (Law 1) -> predicted vs
        measured FAR / FRR, the exact Bhattacharyya bound, and the optimal (Chow) decision
    python3 yale.py"""
import glob, json, os, re, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # my code only; run with -I, the data dir is never on the path
from laws import fit, rates, bhattacharyya, bayes_bounds, SIG2, inverse
from world import sensor, sh9

ROOT = os.path.expanduser("~/neooos/facelaw/data/yaleb/CroppedYale")
DS = 4                                                   # 192 x 168 -> 48 x 42 by box integration (Law 3's pixel)


def read_pgm(path):
    with open(path, "rb") as f:
        data = f.read()
    m = re.match(rb"P5\s+(?:#[^\n]*\n\s*)*(\d+)\s+(?:#[^\n]*\n\s*)*(\d+)\s+(?:#[^\n]*\n\s*)*(\d+)\s", data)
    w, h, mx = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return np.frombuffer(data[m.end():m.end() + w * h], np.uint8).reshape(h, w).astype(float)


def box(img):
    h, w = img.shape[0] // DS * DS, img.shape[1] // DS * DS
    return img[:h, :w].reshape(h // DS, DS, w // DS, DS).mean((1, 3))


def load():
    people = {}
    for d in sorted(glob.glob(ROOT + "/yaleB*")):
        pid = os.path.basename(d)
        amb = glob.glob(d + "/*Ambient.pgm")
        a0 = read_pgm(amb[0]) if amb else None
        A0 = box(a0) if a0 is not None and a0.shape == (192, 168) else 0     # some ambient frames are uncropped: unusable
        n_amb = int(a0 is not None and a0.shape == (192, 168))
        imgs = []
        for f in sorted(glob.glob(d + "/*.pgm")):
            m = re.search(r"A([+-]\d+)E([+-]\d+)\.pgm$", f)
            if not m:
                continue
            a, e = np.radians(int(m.group(1))), np.radians(int(m.group(2)))
            s = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
            theta = np.degrees(np.arccos(np.clip(np.cos(a) * np.cos(e), -1, 1)))
            I = np.maximum(box(read_pgm(f)) - A0, 0)          # light only: the ambient frame subtracted
            imgs.append(dict(s=s, theta=theta, I=I.ravel(), name=os.path.basename(f)))
        people[pid] = imgs
        AMB[pid] = n_amb
    return people


AMB = {}


def subset(theta):
    """the official split (7 / 12 / 12 / 14 / 19 images): theta = 25.0 is subset 2, A+-35 E+40 (51.1 deg) is subset 3"""
    return 1 if theta <= 12 else 2 if theta <= 25.5 else 3 if theta <= 55 else 4 if theta <= 77 else 5


def enrol(imgs, train=(1, 2)):
    """photometric stereo on the enrolment lights: per pixel b = argmin sum_k (I_k - b.s_k)^2 over lit samples"""
    E = [im for im in imgs if subset(im["theta"]) in train]
    S = np.array([im["s"] for im in E]); I = np.array([im["I"] for im in E])        # K x 3, K x P
    lit = I > 0.08 * np.percentile(I, 99)                                           # shadowed samples carry no b
    b = np.zeros((I.shape[1], 3))
    for p in range(I.shape[1]):
        k = lit[:, p]
        if k.sum() >= 3:
            b[p] = np.linalg.lstsq(S[k], I[k, p], rcond=None)[0]
    rho = np.linalg.norm(b, axis=1); n = b / np.maximum(rho[:, None], 1e-9)
    return rho[:, None] * sh9(n), len(E)


def resid(y, B):
    mu, w, c = fit(y, B)
    return float((w * (y - mu) ** 2).sum())


def dark_trial(args):
    """one real probe at Nbar photons: predicted vs measured accept rates, the exact limit, and the optimal decision"""
    pid, k, Nbar, seed = args
    R = np.random.default_rng(seed)
    im = PEOPLE[pid][k]
    unit = im["I"] / im["I"].mean()                                                  # expected photons at Nbar = 1
    Phi = Nbar * unit
    lam = {q: resid(Phi, BASIS[q]) for q in BASIS}                                   # remainders, noiseless
    imp = min((v, q) for q, v in lam.items() if q != pid)[1]                         # the nearest impostor
    pa, _ = rates(Phi, BASIS[pid], 2.326)
    pb, _ = rates(Phi, BASIS[imp], 2.326)
    acc_a = acc_b = 0; right = wrong = answered = 0
    from laws import score
    for r in range(20):
        x = sensor(Phi, R)
        Nh, var, ok = inverse(x)
        idx = np.arange(len(Nh))
        za = score(Nh, ok, idx, BASIS[pid])[0]; zb = score(Nh, ok, idx, BASIS[imp])[0]
        acc_a += za <= 2.326; acc_b += zb <= 2.326
        lp = np.array([-resid(np.where(ok, Nh, 0), BASIS[q]) / 2 for q in IDS])       # optimal: all 38, equal priors
        post = np.exp(lp - lp.max()); post /= post.sum()
        w = int(np.argmax(post))
        if post[w] >= 0.99:
            answered += 1; right += IDS[w] == pid; wrong += IDS[w] != pid
    mA = Phi; mB = np.maximum(fit(Phi, BASIS[imp])[0], 1e-9)
    DB = bhattacharyya(mA, mB)
    lo, hi = bayes_bounds(DB)
    return dict(pid=pid, k=k, Nbar=Nbar, subset=subset(im["theta"]), pred_frr=1 - pa, obs_frr=1 - acc_a / 20,
                pred_far=pb, obs_far=acc_b / 20, lamA=lam[pid], lamB=lam[imp], DB=DB, lower=lo, upper=hi,
                answered=answered / 20, right=right / 20, wrong=wrong / 20)


PEOPLE = load()
IDS = sorted(PEOPLE)
BASIS = {}


def _init(basis):                       # spawned workers re-import this file: hand them the enrolled bases explicitly
    BASIS.update(basis)
if __name__ == "__main__":
    t0 = time.time()
    print("Extended Yale B, cropped: %d people, %d images (%dx box-integrated to 48 x 42); ambient frame subtracted for %d of %d people" % (
        len(PEOPLE), sum(len(v) for v in PEOPLE.values()), DS, sum(AMB.values()), len(AMB)))
    out = {}
    # LAW 2 -- energy in 9 dimensions, all 64 lights of each person (no model needed)
    fr = []
    for pid in IDS:
        M = np.array([im["I"] for im in PEOPLE[pid]])
        sv = np.linalg.svd(M, compute_uv=False) ** 2
        fr.append(sv[:9].sum() / sv.sum())
    fr = np.array(fr)
    print("\nLAW 2  energy of each person's images in the best 9 dimensions (convex Lambertian bound: >= 97.96%%)")
    print("   median %.2f%%   min %.2f%%   max %.2f%%" % (100 * np.median(fr), 100 * fr.min(), 100 * fr.max()))
    out["law2_energy9"] = dict(median=float(np.median(fr)), min=float(fr.min()), max=float(fr.max()))
    # ENROL + LAW 2+4 -- identification on unseen lights by the subspace residual
    nE = {}
    for pid in IDS:
        BASIS[pid], nE[pid] = enrol(PEOPLE[pid])
    print("\nENROL  photometric stereo from lights <= 25 deg off-axis: %d-%d images a person" % (min(nE.values()), max(nE.values())))
    print("\nLAW 2+4  identification on UNSEEN lights by the 9-dim subspace residual (38-way, nothing fitted to them)")
    rows, leftover = {}, {}
    for pid in IDS:
        for im in PEOPLE[pid]:
            sb = subset(im["theta"])
            if sb <= 2:
                continue
            y = im["I"]
            if y.sum() <= 0:
                continue
            # well-lit 8-bit images: plain least squares onto each subspace (the classical harmonic-image test)
            r = {q: float(np.sum((y - BASIS[q] @ np.linalg.lstsq(BASIS[q], y, rcond=None)[0]) ** 2)) for q in IDS}
            rows.setdefault(sb, []).append(min(r, key=r.get) == pid)
            mu = BASIS[pid] @ np.linalg.lstsq(BASIS[pid], y, rcond=None)[0]
            leftover.setdefault(sb, []).append(np.linalg.norm(y - mu) / max(np.linalg.norm(y), 1e-9))
    out["law24"] = {}
    for sb in sorted(rows):
        e = 1 - np.mean(rows[sb])
        print("   subset %d (%s deg): %3d probes   error %5.1f%%   own-subspace leftover |y - Py|/|y| median %.3f" % (
            sb, {3: "26-50", 4: "51-77", 5: "78-130"}[sb], len(rows[sb]), 100 * e, np.median(leftover[sb])))
        out["law24"][sb] = dict(n=len(rows[sb]), error=float(e), leftover=float(np.median(leftover[sb])))
    # LAW 5+6 in the dark -- real faces, real shadows, physical photon thinning through the sensor law
    R = np.random.default_rng(5)
    jobs = []
    for Nbar in (1, 2, 5, 10, 20, 50):
        for t in range(48):
            pid = IDS[int(R.integers(len(IDS)))]
            ks = [i for i, im in enumerate(PEOPLE[pid]) if subset(im["theta"]) == 3]
            jobs.append((pid, int(R.choice(ks)), Nbar, 100 + len(jobs)))
    with Pool(12, initializer=_init, initargs=(BASIS,)) as p:
        res = p.map(dark_trial, jobs)
    out["dark"] = res
    print("\nLAW 5+6  real probes (subset 3 lights) thinned to Nbar photons/pixel, Law 1 sensor; tau = 2.326; 48 probes x 20 draws")
    print("  Nbar   FRR pred  FRR meas   FAR pred  FAR meas   | Law 6 lower <= oracle-free optimal | optimal: answered  wrong/answered")
    for Nbar in (1, 2, 5, 10, 20, 50):
        rr = [r for r in res if r["Nbar"] == Nbar]
        m = lambda k: np.mean([r[k] for r in rr])
        ans = m("answered"); wr = sum(r["wrong"] for r in rr) / max(sum(r["answered"] for r in rr), 1e-9)
        print("  %4d   %6.3f   %6.3f     %6.3f   %6.3f    | bound %.4f  (upper %.4f)          |          %5.2f      %6.4f" % (
            Nbar, m("pred_frr"), m("obs_frr"), m("pred_far"), m("obs_far"), m("lower"), m("upper"), ans, wr))
    json.dump(out, open("yale_results.json", "w"), indent=1, default=float)
    print("\n%.0f s" % (time.time() - t0))
