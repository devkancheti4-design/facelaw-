"""Published illumination-invariant descriptors, run by us on the SAME split as our laws (protocol A: gallery = the 7
subset-1 images of each of the 38 people; probes = subsets 2-5), nearest neighbour, at 48x42 and at full 192x168.
  Gradientfaces (Zhang, Tian, Song, Lu 2009): G = atan2(dI/dy, dI/dx) of the Gaussian-smoothed image; angular L1
  Weberfaces    (Wang, Li, Yang, Liao 2011):  W = atan(alpha sum_{8 nbrs} (I - I_n) / I), alpha = 2, smoothed; L1
  Tan-Triggs    (Tan & Triggs 2010): gamma 0.2, DoG (1, 2), contrast equalisation (a = 0.1, tau = 10); L1
These are reimplementations from the papers' formulas, not the authors' code."""
import glob, os, re, sys, time
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y


def gradientface(I, s=0.75):
    G = gaussian_filter(I, s)
    return np.arctan2(np.gradient(G, axis=0), np.gradient(G, axis=1) + 1e-9)


def weberface(I, s=1.0, alpha=2.0):
    G = gaussian_filter(I, s) + 1.0
    k = -np.ones((3, 3)); k[1, 1] = 8
    return np.arctan(alpha * convolve(G, k, mode="nearest") / G)


def tantriggs(I, gamma=0.2, s0=1.0, s1=2.0, a=0.1, tau=10.0):
    X = np.power(I + 1.0, gamma)
    X = gaussian_filter(X, s0) - gaussian_filter(X, s1)
    X = X / np.mean(np.abs(X) ** a) ** (1 / a)
    X = X / np.mean(np.minimum(tau, np.abs(X)) ** a) ** (1 / a)
    return tau * np.tanh(X / tau)


def angdist(a, B):
    d = np.abs(a[None] - B)
    return np.minimum(d, 2 * np.pi - d).reshape(len(B), -1).sum(1)


def l1(a, B):
    return np.abs(a[None] - B).reshape(len(B), -1).sum(1)


def load_full():
    """full-resolution images (192 x 168), the same files and the same subsets as the laws use"""
    out = {}
    for p in Y.IDS:
        d = os.path.join(Y.ROOT, p)
        ims = []
        for f in sorted(glob.glob(d + "/*.pgm")):
            m = re.search(r"A([+-]\d+)E([+-]\d+)\.pgm$", f)
            if not m:
                continue
            a, e = np.radians(int(m.group(1))), np.radians(int(m.group(2)))
            th = np.degrees(np.arccos(np.clip(np.cos(a) * np.cos(e), -1, 1)))
            ims.append((Y.subset(th), Y.read_pgm(f)))
        out[p] = ims
    return out


if __name__ == "__main__":
    t0 = time.time()
    full = load_full()
    for res_name, prep in (("48x42", lambda I: Y.box(I)), ("192x168", lambda I: I)):
        feats = {}
        for name, fn, dist in (("Gradientfaces", gradientface, angdist), ("Weberfaces", weberface, l1), ("Tan-Triggs", tantriggs, l1)):
            gal, lab = [], []
            for p in Y.IDS:
                for sb, I in full[p]:
                    if sb == 1:
                        gal.append(fn(prep(I))); lab.append(p)
            gal = np.array(gal)
            acc = {}
            for p in Y.IDS:
                for sb, I in full[p]:
                    if sb >= 2:
                        acc.setdefault(sb, []).append(lab[int(np.argmin(dist(fn(prep(I)), gal)))] == p)
            feats[name] = {sb: 100 * np.mean(v) for sb, v in acc.items()}
        print("\n%s, protocol A (gallery: 7 subset-1 images x 38 people), nearest neighbour -- recognition %%" % res_name)
        print("  subset   " + "   ".join("%-13s" % n for n in feats))
        for sb in (2, 3, 4, 5):
            print("     %d     " % sb + "   ".join("%13.2f" % feats[n][sb] for n in feats))
    print("\n%.0f s" % (time.time() - t0))
