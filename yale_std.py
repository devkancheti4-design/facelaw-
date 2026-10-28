"""The standard Yale B protocols, our laws vs published numbers.
A  (Ghazi & Ekenel 2016, arXiv 1606.02894, Table 3): Extended Yale B, 38 people; train on subset 1 (7 images, lights
   <= 12 deg); test subsets 2-5.  Ours: photometric stereo on the 7 training images -> 9 harmonic basis images ->
   the identity is the subspace with the smallest residual (Laws 2 + 4).  Baselines run by us on the SAME split:
   nearest neighbour (correlation) and the 3-dim linear subspace of the training images.
B  (Lee, Ho, Kriegman 2005, TPAMI, Table 2): the original 10 people (yaleB01-10), train subsets 1 + 2, test 3, 4."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y


def lsq_res(y, B):
    return float(np.sum((y - B @ np.linalg.lstsq(B, y, rcond=None)[0]) ** 2))


def run(ids, train, tests):
    basis = {p: Y.enrol(Y.PEOPLE[p], train)[0] for p in ids}
    lin3 = {}
    for p in ids:
        M = np.array([im["I"] for im in Y.PEOPLE[p] if Y.subset(im["theta"]) in train]).T
        U = np.linalg.svd(M, full_matrices=False)[0]
        lin3[p] = U[:, :3]
    trainset = [(p, im["I"] / max(np.linalg.norm(im["I"]), 1e-9)) for p in ids for im in Y.PEOPLE[p] if Y.subset(im["theta"]) in train]
    out = {}
    for sb in tests:
        n = ok_h = ok_nn = ok_l3 = 0
        for p in ids:
            for im in Y.PEOPLE[p]:
                if Y.subset(im["theta"]) != sb:
                    continue
                y = im["I"]; n += 1
                ok_h += min(ids, key=lambda q: lsq_res(y, basis[q])) == p
                ok_l3 += min(ids, key=lambda q: lsq_res(y, lin3[q])) == p
                yn = y / max(np.linalg.norm(y), 1e-9)
                ok_nn += max(trainset, key=lambda t: float(t[1] @ yn))[0] == p
        out[sb] = dict(n=n, ours=100 * ok_h / n, nn=100 * ok_nn / n, lin3=100 * ok_l3 / n)
    return out


if __name__ == "__main__":
    t0 = time.time()
    counts = np.bincount([Y.subset(im["theta"]) for im in Y.PEOPLE["yaleB01"]], minlength=6)[1:]
    print("official subset sizes per person: %s  (expected 7 / 12 / 12 / 14 / 19)" % "/".join(map(str, counts)))
    A = run(Y.IDS, (1,), (2, 3, 4, 5))
    vgg6, vgg7, vgg7p = {2: 100, 3: 88.38, 4: 46.62, 5: 13.85}, {2: 100, 3: 92.32, 4: 52.44, 5: 18.28}, {4: 75.56, 5: 76.32}
    print("\nPROTOCOL A  Extended Yale B, 38 people, train subset 1 (7 images), test 2-5 -- RECOGNITION RATE %")
    print("  subset  probes   ours (Laws 2+4)   VGG-Face FC6   VGG-Face FC7   VGG FC7 +preproc   | same split, run by us: NN-correlation  3-dim subspace")
    for sb, r in A.items():
        print("     %d    %4d       %6.2f          %6.2f         %6.2f           %6s          |                   %6.2f          %6.2f" % (
            sb, r["n"], r["ours"], vgg6[sb], vgg7[sb], ("%.2f" % vgg7p[sb]) if sb in vgg7p else "  -", r["nn"], r["lin3"]))
    orig = [p for p in Y.IDS if int(p[5:]) <= 10]
    B = run(orig, (1, 2), (3, 4))
    lee = [("Correlation", 23.3, 73.6), ("Eigenfaces", 25.8, 75.7), ("Linear subspace", 0.0, 15.0), ("Cones-attached", 0.0, 8.6),
           ("Harmonic images (no cast)", 0.0, 3.571), ("Harmonic images-cast", 0.0, 2.7), ("9PL (simulated)", 0.0, 2.8),
           ("Gradient angle", 0.0, 1.4), ("Cones-cast", 0.0, 0.0), ("9PL (real)", 0.0, 0.0)]
    print("\nPROTOCOL B  original 10 people (%s..%s), train subsets 1+2, test 3, 4 -- ERROR RATE %%" % (orig[0], orig[-1]))
    print("  %-28s subset 3   subset 4" % "method")
    print("  %-28s %7.2f   %7.2f      <- %d / %d probes" % ("OURS (Laws 2+4)", 100 - B[3]["ours"], 100 - B[4]["ours"], B[3]["n"], B[4]["n"]))
    print("  %-28s %7.2f   %7.2f" % ("NN-correlation (run by us)", 100 - B[3]["nn"], 100 - B[4]["nn"]))
    print("  %-28s %7.2f   %7.2f" % ("3-dim subspace (run by us)", 100 - B[3]["lin3"], 100 - B[4]["lin3"]))
    for m, e3, e4 in lee:
        print("  %-28s %7.2f   %7.2f      (Lee et al. 2005, Table 2)" % (m, e3, e4))
    print("\n%.0f s" % (time.time() - t0))
