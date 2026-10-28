"""The sealed PIE test (pie_protocol.txt, hashed before running)."""
import json, os, sys
import numpy as np
import scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yale_gate import sectors
from yale_baselines import gradientface, weberface, tantriggs, angdist, l1

X = sio.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/pie/lights=27.mat"))["fea"]
X = X.astype(float).reshape(-1, 32, 32).transpose(0, 2, 1).reshape(68, 21, 32, 32)
TRAIN = 10
gal_idx, probes = [(p, TRAIN) for p in range(68)], [(p, k) for p in range(68) for k in range(21) if k != TRAIN]
res = {}
# OURS: the lane-authored local law
G = np.array([sectors(X[p, k]) for p, k in gal_idx], np.int16)
ok = 0
for p, k in probes:
    s = sectors(X[p, k]).astype(np.int16)
    d = np.abs(G - s[None]); d = np.minimum(d, 8 - d).sum(1)
    ok += int(np.argmin(d)) == p
res["OURS (synthesis local law)"] = 100 * ok / len(probes)
for name, fn, dist in (("Gradientfaces", gradientface, angdist), ("Weberfaces", weberface, l1), ("Tan-Triggs", tantriggs, l1)):
    G = np.array([fn(X[p, k]) for p, k in gal_idx])
    ok = sum(int(np.argmin(dist(fn(X[p, k]), G))) == p for p, k in probes)
    res[name + " (my run)"] = 100 * ok / len(probes)
res["VGG-Face FC6 (published)"] = 93.16
res["VGG-Face FC7 (published)"] = 92.87
print("SEALED PIE TEST -- 68 people, train 1 frontal image, test 20 lights (%d probes) -- recognition %%" % len(probes))
for k, v in sorted(res.items(), key=lambda t: -t[1]):
    print("   %-28s %6.2f" % (k, v))
ours = res["OURS (synthesis local law)"]
first = all(ours >= v for k, v in res.items() if not k.startswith("OURS"))
print("\nVERDICT (rule fixed in pie_protocol.txt): %s" % ("No. 1 on this test" if first else "NOT No. 1 on this test"))
json.dump(res, open("pie_results.json", "w"), indent=1)
