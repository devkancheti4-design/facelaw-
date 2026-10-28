"""The data law on the CAST-SHADOW remainders (yale_cast_rem.pkl), cross-fitted by people as before -> tables for the
synthesis (yale_cast_tables.json)."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yale_law import table, decide
D = pickle.load(open("yale_cast_rem.pkl", "rb"))
ids = D["ids"]
folds = [[p for i, p in enumerate(ids) if i % 2 == k] for k in (0, 1)]
rem = [[r for r in D["rem"] if r[0] in set(f)] for f in folds]
T = [table(rem[1 - k]) for k in (0, 1)]
acc = {}
for k in (0, 1):
    for sb, v in decide(rem[k], T[k][0]).items():
        acc.setdefault(sb, []).extend(v)
print("cast-shadow remainders + data law (table), cross-fitted -- recognition %")
for sb in sorted(acc):
    print("   subset %d: %6.2f" % (sb, 100 * np.mean(acc[sb])))
json.dump(dict(laws=[t[0].tolist() for t in T], seen=[((t[1] + t[2]) > 2).tolist() for t in T], folds=folds),
          open("yale_cast_tables.json", "w"))
pickle.dump(dict(rem=rem, folds=folds), open("yale_cast_law_rem.pkl", "wb"))
