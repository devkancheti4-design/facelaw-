"""The frozen SPEC 50 trees (LAW 7c rank, LAW 8 dx dy ds) as C: every table an initialised static array, integer leaves
round(4096 v) as frozen by darkface_v2.freeze_model (rounded ONCE); the same integer sums as darkface_v2.score_int
(checked equal on design rows before use).   build(laws, out_dir) -> .so path;  CTrees(path).run(name, X)"""
import ctypes, hashlib, json, os, subprocess
import numpy as np

FIELDS = ("feat", "thr", "left", "right", "val", "leaf")


def build(laws, out_dir):
    blob = json.dumps(laws, sort_keys=True).encode()
    tag = hashlib.sha256(blob).hexdigest()[:16]
    so = os.path.join(out_dir, "ctrees_%s.so" % tag)
    if os.path.exists(so):
        return so
    src = ["#include <stdint.h>"]
    for m, law in sorted(laws.items()):
        starts = np.concatenate([[0], np.cumsum([len(t["leaf"]) for t in law["trees"]])]).astype(int)
        for f in FIELDS:
            vals = [v for t in law["trees"] for v in t[f]]
            src.append("static const int32_t %s_%s[%d] = {%s};" % (m, f, len(vals), ",".join(map(str, vals))))
        src.append("static const int32_t %s_start[%d] = {%s};" % (m, len(starts), ",".join(map(str, starts))))
        src.append("""
void ct_%(m)s(const int32_t *X, int64_t n, int32_t nf, int64_t *out){
  for (int64_t i = 0; i < n; i++){
    const int32_t *x = X + i * nf; int64_t s = %(base)dLL;
    for (int t = 0; t < %(nt)d; t++){
      int32_t b = %(m)s_start[t], j = 0;
      while (!%(m)s_leaf[b + j]) j = (x[%(m)s_feat[b + j]] <= %(m)s_thr[b + j]) ? %(m)s_left[b + j] : %(m)s_right[b + j];
      s += %(m)s_val[b + j];
    }
    out[i] = s;
  }
}""" % dict(m=m, base=law["base"], nt=len(law["trees"])))
    cp = so[:-3] + ".c"
    open(cp, "w").write("\n".join(src) + "\n")
    subprocess.run(["cc", "-O2", "-fwrapv", "-shared", "-fPIC", "-o", so, cp], check=True)
    return so


class CTrees:
    def __init__(self, so):
        self.lib = ctypes.CDLL(so)

    def run(self, m, X):
        X = np.ascontiguousarray(X, np.int32)
        out = np.zeros(len(X), np.int64)
        getattr(self.lib, "ct_" + m)(X.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(len(X)), ctypes.c_int32(X.shape[1]),
                                     out.ctypes.data_as(ctypes.c_void_p))
        return out
