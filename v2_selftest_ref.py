"""The REFERENCE side of v2cpp/selftest.cpp: the same seeded random arrays through v2ref.py / v2ops.py, the same FNV-1a 64
checksums -> v2cpp/selftest_expected.txt (what any machine's build must print).
    python3 -I v2_selftest_ref.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import v2ops as O
import v2ref as R2

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = (1 << 64) - 1
S = [20261028]


def rnd():
    S[0] = (S[0] * 6364136223846793005 + 1442695040888963407) & M64
    return (S[0] >> 11) * (1.0 / 9007199254740992.0)


def rmat(h, w, scale):
    return np.array([rnd() * scale for _ in range(h * w)]).reshape(h, w)


def fnv(b, h=1469598103934665603):
    for x in bytes(b):
        h = ((h ^ x) * 1099511628211) & M64
    return h


if __name__ == "__main__":
    R2.load()
    out = []
    for k in range(25):
        out.append(("gauss_%d" % k, fnv(O.gauss(rmat(37, 29, 255.0), k).astype("<f8").tobytes())))
    out.append(("display_u8", fnv(O.u8(rmat(50, 61, 300.0)).astype("<f8").tobytes())))
    rgb = np.array([int(rnd() * 256.0) for _ in range(31 * 27 * 3)], np.uint8).reshape(31, 27, 3)
    out.append(("linear_of", fnv(O.linear_of(rgb).astype("<f8").tobytes())))
    F = rmat(80, 110, 255.0)
    C = R2.eye_codes(F)
    out.append(("eye_codes", fnv(C.astype("<i4").tobytes())))
    out.append(("score_map", fnv(R2.score_map(C).astype("<i8").tobytes())))
    img = rmat(300, 260, 255.0)
    e = R2.find_eyes(img, (40.0, 50.0, 180.0, 180.0))
    out.append(("find_eyes", fnv(np.array([e[0][0], e[0][1], e[1][0], e[1][1], R2.LAST["evidence"]], "<f8").tobytes())))
    c = R2.crop(img, np.array([101.25, 120.5]), np.array([158.75, 123.0]))
    out.append(("crop", fnv(c.astype("<f8").tobytes())))
    m = R2.map_bins(c)
    out.append(("map_bins", fnv(m.astype(np.uint8).tobytes())))
    pa = R2.probe_all(c)
    b = np.array([p[0] for p in pa], np.uint8); s = np.array([p[1] for p in pa], np.uint8)
    out.append(("probe_all", fnv(s.tobytes(), fnv(b.tobytes()))))
    out.append(("score", fnv(np.array([float(R2.score(pa, m)[0])], "<f8").tobytes())))
    c96 = R2.crop96(img, np.array([101.25, 120.5]), np.array([158.75, 123.0]))      # the full-light law
    out.append(("crop96", fnv(c96.astype("<f8").tobytes())))
    with R2.law96():
        m96 = R2.map_bins(c96); b96, s96 = R2.probe_codes(c96)
        out.append(("map_bins96", fnv(m96.astype(np.uint8).tobytes())))
        out.append(("probe_codes96", fnv(s96.astype(np.uint8).tobytes(), fnv(b96.astype(np.uint8).tobytes()))))
        out.append(("score96", fnv(np.array([float(R2.score([(b96, s96)], m96)[0])], "<f8").tobytes())))
    open(os.path.join(HERE, "v2cpp", "selftest_expected.txt"), "w").write("".join("%s %016x\n" % kv for kv in out))
    print("".join("%s %016x\n" % kv for kv in out), end="")
