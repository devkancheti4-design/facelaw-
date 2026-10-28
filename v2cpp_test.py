"""EXACTNESS of the C++ port (v2cpp/libfacelaw2) against the reference (v2ref.py), stage by stage, on real photos: every
array must be EQUAL (not close).  The face box (own Viola-Jones) is compared with OpenCV's by overlap instead.
    python3 -I v2cpp_test.py [path/to/libfacelaw2.dylib|.so]"""
import ctypes, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import v2ops as O
import v2ref as R2

HERE = os.path.dirname(os.path.abspath(__file__))
D, I, U8 = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint8)


def lib(path=None):
    L = ctypes.CDLL(path or os.path.join(HERE, "v2cpp", "libfacelaw2.dylib"))
    L.fl2_score.restype = ctypes.c_double
    L.fl2_score96.restype = ctypes.c_double
    return L


def dp(a):
    return np.ascontiguousarray(a, np.float64).ctypes.data_as(D)


class C:
    def __init__(self, path=None):
        self.L = lib(path)

    def gauss(self, x, k8):
        o = np.empty_like(x); self.L.fl2_gauss(dp(x), x.shape[0], x.shape[1], k8, o.ctypes.data_as(D)); return o

    def u8(self, x):
        o = np.empty_like(x); self.L.fl2_u8(dp(x), x.shape[0], x.shape[1], o.ctypes.data_as(D)); return o

    def linear_of(self, rgb):
        a = np.ascontiguousarray(rgb, np.uint8); h, w = a.shape[:2]; ch = 1 if a.ndim == 2 else a.shape[2]
        o = np.empty((h, w)); self.L.fl2_linear_of(a.ctypes.data_as(U8), w, h, ch, o.ctypes.data_as(D)); return o

    def eye_codes(self, F):
        o = np.empty(F.shape, np.int32); self.L.fl2_eye_codes(dp(F), F.shape[0], F.shape[1], o.ctypes.data_as(I)); return o.astype(np.int64)

    def find_eyes(self, disp, box):
        o = np.empty(5); ok = self.L.fl2_find_eyes(dp(disp), disp.shape[0], disp.shape[1], dp(np.asarray(box, float)), o.ctypes.data_as(D))
        self.ev = o[4]
        return (o[:2].copy(), o[2:4].copy()) if ok else None

    def crop(self, img, eL, eR):
        o = np.empty((48, 42)); self.L.fl2_crop(dp(img), img.shape[0], img.shape[1], dp(np.r_[eL, eR]), o.ctypes.data_as(D)); return o

    def map_bins(self, c):
        o = np.empty(2652, np.uint8); self.L.fl2_map_bins(dp(c), o.ctypes.data_as(U8)); return o.astype(np.int64)

    def probe_all(self, c):
        b = np.empty((9, 2652), np.uint8); s = np.empty((9, 2652), np.uint8)
        self.L.fl2_probe_all(dp(c), b.ctypes.data_as(U8), s.ctypes.data_as(U8))
        return [(b[p].astype(np.int64), s[p].astype(np.int64)) for p in range(9)]

    def score(self, probes, gb):
        b = np.ascontiguousarray(np.array([p[0] for p in probes], np.uint8)); s = np.ascontiguousarray(np.array([p[1] for p in probes], np.uint8))
        g = np.ascontiguousarray(np.asarray(gb, np.uint8))
        return self.L.fl2_score(b.ctypes.data_as(U8), s.ctypes.data_as(U8), g.ctypes.data_as(U8))

    def crop96(self, img, eL, eR):
        o = np.empty((96, 84)); self.L.fl2_crop96(dp(img), img.shape[0], img.shape[1], dp(np.r_[eL, eR]), o.ctypes.data_as(D)); return o

    def map_bins96(self, c):
        o = np.empty(10584, np.uint8); self.L.fl2_map_bins96(dp(c), o.ctypes.data_as(U8)); return o.astype(np.int64)

    def probe_codes96(self, c):
        b = np.empty(10584, np.uint8); s = np.empty(10584, np.uint8)
        self.L.fl2_probe_codes96(dp(c), b.ctypes.data_as(U8), s.ctypes.data_as(U8)); return b.astype(np.int64), s.astype(np.int64)

    def score96(self, b, s, gb):
        b, s, g = (np.ascontiguousarray(np.asarray(v, np.uint8)) for v in (b, s, gb))
        return self.L.fl2_score96(b.ctypes.data_as(U8), s.ctypes.data_as(U8), g.ctypes.data_as(U8))

    def face_box(self, lin):
        o = np.empty(4); ok = self.L.fl2_face_box(dp(lin), lin.shape[0], lin.shape[1], o.ctypes.data_as(D))
        return o if ok else None

    def template(self, rgb):
        a = np.ascontiguousarray(rgb, np.uint8); h, w = a.shape[:2]; ch = 1 if a.ndim == 2 else a.shape[2]
        e = np.empty(4); bx = np.empty(4); m = np.empty(2652, np.uint8); b = np.empty((9, 2652), np.uint8); s = np.empty((9, 2652), np.uint8)
        r = self.L.fl2_template(a.ctypes.data_as(U8), w, h, ch, e.ctypes.data_as(D), bx.ctypes.data_as(D), m.ctypes.data_as(U8),
                                b.ctypes.data_as(U8), s.ctypes.data_as(U8))
        return r, e, bx, m, b, s


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


if __name__ == "__main__":
    t0 = time.time()
    c = C(sys.argv[1] if len(sys.argv) > 1 else None)
    R2.load()
    rng = np.random.default_rng(5)
    bad = {}
    def check(name, a, b):
        same = np.array_equal(np.asarray(a), np.asarray(b))
        bad.setdefault(name, [0, 0]); bad[name][0] += 1; bad[name][1] += (not same)
        return same
    for k in range(0, O.K8MAX + 1):
        x = rng.random((37, 29)) * 255
        check("gauss (24 sigmas)", c.gauss(x, k), O.gauss(x, k))
    for _ in range(5):
        x = rng.random((50, 61)) * rng.uniform(1, 400)
        check("display bytes", c.u8(x), O.u8(x).astype(float))
        rgb = rng.integers(0, 256, (31, 27, 3)); check("linear light", c.linear_of(rgb), O.linear_of(rgb))
    import yale as Y, eyes2 as E2
    items = []
    TR = json.load(open(os.path.join(HERE, "yale_eyes_truth.json")))["eyes"]
    for p in Y.IDS[::5]:
        if p not in TR:
            continue
        for im in Y.PEOPLE[p][::9]:
            if im["I"].sum() <= 0:
                continue
            Iy = Y.read_pgm(os.path.join(Y.ROOT, p, im["name"])).astype(float)
            tl, tr = np.array(TR[p][0]), np.array(TR[p][1]); w = np.linalg.norm(tr - tl) / 0.45
            items.append(("yale " + im["name"], Iy, Iy, np.array([(tl[0] + tr[0]) / 2 - w / 2, (tl[1] + tr[1]) / 2 - 0.36 * w, w, w])))
    for nm, lin, t in E2.user_items():
        box = c.face_box(lin)
        if box is not None:
            items.append(("owner " + nm, lin, O.u8(lin).astype(float), box))
    n_eq = 0
    for nm, img, disp, box in items:
        F = R2.eye_frame(disp, np.array([box[0] + box[2] / 2, box[1] + 0.36 * box[2]]), 0.45 * box[2], 1.0, 0.0, False)
        check("eye codes", c.eye_codes(F), R2.eye_codes(F))
        ec, ep = c.find_eyes(disp, box), R2.find_eyes(disp, box)
        check("eye evidence (the GATE's input)", c.ev, R2.LAST["evidence"])
        if not check("eyes", np.r_[ec[0], ec[1]], np.r_[ep[0], ep[1]]):
            print("   eyes differ on %s: C++ %s  Python %s" % (nm, np.r_[ec[0], ec[1]], np.r_[ep[0], ep[1]]))
            continue
        cc, cp = c.crop(img, *ep), R2.crop(img, *ep)
        check("crop", cc, cp)
        mb = R2.map_bins(cp); check("map bins", c.map_bins(cp), mb)
        pa_c, pa_p = c.probe_all(cp), R2.probe_all(cp)
        check("probe codes (9 placements)", np.array(pa_c), np.array(pa_p))
        check("score", c.score(pa_c, mb), float(R2.score(pa_p, mb)[0]))
        c96c, c96p = c.crop96(img, *ep), R2.crop96(img, *ep)            # the full-light law
        check("crop 96 x 84", c96c, c96p)
        import unlock_law as U
        for x in (c96p, U.dark_ae(c96p, 5 / 4.0, rng)):                  # in light, and a try at N = 5 (N / 4 per pixel)
            with R2.law96():
                mb96 = R2.map_bins(c96p); pb, pc = R2.probe_codes(x); s96 = float(R2.score([(pb, pc)], mb96)[0])
            check("map bins 96", c.map_bins96(c96p), mb96)
            cb, cs = c.probe_codes96(x)
            check("probe codes 96 (light and dark)", np.r_[cb, cs], np.r_[pb, pc])
            check("score 96", c.score96(cb, cs, mb96), s96)
        n_eq += 1
    print("EXACTNESS on %d real photos (%d Yale, %d the owner's) + random arrays  [%.0fs]" % (
        len(items), sum(i[0].startswith("yale") for i in items), sum(i[0].startswith("owner") for i in items), time.time() - t0))
    for k, (n, b) in bad.items():
        print("   %-28s %4d compared, %d differ" % (k, n, b))
    import finder as FD
    ious = []
    for nm, lin, t in E2.user_items():
        b1 = c.face_box(lin); b2 = E2.face_box(lin)
        if b1 is None or b2 is None:
            print("   face box %-22s C++ %s  OpenCV %s" % (nm, b1 is not None, b2 is not None)); continue
        ious.append(iou(b1, b2))
    print("FACE BOX (own Viola-Jones vs OpenCV) on the owner's %d images: IoU min %.3f, median %.3f" % (len(ious), min(ious), np.median(ious)))
    rgb = (255 * np.clip(E2.user_items()[0][1] / 255.0, 0, 1) ** (1 / 2.2)).astype(np.uint8)
    t1 = time.time(); r = c.template(rgb); dt = time.time() - t1
    print("whole template in C++ (%dx%d image): status %d, %.0f ms" % (rgb.shape[1], rgb.shape[0], r[0], 1000 * dt))
