"""COST of every method in the head-to-head: what it must store, what it computes, what it learns, and whether it fits a
2011-class flip phone.  Measured here (this Mac, one core): the time to describe one face and to compare two; the bytes of
the model and of a template.  Counted by formula: arithmetic per face and per comparison, split into integer and
floating-point work (the formulas are in OPS below).
FLIP PHONE (the stated assumptions): an ARM9 at 200 MHz with NO floating-point unit, 2 MB of working memory, a VGA
camera.  An integer operation costs 1.5 cycles; a double-precision operation done in software about 40 (20-100 in
practice); a fixed-point port turns each into about 3 integer operations.
    python3 -I h2h_cost.py"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import h2h_all as H
import v2ref as R2
import classics as CL

MHZ, CYC_INT, CYC_SOFT, CYC_FIXED = 200e6, 1.5, 40.0, 3.0


def taps(sigma):
    return 2 * int(4 * sigma + 0.5) + 1


def gauss_flops(px, sigma):
    r = taps(sigma) // 2
    return 2 * px * (1 + 3 * r)


def OPS(name, res):
    """(float ops, integer ops) to describe ONE placement of a face, and per comparison (float, integer)"""
    px = 48 * 42 if res == 48 else 96 * 84
    pts = 2652
    if name == "FACE LAW v2":
        var = 504 * 12 + 504 * 9 + gauss_flops(px, 2.0) + px * 3
        sc = sum(gauss_flops(px, s) + px * 9 + px * 8 for s in (0.75, 1.5, 3.0))
        return (px * 3 + var + sc + pts * 3, pts * 30), (0, 2 * 9 * pts * 5)
    if name == "FACE LAW v1":
        var = 504 * 12 + 504 * 9 + gauss_flops(px, 2.0) + px * 3
        sc = sum(gauss_flops(px, s) + px * 12 for s in (0.75, 1.5, 3.0))
        return (var + sc, pts * 30), (0, 9 * pts * 8)
    if name == "Gradientfaces":
        return (gauss_flops(px, 0.75) + px * 2 + px * 50, 0), (px * 5, 0)
    if name == "Weberfaces":
        return (gauss_flops(px, 1.0) + px * 17 + px * 51, 0), (px * 3, 0)
    if name == "Tan-Triggs":
        return (px * 50 + gauss_flops(px, 1.0) + gauss_flops(px, 2.0) + px * 2 * 55 + px * 50, 0), (px * 3, 0)
    if name == "LBP":
        return (px * 28, px * 17), (2891 * 5, 0)
    if name == "LTP":
        return (OPS("Tan-Triggs", res)[0][0] + px * 28, px * 34), (5782 * 5, 0)
    if name == "LGBPHS":
        n = (res + 32) * (res * 7 // 8 + 32)
        return (41 * 5 * n * np.log2(n) + 40 * n * 6 + 40 * px * 3, 40 * px * 25), (0, 115640 * 2)
    if name == "HOG":
        d = 1512 if res == 48 else 7020
        return (px * 62 + d * 6, 0), (d * 3, 0)
    if name == "Eigenfaces" or name == "Eigenfaces-3":
        return (2 * 100 * px, 0), (300, 0)
    if name == "Fisherfaces":
        return (2 * 17 * px, 0), (51, 0)
    raise KeyError(name)


def nbytes(obj):
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(o) for o in obj)
    return 0


if __name__ == "__main__":
    t0 = time.time()
    C48, C96 = H.crops_table()
    ppl = sorted(C48)
    faces48 = [C48[p][k][2] for p in ppl[:6] for k in range(0, len(C48[p]), 9)][:40]
    faces96 = [C96[p][k][2] for p in ppl[:6] for k in range(0, len(C96[p]), 9)][:40]
    D = json.load(open(os.path.join(H.HERE, "v2_design.json")))
    Z = R2.load()
    gmean = {48: float(np.mean([f.mean() for f in faces48])), 96: float(np.mean([f.mean() for f in faces96]))}
    tr48 = [C48[p][k][2] for p in ppl for k in range(len(C48[p]))]; tr96 = [C96[p][k][2] for p in ppl for k in range(len(C96[p]))]
    lab = [p for p in ppl for _ in C48[p]]
    fit = {}
    models = {}
    for name, ctor in H.LEARNED:
        for res, tr in ((48, tr48), (96, tr96)):
            t = time.time(); models["%s@%d" % (name, res)] = ctor(np.array([H.norm_b(c, gmean[res]) for c in tr]), np.array(lab))
            fit["%s@%d" % (name, res)] = time.time() - t
    rows = []
    def timeit(fn, reps=3):
        fn(); t = time.time()
        for _ in range(reps):
            fn()
        return (time.time() - t) / reps
    # FACE LAW v2: the exact reference (numpy) and the C++ port
    import v2cpp_test as VT
    cpp = VT.C()
    x = faces48[0]
    m = R2.map_bins(x); pa = R2.probe_all(x)
    t_py = timeit(lambda: R2.probe_all(x)); t_cpp = timeit(lambda: cpp.probe_all(x), 20)
    import subprocess, glob
    SPB = os.environ.get("BENCHDIR", "")
    cmp_us = float("nan")
    if SPB and os.path.exists(os.path.join(SPB, "bench")):
        o = subprocess.run([os.path.join(SPB, "bench")] + sorted(glob.glob(os.path.join(SPB, "img*.ppm"))), capture_output=True, text=True).stdout
        cmp_us = float(o.split("symmetric comparisons, ")[1].split(" us")[0])
    rows.append(dict(method="FACE LAW v2", res=48, describe_ms_numpy=1000 * t_py, describe_ms_cpp=1000 * t_cpp,
                     compare_us_cpp=cmp_us, template_bytes=2652 + 9 * 2652, template_bytes_packed=(2652 * 3 + 9 * 2652 * 5) // 8 + 1,
                     model_bytes=768 + 31875, model_note="FW 768 + eye law T 31,875 (8-bit entries); + face detector",
                     training="counted log-odds tables from ~1,600 lab photos of 37 people; seconds, no gradient descent"))
    import dark_law as DL, unlock_law as U
    t_v1 = timeit(lambda: U.probe_feats(x))
    rows.append(dict(method="FACE LAW v1", res=48, describe_ms_numpy=9000 * t_v1, template_bytes=2652 + 9 * 2652,
                     template_bytes_packed=(2652 * 4 + 9 * 2652 * 6) // 8 + 1, model_bytes=512, model_note="F 512 (8-bit)",
                     training="a counted 512-entry log-odds table"))
    for res, faces in ((48, faces48), (96, faces96)):
        xb = H.norm_b(faces[0], gmean[res])
        for name, fn, sim, dt in H.CLASSIC:
            d = np.asarray(fn(xb)).astype(dt)
            B = np.array([np.asarray(fn(H.norm_b(f, gmean[res]))).astype(dt) for f in faces])
            rows.append(dict(method=name, res=res, describe_ms_numpy=9000 * timeit(lambda: fn(xb)), compare_us_numpy=1e6 * timeit(lambda: sim(d, B[:1]), 50),
                             template_bytes=int(d.nbytes), model_bytes=(40 * 33 * 33 * 8 if name == "LGBPHS" else 256 if name in ("LBP", "LTP") else 0),
                             model_note={"LGBPHS": "40 complex 33x33 Gabor kernels", "LBP": "u2 table", "LTP": "u2 table"}.get(name, "none"),
                             training="none (hand-designed)"))
        for name, _ in H.LEARNED:
            mdl = models["%s@%d" % (name, res)]
            d = mdl(xb)
            rows.append(dict(method=name, res=res, describe_ms_numpy=9000 * timeit(lambda: mdl(xb)), compare_us_numpy=1e6 * timeit(lambda: CL.cos_sim(d, d[None]), 50),
                             template_bytes=int(d.nbytes), model_bytes=int(mdl.W.astype(np.float32).nbytes + mdl.mu.astype(np.float32).nbytes),
                             model_note="projection + mean (float32)", training="%s on %d lab photos: %.1f s" % ("PCA" if "Eigen" in name else "PCA + LDA", len(tr48), fit["%s@%d" % (name, res)])))
    for r in rows:
        (fl, it), (cfl, cit) = OPS(r["method"], r["res"])
        r.update(flops_per_face=9 * fl, intops_per_face=9 * it, flops_per_compare=cfl, intops_per_compare=cit)
        r["flip_ms_softfloat"] = 1000 * (r["flops_per_face"] * CYC_SOFT + r["intops_per_face"] * CYC_INT) / MHZ
        r["flip_ms_fixedpoint"] = 1000 * ((r["flops_per_face"] * CYC_FIXED + r["intops_per_face"]) * CYC_INT) / MHZ
        r["flip_compare_ms"] = 1000 * (cfl * CYC_SOFT + cit * CYC_INT) / MHZ
    json.dump(rows, open(os.path.join(H.OUT, "cost.json"), "w"), indent=1)
    print("%-14s %3s %9s %9s %10s %9s %9s %11s %11s %9s" % ("method", "res", "ms/face", "us/compare", "template B", "model B", "Mflop/face", "flip ms sf", "flip ms fx", "flip cmp"))
    for r in rows:
        print("%-14s %3d %9.2f %9.1f %10d %9d %9.2f %11.0f %11.0f %9.2f" % (r["method"], r["res"], r.get("describe_ms_cpp", r["describe_ms_numpy"]),
              r.get("compare_us_cpp", r.get("compare_us_numpy", float("nan"))), r["template_bytes"], r["model_bytes"], r["flops_per_face"] / 1e6,
              r["flip_ms_softfloat"], r["flip_ms_fixedpoint"], r["flip_compare_ms"]))
    print("(one face = the 9 placements of the +-1 px search; FACE LAW v2's times are its C++ port, the others numpy)  [%.0fs]" % (time.time() - t0))
