"""end-to-end check of the v7 server path on rendered Yale faces (not the user's photos): JPEG frames -> eye law ->
pose law -> grid enrolment -> NEAR4 tries; the owner's own tries are taken out of the stranger bundle (leave-one-out)"""
import base64, io, json, os, sys, time
import numpy as np
from PIL import Image
FL = os.path.expanduser("~/neooos/facelaw")
sys.path.insert(0, FL); sys.path.insert(0, os.path.join(FL, "phone_faceid"))
os.environ.setdefault("X", "1")
import pose_law as PL
import server as SV
B = SV.B
TR = json.load(open(os.path.join(FL, "yale_eyes_truth.json")))["eyes"]; ids = sorted(TR)
owner = sys.argv[1] if len(sys.argv) > 1 else "yaleB05"
oi = ids.index(owner)
t0 = time.time()
SV.S["T"] = B.tables(); SV.S["pose"] = PL.load_pose()
SV.S["YK"] = np.array([p[0] for p in SV.S["pose"][0]], float); SV.S["PK"] = np.array([p[1] for p in SV.S["pose"][0]], float)
with np.load(os.path.join(FL, "phone_faceid", "pose_bundle.npz")) as Z:
    PB = {k: Z[k] for k in Z.files}
keep = PB["who"] != oi
SV.S["PB"] = {k: (v[keep] if v.shape[:1] == keep.shape else v) for k, v in PB.items()}
PBJ = json.load(open(os.path.join(FL, "phone_faceid", "pose_bars.json"))); SV.S["margin7"] = {"96": PBJ["margin96"], "48": PBJ["margin48"]}
SV.VAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault_test.json")
print("setup %.0fs; bundle without the owner: %d tries" % (time.time() - t0, keep.sum()), flush=True)
Zs = np.load(PL.SHAPES)
def head(p):
    return PL.Head(Zs[p + "_rho"].astype(float), Zs[p + "_n"].astype(float), Zs[p + "_Z"].astype(float), TR[p])
def frame(hd, yaw, pitch, L, amb, R, N=400.0):
    Rm, win, et = hd.turn(yaw, pitch); img, v = hd.shade(Rm, win, L, amb)
    pad = 48; bgv = 0.25 * img[v].mean()
    fr = np.full((PL.H + 2 * pad, PL.W + 2 * pad), bgv); fr[pad:pad + PL.H, pad:pad + PL.W] = np.where(v, img, bgv)
    fr = fr / max(np.percentile(fr, 99.5), 1e-9)
    fr = R.poisson(np.clip(fr, 0, None) * N) / N                       # photon noise in the frame
    rgb = np.repeat(PL.srgb8(fr)[..., None], 3, -1)
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
R = np.random.default_rng(7)
hd = head(owner)
SV.grid_start()
seen = {}
for (yaw, pitch) in PL.POSES:
    if abs(yaw) > 30 or abs(pitch) > 20:
        continue
    for t in range(3):
        r = SV.grid_frame(frame(hd, yaw, pitch, [(1.0, PL.SCREEN)], 0.2, R))
        seen[(yaw, pitch, t)] = (r.get("yaw"), r.get("pitch"), r.get("cell"))
bad = sum(1 for (y, q, t), (ry, rq, c) in seen.items() if ry is None)
err = [abs(ry - y) for (y, q, t), (ry, rq, c) in seen.items() if ry is not None]; errp = [abs(rq - q) for (y, q, t), (ry, rq, c) in seen.items() if rq is not None]
print("enrolment: %d frames, no eyes %d; pose law read yaw within 10 deg %.0f%%, pitch %.0f%%  [%.0fs]" % (
    len(seen), bad, 100 * np.mean(np.array(err) <= 10), 100 * np.mean(np.array(errp) <= 10), time.time() - t0), flush=True)
t1 = time.time(); out = SV.grid_finish()
print("grid_finish: %d views from %d frames, cells %s  [%.0fs for the bars]" % (out["views"], out["frames"], out["cells"], time.time() - t1), flush=True)
def tries(p, poses, L, amb, n=1):
    hd2 = head(p); res = []
    for (yaw, pitch) in poses:
        for _ in range(n):
            r = SV.scan_frame(frame(hd2, yaw, pitch, L, amb, R))
            res.append((yaw, pitch, r.get("match", False), r.get("margin"), r.get("yaw"), r.get("pitch")))
    return res
on = [(y, q) for y in (-30, -20, -10, 0, 10, 20, 30) for q in (-20, -10, 0, 10, 20)]
between = [(y, q) for y in (-25, -15, -5, 5, 15, 25) for q in (-15, -5, 5, 15)]
for name, poses in (("owner, grid angles, enrol light", on), ("owner, in-between angles, enrol light", between)):
    rs = tries(owner, poses, [(1.0, PL.SCREEN)], 0.2)
    byyaw = {}
    for y, q, m, d, ry, rq in rs:
        byyaw.setdefault(y, []).append(m)
    print("%-40s open %d / %d; by yaw %s" % (name, sum(m for *_, m, d, ry, rq in [(a, b, c, d, e, f) for a, b, c, d, e, f in rs]), len(rs),
          " ".join("%+d:%d/%d" % (y, sum(v), len(v)) for y, v in sorted(byyaw.items()))), flush=True)
rs = tries(owner, between, [(0.7, PL.SCREEN), (0.6, (0.3, 0.7, 0.65))], 0.15)
print("%-40s open %d / %d" % ("owner, in-between, room light above-right", sum(r[2] for r in rs), len(rs)), flush=True)
opened = 0; total = 0; worst = -9
for p in ids:
    if p == owner:
        continue
    rs = tries(p, [(0, 0), (-20, 0), (20, 0), (0, 10), (10, -10), (-15, 5), (25, 15)], [(1.0, PL.SCREEN)], 0.2)
    opened += sum(r[2] for r in rs); total += len(rs); worst = max([worst] + [r[3] for r in rs if r[3] is not None])
print("strangers (36 people x 7 angles, enrol light): opened %d / %d; the closest stranger frame %.2f below/above the bar  [%.0fs]" % (opened, total, worst, time.time() - t0))
