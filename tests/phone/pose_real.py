"""the pose law on REAL photographs whose true angle is known: every Extended Yale B person's frontal photos (true
yaw 0, pitch 0), lights within 25 deg, through the phone path (frame, JPEG, eye law, 48 crop) -- against renders of the
same people at 0/0 under the same kind of light (the domain the law was counted on)"""
import base64, io, json, os, sys
import numpy as np
from PIL import Image
FL = os.path.expanduser("~/neooos/facelaw"); T7 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FL); sys.path.insert(0, os.path.join(FL, "phone_faceid"))
import pose_law as PL
import server as SV
B = SV.B
SV.S["pose"] = PL.load_pose()
SV.S["YK"] = np.array([p[0] for p in SV.S["pose"][0]], float); SV.S["PK"] = np.array([p[1] for p in SV.S["pose"][0]], float)
TR = json.load(open(os.path.join(FL, "yale_eyes_truth.json")))["eyes"]; ids = sorted(TR)
Zs = np.load(PL.SHAPES); R = np.random.default_rng(1); PAD = 48


def read(img):
    bgv = 0.25 * img.mean()
    fr = np.full((PL.H + 2 * PAD, PL.W + 2 * PAD), bgv); fr[PAD:PAD + PL.H, PAD:PAD + PL.W] = img
    fr = R.poisson(np.clip(fr / max(np.percentile(fr, 99.5), 1e-9), 0, None) * 400.0) / 400.0
    b = io.BytesIO(); Image.fromarray(np.repeat(PL.srgb8(np.clip(fr, 0, 1))[..., None], 3, -1)).save(b, "JPEG", quality=95)
    e, c, why = SV.frame_codes(SV.linear(SV.rgb_of("data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode())))
    if e is None:
        return None
    k, y, q = SV.pose_read(c[1][0]); return (y, q)


def report(name, rs):
    a = np.array([r for r in rs if r is not None], float)
    edge = (np.abs(a[:, 0]) >= 29) | (np.abs(a[:, 1]) >= 19)
    print("%-44s eyes %4d/%4d  yaw within 10: %3.0f%%  pitch within 10: %3.0f%%  at the grid's edge: %3.0f%%  median yaw %+.0f pitch %+.0f" % (
        name, len(a), len(rs), 100 * np.mean(np.abs(a[:, 0]) <= 10), 100 * np.mean(np.abs(a[:, 1]) <= 10), 100 * edge.mean(),
        np.median(a[:, 0]), np.median(a[:, 1])), flush=True)


real, rend = [], []
for p in ids:
    hd = PL.Head(Zs[p + "_rho"].astype(float), Zs[p + "_n"].astype(float), Zs[p + "_Z"].astype(float), TR[p])
    Rm, win, _ = hd.turn(0, 0)
    for ph in PL.photos(p):
        if ph["theta"] > 25:
            continue
        real.append(read(ph["raw"]))
        im, v = hd.shade(Rm, win, [(1.0, tuple(ph["s"]))], 0.1)
        rend.append(read(np.where(v, im, 0.25 * im[v].mean())))
report("REAL photos, true angle 0/0", real)
report("renders of the same people, same lights, 0/0", rend)
