"""end-to-end check of the v8 server path (mapped angles) on rendered Yale faces, not the user's photos: JPEG frames ->
eye law -> MAPPED lock; the owner is taken out of the stranger bundle (leave-one-out).  Owner maps 3 angles; tries at
them, near them and away from them; one more angle added after a NOT YOU with the (test) password; the password lockout;
the 36 other people at many angles."""
import base64, io, json, os, sys, time
import numpy as np
from PIL import Image
FL = os.path.expanduser("~/neooos/facelaw"); T7 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FL); sys.path.insert(0, os.path.join(FL, "phone_faceid"))
import pose_law as PL
import server as SV
B = SV.B
TR = json.load(open(os.path.join(FL, "yale_eyes_truth.json")))["eyes"]; ids = sorted(TR)
owner = sys.argv[1] if len(sys.argv) > 1 else "yaleB05"; oi = ids.index(owner)
t0 = time.time()
SV.S["T"] = B.tables(); SV.S["pose"] = PL.load_pose()
SV.S["YK"] = np.array([p[0] for p in SV.S["pose"][0]], float); SV.S["PK"] = np.array([p[1] for p in SV.S["pose"][0]], float)
with np.load(os.path.join(FL, "phone_faceid", "pose_bundle.npz")) as Z:
    PB = {k: Z[k] for k in Z.files}
keep = PB["who"] != oi
SV.S["PB"] = {k: (v[keep] if v.shape[:1] == keep.shape else v) for k, v in PB.items()}
MJ = json.load(open(os.path.join(FL, "mapped_match.json"))); SV.S["marginM"] = {"96": MJ["margin96"], "48": MJ["margin48"]}
SV.VAULT = os.path.join(T7, "vault_test8.json"); SV.S["vault"] = None
TEST_PW = "test-only-" + os.urandom(4).hex()                       # a throwaway test value, never a real password
Zs = np.load(PL.SHAPES); R = np.random.default_rng(8)


def head(p):
    return PL.Head(Zs[p + "_rho"].astype(float), Zs[p + "_n"].astype(float), Zs[p + "_Z"].astype(float), TR[p])


def frame(hd, yaw, pitch, L=((1.0, PL.SCREEN),), amb=0.2, N=400.0):
    Rm, win, et = hd.turn(yaw, pitch); img, v = hd.shade(Rm, win, list(L), amb)
    pad = 48; bgv = 0.25 * img[v].mean()
    fr = np.full((PL.H + 2 * pad, PL.W + 2 * pad), bgv); fr[pad:pad + PL.H, pad:pad + PL.W] = np.where(v, img, bgv)
    fr = R.poisson(np.clip(fr / max(np.percentile(fr, 99.5), 1e-9), 0, None) * N) / N
    b = io.BytesIO(); Image.fromarray(np.repeat(PL.srgb8(np.clip(fr, 0, 1))[..., None], 3, -1)).save(b, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


hd = head(owner)
print("setup %.0fs; bundle without the owner: %d tries; MAPPED margins %s" % (time.time() - t0, keep.sum(), SV.S["marginM"]), flush=True)
SV.enrol_start(TEST_PW, TEST_PW)
for (y, q) in ((0, 0), (20, 0), (-20, 0)):                         # a hand-held capture: about 15 frames, +-3 deg of wobble
    SV.S["cap"] = []
    for t in range(15):
        SV.map_frame(frame(hd, y + R.uniform(-3, 3), q + R.uniform(-3, 3)))
    out = SV.map_finish()
    w = SV.S["vault"]["views"][-1]
    print("mapped %+d/%+d: %d frames kept, read as class %s, bars 96 %.2f / 48 %.2f" % (y, q, out["frames"], w["pose"], w["bar96"], w["bar48"]), flush=True)


def tries(p, poses, n=2, attempt=None):
    h2 = head(p); res = []
    for (y, q) in poses:
        for _ in range(n):
            r = SV.scan_frame(frame(h2, y, q), None, attempt)
            res.append((y, q, r.get("match", False), r.get("margin")))
    return res


def rate(rs):
    return "%d / %d" % (sum(r[2] for r in rs), len(rs))


at = [(0, 0), (20, 0), (-20, 0)]
near = [(y + dy, q + dq) for (y, q) in at for dy, dq in ((5, 0), (-5, 0), (0, 5), (0, -5), (10, 0), (-10, 0), (0, 10), (0, -10))]
far = [(0, -20), (0, 20), (30, 15), (-30, -15), (35, 0)]
print("owner at the 3 mapped angles:            open", rate(tries(owner, at, 4)), flush=True)
print("owner within 5-10 deg of them:           open", rate(tries(owner, near, 1)), flush=True)
print("owner far from them (20+ deg):           open", rate(tries(owner, far, 2)), flush=True)
# a NOT YOU at an angle not mapped (30 deg turn, chin down 15), then the password adds it
att = "f-test-1"; rs = tries(owner, [(-30, -15)] * 10, 1, att)
SV.S["failed"] = dict(id=att, t=time.time(), caps=SV.S["tries"][att])       # what /api/end keeps after a NOT YOU
print("try at -30/-15 before adding:           open", rate(rs), flush=True)
try:
    SV.add_angle(att, TEST_PW + "x"); print("WRONG PASSWORD ACCEPTED")
except PermissionError as e:
    print("wrong password refused:", e)
out = SV.add_angle(att, TEST_PW)
print("added with the password: now %d angles (the new one from %d frames)" % (out["angles"], out["frames"]), flush=True)
print("owner at -30/-15 after adding (and 5 deg off):  open", rate(tries(owner, [(-30, -15), (-25, -15), (-30, -10)], 3)), flush=True)
SV.S["pwfail"] = []
refused = 0
for k in range(6):
    try:
        SV.add_angle(att, "wrong%d" % k)
    except PermissionError as e:
        refused += 1; last = str(e)
print("6 wrong passwords: all refused %s; the 6th says: %s" % (refused == 6, last), flush=True)
opened = total = 0; worst = -9
for p in ids:
    if p == owner:
        continue
    rs = tries(p, [(0, 0), (20, 0), (-20, 0), (0, -20), (10, 10), (-10, -10), (15, 5)], 1)
    opened += sum(r[2] for r in rs); total += len(rs); worst = max([worst] + [r[3] for r in rs if r[3] is not None])
print("strangers (36 people x 7 angles): opened %d / %d; the closest stranger frame %.2f from the bar  [%.0fs]" % (opened, total, worst, time.time() - t0))
