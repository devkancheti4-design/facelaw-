"""END TO END: a printed real photograph of the owner (yaleB05, frontal pose, lights within 25 deg) against the phone's
v7.0 lock: the scan frame must match the map (NEAR4), then the flash check (white, black, top, bottom) must call it live.
The print moves 0, 1 or 2 camera pixels per flash (integer), or 0.5 / 1.0 (sub-pixel).  Renders only, no user photo."""
import base64, io, json, os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, shift as subshift
FL = os.path.expanduser("~/neooos/facelaw"); T7 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FL); sys.path.insert(0, os.path.join(FL, "phone_faceid"))
import pose_law as PL
import server as SV
B = SV.B
SV.S["T"] = B.tables(); SV.S["pose"] = PL.load_pose()
SV.S["YK"] = np.array([p[0] for p in SV.S["pose"][0]], float); SV.S["PK"] = np.array([p[1] for p in SV.S["pose"][0]], float)
with np.load(os.path.join(FL, "phone_faceid", "pose_bundle.npz")) as Z:
    SV.S["PB"] = {k: Z[k] for k in Z.files}
SV.VAULT = os.path.join(T7, "vault_test.json"); SV.S["vault"] = None
R = np.random.default_rng(9); PAD = 48
ORDER = ("white", "black", "top", "bottom")
yy = np.linspace(-1, 1, PL.H + 2 * PAD)[:, None] * np.ones((1, PL.W + 2 * PAD))
GAIN = {"white": 1.5, "black": 0.5, "top": 0.5 + 0.25 * (1 + 0.15 * -yy), "bottom": 0.5 + 0.25 * (1 - 0.15 * -yy)}


def frame_of(paper, gain, off, frac):
    bgv = 0.25 * paper.mean()
    fr = np.full((PL.H + 2 * PAD, PL.W + 2 * PAD), bgv); fr[PAD:PAD + PL.H, PAD:PAD + PL.W] = paper
    fr = fr * gain / 1.5
    if frac:
        fr = subshift(fr, off, order=3, mode="nearest")
    else:
        fr = np.roll(fr, (int(off[0]), int(off[1])), (0, 1))
    return fr


def jpeg(fr, scale, N=400.0):
    x = R.poisson(np.clip(fr / scale, 0, None) * N) / N
    rgb = np.repeat(PL.srgb8(np.clip(x, 0, 1))[..., None], 3, -1)
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


photos = [ph for ph in PL.photos("yaleB05") if ph["theta"] <= 25]
print("owner photos with light within 25 deg:", len(photos), flush=True)
out = {}
for name, step, frac in (("still", 0, False), ("0.5 px/flash", 0.5, True), ("1 px/flash", 1.0, True), ("1 px/flash (integer)", 1, False), ("2 px/flash (integer)", 2, False)):
    t = dict(checks=0, matched=0, OPEN=0); why = {}
    for ph in photos:
        x = ph["raw"] / max(np.percentile(ph["raw"], 99.5), 1e-9)
        paper = gaussian_filter(0.06 + 0.86 * np.clip(x, 0, 1), 0.6)
        for rep in range(3):
            w0 = frame_of(paper, GAIN["white"], (0, 0), False); scale = np.percentile(w0, 99.5) / 0.85
            r = SV.scan_frame(jpeg(w0, scale)); t["checks"] += 1
            if not r.get("match"):
                continue
            t["matched"] += 1
            fr = {k: [jpeg(frame_of(paper, GAIN[k], (step * (i + 1), step * (i + 1)), frac), scale)] for i, k in enumerate(ORDER)}
            o = SV.confirm(fr, r["eyes"])
            t["OPEN"] += o["decision"] == "OPEN"
            if o["decision"] != "OPEN":
                k = o["reason"][:30]; why[k] = why.get(k, 0) + 1
    out[name] = dict(t, refused=why)
    print("%-22s scan frames %d, the map matched %d, OPEN %d   refusals %s" % (name, t["checks"], t["matched"], t["OPEN"], json.dumps(why)), flush=True)
json.dump(out, open(os.path.join(T7, "print_attack.json"), "w"), indent=1)
