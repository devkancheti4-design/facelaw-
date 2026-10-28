"""does THE MAP let a photo open?  Real Extended Yale B photographs of the owner (yaleB05; all 64 lights, frontal
pose), printed (paper contrast, a little blur) and held up to the phone's camera, read by the v7 pipeline (eye law,
pose law, NEAR4 against the owner's grid enrolment from renders) -- identity only, the flash check left out.
Controls: the live owner (renders, as enrolled) and prints of the 36 other people's photos (lights within 25 deg)."""
import base64, io, json, os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
FL = os.path.expanduser("~/neooos/facelaw"); T7 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FL); sys.path.insert(0, os.path.join(FL, "phone_faceid"))
import pose_law as PL
import server as SV
B = SV.B
TR = json.load(open(os.path.join(FL, "yale_eyes_truth.json")))["eyes"]; ids = sorted(TR)
owner = "yaleB05"
SV.S["T"] = B.tables(); SV.S["pose"] = PL.load_pose()
SV.S["YK"] = np.array([p[0] for p in SV.S["pose"][0]], float); SV.S["PK"] = np.array([p[1] for p in SV.S["pose"][0]], float)
with np.load(os.path.join(FL, "phone_faceid", "pose_bundle.npz")) as Z:
    SV.S["PB"] = {k: Z[k] for k in Z.files}
SV.VAULT = os.path.join(T7, "vault_test.json"); SV.S["vault"] = None
R = np.random.default_rng(3)
PAD = 48


def jpeg_of(img, R, N=400.0):
    bgv = 0.25 * img.mean()
    fr = np.full((PL.H + 2 * PAD, PL.W + 2 * PAD), bgv); fr[PAD:PAD + PL.H, PAD:PAD + PL.W] = img
    fr = fr / max(np.percentile(fr, 99.5), 1e-9)
    fr = R.poisson(np.clip(fr, 0, None) * N) / N
    rgb = np.repeat(PL.srgb8(np.clip(fr, 0, 1))[..., None], 3, -1)
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def printed(raw):
    """a photo on paper, seen by the camera under even light: ink black ~6%, paper white ~92%, a little blur"""
    x = raw / max(np.percentile(raw, 99.5), 1e-9)
    return gaussian_filter(0.06 + 0.86 * np.clip(x, 0, 1), 0.6)


def scan(img, n=1):
    out = []
    for _ in range(n):
        r = SV.scan_frame(jpeg_of(img, R))
        out.append((r.get("match", False), r.get("margin"), r.get("which"), r.get("yaw"), r.get("pitch")))
    return out


res = {}
# the live owner, as enrolled (renders under the screen's light), straight ahead
Zs = np.load(PL.SHAPES)
hd = PL.Head(Zs[owner + "_rho"].astype(float), Zs[owner + "_n"].astype(float), Zs[owner + "_Z"].astype(float), TR[owner])
Rm, win, _ = hd.turn(0, 0); im, v = hd.shade(Rm, win, [(1.0, PL.SCREEN)], 0.2)
live = scan(np.where(v, im, 0.25 * im[v].mean()), 12)
res["live owner, straight, screen light"] = live
# prints of the owner's real photographs, by the photo's light angle
bands = ((0, 12), (12, 25), (25, 50), (50, 77), (77, 180))
for ph in PL.photos(owner):
    band = next(b for b in bands if b[0] <= ph["theta"] <= b[1])
    res.setdefault("print of the owner, photo light %d-%d deg" % band, []).extend(scan(printed(ph["raw"]), 2))
# prints of everyone else's photographs (lights within 25 deg)
for p in ids:
    if p == owner:
        continue
    for ph in PL.photos(p):
        if ph["theta"] <= 25:
            res.setdefault("print of another person, light <= 25 deg", []).extend(scan(printed(ph["raw"]), 1))
summ = {}
for k, rs in res.items():
    m = [r for r in rs if r[1] is not None]
    summ[k] = dict(frames=len(rs), eyes=len(m), match=sum(r[0] for r in rs),
                   median_margin=round(float(np.median([r[1] for r in m])), 2) if m else None,
                   best=round(float(max(r[1] for r in m)), 2) if m else None)
    print("%-48s frames %3d  eyes %3d  MATCH %3d   median margin %s  best %s" % (k, len(rs), len(m), summ[k]["match"], summ[k]["median_margin"], summ[k]["best"]), flush=True)
json.dump(summ, open(os.path.join(T7, "print_identity.json"), "w"), indent=1)
