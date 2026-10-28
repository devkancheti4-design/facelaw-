"""WHY did strangers' real photographs clear v8's bars?  Owner yaleB01 enrolled through the v8 path (renders, 5 angles,
bundle without the owner).  For the worst hits: where the eye law put the eyes (against the hand-marked truth), the angle
class read, the mask, and which table entries carry the score (by scale, by the probe's signal level, agreement of bins).
Controls: the same stranger in front light, and the owner's own render."""
import base64, io, json, os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
FL = os.path.expanduser("~/neooos/facelaw")
sys.path.insert(0, FL); sys.path.insert(0, os.path.join(FL, "phone_faceid"))
import pose_law as PL
import server as SV
B = SV.B
TR = json.load(open(os.path.join(FL, "yale_eyes_truth.json")))["eyes"]; ids = sorted(TR)
owner = "yaleB01"; oi = ids.index(owner)
SV.S["T"] = B.tables(); SV.S["pose"] = PL.load_pose()
SV.S["YK"] = np.array([p[0] for p in SV.S["pose"][0]], float); SV.S["PK"] = np.array([p[1] for p in SV.S["pose"][0]], float)
with np.load(os.path.join(FL, "phone_faceid", "pose_bundle.npz")) as Z:
    PB = {k: Z[k] for k in Z.files}
keep = PB["who"] != oi
SV.S["PB"] = {k: (v[keep] if v.shape[:1] == keep.shape else v) for k, v in PB.items()}
MJ = json.load(open(os.path.join(FL, "mapped_match.json"))); SV.S["marginM"] = {"96": MJ["margin96"], "48": MJ["margin48"]}
SV.VAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault_why.json"); SV.S["vault"] = None
R = np.random.default_rng(1); PAD = 48
Zs = np.load(PL.SHAPES)
hd = PL.Head(Zs[owner + "_rho"].astype(float), Zs[owner + "_n"].astype(float), Zs[owner + "_Z"].astype(float), TR[owner])


def frame(img, N=400.0):
    bgv = 0.25 * img.mean()
    fr = np.full((PL.H + 2 * PAD, PL.W + 2 * PAD), bgv); fr[PAD:PAD + PL.H, PAD:PAD + PL.W] = img
    fr = R.poisson(np.clip(fr / max(np.percentile(fr, 99.5), 1e-9), 0, None) * N) / N
    b = io.BytesIO(); Image.fromarray(np.repeat(PL.srgb8(np.clip(fr, 0, 1))[..., None], 3, -1)).save(b, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


SV.enrol_start("test-only-pw", "test-only-pw")
for (y, q) in ((0, 0), (20, 0), (-20, 0), (0, 10), (0, -10)):
    SV.S["cap"] = []
    for t in range(15):
        Rm, win, _ = hd.turn(y + R.uniform(-3, 3), q + R.uniform(-3, 3)); img, v = hd.shade(Rm, win, [(1.0, PL.SCREEN)], 0.2)
        SV.map_frame(frame(np.where(v, img, 0.25 * img[v].mean())))
    SV.map_finish()
V = SV.vault()
printed = lambda raw: gaussian_filter(0.06 + 0.86 * np.clip(raw / max(np.percentile(raw, 99.5), 1e-9), 0, 1), 0.6)
fw = SV.S["T"]["fw96"]; M = SV.S["PB"]["M96"]; SC = B.SC[10584]


def explain(name, img):
    f = frame(img)
    lin = SV.linear(SV.rgb_of(f)); e, c, why = SV.frame_codes(lin)
    if e is None:
        print("%-44s no eyes" % name); return
    (m96, p96), (m48, p48) = c
    k, y, q = SV.pose_read(m48)
    best = None
    for i, w in enumerate(V["views"]):
        mp = np.asarray(w["map96"], np.int64)
        E = fw[(mp | (SC << 8)) | ((p96.astype(np.int64) & 7) << 3) | ((p96.astype(np.int64) >> 3) << 6)]
        mk = SV.PL.mapped_mask(M, w["cls"], np.array([k]))[0]
        d = (E * mk).sum() / max(mk.sum(), 1) - w["bar96"]
        if best is None or d > best[0]:
            best = (d, i, E, mk, mp)
    d, i, E, mk, mp = best
    sig = (p96 >> 3).astype(int); agree = (p96 & 7) == mp
    tru = np.array(TR[name.split("_")[0]], float).reshape(2, 2) + PAD if name.split("_")[0] in TR else None
    eye_err = "" if tru is None else " eyes off the truth by %.0f / %.0f px" % tuple(np.hypot(*(np.array(e[j]) - tru[j])) for j in range(2))
    print("%-44s best view %d d=%+.2f (bar %.2f) | read class %s | mask %d pts%s" % (name, i + 1, d, V["views"][i]["bar96"], SV.S["pose"][0][k], mk.sum(), eye_err))
    for s in range(4):
        sel = mk & (sig == s)
        if sel.sum():
            print("      signal %d: %5d pts (%4.0f%%), bins agree %4.0f%%, mean entry %+6.2f, share of the score %+6.2f" % (
                s, sel.sum(), 100 * sel.sum() / mk.sum(), 100 * agree[sel].mean(), E[sel].mean(), E[sel].sum() / max(mk.sum(), 1)))


Rm, win, _ = hd.turn(0, 0); img, v = hd.shade(Rm, win, [(1.0, PL.SCREEN)], 0.2)
explain("owner %s render, straight" % owner, np.where(v, img, 0.25 * img[v].mean()))
for p, nm, kind in (("yaleB17", "yaleB17_P00A+000E-35.pgm", "printed"), ("yaleB22", "yaleB22_P00A-035E-20.pgm", "as is"),
                    ("yaleB29", "yaleB29_P00A-010E-20.pgm", "printed"), ("yaleB17", "yaleB17_P00A+000E+00.pgm", "printed"),
                    ("yaleB22", "yaleB22_P00A+000E+00.pgm", "as is")):
    ph = next(x for x in PL.photos(p) if x["name"] == nm)
    explain("%s_%s %s" % (p, nm.split("_")[1][:-4], kind), printed(ph["raw"]) if kind == "printed" else ph["raw"])
os.remove(SV.VAULT)
