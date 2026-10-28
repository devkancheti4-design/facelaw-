"""the PUBLIC CHALLENGE's flash check (unlock_server.attempt: one crop at the eyes, v1 liveness law), on rendered Yale faces moving during the check (not the user's photos):
a live face drifting / shaking / turning a little, and the attacks the check exists for (a flat print, a screen replay)
-- both versions get the same frames"""
import base64, io, json, os, sys, importlib.util
import numpy as np
from PIL import Image
FL = os.path.expanduser("~/neooos/facelaw"); T7 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FL); sys.path.insert(0, os.path.join(FL, "phone_faceid"))
import pose_law as PL
import server as SV


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


sys.path.insert(0, FL)
import unlock_server as US, phone_eyes as PE
US.LAW["cL"], US.LAW["cR"] = PE.canonical_eyes()
TR = json.load(open(os.path.join(FL, "yale_eyes_truth.json")))["eyes"]
Zs = np.load(PL.SHAPES)
ROOM, AMB = [(0.5, (0.3, 0.7, 0.65))], 0.1
nrm = lambda v: tuple(np.asarray(v, float) / np.linalg.norm(v))
TOP, BOT = nrm((0.0, -0.04, 1.0)), nrm((0.0, -0.30, 0.95))
PHASES = {"white": ROOM + [(1.0, PL.SCREEN)], "black": ROOM, "top": ROOM + [(0.5, TOP)], "bottom": ROOM + [(0.5, BOT)]}
ORDER = ("white", "black", "top", "bottom")
PAD = 48


def head(p):
    return PL.Head(Zs[p + "_rho"].astype(float), Zs[p + "_n"].astype(float), Zs[p + "_Z"].astype(float), TR[p])


def place(img, v, off, scale):
    """the face in the camera frame at an offset (dy, dx); one exposure for the whole check (the scan frame's)"""
    bg = 0.25 * scale
    fr = np.full((PL.H + 2 * PAD, PL.W + 2 * PAD), bg)
    dy, dx = off
    fr[PAD + dy:PAD + dy + PL.H, PAD + dx:PAD + dx + PL.W] = np.where(v, img, bg)
    return fr


def jpeg(fr, scale, R, N=400.0):
    x = R.poisson(np.clip(fr / scale, 0, None) * N) / N
    rgb = np.repeat(PL.srgb8(np.clip(x, 0, 1))[..., None], 3, -1)
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def check(hd, yaw, pitch, offs, turns, R, kind="live"):
    """-> (scan frame data URL, {phase: [data URL]}) for one check; offs/turns per phase (white, black, top, bottom)"""
    Rm, win, _ = hd.turn(yaw, pitch); img, v = hd.shade(Rm, win, PHASES["white"], AMB)
    scale = np.percentile(img[v], 99.5) / 0.85
    scan = jpeg(place(img, v, (0, 0), scale), scale, R)
    fr = {}
    yy = np.linspace(-1, 1, PL.H)[:, None] * np.ones((1, PL.W))
    for k, off, dt in zip(ORDER, offs, turns):
        if kind == "live":
            Rm, win, _ = hd.turn(yaw + dt, pitch); im, vv = hd.shade(Rm, win, PHASES[k], AMB)
        elif kind == "print":                                  # a flat print of the scan frame: the screen's light on paper
            gain = {"white": 1.0 + 0.5, "black": 0.5, "top": 0.5 + 0.25 * (1 + 0.15 * -yy), "bottom": 0.5 + 0.25 * (1 - 0.15 * -yy)}[k]
            im, vv = img * gain / 1.5, v
        else:                                                   # a screen replay: its own light, 5% of the flash reflected
            refl = {"white": 0.05, "black": 0.0, "top": 0.025, "bottom": 0.025}[k]
            im, vv = img * (1.0 + refl), v
        fr[k] = [jpeg(place(im, vv, off, scale), scale, R)]
    return scan, fr



from scipy.ndimage import shift as subshift
def check_frac(hd, yaw, pitch, offs, R, kind):
    """like check(), but the motion is fractional (sub-pixel): each phase's frame resampled by (dy, dx)"""
    Rm, win, _ = hd.turn(yaw, pitch); img, v = hd.shade(Rm, win, PHASES["white"], AMB)
    scale = np.percentile(img[v], 99.5) / 0.85
    base = place(img, v, (0, 0), scale)
    scan = jpeg(base, scale, R)
    yy = np.linspace(-1, 1, PL.H)[:, None] * np.ones((1, PL.W)); fr = {}
    for k, off in zip(ORDER, offs):
        if kind == "live":
            im, vv = hd.shade(Rm, win, PHASES[k], AMB)
        else:
            gain = {"white": 1.0 + 0.5, "black": 0.5, "top": 0.5 + 0.25 * (1 + 0.15 * -yy), "bottom": 0.5 + 0.25 * (1 - 0.15 * -yy)}[k]
            im, vv = img * gain / 1.5, v
        f0 = place(im, vv, (0, 0), scale)
        fr[k] = [jpeg(subshift(f0, off, order=3, mode="nearest"), scale, R)]
    return scan, fr
people = sorted(TR)[:8]; R = np.random.default_rng(5); res = {}
for kind in ("live", "print"):
    for step in (0.0, 0.25, 0.5, 1.0, 1.5):
        offs = [(step * (i + 1), step * (i + 1)) for i in range(4)]
        tally = {}; r3 = []
        for p in people:
            hd = head(p)
            for yaw, pitch in ((0, 0), (-20, 0), (20, 10)):
                scan, fr = check_frac(hd, yaw, pitch, offs, R, kind)
                e, _ = SV.eyes_on(SV.linear(SV.rgb_of(scan)))
                if e is None:
                    tally["no eyes"] = tally.get("no eyes", 0) + 1; continue
                Lm = {k: US.mean_of(fr[k]) for k in ORDER}
                K = {k: US.face_crop(Lm[k], [e[0].tolist(), e[1].tolist()]) for k in ORDER}
                lv = US.liveness(K)
                key = "LIVE" if lv["live"] else lv["fail"]
                tally[key] = tally.get(key, 0) + 1
                if lv.get("r3d") is not None: r3.append(lv["r3d"])
        print("%-6s drift %.2f px/phase: %s  r3d median %s" % (kind, step, json.dumps(tally), round(float(np.median(r3)), 3) if r3 else None), flush=True)
