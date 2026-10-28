"""the flash check, v7.0 against v7.1 (FOLLOW), on rendered Yale faces moving during the check (not the user's photos):
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


SV0 = load("server_v70", os.path.join(T7, "server_v70.py"))
for M in (SV, SV0):
    M.VAULT = os.path.join(T7, "vault_test.json"); M.S["vault"] = None
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


MOTION = {
    "still":            ([(0, 0)] * 4, [0, 0, 0, 0]),
    "drift 1 px/phase": ([(1, 1), (2, 2), (3, 3), (4, 4)], [0, 0, 0, 0]),
    "drift 2 px/phase": ([(2, 2), (4, 4), (6, 6), (8, 8)], [0, 0, 0, 0]),
    "drift 3 px/phase": ([(3, 3), (6, 6), (9, 9), (12, 12)], [0, 0, 0, 0]),
    "shake +-4 px":     (None, [0, 0, 0, 0]),
    "turn 1 deg/phase": ([(0, 0)] * 4, [1, 2, 3, 4]),
}
people = sorted(TR)[:8]
R = np.random.default_rng(11)
res = {}
for kind in ("live", "print", "replay"):
    for mname, (offs, turns) in MOTION.items():
        if kind != "live" and mname not in ("still", "drift 2 px/phase", "shake +-4 px"):
            continue
        for p in people:
            hd = head(p)
            for yaw, pitch in ((0, 0), (-20, 0), (20, 10)):
                o = offs if offs is not None else [tuple(R.integers(-4, 5, 2)) for _ in range(4)]
                scan, fr = check(hd, yaw, pitch, o, turns, R, kind)
                e, _ = SV.eyes_on(SV.linear(SV.rgb_of(scan)))
                if e is None:
                    res.setdefault((kind, mname), {}).setdefault("no eyes on the scan frame", [0, 0])[0] += 1
                    continue
                for ver, M in (("v7.0", SV0), ("v7.1", SV)):
                    out = M.confirm(fr, [e[0].tolist(), e[1].tolist()])
                    key = out["decision"] + ("" if out["decision"] == "OPEN" else " (" + out["reason"][:34] + ")")
                    d = res.setdefault((kind, mname), {}).setdefault(ver, {})
                    d[key] = d.get(key, 0) + 1
        print(kind, mname, json.dumps(res[(kind, mname)]), flush=True)
json.dump({"%s | %s" % k: v for k, v in res.items()}, open(os.path.join(T7, "check_test.json"), "w"), indent=1)
