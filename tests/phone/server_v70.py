"""FACE ID HEAD-TO-HEAD: the phone's own face unlock against Face Law, the same people, light and attacks.  The phone's
page captures camera frames lit by its own screen (black / white pairs; for a try also the top and bottom halves) and
sends them over the USB cable (adb reverse tcp:8080 tcp:8080) to this server on 127.0.0.1, which runs Face Law:
    ENROL  8 black/white pairs -> the screen-lit face (white minus black: the room's light cancels) -> automatic eyes on
           the white frame (face box + eye law + gate) -> both laws' maps (per-point majority of the frames) -> each
           law's bar for this owner: the best score of the Yale stranger bundle (bundle.npz) + the margin (bars.json)
    TRY    liveness (v1's law: the screen's light must reach the face; the top/bottom ratio must carry 3D shape) -> both
           laws on the screen-lit face -> OPEN if either law clears its bar, else NOT YOU; NOT NOW if not live
    LOG    per trial: time, who (owner / stranger / photo / screen), light, Face Law's answer and scores, and the phone's
           own face unlock result as the tester taps it.  No image is ever written to disk.
  v3 (2026-10-28, after the first owner block: 5 of 24 opens, 13 NOT NOW from 1-pixel movement and weak flashes)
    FOLLOW every frame is cropped at the eyes of the try's first white frame and lined up with it (the best integer shift
           within +-3 law pixels on log-gradient maps), so a hand-held phone's small movement no longer stops a try
    MORE   a try is 3 black/white pairs and 2 top + 2 bottom half-screen flashes; the screen-lit face is the mean of
           the 3 aligned pairs, and the liveness ratio uses the mean of the 2 + 2 half flashes
    PLAIN  a second map per law from the bright (white-screen) frame itself, which in a lit room carries 3-5 times the
           light of the flash difference; each map has its own bar, the margins re-set for two chances (bars.json
           margin_union2; each law keeps the larger of its two margins)
  v4 (2026-10-28, the owner: "same angle opens, eye position different is NOT YOU"; "capture frames until it opens")
    ANGLES enrolment at 5 angles (straight, phone higher, lower, left, right), each with its screen-lit and bright map
    READS  every try window gives 8 reads (the 3 pairs and the 3 bright frames one by one, and both means), all inside one
           flash-checked, lined-up window, so a face cannot be swapped for a photo after the liveness check
    AGAIN  the page opens a second window by itself when the first says NOT YOU; the attempt is NOT YOU only if both do
    BARS   margins for 10 maps per law (bars.json margin_union10; each law keeps its strictest margin)
  v5 FAST (2026-10-28, the owner: "recursive but fast": frame after frame until it opens)
    ENROL  about 3 s of frames while the owner moves the phone in a small circle -> 5 maps per law from consecutive groups
    SCAN   the screen stays white; every frame is read as it arrives (eyes, both laws, every map) -- about 6-8 per second
    LIVE   on the first matching frame: a 1.5 s flash check (black, white, top, bottom, top, bottom) at the same eyes; OPEN
           only if it is live and the face there still matches; no match in 3 s -> NOT YOU
  v7 POSE LAW (2026-10-28, the owner: "map the enrolment correctly, like Face ID"; measured: a view holds only ~10 deg)
    POSE   every frame's head angle read by the pose law (the synthesis's table: 35 classes, yaw -30..30 x pitch -20..20)
    ENROL  a grid of angles at 10 deg steps fills as the owner turns the head; one view per cell with 3+ frames
    MATCH  a try is compared only with the 4 enrolled views nearest the angle the pose law reads (NEAR4), each view with
           its own bar from the rendered stranger bundle (pose_bundle.npz) + the NEAR4 margins (pose_bars.json)
    python3 -I phone_faceid/server.py"""
import base64, hashlib, io, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from PIL import Image, ImageOps
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bars as B                                     # the C++ library (ctypes), the codes, the tables, scores(), mode_map()
import unlock_law as U
import pose_law as PL                                # the pose law (pose_design.json, the authored lanes_pose.json)
VAULT, LOG = os.path.join(HERE, "vault.json"), os.path.join(HERE, "trials.jsonl")
LIVE = dict(REL_MIN=0.15, R3D_MIN=0.12, MIN_PX=150, SHIFT_MAX=1)   # v1's liveness law (data/webapp/liveness.json)
GATE = -6000.0
LOCK = threading.Lock()
S = {}


def rgb_of(b64):
    """a camera frame (JPEG data URL) -> RGB bytes, upright"""
    raw = base64.b64decode(b64.split(",", 1)[-1])
    return np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")), np.uint8)


def linear(rgb):
    h, w = rgb.shape[:2]
    o = np.empty((h, w)); B.LIB.fl2_linear_of(rgb.ctypes.data_as(B.U8), w, h, 3, o.ctypes.data_as(B.D)); return o


def eyes_on(lin, box=None):
    """the face box (largest face; or the box the previous frame's eyes give: tracking) and the eye law with its gate"""
    h, w = lin.shape
    if box is not None:
        box = np.asarray(box, float)
    else:
        box = np.empty(4)
        if not B.LIB.fl2_face_box(B.dp(lin), h, w, box.ctypes.data_as(B.D)):
            return None, "no face"
    disp = np.empty_like(lin); B.LIB.fl2_u8(B.dp(lin), h, w, disp.ctypes.data_as(B.D))
    o = np.empty(5)
    if not B.LIB.fl2_find_eyes(B.dp(disp), h, w, B.dp(box), o.ctypes.data_as(B.D)) or o[4] < GATE:
        return None, "no eyes"
    return (o[:2].copy(), o[2:4].copy()), ""


def crop(img, e, big=False):
    o = np.empty((96, 84) if big else (48, 42))
    (B.LIB.fl2_crop96 if big else B.LIB.fl2_crop)(B.dp(img), img.shape[0], img.shape[1], B.dp(np.r_[e[0], e[1]]), o.ctypes.data_as(B.D))
    return o


def best_shift(a, b):
    ga = np.hypot(*np.gradient(np.log(np.maximum(a, 1e-3)))); gb = np.hypot(*np.gradient(np.log(np.maximum(b, 1e-3))))
    best, arg = -2.0, 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c = np.corrcoef(ga[3 + dy:45 + dy, 3 + dx:39 + dx].ravel(), gb[3:45, 3:39].ravel())[0, 1]
            if c > best:
                best, arg = c, max(abs(dy), abs(dx))
    return arg


def shift_to(a, b, r=3):
    """the integer (dy, dx) that lines frame b up with frame a (log-gradient correlation), and how well"""
    ga = np.hypot(*np.gradient(np.log(np.maximum(a, 1e-3)))); gb = np.hypot(*np.gradient(np.log(np.maximum(b, 1e-3))))
    best, arg = -2.0, (0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            c = np.corrcoef(ga[3 + dy:45 + dy, 3 + dx:39 + dx].ravel(), gb[3:45, 3:39].ravel())[0, 1]
            if c > best:
                best, arg = c, (dy, dx)
    return arg, best


def moved(x, dy, dx):
    h, w = x.shape
    return x[np.clip(np.arange(h) - dy, 0, h - 1)][:, np.clip(np.arange(w) - dx, 0, w - 1)]


def aligned_crops(frames, e, ref48):
    """each linear frame cropped at the eyes e, lined up with the reference 48 x 42 crop -> (48 x 42, 96 x 84, shift)"""
    out = []
    for L in frames:
        c48, c96 = crop(L, e), crop(L, e, True)
        if c48.mean() < 0.1 * max(ref48.mean(), 1e-9):            # a black-screen frame in the dark: nothing to line up
            out.append((c48, c96, (0, 0))); continue
        (dy, dx), _ = shift_to(ref48, c48)
        out.append((moved(c48, dy, dx), moved(c96, 2 * dy, 2 * dx), (dy, dx)))
    return out


def liveness(K):
    """v1's law (unlock_server.liveness): the screen's share of the face's light, and 3D shape in the top/bottom ratio"""
    b, w, t, d = (K[k] for k in ("black", "white", "top", "bottom"))
    Dw, Dt, Dd = w - b, t - b, d - b
    yy, xx = np.mgrid[0:48, 0:42].astype(float)
    oval = ((xx - 20.5) / 18.0) ** 2 + ((yy - 24.0) / 23.0) ** 2 <= 1.0
    rel = float(np.median(Dw[oval] / np.maximum(w[oval], 1e-6)))
    shift = max(best_shift(w, t), best_shift(w, d))                 # after the lining-up: what is left
    sd = np.sqrt(U.measured_var(b).reshape(48, 42) + np.maximum(U.measured_var(t), U.measured_var(d)).reshape(48, 42))
    ok = oval & (Dt > 3 * sd) & (Dd > 3 * sd)
    n = int(ok.sum())
    out = dict(rel=round(rel, 3), shift=int(shift), n=n, r3d=None)
    if shift > LIVE["SHIFT_MAX"]:
        return dict(out, live=False, why="The face moved during the flashes. Hold still and try again.")
    if rel < LIVE["REL_MIN"] or n < LIVE["MIN_PX"]:
        return dict(out, live=False, why="Too little of the screen's light reaches the face. Brightness up, hold the phone closer.")
    Q = np.log(Dt[ok] / Dd[ok]); u, v = (xx[ok] - 20.5) / 21.0, (yy[ok] - 24.0) / 24.0
    X = np.stack([np.ones(n), u, v, u * u, u * v, v * v], 1)
    r3d = float(np.sqrt(np.mean((Q - X @ np.linalg.lstsq(X, Q, rcond=None)[0]) ** 2)))
    out["r3d"] = round(r3d, 4)
    if r3d < LIVE["R3D_MIN"]:
        return dict(out, live=False, why="This looks flat, like a photo or a screen.")
    return dict(out, live=True, why="")


def box_of(e):
    """the face box the eyes give (the eye law's own frame: width = eye distance / 0.45)"""
    wd = float(np.linalg.norm(e[1] - e[0])) / 0.45; cx, cy = (e[0] + e[1]) / 2.0
    return [cx - wd / 2, cy - 0.36 * wd, wd, wd]


def frame_codes(lin, box=None):
    e, why = eyes_on(lin, box)
    if e is None and box is not None:
        e, why = eyes_on(lin)                                      # tracking lost: search the whole frame
    if e is None:
        return None, None, why
    return e, (B.codes96(crop(lin, e, True)), B.codes48(crop(lin, e))), ""


def sharpness(lin, e):
    c = crop(lin, e)
    return float(np.mean(np.hypot(*np.gradient(c))) / max(c.mean(), 1e-9))


def pose_read(m48):
    """THE POSE LAW (authored as integer lanes; rebuilt from its lanes and checked at start): the pose class with the most
    counted evidence in the 48 x 42 map bins, and the soft angle (the classes weighted by 2^(evidence / QP))"""
    poses, pts, W = S["pose"]
    sc = W[:, np.arange(len(pts)), np.asarray(m48)[pts]].sum(1).astype(float)
    pr = 2.0 ** ((sc - sc.max()) / PL.QP); pr /= pr.sum()
    return int(np.argmax(sc)), float(pr @ S["YK"]), float(pr @ S["PK"])


def masked_scores(law, mp, cv, P, est, chunk=400):
    """each try's mean table entry against a view, over the points the view's angle and the try's read angle both show"""
    fw, M = S["T"]["fw" + law], S["PB"]["M" + law]
    base = np.asarray(mp, np.int64) | (B.SC[len(mp)] << 8)
    out = np.empty(len(P))
    for a in range(0, len(P), chunk):
        Q = P[a:a + chunk].astype(np.int64)
        E = S["T"]["fw" + law][base[None, :] | ((Q & 7) << 3) | ((Q >> 3) << 6)]
        mk = M[cv][None, :] & M[np.asarray(est[a:a + chunk])]
        out[a:a + chunk] = (E * mk).sum(1) / np.maximum(mk.sum(1), 1)
    return out


def nearest(y, q, vp, k=4):
    """the k enrolled views nearest the read angle (ties: the earlier view), for one try or a column of tries"""
    d = np.hypot(np.atleast_1d(y)[None, :] - vp[:, :1], np.atleast_1d(q)[None, :] - vp[:, 1:2])
    return np.argsort(d, 0, kind="stable")[:k]


def grid_start():
    S["grid"] = dict(cells={}, n=0)
    return dict(ok=True, poses=[list(p) for p in S["pose"][0]])


def grid_frame(f, box=None):
    """v7 enrolment: every frame's angle read by the pose law; the frame joins the grid cell (10 deg steps) nearest its
    angle when it is within 7 deg of it in both yaw and pitch"""
    G = S["grid"]
    lin = linear(rgb_of(f)); e, c, why = frame_codes(lin, box)
    G["n"] += 1
    if e is None:
        return dict(ok=False, why=why, counts={str(i): len(v) for i, v in G["cells"].items()})
    k, y, q = pose_read(c[1][0])
    poses = S["pose"][0]
    j = min(range(len(poses)), key=lambda i: ((poses[i][0] - y) ** 2 + (poses[i][1] - q) ** 2, i))
    inside = abs(poses[j][0] - y) <= 7 and abs(poses[j][1] - q) <= 7
    if inside:
        G["cells"].setdefault(j, []).append((sharpness(lin, e), c))
    return dict(ok=True, yaw=round(y, 1), pitch=round(q, 1), cell=j if inside else None,
                counts={str(i): len(v) for i, v in G["cells"].items()}, box=box_of(e))


def grid_finish():
    """one view per cell with 3 or more frames (the sharp ones: >= 0.7 of the cell's median sharpness); each view's bar
    per law = the best stranger try of the pose bundle the NEAR4 rule compares with it + that law's margin"""
    G = S.get("grid") or {}
    poses = S["pose"][0]; c0 = poses.index((0, 0))
    cells = {j: v for j, v in G.get("cells", {}).items() if len(v) >= 3}
    if c0 not in cells:
        raise ValueError("The straight-ahead view was not captured. Start again and look at the screen first.")
    views = []
    for j in sorted(cells):
        got = cells[j]; med = float(np.median([g[0] for g in got]))
        keep = [c for sh, c in got if sh >= 0.7 * med]
        views.append(dict(cls=j, pose=list(poses[j]), frames=len(keep),
                          map96=B.mode_map([c[0][0] for c in keep]), map48=B.mode_map([c[1][0] for c in keep])))
    PB = S["PB"]; vp = np.array([w["pose"] for w in views], float)
    order = nearest(PB["soft"][:, 0], PB["soft"][:, 1], vp, min(4, len(views)))
    for i, w in enumerate(views):
        rows = np.flatnonzero((order == i).any(0))
        if len(rows) == 0:
            rows = np.flatnonzero(PB["cls"] == w["cls"])
        for law in ("96", "48"):
            w["bar" + law] = float(masked_scores(law, w["map" + law], w["cls"], PB["P" + law][rows], PB["est"][rows]).max()) + S["margin7"][law]
            w["map" + law] = w["map" + law].tolist()
        w["bundle_tries"] = int(len(rows))
    v = dict(version=7, made=time.strftime("%Y-%m-%d %H:%M:%S"), frames=sum(w["frames"] for w in views), views=views,
             margins=S["margin7"], enrolment="pose grid", seen=G.get("n", 0))
    json.dump(v, open(VAULT, "w")); S["vault"] = v; S["grid"] = None
    return dict(ok=True, views=len(views), frames=v["frames"], cells=[w["pose"] for w in views])


def match7(codes, v):
    """NEAR4: the try's angle read by the pose law; compared with the 4 enrolled views nearest it, both laws; the best
    (score - bar) -> (margin, which, angle)"""
    (m96, p96), (m48, p48) = codes
    k, y, q = pose_read(m48)
    vp = np.array([w["pose"] for w in v["views"]], float)
    best = (-1e9, "")
    for i in nearest(y, q, vp, min(4, len(v["views"])))[:, 0]:
        w = v["views"][i]
        for law, p in (("96", p96), ("48", p48)):
            d = float(masked_scores(law, w["map" + law], w["cls"], p[None], [k])[0]) - w["bar" + law]
            if d > best[0]:
                best = (d, "%s %+d/%+d" % (law, w["pose"][0], w["pose"][1]))
    return best[0], best[1], (y, q)


def vault():
    if S.get("vault") is None and os.path.exists(VAULT):
        S["vault"] = json.load(open(VAULT))
    v = S.get("vault")
    if not v or v.get("version") != 7:
        raise ValueError("Enrol again: the pose-law lock needs a new enrolment.")
    return v


def scan_frame(f, box=None):
    e, c, why = frame_codes(linear(rgb_of(f)), box)
    if e is None:
        return dict(match=False, why=why)
    d, which, (y, q) = match7(c, vault())
    return dict(match=d > 0, margin=round(d, 2), which=which, yaw=round(y, 1), pitch=round(q, 1),
                eyes=[e[0].tolist(), e[1].tolist()], box=box_of(e))


def confirm(frames, eyes, which=None):
    """the flash check at the matching frame's eyes: live, and the face there still matches"""
    v = vault()
    L = {k: [linear(rgb_of(x)) for x in frames[k]] for k in ("black", "white", "top", "bottom")}
    e = (np.array(eyes[0]), np.array(eyes[1]))
    e2, why = eyes_on(L["white"][0])
    if e2 is not None and max(np.abs(e2[0] - e[0]).max(), np.abs(e2[1] - e[1]).max()) < 0.25 * np.linalg.norm(e[1] - e[0]):
        e = e2                                                     # the same face, a moment later
    ref48 = crop(L["white"][0], e)
    A = {k: aligned_crops(a, e, ref48) for k, a in L.items()}
    if max(max(abs(s[0]), abs(s[1])) for a in A.values() for _, _, s in a) >= 3:
        return dict(decision="NOT NOW", reason="The face moved during the check flashes. Hold the phone steady.")
    K = {k: np.mean([c48 for c48, _, _ in a], 0) for k, a in A.items()}
    lv = liveness(K)
    if not lv["live"]:
        return dict(decision="NOT NOW", reason=lv["why"], live=lv)
    # v6.1: the check proves the SAME live face in the SAME place as the matching scan frame (continuity), not a second
    # identity pass -- the owner's frames swing around the bar, and a second pass refused true matches by 0.01 and 0.23
    if e2 is None or max(np.abs(e2[0] - np.array(eyes[0])).max(), np.abs(e2[1] - np.array(eyes[1])).max()) >= 0.25 * np.linalg.norm(np.array(eyes[1]) - np.array(eyes[0])):
        return dict(decision="NOT NOW", reason="The face was not in the same place during the check flashes. Hold the phone steady.", live=lv)
    return dict(decision="OPEN", reason="", live=lv, which=which)

def log(entry):
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def summary():
    rows = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    since = json.load(open(VAULT)).get("made_first", "") if os.path.exists(VAULT) else ""
    trials = {}
    for r in rows:                                    # an attempt = its windows; OPEN if any window opened
        if r.get("kind") != "try" or not r.get("attempt"):
            continue
        t = trials.setdefault(r["attempt"], dict(who=r["who"], light=r["light"], decision="NOT YOU" if r["decision"] != "NOT NOW" else "NOT NOW"))
        if r["decision"] == "OPEN":
            t["decision"] = "OPEN"
        elif r["decision"] == "NOT YOU" and t["decision"] == "NOT NOW":
            t["decision"] = "NOT YOU"
    for r in rows:
        if r.get("kind") == "phone" and r.get("id") in trials:
            trials[r["id"]]["phone"] = r["phone"]
    for r in rows:                                    # v5 attempts are logged once, with their final answer
        if r.get("kind") == "attempt":
            trials[r["id"]] = dict(who=r["who"], light=r["light"], decision=r["decision"])
    for r in rows:
        if r.get("kind") == "phone" and r.get("id") in trials:
            trials[r["id"]]["phone"] = r["phone"]
    out = {}
    for t in trials.values():
        k = "%s|%s" % (t["who"], t["light"])
        o = out.setdefault(k, dict(tries=0, facelaw_open=0, phone_tried=0, phone_open=0))
        o["tries"] += 1; o["facelaw_open"] += t["decision"] == "OPEN"
        if t.get("phone") in ("opened", "refused"):
            o["phone_tried"] += 1; o["phone_open"] += t["phone"] == "opened"
    return dict(trials=len(trials), cells=out)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            return self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if self.path == "/api/status":
            v = json.load(open(VAULT)) if os.path.exists(VAULT) else {}
            st = dict(enrolled=v.get("version") == 7)                  # an older enrolment cannot be used by the pose-law lock
            if st["enrolled"]:
                st.update(made=v["made"], frames=v["frames"], angles=len(v["views"]))
            return self.send(200, dict(st, **summary()))
        self.send(404, dict(error="not found"))

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                if self.path == "/api/grid_start":
                    return self.send(200, grid_start())
                if self.path == "/api/grid_frame":
                    return self.send(200, grid_frame(req["frame"], req.get("box")))
                if self.path == "/api/grid_finish":
                    return self.send(200, grid_finish())
                if self.path == "/api/frame":
                    return self.send(200, scan_frame(req["frame"], req.get("box")))
                if self.path == "/api/confirm":
                    out = confirm(req["frames"], req["eyes"], req.get("which"))
                    log(dict(kind="attempt", id=req["attempt"], t=time.strftime("%Y-%m-%d %H:%M:%S"), who=req.get("who"), light=req.get("light"),
                             decision=out["decision"], reason=out.get("reason"), frames=req.get("scanned"), secs=req.get("secs"),
                             margin=out.get("margin"), which=out.get("which"), live=out.get("live")))
                    return self.send(200, out)
                if self.path == "/api/end":
                    log(dict(kind="attempt", id=req["attempt"], t=time.strftime("%Y-%m-%d %H:%M:%S"), who=req.get("who"), light=req.get("light"),
                             decision="NOT YOU" if req.get("faces") else "NOT NOW", reason="no frame matched in %s s" % req.get("secs"),
                             frames=req.get("scanned"), faces=req.get("faces"), best=req.get("best"), why=req.get("why")))
                    return self.send(200, dict(ok=True))
                if self.path == "/api/phone":
                    log(dict(kind="phone", id=req["id"], phone=req["phone"], t=time.strftime("%Y-%m-%d %H:%M:%S")))
                    return self.send(200, summary())
                if self.path == "/api/reset":
                    if os.path.exists(VAULT):
                        os.remove(VAULT)
                    return self.send(200, dict(ok=True))
            self.send(404, dict(error="not found"))
        except ValueError as e:
            self.send(400, dict(error=str(e)))
        except Exception as e:
            sys.stderr.write("error: %r\n" % (e,)); self.send(500, dict(error=type(e).__name__))


if __name__ == "__main__":
    S["T"] = B.tables()
    S["pose"] = PL.load_pose()                                     # rebuilt from the authored lanes, checked against the design
    S["YK"] = np.array([p[0] for p in S["pose"][0]], float); S["PK"] = np.array([p[1] for p in S["pose"][0]], float)
    with np.load(os.path.join(HERE, "pose_bundle.npz")) as Z:
        S["PB"] = {k: Z[k] for k in Z.files}
    PBJ = json.load(open(os.path.join(HERE, "pose_bars.json")))
    S["margin7"] = {"96": PBJ["margin96"], "48": PBJ["margin48"]}
    print("Face ID test v7 (pose law) on http://127.0.0.1:8080 (pose bundle %d stranger tries; margins %s)" % (len(S["PB"]["P96"]), S["margin7"]), flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8080), H).serve_forever()
