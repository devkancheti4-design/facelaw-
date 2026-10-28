"""THE FACE LOCK: local web app + the public challenge.  The page only captures camera frames (or, on this Mac, a photo)
and two eye taps; this server runs the law.
    THE LIGHT  every camera capture pairs a black-screen frame with a white-screen frame: white minus black is the face
               lit by the SCREEN ALONE -- the room's own light cancels, so scan and unlock see the same light anywhere
    SCAN       (admin) a burst of 8 black/white pairs: eyes tapped once, the first frame's map finds the eyes in the other
               frames (identity search, scan time only) -> up to 8 maps (numbers; never shown, never sent back), each
               with its own bars: the best of 19,168 lab-stranger tries per cell + 0.20, authored as integer lanes; a fresh
               check on 11,424 CMU PIE tries
    TRY        black, white, top half, bottom half (random order), black, white -> LIVENESS (the screen's light must reach
               the face; the top/bottom light ratio must carry 3D shape a flat photo or a screen cannot) -> the law on the
               screen-lit face against every map: OPEN if it clears any map's bar, else NOT NOW / NOT YOU
    UNLOCK     (admin) one photo -> the law, no liveness
MODE=admin (default): everything, on 127.0.0.1, and the owner's own tries are kept on this Mac (data/webapp/debug) for
diagnosis.  MODE=challenge: TRY only, scores hidden, consent required, rate-limited, a real try counter, nothing kept.
    python3 -I unlock_server.py [port]"""
import base64, io, json, os, shutil, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))           # the law's lane files are read from its own folder
import dark_law as DL
import unlock_law as U
import unlock_v4 as V4
import personal as P
import phone_eyes as PE
import align as AL

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "data", "webapp")
SITE, VAULT, DEBUG = os.path.join(HERE, "webapp"), os.path.join(WEB, "vault.json"), os.path.join(WEB, "debug")
STATS, CONF, LIVECONF = os.path.join(WEB, "challenge_stats.json"), os.path.join(WEB, "challenge.json"), os.path.join(WEB, "liveness.json")
EVENTS = os.path.join(WEB, "challenge_log.jsonl")               # per try: the time and the answer only (no image, no IP, no score)
MODE = os.environ.get("MODE", "admin")
PUBLIC = MODE == "challenge"
MARGIN = 0.20
TRACK_MIN = 2.0                                                 # a burst frame joins the scan only if the first map finds it this well
NAME = {0: "NOT YOU", 1: "OPEN", 2: "NOT NOW"}
LIGHT = ("front", "oblique", "side")
LIVE = dict(REL_MIN=0.15, R3D_MIN=0.12, MIN_PX=150, SHIFT_MAX=1)  # set 2026-10-27 from the owner's live tries (screen share
                                                                # 0.37-0.75, 3D 0.21-0.26), a phone replay (0.02) and a flat print (3D 0.04)
RATE = (15, 600.0)                                              # tries per visitor per 10 minutes (challenge)
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
LAW, LOCK, SEEN = {}, threading.Lock(), {}


def load_law():
    import yale as Y
    Z3 = json.load(open(os.path.join(HERE, "unlock_lanes.json"))); Z4 = json.load(open(os.path.join(HERE, "unlock_v4_lanes.json")))
    front = [next(im["I"] for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")) for p in Y.IDS]
    cL, cR = PE.canonical_eyes()
    LAW.update(F=np.array(Z3["F"]["table"]), EB=Z3["EBINS"], D1=Z4["D1"]["value"], REF=float(np.mean(front)), cL=cL, cR=cR, Y=Y)
    if os.path.exists(LIVECONF):
        LIVE.update(json.load(open(LIVECONF)))


def readj(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def writej(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w")); os.replace(tmp, path)


def raw(b64):
    return base64.b64decode(b64.split(",", 1)[1] if "," in b64[:100] else b64)


def decode(b64):
    """the photo exactly as the law's own loader reads it: upright, sRGB -> linear light, mean of the channels"""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw(b64)))).convert("RGB")
    a = np.asarray(im, float) / 255
    return 255 * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).mean(2)


def mean_of(x):
    xs = [decode(s) for s in (x if isinstance(x, list) else [x])]
    if len({a.shape for a in xs}) != 1:
        raise ValueError("The camera frames differ in size.")
    return np.mean(xs, 0)


def screen_lit(white, black):
    """the face lit by the screen alone: the room's own light is in both frames and cancels"""
    return np.maximum(white - black, 0.0)


def eyes_of(eyes):
    (x1, y1), (x2, y2) = sorted((tuple(map(float, e)) for e in eyes))     # image-left eye first
    return np.array([x1, y1]), np.array([x2, y2])


def face_crop(lin, eyes):
    eL, eR = eyes_of(eyes)
    H, W = lin.shape
    if not all(0 <= x < W and 0 <= y < H for x, y in (eL, eR)) or np.linalg.norm(eR - eL) < 12:
        raise ValueError("Tap both eyes on the photo, apart from each other.")
    return PE.crop(lin, eL, eR, LAW["cL"], LAW["cR"])


def expose(x):
    return x * (LAW["REF"] / max(float(x.mean()), 1e-9))           # a scan at the lab reference exposure


def cell_of(sh, l):
    return l * U.NE + U.ebin(sh, LAW["EB"][l])


def vault_maps(v):
    return v.get("maps") or [dict(gs=v["gs"], gf=v["gf"], bars=v["bars"], balance=v["balance"])]


def scan(imgs, eyes, secret):
    """imgs: the screen-lit frames (or one photo), eyes tapped on imgs[0] -> the vault"""
    t0 = time.time()
    crops = [face_crop(imgs[0], eyes)]
    if len(imgs) > 1:                                             # the first frame's map finds the eyes in the rest
        eL, eR = eyes_of(eyes)
        g0 = DL.gallery_feats(expose(crops[0])); gmap = [np.array([g0[0]]), np.array([g0[1]])]
        with Pool(3) as pool:
            found = pool.starmap(AL.align, [(im, (eL + eR) / 2, float(np.linalg.norm(eR - eL)), LAW["F"], gmap) for im in imgs[1:]])
        for im, (a, b, sc, _) in zip(imgs[1:], found):
            if sc >= TRACK_MIN:
                crops.append(PE.crop(im, a, b, LAW["cL"], LAW["cR"]))
    maps = [DL.gallery_feats(expose(x)) for x in crops]
    gs = np.array([a for a, _ in maps]); gf = np.array([b for _, b in maps])
    Y, R = LAW["Y"], np.random.default_rng(1374)
    cal = [U.jitter(U.dark_ae(im["I"].reshape(48, 42), N, R), R) for p in Y.IDS[0:19] for im in Y.PEOPLE[p] if im["I"].sum() > 0
           for N in U.NBARS for _ in range(2)]
    pie = [t[2] for t in P.pie_strangers(np.random.default_rng(7))[0]]
    with Pool(3, V4.init, (LAW["F"], gs, gf, (48, 42))) as pool:
        oc = pool.map(V4.score_task, cal, chunksize=64)
        op = pool.map(V4.score_task, pie, chunksize=64)
    del cal, pie
    K, nc = len(maps), U.NL * U.NE
    cells = np.array([cell_of(sh, l) for (_, sh, l) in oc]); SC = np.array([S for (S, _, _) in oc])
    flat = []
    for j in range(K):
        for c in range(nc):
            m = cells == c
            flat.append(float(SC[m, j].max()) + MARGIN if m.sum() >= P.MIN_CAL else 999999.0)
    with Pool(1) as pp:                                          # program synthesis writes in its own process: it needs a main thread
        RB, nl = pp.apply(P.author_bars, (flat,))
    bars = [[b if b < 999999.0 - 1e-6 else None for b in RB[j * nc:(j + 1) * nc]] for j in range(K)]
    fresh = 0
    for (S, sh, l) in op:
        c = cell_of(sh, l)
        fresh += any(bars[j][c] is not None and float(S[j]) > bars[j][c] for j in range(K))
    if not secret and os.path.exists(VAULT):
        secret = json.load(open(VAULT)).get("secret", "")
    writej(VAULT, dict(maps=[dict(gs=gs[j].tolist(), gf=gf[j].tolist(), bars=bars[j], balance=V4.balance(crops[j]).tolist()) for j in range(K)],
                       secret=secret, made=time.strftime("%Y-%m-%d %H:%M"), cal_n=len(oc), fresh=fresh, fresh_n=len(op), lanes=nl,
                       frames=len(imgs)))
    return dict(ok=True, maps=K, frames=len(imgs), cal_n=len(oc), fresh=fresh, fresh_n=len(op), lanes=nl, seconds=round(time.time() - t0))


def judge(x):
    """the law on one face crop against every map in the vault -> (decision code, reason, details, secret)"""
    v = json.load(open(VAULT)); maps = vault_maps(v)
    V4.init(LAW["F"], np.array([m["gs"] for m in maps]), np.array([m["gf"] for m in maps]), (48, 42))
    S, sh, l = V4.score_task(x)
    c = cell_of(sh, l)
    # no light gets a lower bar than front light (the screen's own light): 11 of 228 lab strangers cleared the low
    # side/oblique bars through the app's crop path (pose_test.py, 2026-10-27); this can only turn opens into refusals
    front = lambda m: max(b for b in m["bars"][:U.NE] if b is not None) if any(b is not None for b in m["bars"][:U.NE]) else None
    eff = lambda m: None if m["bars"][c] is None else max(m["bars"][c], front(m) or m["bars"][c])
    marg = [(float(S[j]) - eff(m), j) for j, m in enumerate(maps) if eff(m) is not None]
    j = max(marg)[1] if marg else int(np.argmax(S))
    bar = eff(maps[j])
    mis = min(float(np.abs(V4.balance(x) - np.array(m["balance"])).sum()) for m in maps) >= LAW["D1"]
    if marg and max(marg)[0] > 0:
        a, why = 1, "The face matches the map."
    elif not marg:
        a, why = 2, "There are too few lab tries in this light to set a safe bar. Let the screen light the face."
    elif mis:
        a, why = 2, "The light on the face differs from the scan. Face the screen, as in the scan."
    else:
        a, why = 0, "This face does not match the map."
    return a, why, dict(score=round(float(S[j]), 3), bar=None if bar is None else round(bar, 3), light=LIGHT[l], maps=len(maps)), v.get("secret", "")


def best_shift(a, b):
    """how far the face moved between two frames: the shift that best lines up their (light-free) edge maps"""
    ga = np.hypot(*np.gradient(np.log(np.maximum(a, 1e-3)))); gb = np.hypot(*np.gradient(np.log(np.maximum(b, 1e-3))))
    best, arg = -2.0, 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c = np.corrcoef(ga[3 + dy:45 + dy, 3 + dx:39 + dx].ravel(), gb[3:45, 3:39].ravel())[0, 1]
            if c > best:
                best, arg = c, max(abs(dy), abs(dx))
    return arg


def liveness(K):
    """the screen is the light: (1) its light must make up a real share of the face's light (a screen showing a photo
    makes its own); (2) top-half vs bottom-half light: log(D_top / D_bottom) over the face, minus its best smooth
    (quadratic) fit, must keep 3D structure (nose, brows, eye sockets) -- on a flat photo the ratio is smooth"""
    b, w, t, d = (K[k] for k in ("black", "white", "top", "bottom"))
    Dw, Dt, Dd = w - b, t - b, d - b
    yy, xx = np.mgrid[0:48, 0:42].astype(float)
    oval = ((xx - 20.5) / 18.0) ** 2 + ((yy - 24.0) / 23.0) ** 2 <= 1.0
    rel = float(np.median(Dw[oval] / np.maximum(w[oval], 1e-6)))
    shift = max(best_shift(w, t), best_shift(w, d))
    sd = np.sqrt(U.measured_var(b).reshape(48, 42) + np.maximum(U.measured_var(t), U.measured_var(d)).reshape(48, 42))
    ok = oval & (Dt > 3 * sd) & (Dd > 3 * sd)
    n = int(ok.sum())
    out = dict(rel=round(rel, 3), shift=int(shift), n=n, r3d=None)
    if shift > LIVE["SHIFT_MAX"]:
        return dict(out, live=False, fail="moved", why="The face moved during the flashes. Hold still and try again.")
    if rel < LIVE["REL_MIN"] or n < LIVE["MIN_PX"]:
        return dict(out, live=False, fail="light", why="Too little of the screen's light reaches the face. Dim the room, turn the brightness up and hold the screen closer.")
    Q = np.log(Dt[ok] / Dd[ok]); u, v = (xx[ok] - 20.5) / 21.0, (yy[ok] - 24.0) / 24.0
    X = np.stack([np.ones(n), u, v, u * u, u * v, v * v], 1)
    coef = np.linalg.lstsq(X, Q, rcond=None)[0]
    r3d = float(np.sqrt(np.mean((Q - X @ coef) ** 2)))
    out["r3d"] = round(r3d, 4)
    if r3d < LIVE["R3D_MIN"]:
        return dict(out, live=False, fail="flat", why="This looks flat, like a photo or a screen, not a live face.")
    return dict(out, live=True, why="")


def attempt(frames, eyes):
    L_ = {k: mean_of(frames[k]) for k in ("black", "white", "top", "bottom")}
    if len({a.shape for a in L_.values()}) != 1:
        raise ValueError("The camera frames differ in size.")
    K = {k: face_crop(a, eyes) for k, a in L_.items()}
    Lv = liveness(K)
    if not Lv["live"]:
        return dict(decision="NOT NOW", reason=Lv["why"], live=Lv, fail=Lv["fail"]), "not_live"
    a, why, det, secret = judge(face_crop(screen_lit(L_["white"], L_["black"]), eyes))
    out = dict(decision=NAME[a], reason=why, live=Lv, **det)
    if a == 1:
        out["secret"] = secret
    return out, {1: "opened", 0: "not_you", 2: "not_now"}[a]


def keep(kind, req, out):
    """admin only: the owner's own try, kept on this Mac so a failure can be diagnosed"""
    d = os.path.join(DEBUG, time.strftime("%Y%m%d-%H%M%S-") + kind)
    os.makedirs(d, exist_ok=True)
    fr = req.get("frames") or {}
    items = [("%s%d" % (k, i), s) for k in fr for i, s in enumerate(fr[k] if isinstance(fr[k], list) else [fr[k]])]
    items += [("pair%d_%s" % (i, ("black", "white")[j]), s) for i, p in enumerate(req.get("pairs") or []) for j, s in enumerate(p)]
    if req.get("image"):
        items.append(("photo", req["image"]))
    for nm, s in items:
        open(os.path.join(d, nm + ".jpg"), "wb").write(raw(s))
    json.dump(dict(eyes=req.get("eyes"), out={k: v for k, v in out.items() if k != "secret"}), open(os.path.join(d, "result.json"), "w"))
    runs = sorted(os.listdir(DEBUG))
    for old in runs[:-30]:
        shutil.rmtree(os.path.join(DEBUG, old), ignore_errors=True)


def allowed(ip):
    now = time.time(); q = [t for t in SEEN.get(ip, []) if now - t < RATE[1]]
    SEEN[ip] = q
    if len(q) >= RATE[0]:
        return False
    q.append(now); return True


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer"); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/status":
            st = dict(mode=MODE, has_map=os.path.exists(VAULT))
            if st["has_map"]:
                v = json.load(open(VAULT))
                st.update(made=v["made"], fresh=v["fresh"], fresh_n=v["fresh_n"], cal_n=v["cal_n"], maps=len(vault_maps(v)))
            if PUBLIC:
                st.update(counts=readj(STATS, dict(tries=0, opened=0)), **readj(CONF, {}))
            else:
                st.update(live=LIVE)
            return self.send(200, st)
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        fp = os.path.realpath(os.path.join(SITE, name))
        if not fp.startswith(os.path.realpath(SITE) + os.sep) or not os.path.isfile(fp):
            return self.send(404, dict(error="not found"))
        self.send(200, open(fp, "rb").read(), TYPES.get(os.path.splitext(fp)[1], "application/octet-stream"))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > (30_000_000 if PUBLIC else 120_000_000):
            return self.send(413, dict(error="The upload is too large."))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/try":
                if PUBLIC:
                    if readj(CONF, {}).get("paused"):                # 2026-10-28: paused while the photo check is rebuilt
                        return self.send(503, dict(error="The challenge is paused while the photo check is upgraded. No tries are taken right now."))
                    if not req.get("consent"):
                        return self.send(400, dict(error="Please agree to the camera notice first."))
                    if not allowed(self.headers.get("CF-Connecting-IP") or self.headers.get("X-Forwarded-For", "").split(",")[0].strip() or self.client_address[0]):
                        return self.send(429, dict(error="That's a lot of tries. Take a break and come back in a few minutes."))
                if not os.path.exists(VAULT):
                    return self.send(409, dict(error="There is no face lock yet."))
                with LOCK:
                    out, kind = attempt(req["frames"], req["eyes"])
                    if PUBLIC:
                        st = readj(STATS, dict(tries=0, opened=0, not_you=0, not_now=0, not_live=0))
                        st["tries"] += 1; st[kind] = st.get(kind, 0) + 1
                        writej(STATS, st)
                        with open(EVENTS, "a") as fh:
                            fh.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), answer=kind, why=out.get("fail"))) + "\n")
                    else:
                        keep("try", req, out)
                return self.send(200, {k: out[k] for k in ("decision", "reason", "secret") if k in out} if PUBLIC else out)
            if PUBLIC:
                return self.send(404, dict(error="not found"))
            if self.path == "/api/reset":
                with LOCK:
                    if os.path.exists(VAULT):
                        os.remove(VAULT)
                return self.send(200, dict(ok=True))
            if self.path == "/api/scan":
                if req.get("pairs"):
                    imgs = [screen_lit(decode(w), decode(b)) for b, w in req["pairs"]]
                else:
                    imgs = [decode(req["image"])]
                with LOCK:
                    out = scan(imgs, req["eyes"], str(req.get("secret", ""))[:2000])
                    keep("scan", req, out)
                return self.send(200, out)
            if self.path == "/api/unlock":
                if not os.path.exists(VAULT):
                    return self.send(409, dict(error="No map yet. Scan a face first."))
                with LOCK:
                    a, why, det, secret = judge(face_crop(decode(req["image"]), req["eyes"]))
                    out = dict(decision=NAME[a], reason=why, **det)
                    keep("photo", req, out)
                if a == 1:
                    out["secret"] = secret
                return self.send(200, out)
            return self.send(404, dict(error="not found"))
        except ValueError as e:
            return self.send(400, dict(error=str(e)))
        except Exception as e:
            sys.stderr.write("error: %r\n" % (e,))
            return self.send(500, dict(error="The law could not read this: %s" % type(e).__name__))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else (8780 if PUBLIC else 8777)
    load_law()
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print("Face Lock (%s) on http://127.0.0.1:%d" % (MODE, port), flush=True)
    srv.serve_forever()
