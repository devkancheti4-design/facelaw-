"""EYE AND FACE FINDING HEAD-TO-HEAD on the owner's phone, the way of the 300-photo test (one device, no camera, no
screen): every real Extended Yale B photograph (37 people x 64 lights, frontal; the eyes marked by hand) is sent over the
USB cable to Android's own face engine (Chrome's FaceDetector, which calls the phone's detector: boxes and landmarks), and
Face Law reads the very same picture (its face box, its eye finder, and v9's eye check).  Each photo is scaled 2x and
framed as the phone frames faces (a border of a quarter of its mean).
    serve     python3 -I phone_h2h/eyes_h2h.py serve      -> the phone's results: phone_h2h/eyes_android.jsonl
    facelaw   python3 -I phone_h2h/eyes_h2h.py facelaw    -> Face Law's results: phone_h2h/eyes_facelaw.jsonl
    report    python3 -I phone_h2h/eyes_h2h.py report     -> phone_h2h/eyes_h2h.json
A pair of eyes is RIGHT when both are within 0.12 eye distances of the marked eyes, a GROSS MISS when one is more than
0.30 off or the spacing is off by 1.33x or more (the misses that let strangers into v8)."""
import io, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "phone_faceid"))
OUT_A, OUT_F, REPORT = (os.path.join(HERE, f) for f in ("eyes_android.jsonl", "eyes_facelaw.jsonl", "eyes_h2h.json"))
PORT, SCALE, PAD = 8081, 2, 96


def photos():
    """[(key, person, light angle, truth eyes in the framed picture, png bytes)]"""
    import pose_law as PL
    from PIL import Image
    TR = json.load(open(os.path.join(ROOT, "yale_eyes_truth.json")))["eyes"]
    out = []
    for p in sorted(TR):
        t = np.array(TR[p], float).reshape(2, 2) * SCALE + PAD
        for ph in PL.photos(p):
            x = ph["raw"] / max(np.percentile(ph["raw"], 99.5), 1e-9)
            big = np.array(Image.fromarray((255 * np.clip(x, 0, 1)).astype(np.uint8)).resize((PL.W * SCALE, PL.H * SCALE), Image.BICUBIC))
            fr = np.full((PL.H * SCALE + 2 * PAD, PL.W * SCALE + 2 * PAD), int(0.25 * big.mean()), np.uint8)
            fr[PAD:PAD + big.shape[0], PAD:PAD + big.shape[1]] = big
            b = io.BytesIO(); Image.fromarray(fr).save(b, "PNG")
            out.append((ph["name"][:-4], p, float(ph["theta"]), t.tolist(), b.getvalue()))
    return out


PAGE = b"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eye finding test</title><style>body{margin:0;font:16px/1.45 system-ui;background:#f4f5f3;color:#151816}main{max-width:520px;margin:0 auto;padding:20px 16px}
progress{width:100%}.big{font:700 28px system-ui}</style></head><body><main><h1>Eye finding test</h1>
<p>This phone's own face engine reads each photo; nothing leaves the cable. Keep this page open.</p>
<p id="state">Starting...</p><progress id="bar" value="0" max="1"></progress><p>Photos done: <span class="big" id="done">0</span></p></main>
<script>
const $ = (id) => document.getElementById(id);
(async () => {
  if (!('FaceDetector' in window)) { $('state').textContent = 'FaceDetector is not available in this Chrome.'; return; }
  try { const w = await navigator.wakeLock.request('screen'); } catch (e) {}
  const fd = new FaceDetector({ maxDetectedFaces: 10, fastMode: false });
  const keys = await (await fetch('/list')).json(); $('bar').max = keys.length;
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const bmp = await createImageBitmap(await (await fetch('/img/' + encodeURIComponent(k) + '.png')).blob());
    const t0 = performance.now(); let faces = [];
    try { faces = await fd.detect(bmp); } catch (e) {}
    const ms = performance.now() - t0;
    const out = faces.map((f) => ({ box: [f.boundingBox.x, f.boundingBox.y, f.boundingBox.width, f.boundingBox.height],
      marks: (f.landmarks || []).map((l) => ({ type: l.type, pts: l.locations.map((p) => [p.x, p.y]) })) }));
    await fetch('/result', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: k, faces: out, ms }) });
    bmp.close(); $('done').textContent = i + 1; $('bar').value = i + 1; $('state').textContent = 'Photo ' + (i + 1) + ' of ' + keys.length;
  }
  $('state').textContent = 'Done. You can close this page.';
  await fetch('/done', { method: 'POST' });
})();
</script></body></html>"""


def serve():
    P = photos(); IMG = {k: png for k, _, _, _, png in P}; KEYS = [k for k, *_ in P]
    done = set()
    if os.path.exists(OUT_A):
        done = {json.loads(l)["key"] for l in open(OUT_A)}
    todo = [k for k in KEYS if k not in done]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            from urllib.parse import unquote
            if self.path == "/":
                return self.send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/list":
                return self.send(200, json.dumps(todo).encode(), "application/json")
            if self.path.startswith("/img/") and self.path.endswith(".png") and unquote(self.path[5:-4]) in IMG:
                return self.send(200, IMG[unquote(self.path[5:-4])], "image/png")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
            if self.path == "/done":
                print("ANDROID DONE", flush=True); return self.send(200, b"ok", "text/plain")
            if self.path != "/result":
                return self.send(404, b"not found", "text/plain")
            d = json.loads(body)
            if d.get("key") not in IMG:
                return self.send(400, b"bad key", "text/plain")
            with open(OUT_A, "a") as f:
                f.write(json.dumps(dict(d, t=time.time())) + "\n")
            self.send(200, b"ok", "text/plain")
    print("serving %d photos (%d still to do) on http://127.0.0.1:%d -> %s" % (len(KEYS), len(todo), PORT, OUT_A), flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


def facelaw():
    """Face Law on the same pictures: its face box, its eye finder alone (v8), and v9 (candidates + the eye-check law: the
    deployed law, and the half-law that never counted that person)"""
    import bars as B, eye_check as EC, server as SV
    from PIL import Image
    W, WG, TAU, TAUG = EC.load(); SV.EC = EC
    Hz = np.load(os.path.join(ROOT, "data", "eyecheck", "eyecheck_halves.npz"))
    TR = json.load(open(os.path.join(ROOT, "yale_eyes_truth.json")))["eyes"]; ids = sorted(TR)
    with open(OUT_F, "w") as f:
        for k, p, th, t, png in photos():
            rgb = np.ascontiguousarray(np.repeat(np.array(Image.open(io.BytesIO(png)))[..., None], 3, -1))
            lin = SV.linear(rgb); h, w = lin.shape
            box = np.empty(4); hasbox = bool(B.LIB.fl2_face_box(B.dp(lin), h, w, box.ctypes.data_as(B.D)))
            raw = None
            if hasbox:
                disp = np.empty_like(lin); B.LIB.fl2_u8(B.dp(lin), h, w, disp.ctypes.data_as(B.D)); o = np.empty(5)
                if B.LIB.fl2_find_eyes(B.dp(disp), h, w, B.dp(box), o.ctypes.data_as(B.D)) and o[4] >= SV.GATE:
                    raw = [o[:2].tolist(), o[2:4].tolist()]
            res = {}
            for name, ec in (("v9_deployed", (W, WG, TAU, TAUG)), ("v9_held_out", (Hz["W%d" % (1 - ids.index(p) % 2)], Hz["G%d" % (1 - ids.index(p) % 2)], TAU, TAUG))):
                SV.S["EC"] = ec
                e, _, why = SV.eyes_checked(lin)
                res[name] = None if e is None else [e[0].tolist(), e[1].tolist()]
            f.write(json.dumps(dict(key=k, person=p, theta=th, truth=t, box=box.tolist() if hasbox else None, finder=raw, **res)) + "\n")
    print("Face Law: %s" % OUT_F, flush=True)


def judge(e, t):
    """'right', 'near', 'gross' or 'none' for an eye pair against the marked eyes"""
    if e is None:
        return "none"
    e, t = np.array(e, float), np.array(t, float); d = np.hypot(*(t[1] - t[0]))
    err = min(max(np.hypot(*(e[0] - t[0])), np.hypot(*(e[1] - t[1]))), max(np.hypot(*(e[0] - t[1])), np.hypot(*(e[1] - t[0])))) / d
    ratio = np.hypot(*(e[1] - e[0])) / d
    if err > 0.30 or ratio >= 1.33 or ratio <= 1 / 1.33:
        return "gross"
    return "right" if err <= 0.12 else "near"


def android_eyes(faces, t):
    """the phone's eye pair: of the faces it found, the one whose box holds the marked eyes' midpoint (else the largest);
    its two 'eye' landmarks (each the mean of its points)"""
    if not faces:
        return None, False
    mid = np.mean(np.array(t, float), 0)
    inside = [f for f in faces if f["box"][0] <= mid[0] <= f["box"][0] + f["box"][2] and f["box"][1] <= mid[1] <= f["box"][1] + f["box"][3]]
    f = inside[0] if inside else max(faces, key=lambda f: f["box"][2] * f["box"][3])
    eyes = [np.mean(np.array(m["pts"], float), 0).tolist() for m in f.get("marks", []) if m["type"] == "eye" and m["pts"]]
    return (eyes[:2] if len(eyes) >= 2 else None), bool(inside)


def report():
    A = {json.loads(l)["key"]: json.loads(l) for l in open(OUT_A)}
    F = [json.loads(l) for l in open(OUT_F)]
    bands = (("light within 25 deg", 0, 25), ("25-50 deg", 25, 50), ("50-77 deg", 50, 77), ("beyond 77 deg (harsh)", 77, 181))
    rows = {}
    for r in F:
        if r["key"] not in A:
            continue
        b = next(n for n, lo, hi in bands if lo <= r["theta"] <= hi or (lo == 0 and r["theta"] <= hi))
        ae, aface = android_eyes(A[r["key"]]["faces"], r["truth"])
        mid = np.mean(np.array(r["truth"], float), 0); bx = r["box"]
        fl_face = bool(bx) and bx[0] <= mid[0] <= bx[0] + bx[2] and bx[1] <= mid[1] <= bx[1] + bx[3]
        for sys_, face, eyes in (("Android", aface, ae), ("Face Law eye finder (v8)", fl_face, r["finder"]),
                                 ("Face Law v9 (eye check, held out)", fl_face, r["v9_held_out"]), ("Face Law v9 (deployed law)", fl_face, r["v9_deployed"])):
            for band in ("all", b):
                d = rows.setdefault(sys_, {}).setdefault(band, dict(n=0, face=0, right=0, near=0, gross=0, none=0))
                d["n"] += 1; d["face"] += int(bool(face)); d[judge(eyes, r["truth"])] += 1
    ms = [A[k]["ms"] for k in A]
    out = dict(photos=len([r for r in F if r["key"] in A]), android_median_ms=float(np.median(ms)), systems=rows)
    json.dump(out, open(REPORT, "w"), indent=1)
    print("%d photos; Android's median time per photo %.0f ms" % (out["photos"], out["android_median_ms"]))
    for s, by in rows.items():
        print(s)
        for band in ["all"] + [n for n, _, _ in bands]:
            d = by.get(band)
            if d:
                print("   %-24s %5d photos | face found %5.1f%% | eyes right %5.1f%%, near %4.1f%%, GROSS MISS %5.1f%%, none %5.1f%%" % (
                    band, d["n"], 100 * d["face"] / d["n"], 100 * d["right"] / d["n"], 100 * d["near"] / d["n"], 100 * d["gross"] / d["n"], 100 * d["none"] / d["n"]))


if __name__ == "__main__":
    {"serve": serve, "facelaw": facelaw, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "serve"]()
