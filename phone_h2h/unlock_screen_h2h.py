"""UNLOCK HEAD-TO-HEAD, SCREEN TEST: the phone's own face unlock (Android, convenience class) against Face Law v9, on the
same faces shown on the Mac's screen to the phone's front camera (every one of them a photo-on-a-screen attack):
    strangers   Extended Yale B real photographs (frontal, light within 25 deg), a fixed draw of N_STRANGERS
    owner       the owner's own photographs already on this Mac (data/phone_user, phone_jpeg, phone_heic, iphone);
                shown, never copied anywhere else; one after every OWNER_EVERY strangers
    phase A     Face Law: for each face, one try on the owner's page (driven over the USB cable, Chrome DevTools)
    phase B     Android: for each face, the phone is locked and woken (adb keyevents) and its lock state read 3.5 s
                later (dumpsys trust: deviceLocked; strongAuthRequired != 0 = Android asks for the PIN: the run waits
                for the owner to unlock it, then goes on); after 4 refusals in a row a 31 s pause (Android's timed
                lockout would otherwise refuse everything)
Results: phone_h2h/unlock_screen_h2h.jsonl (one line per face and phase; no image is written).
    python3 -I phone_h2h/unlock_screen_h2h.py [pilot|A|B] [n_strangers]"""
import base64, glob, io, json, os, socket, struct, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import pose_law as PL
ADB = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
OUT = os.path.join(HERE, "unlock_screen_h2h.jsonl")
TRIALS = os.path.join(ROOT, "phone_faceid", "trials.jsonl")
PORT, OWNER_EVERY = 8792, 4
CUR = {"id": None}
IMGS = {}


def faces(n_str):
    """the fixed list: (id, kind, png/jpeg bytes, mime) -- strangers' Yale photos, the owner's photos interleaved"""
    from PIL import Image
    R = np.random.default_rng(2026)
    TR = json.load(open(os.path.join(ROOT, "yale_eyes_truth.json")))["eyes"]
    pool = [(p, ph) for p in sorted(TR) for ph in PL.photos(p) if ph["theta"] <= 25]
    pick = [pool[i] for i in sorted(R.choice(len(pool), n_str, replace=False))]
    R.shuffle(pick)
    strangers = []
    for p, ph in pick:
        x = ph["raw"] / max(np.percentile(ph["raw"], 99.5), 1e-9)
        im = Image.fromarray((255 * np.clip(x, 0, 1)).astype(np.uint8)).resize((168 * 4, 192 * 4), Image.BICUBIC)
        b = io.BytesIO(); im.save(b, "PNG")
        strangers.append(("stranger:" + ph["name"][:-4], "stranger", b.getvalue(), "image/png"))
    own = []
    D = os.path.join(ROOT, "data")
    files = sorted(glob.glob(os.path.join(D, "phone_user", "IMG_[0-9]*.jpg")) + glob.glob(os.path.join(D, "phone_jpeg", "*.jp*g")) +
                   glob.glob(os.path.join(D, "iphone", "*.jp*g")) + glob.glob(os.path.join(D, "phone_heic", "*.[hH][eE][iI][cC]")))
    files = [f for f in files if os.path.basename(f).split(".")[0].split("_")[-1].isdigit() or "phone_user" not in f]
    for f in files:
        if f.lower().endswith(".heic"):
            tmp = os.path.join(os.environ.get("SCRATCH", HERE), "h2h_%d.jpg" % (abs(hash(f)) % 10 ** 8))
            subprocess.run(["sips", "-s", "format", "jpeg", f, "--out", tmp], capture_output=True)
            data = open(tmp, "rb").read(); os.remove(tmp)
        else:
            data = open(f, "rb").read()
        own.append(("owner:%d" % len(own), "owner", data, "image/jpeg"))
    out, k = [], 0
    for i, s in enumerate(strangers):
        out.append(s)
        if (i + 1) % OWNER_EVERY == 0 and own:
            out.append(own[k % len(own)]); k += 1
    return out


PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>face</title><style>html,body{margin:0;height:100%;background:#000}
img{position:fixed;inset:0;margin:auto;max-width:100%;max-height:100%;width:auto;height:100%;object-fit:contain}</style></head>
<body><img id="f"><script>let cur=null;setInterval(async()=>{try{const j=await (await fetch('/cur',{cache:'no-store'})).json();
if(j.id!==cur){cur=j.id;document.getElementById('f').src=j.id?('/img/'+encodeURIComponent(j.id)+'?'+Date.now()):'';}}catch(e){}},200);</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            b, t = PAGE, "text/html; charset=utf-8"
        elif self.path.startswith("/cur"):
            b, t = json.dumps(CUR).encode(), "application/json"
        elif self.path.startswith("/img/"):
            k = urllib.request.unquote(self.path[5:].split("?")[0])
            if k not in IMGS:
                self.send_response(404); self.end_headers(); return
            b, t = IMGS[k][0], IMGS[k][1]
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", t); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)


class CDP:
    """a tiny Chrome DevTools client over the adb-forwarded port, for the Face Law page tab (text only, no screenshots)"""
    def __init__(self):
        subprocess.run([ADB, "forward", "tcp:9333", "localabstract:chrome_devtools_remote"], capture_output=True)
        tabs = [t for t in json.load(urllib.request.urlopen("http://127.0.0.1:9333/json", timeout=5)) if t.get("url", "").startswith("http://localhost:8080")]
        path = "/" + tabs[0]["webSocketDebuggerUrl"].split("/", 3)[3]
        self.s = socket.create_connection(("127.0.0.1", 9333), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall(("GET %s HTTP/1.1\r\nHost: 127.0.0.1:9333\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
                        "Sec-WebSocket-Version: 13\r\n\r\n" % (path, key)).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        self.rest = buf.split(b"\r\n\r\n", 1)[1]; self.n = 0

    def _need(self, n):
        while len(self.rest) < n:
            self.rest += self.s.recv(65536)
        out, self.rest = self.rest[:n], self.rest[n:]; return out

    def eval(self, expr):
        self.n += 1
        data = json.dumps(dict(id=self.n, method="Runtime.evaluate", params=dict(expression=expr, returnByValue=True, awaitPromise=True))).encode()
        mask = os.urandom(4)
        hdr = bytes([0x81]) + (bytes([0x80 | len(data)]) if len(data) < 126 else bytes([0x80 | 126]) + struct.pack(">H", len(data)))
        self.s.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
        while True:
            h = self._need(2); n = h[1] & 0x7F
            if n == 126: n = struct.unpack(">H", self._need(2))[0]
            elif n == 127: n = struct.unpack(">Q", self._need(8))[0]
            m = json.loads(self._need(n))
            if m.get("id") == self.n:
                return m.get("result", {}).get("result", {}).get("value")


def adb(*a):
    return subprocess.run([ADB, "shell"] + list(a), capture_output=True, text=True).stdout


def lock_state():
    t = adb("dumpsys", "trust"); w = adb("dumpsys", "window")
    dl = "deviceLocked=1" in t
    sa = t.split("strongAuthRequired=")[1].split()[0].strip(",") if "strongAuthRequired=" in t else "?"
    return dict(locked=dl, keyguard="isKeyguardShowing=true" in w, strong_auth=sa)


def record(**kw):
    with open(OUT, "a") as f:
        f.write(json.dumps(dict(kw, t=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")


def show(fid, wait=1.5):
    CUR["id"] = fid; time.sleep(wait)


def unlocked_first(why):
    """Face Law runs on the owner's page, so the phone must be unlocked: ask once, then wait (the owner types the PIN)"""
    if lock_state()["locked"]:
        print("UNLOCK NEEDED (%s): unlock the phone with your PIN and leave Chrome on the Face Law page; the run waits." % why, flush=True)
        while lock_state()["locked"]:
            time.sleep(2)
        time.sleep(2)


def phase_a(L):
    unlocked_first("Face Law's turn")
    cdp = CDP()
    cdp.eval("document.querySelector('#who [data-v=screen]').click(); document.querySelector('#light [data-v=room]').click(); 'ok'")
    for fid, kind, _, _ in L:
        unlocked_first("the phone locked itself")
        show(fid)
        n0 = sum(1 for _ in open(TRIALS))
        cdp.eval("document.getElementById('try').click(); 'go'")
        t0 = time.time()
        while time.time() - t0 < 30:                                      # the try: up to 8 s of scanning + checks
            time.sleep(0.7)
            if cdp.eval("document.getElementById('try').disabled") is False and time.time() - t0 > 2:
                break
        rows = [json.loads(l) for l in open(TRIALS)][n0:]
        rows = [r for r in rows if r.get("kind") == "attempt"]
        dec = "OPEN" if any(r["decision"] == "OPEN" for r in rows) else (rows[-1]["decision"] if rows else "ERROR")
        matched = any("faces" not in r for r in rows)
        record(phase="A", face=fid, kind=kind, facelaw=dec, identity_matched=matched,
               reasons=[(r.get("reason") or "")[:60] for r in rows], which=[r.get("which") for r in rows if r.get("which")])
        print("A %-40s %-8s Face Law %-8s%s" % (fid, kind, dec, " (its face matched, the flash check refused)" if matched and dec != "OPEN" else ""), flush=True)
        cdp.eval("for (const b of document.querySelectorAll('#fid button')) b.disabled = true; document.getElementById('res').hidden = true; 'ok'")


def sleep_locked():
    """screen off (the sleep key: never toggles the screen on) and wait until Android reports the phone LOCKED, so a
    phone that did not lock can never be counted as 'Android opened'"""
    adb("input", "keyevent", "223")
    t0 = time.time()
    while time.time() - t0 < 12:
        time.sleep(1.0)
        if lock_state()["locked"]:
            return True
    return False


def phase_b(L):
    refusals = 0
    if not sleep_locked():
        print("STOP: the phone does not lock when its screen goes off, so Android's face unlock cannot be tested this way.", flush=True)
        return
    for fid, kind, _, _ in L:
        st = lock_state()
        if st["strong_auth"] not in ("0x0", "?"):
            print("PIN NEEDED: Android wants your PIN before face unlock works again. Unlock the phone with your PIN; the run waits.", flush=True)
            while lock_state()["locked"]:
                time.sleep(2)
            if not sleep_locked():
                print("STOP: the phone did not lock again.", flush=True); return
            refusals = 0
        if refusals and refusals % 4 == 0:
            time.sleep(31)                                               # Android's timed lockout after repeated failures
        show(fid, 1.0)
        adb("input", "keyevent", "224"); time.sleep(3.5)                   # wake: Android's face unlock looks at the screen
        st = lock_state()
        opened = not st["locked"]
        record(phase="B", face=fid, kind=kind, android="OPENED" if opened else "REFUSED", **st)
        print("B %-40s %-8s Android %s" % (fid, kind, "OPENED" if opened else "refused"), flush=True)
        refusals = 0 if opened else refusals + 1
        if not sleep_locked():                                           # lock again for the next face
            print("STOP: the phone did not lock again.", flush=True); return


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    n_str = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    L = faces(n_str)
    if mode == "pilot":
        L = [x for x in L if x[1] == "stranger"][:3] + [x for x in L if x[1] == "owner"][:2]
    if mode.startswith("repeat:"):                                       # the same faces again, several times each
        want = mode.split(":", 1)[1].split(","); k = int(os.environ.get("REPEAT", "3"))
        L = [x for w in want for x in L if x[0] == w] * k
        mode = os.environ.get("REPEAT_PHASES", "AB")
    for fid, kind, data, mime in L:
        IMGS[fid] = (data, mime)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("faces: %d (%d strangers, %d owner photos); the Mac shows them at http://127.0.0.1:%d/" % (
        len(L), sum(k == "stranger" for _, k, _, _ in L), sum(k == "owner" for _, k, _, _ in L), PORT), flush=True)
    if os.environ.get("WAIT_READY"):
        while not os.path.exists(os.environ["WAIT_READY"]):
            time.sleep(1)
    if mode in ("pilot", "A", "AB"):
        phase_a(L)
    if mode in ("pilot", "B", "AB"):
        phase_b(L)
    CUR["id"] = None
    print("DONE", flush=True)
