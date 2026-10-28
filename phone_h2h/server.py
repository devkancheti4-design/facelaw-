"""FACE DETECTION HEAD-TO-HEAD on the owner's phone: Android's own face detector (Chrome's FaceDetector, which calls the
phone's detector) against Face Law's dark-face law, on the DARK FACE design-validation images (300 night photos, 2,621
faces; the sealed test images are not used).  The server listens on 127.0.0.1 only: the phone reaches it through the USB
cable (adb reverse tcp:8080 tcp:8080).  It sends the phone the images and stores the phone's boxes; nothing else.
    python3 -I phone_h2h/server.py"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEYS = json.load(open(os.path.join(ROOT, "darkface", "split.json")))["design_val"]
IMG = {k: os.path.join(ROOT, "data", "darkface", "image", k + ".png") for k in KEYS}      # as darkface_run.items
OUT = os.environ.get("OUTFILE", os.path.join(ROOT, "phone_h2h", "android_dets.jsonl"))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if self.path == "/list":
            return self.send(200, json.dumps(KEYS).encode(), "application/json")
        if self.path.startswith("/img/") and self.path[5:-4] in IMG and self.path.endswith(".png"):
            return self.send(200, open(IMG[self.path[5:-4]], "rb").read(), "image/png")
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/result":
            return self.send(404, b"not found", "text/plain")
        d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if d.get("key") not in IMG:
            return self.send(400, b"bad key", "text/plain")
        with open(OUT, "a") as f:
            f.write(json.dumps(dict(d, t=time.time())) + "\n")
        self.send(200, b"ok", "text/plain")


if __name__ == "__main__":
    print("serving %d design-validation images on http://127.0.0.1:8080  -> %s" % (len(KEYS), OUT), flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8080), H).serve_forever()
