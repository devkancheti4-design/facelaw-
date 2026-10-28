"""a tiny Chrome DevTools client over the adb-forwarded port: evaluate one JS expression in the Face ID test tab (text
only -- never a screenshot, the camera preview shows the owner's face)"""
import base64, json, os, socket, struct, sys, urllib.request
tabs = [t for t in json.load(urllib.request.urlopen("http://127.0.0.1:9333/json", timeout=5)) if t.get("url", "").startswith("http://localhost:8080")]
ws = tabs[0]["webSocketDebuggerUrl"]; path = ws.split("9333", 1)[1] if "9333" in ws else "/" + ws.split("/", 3)[3]
s = socket.create_connection(("127.0.0.1", 9333), timeout=10)
key = base64.b64encode(os.urandom(16)).decode()
s.sendall(("GET %s HTTP/1.1\r\nHost: 127.0.0.1:9333\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" % (path, key)).encode())
buf = b""
while b"\r\n\r\n" not in buf:
    buf += s.recv(4096)
rest = buf.split(b"\r\n\r\n", 1)[1]
def send(obj):
    data = json.dumps(obj).encode(); mask = os.urandom(4)
    hdr = bytes([0x81]) + (bytes([0x80 | len(data)]) if len(data) < 126 else bytes([0x80 | 126]) + struct.pack(">H", len(data)))
    s.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
def recv():
    global rest
    def need(n):
        global rest
        while len(rest) < n:
            rest += s.recv(65536)
        out, rest = rest[:n], rest[n:]; return out
    h = need(2); n = h[1] & 0x7F
    if n == 126: n = struct.unpack(">H", need(2))[0]
    elif n == 127: n = struct.unpack(">Q", need(8))[0]
    return json.loads(need(n))
send(dict(id=1, method="Runtime.evaluate", params=dict(expression=sys.argv[1], returnByValue=True, awaitPromise=True)))
while True:
    m = recv()
    if m.get("id") == 1:
        print(json.dumps(m.get("result", {}).get("result", {}).get("value"), ensure_ascii=False)); break
