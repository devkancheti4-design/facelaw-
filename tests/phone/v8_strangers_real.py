"""THE CLAIM "never a stranger" against v8's own bars, on REAL photographs (not renders): 6 owners, each enrolled through
the v8 server path from renders at 5 mapped angles (bundle without the owner); every other person's 64 real Extended
Yale B photographs (frontal pose, every light), shown as they are and printed -> any frame that clears a bar?
The owner's own real photographs are the control (a photo of the owner may get in: the other half of the claim)."""
import base64, io, json, os, sys
import numpy as np
from multiprocessing import Pool
FL = os.path.expanduser("~/neooos/facelaw"); T7 = os.path.dirname(os.path.abspath(__file__))


def job(owner):
    sys.path.insert(0, FL); sys.path.insert(0, os.path.join(FL, "phone_faceid"))
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    import pose_law as PL
    import server as SV
    B = SV.B
    TR = json.load(open(os.path.join(FL, "yale_eyes_truth.json")))["eyes"]; ids = sorted(TR); oi = ids.index(owner)
    SV.S["T"] = B.tables(); SV.S["pose"] = PL.load_pose()
    SV.S["YK"] = np.array([p[0] for p in SV.S["pose"][0]], float); SV.S["PK"] = np.array([p[1] for p in SV.S["pose"][0]], float)
    with np.load(os.path.join(FL, "phone_faceid", "pose_bundle.npz")) as Z:
        PB = {k: Z[k] for k in Z.files}
    keep = PB["who"] != oi
    SV.S["PB"] = {k: (v[keep] if v.shape[:1] == keep.shape else v) for k, v in PB.items()}
    MJ = json.load(open(os.path.join(FL, "mapped_match.json"))); SV.S["marginM"] = {"96": MJ["margin96"], "48": MJ["margin48"]}
    SV.VAULT = os.path.join(T7, "vault_real_%s.json" % owner); SV.S["vault"] = None
    R = np.random.default_rng(oi); PAD = 48
    Zs = np.load(PL.SHAPES)
    hd = PL.Head(Zs[owner + "_rho"].astype(float), Zs[owner + "_n"].astype(float), Zs[owner + "_Z"].astype(float), TR[owner])
    def jpeg(img, N=400.0):
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
            SV.map_frame(jpeg(np.where(v, img, 0.25 * img[v].mean())))
        SV.map_finish()
    printed = lambda raw: gaussian_filter(0.06 + 0.86 * np.clip(raw / max(np.percentile(raw, 99.5), 1e-9), 0, 1), 0.6)
    out = dict(owner=owner, stranger=dict(frames=0, eyes=0, match=0, best=-99.0, hits=[]), own=dict(frames=0, eyes=0, match=0))
    for p in ids:
        for ph in PL.photos(p):
            for kind, img in (("as is", ph["raw"]), ("printed", printed(ph["raw"]))):
                r = SV.scan_frame(jpeg(img))
                d = out["own"] if p == owner else out["stranger"]
                d["frames"] += 1
                if r.get("margin") is None:
                    continue
                d["eyes"] += 1; d["match"] += bool(r["match"])
                if p != owner:
                    d["best"] = max(d["best"], r["margin"])
                    if r["match"]:
                        d["hits"].append((p, ph["name"], kind, r["margin"], r["which"]))
    os.remove(SV.VAULT)
    return out


if __name__ == "__main__":
    TR = json.load(open(os.path.join(FL, "yale_eyes_truth.json")))["eyes"]; ids = sorted(TR)
    owners = [ids[i] for i in (0, 6, 12, 18, 24, 30)]
    with Pool(3) as pool:
        res = pool.map(job, owners)
    tot = dict(frames=0, eyes=0, match=0)
    for r in res:
        s, o = r["stranger"], r["own"]
        for k in tot: tot[k] += s[k]
        print("owner %s: strangers' real photos %d frames (%d with eyes): MATCH %d, closest %.2f from the bar | the owner's own photos: %d of %d match"
              % (r["owner"], s["frames"], s["eyes"], s["match"], s["best"], o["match"], o["eyes"]), flush=True)
        for h in s["hits"]:
            print("   stranger frame cleared a bar:", h)
    print("ALL: strangers' real photographs %d frames (%d with eyes), matched %d" % (tot["frames"], tot["eyes"], tot["match"]))
    json.dump(res, open(os.path.join(T7, "v8_strangers_real.json"), "w"), indent=1)
