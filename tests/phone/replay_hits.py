"""THE CLAIM "never a stranger" against v8's own bars, on REAL photographs (not renders): 6 owners, each enrolled through
the v8 server path from renders at 5 mapped angles (bundle without the owner); every other person's 64 real Extended
Yale B photographs (frontal pose, every light), shown as they are and printed -> any frame that clears a bar?
The owner's own real photographs are the control (a photo of the owner may get in: the other half of the claim)."""
import base64, io, json, os, sys
import numpy as np
from multiprocessing import Pool
FL = os.path.expanduser("~/neooos/facelaw"); T7 = os.path.dirname(os.path.abspath(__file__))
DIAG = []


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
                fj = jpeg(img)
                r = SV.scan_frame(fj)
                if r.get("match") and p != owner:              # replay diagnostics: where were the eyes, what was cropped
                    lin = SV.linear(SV.rgb_of(fj)); e, c, why = SV.frame_codes(lin)
                    tru = np.array(TR[p], float).reshape(2, 2) + PAD
                    c96 = SV.crop(lin, e, True)
                    k, y, q = SV.pose_read(c[1][0])
                    DIAG.append(dict(p=p, name=ph["name"], kind=kind, margin=r["margin"], which=r["which"],
                                     eyes=[list(map(float, e[0])), list(map(float, e[1]))], truth=tru.tolist(),
                                     eye_dist=float(np.hypot(*(e[1] - e[0]))), true_dist=float(np.hypot(*(tru[1] - tru[0]))),
                                     off=[float(np.hypot(*(e[j] - tru[j]))) for j in range(2)], read=list(SV.S["pose"][0][k]),
                                     crop_mean=float(c96.mean()), crop_cv=float(c96.std() / max(c96.mean(), 1e-9)),
                                     sig_share=[float(np.mean((c[0][1] >> 3) == s)) for s in range(4)]))
                    np.save(os.path.join(T7, "hit_%02d_crop96.npy" % len(DIAG)), c96)
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
    json.dump(DIAG, open(os.path.join(T7, "hits_%s.json" % owner), "w"), indent=1)
    return out


if __name__ == "__main__":
    r = job("yaleB01")
    print("owner yaleB01 replay: strangers matched %d (first run: 12)" % r["stranger"]["match"])
    for d in json.load(open(os.path.join(T7, "hits_yaleB01.json"))):
        print("%-28s %-8s margin %6.2f | eyes off truth %5.1f / %5.1f px | eye distance %5.1f (true %5.1f) | read %s | crop cv %.2f | signal shares %s" % (
            d["name"], d["kind"], d["margin"], d["off"][0], d["off"][1], d["eye_dist"], d["true_dist"], d["read"], d["crop_cv"],
            " ".join("%.2f" % x for x in d["sig_share"])))
