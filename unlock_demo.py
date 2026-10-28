"""Dark Unlock demo (UNLOCK_SPEC.md): map sample faces, then try to unlock each map with its owner's own dark photos and
with strangers' photos.  The law: sealed v4 exactly (F, ACT4, D1 rebuilt from the authored lanes).  Local only.
OWNERS   the user (scan = photo2, the room-lit one) and three Yale B people (scan = their frontal photo)
OWNER TRIES   the owner's photos lit within 12 deg of the scan (the screen case), darkened to 100 photons per face pixel
STRANGER TRIES   10 other Yale B people, the same lights, darkened the same way
COHORT   18 further Yale B faces (fixed)
    python3 -I unlock_demo.py   -> results + data/phone_user/demo_panel.png (the maps)"""
import colorsys, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
import dark_law as DL
import unlock_law as U
import unlock_v4 as V4
import yale as Y
from yale_gate import sectors

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "phone_user")
NAME = {0: "NOT YOU", 1: "OPEN", 2: "NOT NOW"}


def map_image(I, k=6):
    """the map as the scanner stores it at the finest scale: sector 0..7 as a colour wheel, flat points black"""
    s = sectors(I, DL.SCALES[0]).reshape(I.shape)
    G = DL.gaussian_filter(I, DL.SCALES[0]); flat = np.hypot(np.gradient(G, axis=1), np.gradient(G, axis=0)) < 1.0
    rgb = np.zeros(I.shape + (3,))
    for c in range(8):
        rgb[s == c] = colorsys.hsv_to_rgb(c / 8, 0.9, 0.95)
    rgb[flat] = 0
    return Image.fromarray((np.kron(rgb, np.ones((k, k, 1))) * 255).astype(np.uint8))


def gray_image(I, k=6):
    return Image.fromarray(np.kron(I / max(I.max(), 1e-9) * 255, np.ones((k, k))).astype(np.uint8)).convert("RGB")


if __name__ == "__main__":
    Z3 = json.load(open(os.path.join(HERE, "unlock_lanes.json"))); Z4 = json.load(open(os.path.join(HERE, "unlock_v4_lanes.json")))
    F, EB, A4, D1 = np.array(Z3["F"]["table"]), Z3["EBINS"], Z4["ACT4"]["table"], Z4["D1"]["value"]
    ids = Y.IDS
    front = {p: next(im for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")) for p in ids}
    owners_y = [ids[0], ids[4], ids[11]]
    strangers = [p for p in ids[:20] if p not in owners_y][:10]
    cohort = ids[20:38]
    C = np.load(os.path.join(OUT, "crops.npz"))
    R = np.random.default_rng(2710)
    cg = [DL.gallery_feats(front[p]["I"].reshape(48, 42)) for p in cohort]
    near = lambda p: [im["I"].reshape(48, 42) for im in Y.PEOPLE[p] if im["I"].sum() > 0 and im["theta"] <= 12.5 and im is not front[p]]
    stranger_imgs = [I for p in strangers for I in near(p)]
    owners = [("you (photo2 scan)", C["photo2"], [C["photo2"], C["photo1"]])] + [
        ("Yale %s" % p, front[p]["I"].reshape(48, 42), near(p)) for p in owners_y]
    panel = Image.new("RGB", (4 * 42 * 6 + 50, len(owners) * (48 * 6 + 30) + 10), (245, 245, 245))
    dr = ImageDraw.Draw(panel)
    print("SEALED v4 law; 100 photons per face pixel (a dark room lit by the screen); cohort 18 Yale B faces")
    for row, (name, scan, own) in enumerate(owners):
        gal = [DL.gallery_feats(scan)] + cg
        gs = np.array([g[0] for g in gal]); gf = np.array([g[1] for g in gal])
        coh = list(range(1, len(gal)))
        def decide(x):
            S, pc = U.scores(F, x, gs, gf)
            l = U.lbin(U.imbalance(x)); e = U.ebin(U.eshare(pc), EB[l]); mi = int(V4.mismatch(x, scan) >= D1)
            return A4[V4.state4(l, e, mi, U.mbin(U.margins(S, 0, coh)))]
        own_tries = [U.jitter(U.dark_ae(I, 100, R), R) for I in own for _ in range(max(1, 10 // len(own)))]
        st_tries = [U.jitter(U.dark_ae(I, 100, R), R) for I in stranger_imgs]
        oa = [decide(x) for x in own_tries]; sa = [decide(x) for x in st_tries]
        print("  %-18s owner tries: OPEN %2d, NOT NOW %2d, NOT YOU %2d of %2d  |  strangers: OPEN %d of %d" % (
            name, oa.count(1), oa.count(2), oa.count(0), len(oa), sa.count(1), len(sa)))
        y0 = 10 + row * (48 * 6 + 30)
        for col, (img, cap) in enumerate(((gray_image(scan), "scan"), (map_image(scan), "map (directions)"),
                                          (gray_image(own_tries[0]), "owner try: %s" % NAME[oa[0]]),
                                          (gray_image(st_tries[row * 7 % len(st_tries)]), "stranger: %s" % NAME[sa[row * 7 % len(sa)]]))):
            x0 = 10 + col * (42 * 6 + 10)
            panel.paste(img, (x0, y0 + 16)); dr.text((x0, y0), (name + " - " if col == 0 else "") + cap, fill=(0, 0, 0))
    panel.save(os.path.join(OUT, "demo_panel.png"))
    print("panel -> data/phone_user/demo_panel.png (local)")
