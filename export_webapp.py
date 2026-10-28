"""Exports what the web app needs (LOCAL ONLY, under data/webapp, which git ignores): the law's constants and tables, the
sector lanes as exhaustively-checked truth tables, the lab strangers for on-device calibration (Yale B, the 19 calibration
people) and the fresh check (CMU PIE, 68 people), the mean frontal face for the identity-blind aligner, and test vectors
so the JavaScript port is checked against this Python law.
    python3 -I export_webapp.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/the private toolkit"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import dark_law as DL
import unlock_law as U
import unlock_v4 as V4
import phone_eyes as PE
import personal as P
import yale_gate as YG
import yale as Y
import align as AL
from world import B_DN, G_DN, SIG_R, W_DN

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "webapp")
FIX = 128                                                # stranger crops stored as uint16 = round(128 x value)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    Z3 = json.load(open("unlock_lanes.json")); Z4 = json.load(open("unlock_v4_lanes.json"))
    F = np.array(Z3["F"]["table"])
    front = np.array([next(im["I"] for im in Y.PEOPLE[p] if im["name"].endswith("A+000E+00.pgm")).reshape(48, 42) for p in Y.IDS])
    REF = float(front.mean())
    cL, cR = PE.canonical_eyes()
    PERM = [int(YG.ev(YG.LS["O_perm"]["expr"], np.array([k]))[0]) for k in range(8)]
    # the table form of the sector lanes must equal the lanes on every gradient pair we can throw at it
    rng = np.random.default_rng(0)
    g = np.concatenate([rng.normal(0, s, (200000, 2)) for s in (0.01, 0.1, 1, 10, 100, 1000)])
    g = np.concatenate([g, np.array([[0, 0], [1 / 32, -1 / 32], [-1 / 32, 1 / 32], [200, -200], [-128, 0], [0, 127.97]])])
    q = lambda v: (np.clip(np.rint(16 * v), -2048, 2047) + 2048).astype(np.int64)
    x = q(g[:, 0]) | (q(g[:, 1]) << 12)
    lanes = YG.ev(YG.LS["O_perm"]["expr"], 4 * YG.ev(YG.LS["O_sy"]["expr"], x) + 2 * YG.ev(YG.LS["O_sx"]["expr"], x)
                  + YG.ev(YG.CAST, YG.ev(YG.LS["O_ay"]["expr"], x) | (YG.ev(YG.LS["O_ax"]["expr"], x) << 12)))
    qx, qy = q(g[:, 0]), q(g[:, 1])
    table = np.array(PERM)[4 * (qy < 2048) + 2 * (qx < 2048) + (np.abs(qy - 2048) > np.abs(qx - 2048))]
    assert (lanes == table).all(), "sector table differs from the lanes"
    law = dict(F=[int(v) for v in F], EBINS=Z3["EBINS"], D1=Z4["D1"]["value"], REF=REF, cL=cL.tolist(), cR=cR.tolist(),
               E_H=list(DL.E_H), SCALES=list(DL.SCALES), STRIDE=list(DL.STRIDE), NBARS=list(U.NBARS), LBANDS=list(U.LBANDS),
               NE=U.NE, NL=U.NL, MIN_CAL=P.MIN_CAL, MARGIN=0.20, JITTER=U.JITTER, PERM=PERM,
               SENSOR=dict(B_DN=B_DN, G_DN=G_DN, SIG_R=SIG_R, W_DN=W_DN), FIX=FIX,
               MEAN_FACE=(front.mean(0) * (REF / front.mean(0).mean())).ravel().tolist(),
               ALIGN=dict(WIN=AL.WIN, SRANGE=AL.SRANGE, TILTS=list(AL.TILTS)))
    json.dump(law, open(os.path.join(OUT, "law.json"), "w"))
    cal = [im["I"] for p in Y.IDS[0:19] for im in Y.PEOPLE[p] if im["I"].sum() > 0]
    pie, _ = P.pie_crops()
    pie = pie.reshape(-1, 2016)
    allc = np.vstack([np.array(cal), pie])
    assert allc.max() * FIX < 65535 and allc.min() >= 0
    hdr = np.array([len(cal), len(pie), 48, 42], np.uint32)
    with open(os.path.join(OUT, "strangers.bin"), "wb") as fh:
        fh.write(hdr.tobytes()); fh.write(np.round(allc * FIX).astype(np.uint16).tobytes())
    # test vectors: lab crops only (the user's face never goes into a test file)
    R = np.random.default_rng(42)
    ims = [front[0], front[1], next(im["I"] for im in Y.PEOPLE[Y.IDS[2]] if im["name"].endswith("A+035E+15.pgm")).reshape(48, 42),
           U.jitter(U.dark_ae(front[3], 20, R), R), U.dark_ae(Y.PEOPLE[Y.IDS[4]][7]["I"].reshape(48, 42), 5, R)]
    tv = dict(crops=[x.ravel().tolist() for x in ims], gallery=[], probe=[], V=[], imb=[], bal=[])
    for x in ims:
        gs, gf = DL.gallery_feats(x); ps, pc = U.probe_feats(x)
        tv["gallery"].append(dict(gs=gs.tolist(), gf=gf.tolist())); tv["probe"].append(dict(ps=ps.tolist(), pc=pc.tolist()))
        tv["V"].append(U.measured_var(x).tolist()); tv["imb"].append(U.imbalance(x)); tv["bal"].append(V4.balance(x).tolist())
    gal = [DL.gallery_feats(x * (REF / x.mean())) for x in ims[:2]]
    gs = np.array([a for a, _ in gal]); gf = np.array([b for _, b in gal])
    tv["scores"] = []
    for x in ims[2:]:
        S, pc = U.scores(F, x, gs, gf)
        tv["scores"].append(dict(S=S.tolist(), eshare=U.eshare(pc), lbin=U.lbin(U.imbalance(x))))
    I = Y.read_pgm(os.path.join(Y.ROOT, Y.IDS[5], Y.PEOPLE[Y.IDS[5]][0]["name"])).astype(float)
    lin = np.pad(255 * (I / 255) ** 2.2, 48, mode="edge")
    eL, eR = np.array([33.5 + 48 + 3.2, 45.5 + 48 - 1.7]), np.array([133.5 + 48 - 2.1, 45.5 + 48 + 2.4])
    tv["crop"] = dict(img=lin.ravel().tolist(), H=lin.shape[0], W=lin.shape[1], eL=eL.tolist(), eR=eR.tolist(),
                      out=PE.crop(lin, eL, eR, cL, cR).ravel().tolist())
    json.dump(tv, open(os.path.join(OUT, "test_vectors.json"), "w"))
    print("law.json, strangers.bin (%d Yale calibration + %d PIE fresh crops, %.1f MB), test_vectors.json -> %s; REF %.4f; PERM %s"
          % (len(cal), len(pie), os.path.getsize(os.path.join(OUT, "strangers.bin")) / 1e6, OUT, REF, PERM))
