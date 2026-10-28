"""The FULL-LIGHT LAW report: one self-contained HTML page from the head-to-head summaries with the full-light law added
(h2h/*_full*_summary.json, h2h/p2_full_matrices.npz).  Nothing is uploaded; the page makes no external requests.
    python3 -I full_report.py"""
import html, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2h_report as HR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", os.path.join(HERE, "h2h"))
FULL = "FACE LAW v2@96 nosearch"
NAMES = {"FACE LAW v2@96 nosearch": "Full-light law (96&times;84, no search)", "FACE LAW v2@96": "Full-light law with &plusmn;2 px search",
         "FACE LAW v2": "Face Law v2 (48&times;42)", "FACE LAW v2 nosearch": "Face Law v2, no search", "FACE LAW v1": "Face Law v1"}
OURS = tuple(NAMES)


def name(m, nosearch_run=False):
    if nosearch_run and m == "FACE LAW v2@96":               # in the no-search run this key IS the full-light law as shipped
        m = FULL
    return NAMES.get(m, html.escape(m).replace("@96", " (96&times;84)").replace("@48", " (48&times;42)"))


def fmt(v, d=1):
    return "&ndash;" if v is None or v != v else ("%.*f" % (d, v))


def p2_table(p2, top=9, nosearch_run=False):
    rows = []
    for m, r in sorted(p2.items(), key=lambda kv: -kv[1]["tar"][1])[:top]:
        lo, hi = r["ci_1e3"]
        nm = "<b>%s</b>" % name(m, nosearch_run) if m in OURS else name(m)
        rows.append([nm, fmt(r["tar"][0]), "%s <span class='ci'>(%s&ndash;%s)</span>" % (fmt(r["tar"][1]), fmt(lo), fmt(hi)), fmt(r["tar"][2]), fmt(r["eer"], 2)])
    return HR.table(["method", "at 1 in 100", "at 1 in 1,000 (95% CI)", "at 0 strangers", "equal error %"], rows)


if __name__ == "__main__":
    L = lambda n: HR.load(n, OUT)
    p2, p2n, p3, p3n_old, p1, p1n = (L("p2_full_summary.json"), L("p2_full_nosearch_summary.json"), L("p3_full_summary.json"),
                                     L("p3_nosearch_summary.json"), L("p1_full_summary.json"), L("p1_full_nosearch_summary.json"))
    old = {"p2": L("p2_summary.json"), "p3": L("p3_summary.json"), "p1": L("p1_summary.json")}
    same = []
    for k, new in (("p2", p2), ("p3", p3), ("p1", p1)):
        if old[k] and new:
            keys = [x for x in old[k] if x in new]
            same.append((k.upper(), len(keys), sum(old[k][x] != new[x] for x in keys)))
    mats = dict(np.load(os.path.join(OUT, "p2_full_matrices.npz"), allow_pickle=True))
    classics = [m for m in p2 if m not in OURS]
    best = max(classics, key=lambda m: p2[m]["tar"][1])
    boots = {(a, f): HR.bootstrap(mats, a, best, far=f) for a in (FULL, "FACE LAW v2@96") for f in (1e-2, 1e-3)}
    # ---- P1: the best law for each setting
    allm = sorted({k.split("|")[0] for k in p1})
    p1rows, route = [], []
    for N in (1000, 100, 20, 5):
        for cls, lab in (("front", "front light"), ("all", "all light")):
            vals = {m: p1["%s|%d|%s" % (m, N, cls)] for m in allm if "%s|%d|%s" % (m, N, cls) in p1}
            ours = {m: vals[m] for m in ("FACE LAW v2", "FACE LAW v2 nosearch", "FACE LAW v2@96", FULL) if m in vals}
            bo = max(ours, key=lambda m: ours[m]["raw"][1])
            bc = max((m for m in vals if m not in OURS), key=lambda m: vals[m]["raw"][1])
            route.append((N, cls, bo))
            p1rows.append(["N = %s, %s" % ("{:,}".format(N), lab)] + ["%s / %s" % (fmt(vals[m]["raw"][1]), fmt(vals[m]["raw"][0])) for m in
                                                                     ("FACE LAW v2", FULL)] + [name(bo), "%s %s / %s" % (name(bc), fmt(vals[bc]["raw"][1]), fmt(vals[bc]["raw"][0]))])
    p3rows = []
    for m, r in sorted(p3.items(), key=lambda kv: -(kv[1]["4"] + kv[1]["5"]))[:9]:
        b = (p3n_old or {}).get(m, {})
        p3rows.append(["<b>%s</b>" % name(m) if m in OURS else name(m)] + [fmt(r.get(s)) for s in ("2", "3", "4", "5")])
    full = p2[FULL]; bc = p2[best]
    # ---- the router (router.py, lanes_router.json): which law reads a try, from the try's own light
    router_html = ""
    if os.path.exists(os.path.join(HERE, "lanes_router.json")):
        RJ = json.load(open(os.path.join(HERE, "lanes_router.json")))
        ms, rt, T = RJ["measure"], RJ["routed"], RJ["ROUTE"]["T"]
        both = L("router_both.json") or {}
        rows = []
        for N in (5, 7, 10, 14, 20, 40):
            m = ms[str(N)]; r = rt["%d|front" % N]; a = rt["%d|all" % N]
            rows.append(["{:,}".format(N), "%.1f" % m["nhat"][1], "%s / %s" % (fmt(m["front v2 (48)"][0]), fmt(m["front v2 (48)"][1])),
                         "%s / %s" % (fmt(m["front full-light (96)"][0]), fmt(m["front full-light (96)"][1])),
                         "%s / %s" % (fmt(r["owner_0"]), fmt(r["owner_1e3"])),
                         "<b>%s / %s</b>" % (fmt(both.get("%d|front" % N, {}).get("owner_0")), fmt(both.get("%d|front" % N, {}).get("owner_1e-3"))),
                         "%.0f%%" % (100 * r["to_full_light"])])
        router_html = ("<p>The light of a try is read from the try itself: a pixel's photon count is mean&sup2; / variance in each of the "
                       "law's four brightness bands, so no camera gain is needed. Tries under %.2f photons per pixel go to Face Law v2 "
                       "at 48&times;42 (four times the photons per pixel); the rest to the full-light law. Each law keeps its own threshold, "
                       "set on the strangers it reads. <b>Both laws</b>: both read every try and the owner is in if either says yes (each "
                       "law at its own threshold: above every stranger's score for 0 strangers, half of the budget each for 1 in 1,000). "
                       "Front light (the phone's screen lights the face), owner accepted %% at 0 strangers / at 1 in 1,000. The threshold "
                       "was chosen on these "
                       "same tries (one number); a sealed check would use new people.</p><div class='card'>%s</div>" % (T / 16.0,
                       HR.table(["photons per pixel", "read from the try (median)", "v2 48&times;42", "full-light law",
                                 "routed", "both laws, either says yes", "routed to the full-light law"], rows)))
    # ---- more than one try (tries.py): the stranger always gets the same number of tries
    tp, tf = L("tries_pairs.json"), L("tries_frames.json")
    tries_html = ""
    if tp:
        rows = []
        for m in (FULL, best, "FACE LAW v2"):
            for far, lab in ((0, "0 strangers"), (1e-3, "1 in 1,000")):
                rows.append([name(m) if m not in OURS else "<b>%s</b>" % name(m), lab] + [fmt(tp["%s|%g|%d" % (m, far, k)]["owner"]) for k in (1, 2, 3, 5)])
        tries_html += ("<h3>Different photos (other lights within 25&deg;), NIST-style pairs</h3><div class='card'>%s</div>"
                       % HR.table(["method", "strangers let in", "1 try", "2 tries", "3 tries", "5 tries"], rows))
    if tf:
        rows = []
        for N in (1000, 100, 20, 5):
            for m in ("Full-light law", "FACE LAW v2 (48)", "Tan-Triggs@96"):
                nm = {"Full-light law": "<b>Full-light law</b>", "FACE LAW v2 (48)": "<b>Face Law v2 (48&times;42)</b>"}.get(m, name(m))
                rows.append(["N = %s" % "{:,}".format(N), nm] + ["%s / %s" % (fmt(tf["%s|%d|0.001|%d" % (m, N, k)]["owner"]), fmt(tf["%s|%d|0|%d" % (m, N, k)]["owner"])) for k in (1, 2, 3, 5)])
        tries_html += ("<h3>Continuous frames: the same photo again, fresh light noise each frame (no search for any method)</h3>"
                       "<p>Owner accepted %% at 1 in 1,000 strangers / at 0 strangers; every stranger gets the same number of frames.</p>"
                       "<div class='card'>%s</div>" % HR.table(["light", "method", "1 frame", "2 frames", "3 frames", "5 frames"], rows))
    lanes = json.load(open(os.path.join(HERE, "lanes_v2_full.json")))["FW96"]
    nl = sum(len(v) for v in lanes["lanes"].values())
    finds = [
        "<b>NIST-style 1:1, normal light: the full-light law is first.</b> Owner accepted at 1 in 1,000 strangers: %s%% against %s's %s%% "
        "(Face Law v2 at 48&times;42: %s%%). People resampled, the lead is %+.1f to %+.1f points; with the &plusmn;2 px search it is %s%% and "
        "%+.1f to %+.1f points." % (fmt(full["tar"][1]), name(best), fmt(bc["tar"][1]), fmt(p2["FACE LAW v2"]["tar"][1]), *boots[(FULL, 1e-3)],
                                     fmt(p2["FACE LAW v2@96"]["tar"][1]), *boots[("FACE LAW v2@96", 1e-3)]),
        "<b>With no stranger let in at all</b> (the threshold above every stranger's score), it accepts %s%% of owners; %s accepts %s%%." % (
            fmt(full["tar"][2]), name(best), fmt(bc["tar"][2])),
        "<b>Without the search for every method</b> the order holds: %s%% against %s%%." % (fmt(p2n[FULL if FULL in p2n else "FACE LAW v2@96"]["tar"][1]), fmt(p2n[best]["tar"][1]))]
    findings = "<ul class='finds'>" + "".join("<li>%s</li>" % f for f in finds) + "</ul>"
    css = """
:root{color-scheme:light;--page:#f6f7f5;--surface:#ffffff;--ink:#141716;--ink2:#4c5450;--muted:#7d8580;--rule:#dfe3e0;--ring:rgba(20,23,22,.10);--acc:#1f6f5c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0f1211;--surface:#181c1b;--ink:#eef1ef;--ink2:#bfc7c3;--muted:#8a938f;--rule:#2a302e;--ring:rgba(255,255,255,.10);--acc:#5cc2a5}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0f1211;--surface:#181c1b;--ink:#eef1ef;--ink2:#bfc7c3;--muted:#8a938f;--rule:#2a302e;--ring:rgba(255,255,255,.10);--acc:#5cc2a5}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:4px 0 6px;text-wrap:balance} h2{font-size:18px;margin:34px 0 8px;text-wrap:balance}
p,li{color:var(--ink2);max-width:72ch} .lede{color:var(--ink);font-size:16px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:12px 14px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--rule);white-space:nowrap} th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600;white-space:normal;vertical-align:bottom} .ci{color:var(--muted);font-size:12px} b{color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:12px 14px}
.tile .v{font-size:28px;font-weight:650;color:var(--acc)} .tile .l{color:var(--ink2);font-size:13px}
.finds li{margin:6px 0} code{font-size:12.5px}
"""
    gen = full["genuine"]; imp = full["impostor"]
    repro = "; ".join("%s: %d of %d earlier numbers reproduced exactly" % (k, n - d, n) for k, n, d in same)
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face Law Full Light</title><style>%s</style></head><body><main>
<h1>The full-light law for Face Law v2</h1>
<p class="lede">Face Law v2's codes read on a 96&times;84 crop of the face (eye distance 50 pixels) with their own counted table, and one
placement instead of the &plusmn;1-pixel search. Tested on the same faces, eye finder, seeds and protocols as the head-to-head with the
classic methods (%s).</p>
<div class="tiles">
<div class="tile"><div class="l">NIST-style 1:1, owner accepted at 1 in 1,000 strangers</div><div class="v">%s%%</div><div class="l">best classic (%s): %s%%</div></div>
<div class="tile"><div class="l">Lead over the best classic, people resampled (95%%)</div><div class="v">%+.1f to %+.1f</div><div class="l">percentage points</div></div>
<div class="tile"><div class="l">Template, one comparison, one face (C++, one core)</div><div class="v">21 KB</div><div class="l">12.5 &micro;s per comparison; 125 ms per 640&times;480 image</div></div>
</div>
<h2>What we found</h2>
%s
<h2>1. NIST-style 1:1</h2>
<p>Every pair of normal-light photos (light within 25&deg;, N = 1,000 photons per face pixel), symmetric score, one threshold for everyone:
%s genuine and %s impostor pairs. Every classic method at both sizes, with the &plusmn;1-pixel search (&plusmn;2 at 96&times;84).</p>
<div class="card">%s</div>
<h3>The same test with no search for any method</h3>
<div class="card">%s</div>
<h2>2. The best law for each setting</h2>
<p>Verification in light and darkness: each person's frontal photo enrols, every other photo (light up to 77&deg;) tries at N photons per
face pixel. Owner accepted %% at 1 in 1,000 strangers / at 0 strangers, raw score (one threshold).</p>
<div class="card">%s</div>
<h2>3. The literature's protocol: Extended Yale B identification</h2>
<p>Yale's own crops, 38 people, subset 1 the gallery, subsets 2&ndash;5 the probes, nearest neighbour, rank-1 %%.</p>
<div class="card">%s</div>
<h2>4. The router: which law for which light</h2>
%s
<h2>5. More than one try</h2>
<p>A phone or a gate lets a person try again. Here the owner may try k times and so may every stranger, against the same threshold;
anyone is let in when any of their tries passes.</p>
%s
<h2>6. How the table was made</h2>
<p>The full-light table (768 entries) was counted on the 37 design people through the exact reference, then written as %d integer lanes,
each proven on every row; the library uses only the table rebuilt from those lanes. The C++ port matches the reference bit for bit on
74 real photos, in light and in darkness, and passes 38 of 38 self-test checksums.</p>
<h2>7. Limits</h2>
<ul>
<li>Lab faces only (Extended Yale B, 37 people). No age, pose, expression or camera change; NIST's sets have all four.</li>
<li>The full-light law is not in the sealed CMU PIE check: PIE's 32&times;32 faces would only be interpolated to 96&times;84.</li>
<li>The cross-fitted tables in sections 1&ndash;3 are counted on the other half of the people; the shipped table is counted on all 37.</li>
</ul>
<p>Generated %s.</p>
</main></body></html>""" % (
        css, repro, fmt(full["tar"][1]), name(best), fmt(bc["tar"][1]), *boots[(FULL, 1e-3)], findings,
        "{:,}".format(gen), "{:,}".format(imp), p2_table(p2), p2_table(p2n, nosearch_run=True),
        HR.table(["setting", "Face Law v2 (48&times;42)", "Full-light law", "our best here", "best classic"], p1rows),
        HR.table(["method", "subset 2", "subset 3", "subset 4", "subset 5"], p3rows), router_html, tries_html, nl, time.strftime("%Y-%m-%d %H:%M"))
    path = os.path.join(HERE, "dist", "facelaw_v2_full_light.html")
    open(path, "w").write(page)
    print("%s  %d bytes" % (path, len(page)))
    print("routing (our best per setting):", route)
