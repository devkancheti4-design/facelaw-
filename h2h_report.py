"""The head-to-head REPORT: one self-contained HTML page (and a Markdown twin) from h2h/*.json, pie_v2_results.json and
h2h/p2_matrices.npz.  Nothing is uploaded; the page has no external requests.
    python3 -I h2h_report.py"""
import hashlib, html, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", os.path.join(HERE, "h2h"))
DIST = os.path.join(HERE, "dist")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]           # the validated categorical order (light)
SERIES_DK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
CLASSIC = ["Gradientfaces", "Weberfaces", "Tan-Triggs", "LBP", "LTP", "LGBPHS", "HOG", "Eigenfaces", "Eigenfaces-3", "Fisherfaces"]
REFS = {"FACE LAW v2": "this work", "FACE LAW v1": "this work (2026-10-27)",
        "Gradientfaces": "Zhang, Tian, Song, Lu 2009", "Weberfaces": "Wang, Li, Yang, Liao 2011", "Tan-Triggs": "Tan & Triggs 2010 (preprocessing, L1)",
        "LBP": "Ahonen, Hadid, Pietikainen 2006", "LTP": "Tan & Triggs 2010", "LGBPHS": "Zhang, Shan, Gao, Chen, Zhang 2005", "HOG": "Dalal & Triggs 2005; Deniz et al. 2011",
        "Eigenfaces": "Turk & Pentland 1991", "Eigenfaces-3": "Belhumeur, Hespanha, Kriegman 1997", "Fisherfaces": "Belhumeur, Hespanha, Kriegman 1997"}


def load(name, where=OUT):
    p = os.path.join(where, name)
    return json.load(open(p)) if os.path.exists(p) else None


def best_res(name, p2):
    """a classic method at its better resolution (chosen on P2's 1-in-1,000 column: in the classics' favour)"""
    c = [("%s@%d" % (name, r), p2["%s@%d" % (name, r)]["tar"][1]) for r in (48, 96) if "%s@%d" % (name, r) in p2]
    return max(c, key=lambda t: t[1])[0] if c else None


def weighted_tar(g, gw, i, iw, far):
    o = np.argsort(i); ci = np.cumsum(iw[o]) / iw.sum()
    k = np.searchsorted(ci, 1 - far)
    thr = i[o][min(k, len(o) - 1)]
    return float((gw * (g > thr)).sum() / gw.sum())


def bootstrap(mats, a, b, far=1e-3, reps=1000, seed=7):
    """people resampled with replacement (pairs weighted by how often both people were drawn): the 95% interval of
    method a's accepted rate minus method b's at the same stranger rate"""
    rng = np.random.default_rng(seed)
    folds = sorted({int(k.split("|")[1]) for k in mats if "|" in k})
    prep = []
    for f in folds:
        lab = mats["labels_%d" % f]; ppl = sorted(set(lab)); idx = np.array([ppl.index(x) for x in lab])
        iu = np.triu_indices(len(lab), 1)
        prep.append((len(ppl), idx[iu[0]], idx[iu[1]], mats["%s|%d" % (a, f)][iu], mats["%s|%d" % (b, f)][iu]))
    diffs = []
    for _ in range(reps):
        G = {"a": [], "b": [], "gw": [], "ia": [], "ib": [], "iw": []}
        for n, pi, pj, sa, sb in prep:
            c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            w = c[pi] * c[pj]; same = pi == pj
            wg = np.where(same, c[pi], 0.0)
            G["a"].append(sa[same]); G["b"].append(sb[same]); G["gw"].append(wg[same])
            G["ia"].append(sa[~same]); G["ib"].append(sb[~same]); G["iw"].append(w[~same])
        cat = {k: np.concatenate(v) for k, v in G.items()}
        diffs.append(weighted_tar(cat["a"], cat["gw"], cat["ia"], cat["iw"], far) - weighted_tar(cat["b"], cat["gw"], cat["ib"], cat["iw"], far))
    d = 100 * np.array(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def det_points(g, i, n=60):
    fm = np.logspace(-4.5, 0, n)
    o = np.sort(i)
    out = []
    for f in fm:
        thr = o[min(int(np.ceil((1 - f) * len(o))), len(o) - 1)]
        out.append((f, float((g <= thr).mean())))
    return out


def det_svg(curves):
    """FNMR against FMR (log), one 2 px line per method, a legend, direct end labels, hover readout"""
    W, H, L, R, T, B = 720, 360, 56, 150, 16, 44
    x = lambda f: L + (np.log10(f) + 4.5) / 4.5 * (W - L - R)
    y = lambda v: T + (1 - v) * (H - T - B)
    s = ['<svg class="det" viewBox="0 0 %d %d" role="img" aria-label="Error trade-off: false non-match rate against false match rate">' % (W, H)]
    for k in range(6):
        v = k / 5.0
        s.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/><text class="tick" x="%d" y="%.1f" text-anchor="end">%d%%</text>' % (L, W - R, y(v), y(v), L - 8, y(v) + 4, round(100 * v)))
    for e in range(-4, 1):
        s.append('<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/><text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (
            x(10.0 ** e), x(10.0 ** e), T, H - B, x(10.0 ** e), H - B + 18, "1" if e == 0 else "10<tspan dy='-6' font-size='9'>%d</tspan>" % e))
    s.append('<text class="axis" x="%d" y="%d" text-anchor="middle">false match rate (strangers let in)</text>' % ((L + W - R) // 2, H - 6))
    s.append('<text class="axis" transform="translate(14,%d) rotate(-90)" text-anchor="middle">owner refused</text>' % ((T + H - B) // 2))
    for k, (name, pts) in enumerate(curves):
        d = " ".join(("M" if j == 0 else "L") + "%.1f,%.1f" % (x(f), y(v)) for j, (f, v) in enumerate(pts))
        s.append('<path class="s%d" d="%s" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>' % (k, d))
        s.append('<circle class="d%d" cx="%.1f" cy="%.1f" r="4"><title>%s</title></circle>' % (k, x(pts[0][0]), y(pts[0][1]), html.escape(name)))
    s.append('<text class="lab" x="%.1f" y="%.1f">%s</text>' % (x(curves[0][1][0][0]) + 8, y(curves[0][1][0][1]) - 8, html.escape(curves[0][0])))
    s.append('<g class="hover"><line class="cross" y1="%d" y2="%d"/><text class="readout" x="%d" y="%d"></text></g>' % (T, H - B, L + 6, T + 12))
    s.append("</svg>")
    return "\n".join(s)


def fmt(v, d=1):
    return "&ndash;" if v is None or (isinstance(v, float) and np.isnan(v)) else ("%.*f" % (d, v))


def table(head, rows, cls=""):
    h = "<table class='%s'><thead><tr>%s</tr></thead><tbody>" % (cls, "".join("<th>%s</th>" % c for c in head))
    for r in rows:
        h += "<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in r)
    return h + "</tbody></table>"


if __name__ == "__main__":
    p1, p2, p3, p3n = load("p1_summary.json"), load("p2_summary.json"), load("p3_summary.json"), load("p3_nosearch_summary.json")
    pie = load("pie_v2_results.json", HERE); cost = load("cost.json")
    mats = dict(np.load(os.path.join(OUT, "p2_matrices.npz"), allow_pickle=True)) if os.path.exists(os.path.join(OUT, "p2_matrices.npz")) else None
    chosen = ["FACE LAW v2", "FACE LAW v1"] + [best_res(c, p2) for c in CLASSIC]
    chosen = [c for c in chosen if c]
    # ---- P2 headline
    p2rows = []
    order = sorted(chosen, key=lambda m: -p2[m]["tar"][1])
    for m in order:
        r = p2[m]; lo, hi = r["ci_1e3"]
        p2rows.append(["<b>%s</b>" % m if m.startswith("FACE LAW v2") else m, fmt(r["tar"][0]), "%s <span class='ci'>(%s&ndash;%s)</span>" % (fmt(r["tar"][1]), fmt(lo), fmt(hi)),
                       fmt(r["tar"][2]), fmt(r["eer"], 2)])
    best_classic = max((m for m in chosen if not m.startswith("FACE LAW")), key=lambda m: p2[m]["tar"][1])
    boot = bootstrap(mats, "FACE LAW v2", best_classic) if mats is not None else (float("nan"), float("nan"))
    top = ["FACE LAW v2"] + [m for m in order if not m.startswith("FACE LAW")][:4]
    def pairs(m):
        g, i = [], []
        for f in (0, 1):
            lab = mats["labels_%d" % f]; iu = np.triu_indices(len(lab), 1); same = lab[iu[0]] == lab[iu[1]]
            v = mats["%s|%d" % (m, f)][iu]; g.append(v[same]); i.append(v[~same])
        return np.concatenate(g), np.concatenate(i)
    curves = [(m, det_points(*pairs(m))) for m in top]
    # ---- P1
    def p1row(m, N, cls):
        r = p1.get("%s|%d|%s" % (m, N, cls)); return r
    p1rows = []
    for m in chosen:
        cells = [m]
        for N in (1000, 100, 20, 5):
            r = p1row(m, N, "front"); cells.append("%s / %s" % (fmt(r["raw"][1]), fmt(r["tnorm"][1])) if r else "&ndash;")
        r = p1row(m, 1000, "all"); cells.append("%s / %s" % (fmt(r["raw"][1]), fmt(r["tnorm"][1])) if r else "&ndash;")
        p1rows.append(cells)
    n1 = p1.get("FACE LAW v2|1000|front", {})
    # ---- P3
    p3rows = []
    for m in chosen:
        a = p3.get(m, {}); b = (p3n or {}).get(m, {})
        p3rows.append([m] + [fmt(a.get(str(s))) for s in (2, 3, 4, 5)] + [fmt(b.get(str(s))) for s in (4, 5)])
    # ---- P4
    p4rows = []
    if pie:
        for m in ["FACE LAW v2", "FACE LAW v1"] + ["%s@48" % c for c in CLASSIC]:
            cells = [m.replace("@48", "")]
            for N in (1000, 20):
                r = pie.get("%s|%d|all" % (m, N)); cells.append("%s / %s" % (fmt(r["raw"][1]), fmt(r["tnorm"][1])) if r else "&ndash;")
            p4rows.append(cells)
    proto_sha = open(os.path.join(HERE, "pie_v2_protocol.sha256")).read().split()[0] if os.path.exists(os.path.join(HERE, "pie_v2_protocol.sha256")) else "&ndash;"
    res_sha = open(os.path.join(HERE, "pie_v2_results.sha256")).read().split()[0] if os.path.exists(os.path.join(HERE, "pie_v2_results.sha256")) else "&ndash;"
    # ---- cost
    crows = []
    for r in (cost or []):
        key = r["method"] if r["method"].startswith("FACE LAW") else "%s@%d" % (r["method"], r["res"])
        if key not in chosen:
            continue
        ms = r.get("describe_ms_cpp", r["describe_ms_numpy"])
        cmpu = r.get("compare_us_cpp", r.get("compare_us_numpy"))
        crows.append([key, "{:,}".format(int(r["template_bytes"])) + ((" (%s packed)" % "{:,}".format(int(r["template_bytes_packed"]))) if r.get("template_bytes_packed") else ""),
                      "{:,}".format(int(r["model_bytes"])), html.escape(r["training"]), fmt(ms, 2) + (" C++" if "describe_ms_cpp" in r else ""),
                      fmt(cmpu, 1), "%.2f / %.2f" % (r["flops_per_face"] / 1e6, r["intops_per_face"] / 1e6),
                      "%s / %s" % ("{:,}".format(int(r["flip_ms_softfloat"])), "{:,}".format(int(r["flip_ms_fixedpoint"]))), fmt(r["flip_compare_ms"], 2)])
    v2 = p2["FACE LAW v2"]; bc = p2[best_classic]
    # ---- the findings, from the data
    best48 = max((m for m in p2 if m.endswith("@48")), key=lambda m: p2[m]["tar"][1])
    allm = sorted({k.split("|")[0] for k in p1})
    dark = sorted(((p1["%s|5|front" % m]["raw"][1], m) for m in allm if "%s|5|front" % m in p1), reverse=True)
    nxt = next(t for t in dark if t[1] != "FACE LAW v2")
    p3s = sorted(((p3[m]["4"] + p3[m]["5"], m) for m in p3), reverse=True)
    v3, v3n = p3["FACE LAW v2"], (p3n or {}).get("FACE LAW v2", {})
    pierank = sorted(((pie["%s|1000|all" % m]["raw"][1], m) for m in ["FACE LAW v2", "FACE LAW v1"] + ["%s@48" % c for c in CLASSIC] if "%s|1000|all" % m in pie), reverse=True) if pie else []
    pie20 = sorted(((pie["%s|20|all" % m]["raw"][1], m) for m in ["FACE LAW v2", "FACE LAW v1"] + ["%s@48" % c for c in CLASSIC] if "%s|20|all" % m in pie), reverse=True) if pie else []
    finds = [
        "<b>Same image size (48&times;42): v2 leads every classic.</b> NIST-style, owner accepted at 1 in 1,000: v2 %s%%, the best classic at that size (%s) %s%%." % (fmt(v2["tar"][1]), best48.replace("@48", ""), fmt(p2[best48]["tar"][1])),
        "<b>Classics given four times the pixels (96&times;84) beat it in normal light.</b> %s %s%% against v2's %s%%; with people resampled, v2 minus %s is %+.1f to %+.1f points." % (best_classic.replace("@96", " (96&times;84)"), fmt(bc["tar"][1]), fmt(v2["tar"][1]), best_classic.split("@")[0], boot[0], boot[1]),
        "<b>In the dark v2 is clearly first.</b> At 5 photons per face pixel (front light, raw score, 1 in 1,000): v2 %s%%, next %s %s%%." % (fmt(dark[0][0] if dark[0][1] == "FACE LAW v2" else p1["FACE LAW v2|5|front"]["raw"][1]), nxt[1].replace("@", " at ").replace("96", "96&times;84").replace("48", "48&times;42"), fmt(nxt[0])),
        "<b>The literature's protocol</b> (Yale B subsets 4 and 5): %s leads with %s / %s; v2 %s / %s with the search and %s / %s without it." % (p3s[0][1].replace("@96", " (96&times;84)"), fmt(p3[p3s[0][1]]["4"]), fmt(p3[p3s[0][1]]["5"]), fmt(v3["4"]), fmt(v3["5"]), fmt(v3n.get("4")), fmt(v3n.get("5")))]
    if pierank:
        finds.append("<b>Sealed PIE</b> (every method at 48&times;42): normal light, v2 is %s (%s%%, next %s %s%%); dark (N = 20), v2 is %s (%s%%; first %s %s%%)." % (
            "first" if pierank[0][1] == "FACE LAW v2" else "#%d" % (1 + [m for _, m in pierank].index("FACE LAW v2")), fmt(pie["FACE LAW v2|1000|all"]["raw"][1]),
            [m for _, m in pierank if m != "FACE LAW v2"][0].replace("@48", ""), fmt([v for v, m in pierank if m != "FACE LAW v2"][0]),
            "first" if pie20[0][1] == "FACE LAW v2" else "#%d" % (1 + [m for _, m in pie20].index("FACE LAW v2")), fmt(pie["FACE LAW v2|20|all"]["raw"][1]),
            pie20[0][1].replace("@48", ""), fmt(pie20[0][0])))
    finds.append("<b>Cost.</b> v2 reads a face in about 4.5 million arithmetic operations and compares two in about 50 thousand table look-ups; Tan-Triggs at 96&times;84 needs about 21 million per face, much of it transcendental (powers, tanh).")
    findings_html = "<ul class='finds'>" + "".join("<li>%s</li>" % f for f in finds) + "</ul>"
    js = """<script>
document.querySelectorAll('svg.det').forEach(function(svg){
  var cross=svg.querySelector('.cross'), out=svg.querySelector('.readout');
  svg.addEventListener('mousemove',function(e){var p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;var q=p.matrixTransform(svg.getScreenCTM().inverse());
    if(q.x<56||q.x>570){cross.style.opacity=0;out.textContent='';return;} cross.setAttribute('x1',q.x);cross.setAttribute('x2',q.x);cross.style.opacity=1;
    var f=Math.pow(10,(q.x-56)/(570-56)*4.5-4.5); out.textContent='strangers let in: '+(f<0.001?f.toExponential(1):(100*f).toFixed(f<0.01?2:1)+'%');});
  svg.addEventListener('mouseleave',function(){cross.style.opacity=0;out.textContent='';});
});
</script>"""
    css = """
:root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
%s}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);%s}}
:root[data-theme="dark"]{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);%s}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:8px 0 4px} h2{font-size:19px;margin:36px 0 8px} h3{font-size:15px;margin:20px 0 6px;color:var(--ink2)}
p,li{color:var(--ink2)} .lede{font-size:16px;color:var(--ink)}
.card{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:14px 16px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid);white-space:nowrap} th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600} .ci{color:var(--muted);font-size:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px} .tile{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}
.tile .v{font-size:28px;font-weight:600;color:var(--ink)} .tile .l{color:var(--ink2);font-size:13px}
svg.det{width:100%%;height:auto} .grid{stroke:var(--grid);stroke-width:1} .tick,.axis,.lab,.readout{fill:var(--muted);font-size:11px} .axis{fill:var(--ink2);font-size:12px}
.cross{stroke:var(--axis);stroke-width:1;opacity:0} .legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:6px 0 0;font-size:13px;color:var(--ink2)}
.key{display:inline-block;width:18px;height:2px;vertical-align:middle;margin-right:6px}
code{font-size:12.5px} .finds li{margin:6px 0;color:var(--ink2)} .finds b{color:var(--ink)}
""" % ("".join("--s%d:%s;" % (k, c) for k, c in enumerate(SERIES)), "".join("--s%d:%s;" % (k, c) for k, c in enumerate(SERIES_DK)),
       "".join("--s%d:%s;" % (k, c) for k, c in enumerate(SERIES_DK)))
    css += "".join(".s%d{stroke:var(--s%d)} .d%d{fill:var(--s%d);stroke:var(--surface-1);stroke-width:2} .k%d{background:var(--s%d)}\n" % (k, k, k, k, k, k) for k in range(5))
    legend = "<div class='legend'>" + "".join("<span><i class='key k%d'></i>%s</span>" % (k, html.escape(m)) for k, (m, _) in enumerate(curves)) + "</div>"
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face Law Head-to-Head</title><style>%s</style></head><body><main>
<h1>Face Law v2 against the classic face recognisers</h1>
<p class="lede">Every method saw the same faces, cut by the same automatic eye finder, with the same &plusmn;1-pixel search, trained (where it learns) on
people it is never tested on. Extended Yale B and a sealed CMU PIE check. %s.</p>
<div class="tiles">
<div class="tile"><div class="l">NIST-style 1:1, owner accepted at 1 in 1,000 strangers</div><div class="v">%s%%</div><div class="l">best classic (%s): %s%%</div></div>
<div class="tile"><div class="l">v2 minus best classic, people resampled (95%%)</div><div class="v">%+.1f to %+.1f</div><div class="l">percentage points</div></div>
<div class="tile"><div class="l">Model, template, one comparison</div><div class="v">33 KB</div><div class="l">%s-byte template; %s &micro;s per comparison (C++)</div></div>
</div>
<h2>What we found</h2>
%s
<h2>1. NIST-style 1:1 (the headline)</h2>
<p>Every pair of normal-light photos (light within 25&deg;, N = 1,000 photons per face pixel), symmetric score, one threshold for everyone.
%d genuine and %d impostor pairs. Each classic method at its better of 48&times;42 and 96&times;84 (chosen on this table, in its favour).</p>
<div class="card">%s</div>
<div class="card"><b>Error trade-off</b> &mdash; owner refused against strangers let in (lower is better; logarithmic left to right)%s%s</div>
<h2>2. Verification under light and darkness</h2>
<p>Each person's frontal photo enrols; every other photo (light up to 77&deg;) tries, made dark through the sensor model. Owner accepted %% at 1 in
1,000 strangers: raw score (one threshold) / T-norm against the other half's faces. %d tries per cell (front light).</p>
<div class="card">%s</div>
<h2>3. The literature's protocol: Extended Yale B identification</h2>
<p>Yale's own crops, 38 people, subset 1 (7 photos each) the gallery, subsets 2&ndash;5 the probes, nearest neighbour, rank-1 %%. With the
&plusmn;1 px search, and without it for subsets 4 and 5 (the literature's setting).</p>
<div class="card">%s</div>
<h2>4. Sealed CMU PIE confirmation</h2>
<p>68 people &times; 21 flashes (32&times;32, warped to the law's frame at 48&times;42). v2 was designed on Yale only and never run on PIE before this. The
protocol was written and hashed before the run (<code>%s</code>); results <code>%s</code>. Owner accepted %% at 1 in 1,000: raw / T-norm.</p>
<div class="card">%s</div>
<h2>5. Size, cost, training &mdash; and a 2011 flip phone</h2>
<p>One face = the 9 placements of the &plusmn;1 px search, after the face is found and cut (that front end is shared). Times on this Mac, one core: Face Law v2 in its C++ port, the others in numpy, so compare the counted arithmetic, not the times.
Arithmetic counted by formula (float / integer, millions per face). <b>Flip phone (assumed):</b> ARM9 at 200&nbsp;MHz with no floating-point unit:
an integer operation 1.5 cycles, a software double 40 cycles (20&ndash;100 in practice); a fixed-point port about 3 integer operations per float.
Face detection and the eye finder are shared by every method and not included.</p>
<div class="card">%s</div>
<h2>6. Limits</h2>
<ul>
<li>Lab faces only: 37 people under 64 lights (Yale), 68 people under 21 flashes (PIE). No age, pose, expression or camera change: NIST's sets
have all four, at false-match rates down to one in a million, which 116,000 impostor pairs cannot measure.</li>
<li>The classic methods are reimplemented from their papers with their stated parameters, not the authors' code; LGBPHS pads its borders by
repetition; LTP uses chi-square histograms (the paper also gives a distance-transform match). Learned baselines were trained on the same 18&ndash;19
people per fold as v2's tables.</li>
<li>Strong learned methods of the classic era (high-dimensional LBP, Fisher-vector faces, joint Bayesian) and deep networks were not run.</li>
<li>The flip-phone column is an estimate from counted arithmetic under the stated assumptions, not a measurement on a phone.</li>
</ul>
<h2>7. Reproduce</h2>
<p><code>h2h_all.py</code> (prep, p1, p2, p3, p4protocol, p4), <code>h2h_cost.py</code>, <code>h2h_report.py</code>; methods in <code>classics.py</code>
and <code>yale_baselines.py</code>; v2 in <code>v2ref.py</code> (exact) and <code>v2cpp/</code>. Seeds 1001&ndash;1003, 20261028. Generated %s.</p>
</main>%s</body></html>""" % (
        css, time.strftime("%Y-%m-%d"), fmt(v2["tar"][1]), best_classic, fmt(bc["tar"][1]), boot[0], boot[1],
        "{:,}".format(26528), fmt(next((r.get("compare_us_cpp") for r in (cost or []) if r["method"] == "FACE LAW v2"), float("nan")), 1), findings_html,
        v2["genuine"], v2["impostor"], table(["method", "at 1 in 100", "at 1 in 1,000 (95% CI)", "at 0 of %s" % "{:,}".format(v2["impostor"]), "equal error %"], p2rows),
        det_svg(curves), legend,
        n1.get("owners", 0), table(["method", "N=1000", "N=100", "N=20", "N=5", "all light, N=1000"], p1rows),
        table(["method", "subset 2", "subset 3", "subset 4", "subset 5", "4, no search", "5, no search"], p3rows),
        proto_sha[:16], res_sha[:16], table(["method", "N=1000", "N=20"], p4rows) if p4rows else "not run yet",
        table(["method", "template bytes", "model bytes", "training", "ms per face", "&micro;s per comparison", "Mflop / Mint per face", "flip phone ms per face (soft-float / fixed-point)", "flip ms per comparison"], crows),
        time.strftime("%Y-%m-%d %H:%M"), js)
    os.makedirs(DIST, exist_ok=True)
    path = os.path.join(DIST, "facelaw_v2_head_to_head.html")
    open(path, "w").write(page)
    print("%s  %d bytes;  v2 vs %s at 1 in 1,000: %.1f vs %.1f (bootstrap difference %+.1f to %+.1f)" % (path, len(page), best_classic, v2["tar"][1], bc["tar"][1], *boot))
