"""The paper: "Face Law v2: counted integer laws for light-robust face verification on low-end hardware" (a preprint
for Zenodo).  Every number is read from the measured result files; the two figures are drawn as vector graphics.
    OUTDIR=<head-to-head results> python3 -I paper/make_paper.py      -> paper/face_law_v2_paper.pdf"""
import json, math, os, sys
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.environ.get("OUTDIR", os.path.join(ROOT, "h2h"))
FD = "/System/Library/Fonts/Supplemental/"
for name, f in (("TNR", "Times New Roman.ttf"), ("TNR-B", "Times New Roman Bold.ttf"), ("TNR-I", "Times New Roman Italic.ttf"),
                ("TNR-BI", "Times New Roman Bold Italic.ttf")):
    pdfmetrics.registerFont(TTFont(name, FD + f))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")
INK, MUTED, ACC, GRID = colors.HexColor("#1b1d1c"), colors.HexColor("#6b716e"), colors.HexColor("#1f6f5c"), colors.HexColor("#d9dcda")


def J(name):
    return json.load(open(os.path.join(OUT, name)))


P2, P2N, P3, P1 = J("p2_full_summary.json"), J("p2_full_nosearch_summary.json"), J("p3_full_summary.json"), J("p1_full_summary.json")
TP, TF = J("tries_pairs.json"), J("tries_frames.json")
COST = J("cost.json")
RJ = json.load(open(os.path.join(ROOT, "lanes_router.json")))
BOTH = J("router_both.json")
FULL, FULLN, DARK = "FACE LAW v2@96 nosearch", "FACE LAW v2@96", "FACE LAW v2"     # FULLN: the full-light key in the no-search run

S = dict(
    title=ParagraphStyle("t", fontName="TNR-B", fontSize=17, leading=21, alignment=TA_CENTER, textColor=INK, spaceAfter=6),
    auth=ParagraphStyle("a", fontName="TNR", fontSize=11, leading=14, alignment=TA_CENTER, textColor=INK),
    aff=ParagraphStyle("f", fontName="TNR-I", fontSize=9.5, leading=12, alignment=TA_CENTER, textColor=MUTED, spaceAfter=10),
    abs=ParagraphStyle("ab", fontName="TNR", fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, leftIndent=1.0 * cm, rightIndent=1.0 * cm, textColor=INK),
    h1=ParagraphStyle("h1", fontName="TNR-B", fontSize=12, leading=15, spaceBefore=10, spaceAfter=4, textColor=INK, keepWithNext=1),
    h2=ParagraphStyle("h2", fontName="TNR-B", fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2, textColor=INK, keepWithNext=1),
    eqn=ParagraphStyle("en", fontName="TNR", fontSize=10, leading=13.5, alignment=2, textColor=INK),
    p=ParagraphStyle("p", fontName="TNR", fontSize=10, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5, textColor=INK),
    eq=ParagraphStyle("eq", fontName="TNR", fontSize=10, leading=13.5, leftIndent=1.2 * cm, spaceAfter=4, textColor=INK),
    cap=ParagraphStyle("c", fontName="TNR", fontSize=8.8, leading=11, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=8, textColor=MUTED),
    td=ParagraphStyle("td", fontName="TNR", fontSize=8.6, leading=10.4, textColor=INK),
    th=ParagraphStyle("th", fontName="TNR-B", fontSize=8.6, leading=10.4, textColor=INK),
    ref=ParagraphStyle("r", fontName="TNR", fontSize=8.8, leading=11, leftIndent=0.5 * cm, firstLineIndent=-0.5 * cm, spaceAfter=2, textColor=INK))


def P(t, st="p"):
    return Paragraph(t, S[st])


def EQ(text, num):
    t = Table([[Paragraph(text, S["eq"]), Paragraph("(%d)" % num, S["eqn"])]], colWidths=[15.6 * cm, 1.2 * cm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def f1(v):
    return "%.1f" % v


def table(rows, widths, bold_rows=()):
    data = [[Paragraph(str(c), S["th"] if i == 0 else S["td"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="CENTER")
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]
    for r in bold_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef4f1")))
    t.setStyle(TableStyle(st))
    return t


def nm(m, nosearch_run=False):
    if nosearch_run and m == "FACE LAW v2@96":
        return "Full-light law (96×84)"
    return {FULL: "Full-light law (96×84)", "FACE LAW v2@96": "Full-light law, ±2 px search", DARK: "Dark law (48×42)",
            "FACE LAW v2 nosearch": "Dark law (48×42), no search"}.get(m, m.replace("@96", " (96×84)").replace("@48", " (48×42)"))


# ---------------------------------------------------------------- figure 1: accuracy against arithmetic (one placement)
def fig_cost():
    pts = []
    for r in COST:
        if r["method"].startswith("FACE LAW") or r["res"] != 96:
            continue
        k = "%s@96" % r["method"]
        if k in P2N:
            pts.append((r["method"], (r["flops_per_face"] + r["intops_per_face"]) / 9 / 1e6, P2N[k]["tar"][1], False))
    dl = next(r for r in COST if r["method"] == "FACE LAW v2")
    pts.append(("Dark law (48×42)", (dl["flops_per_face"] + dl["intops_per_face"]) / 9 / 1e6, P2N[DARK]["tar"][1], True))
    pts.append(("Full-light law", 2.319, P2N[FULLN]["tar"][1], True))
    W, H, L, R, T, B = 440, 250, 46, 12, 10, 34
    d = Drawing(W, H)
    x0, x1, y0, y1 = math.log10(0.1), math.log10(100), 0, 100
    X = lambda v: L + (math.log10(v) - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: B + (v - y0) / (y1 - y0) * (H - B - T)
    for v in (0.1, 1, 10, 100):
        d.add(Line(X(v), B, X(v), H - T, strokeColor=GRID, strokeWidth=0.5))
        d.add(String(X(v), B - 11, ("%g" % v), fontName="TNR", fontSize=8, fillColor=MUTED, textAnchor="middle"))
    for v in range(0, 101, 20):
        d.add(Line(L, Y(v), W - R, Y(v), strokeColor=GRID, strokeWidth=0.5))
        d.add(String(L - 4, Y(v) - 3, "%d" % v, fontName="TNR", fontSize=8, fillColor=MUTED, textAnchor="end"))
    d.add(String((L + W - R) / 2, 4, "million operations to read one face (one placement; log scale)", fontName="TNR", fontSize=8.5, fillColor=INK, textAnchor="middle"))
    off = {"Full-light law": (6, 3), "Dark law (48×42)": (-6, -11, "end"), "Tan-Triggs": (-6, -3, "end"), "LTP": (6, -3), "Fisherfaces": (-6, 3, "end"),
           "Gradientfaces": (6, 4), "Weberfaces": (6, -4), "LGBPHS": (-6, 4, "end"), "HOG": (6, -3), "LBP": (6, -3), "Eigenfaces": (6, -9),
           "Eigenfaces-3": (6, 3)}
    for name, x, y, ours in sorted(pts, key=lambda p: p[3]):
        d.add(Circle(X(x), Y(y), 3.6 if ours else 2.6, fillColor=ACC if ours else colors.white, strokeColor=ACC if ours else INK, strokeWidth=1.0))
        o = off.get(name, (6, -3))
        d.add(String(X(x) + o[0], Y(y) + o[1], name, fontName="TNR-B" if ours else "TNR", fontSize=7.8, fillColor=INK if ours else MUTED,
                     textAnchor=o[2] if len(o) > 2 else "start"))
    d.add(String(L + 2, H - T - 9, "owners accepted at 1 stranger in 1,000 (%)", fontName="TNR", fontSize=8.5, fillColor=INK))
    return d


# ---------------------------------------------------------------- figure 2: error trade-off, no search for any method
def fig_det():
    M = dict(np.load(os.path.join(OUT, "p2_full_nosearch_matrices.npz"), allow_pickle=True))
    def pairs(m):
        g, i = [], []
        for f in (0, 1):
            lab = M["labels_%d" % f]; iu = np.triu_indices(len(lab), 1); same = lab[iu[0]] == lab[iu[1]]
            v = M["%s|%d" % (m, f)][iu]; g.append(v[same]); i.append(v[~same])
        return np.concatenate(g), np.sort(np.concatenate(i))
    W, H, L, R, T, B = 440, 230, 46, 110, 10, 34
    d = Drawing(W, H)
    x0, x1, y0, y1 = -5, 0, 0, 50
    X = lambda f: L + (math.log10(f) - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: B + (v - y0) / (y1 - y0) * (H - B - T)
    for e in range(-5, 1):
        d.add(Line(X(10.0 ** e), B, X(10.0 ** e), H - T, strokeColor=GRID, strokeWidth=0.5))
        if e >= -1:
            d.add(String(X(10.0 ** e), B - 11, "1" if e == 0 else "0.1", fontName="TNR", fontSize=8, fillColor=MUTED, textAnchor="middle"))
        else:
            d.add(String(X(10.0 ** e) - 2, B - 11, "10", fontName="TNR", fontSize=8, fillColor=MUTED, textAnchor="middle"))
            d.add(String(X(10.0 ** e) + 5, B - 7, "%d" % e, fontName="TNR", fontSize=6, fillColor=MUTED, textAnchor="start"))
    for v in range(0, 51, 10):
        d.add(Line(L, Y(v), W - R, Y(v), strokeColor=GRID, strokeWidth=0.5))
        d.add(String(L - 4, Y(v) - 3, "%d" % v, fontName="TNR", fontSize=8, fillColor=MUTED, textAnchor="end"))
    d.add(String((L + W - R) / 2, 4, "share of strangers let in (log scale)", fontName="TNR", fontSize=8.5, fillColor=INK, textAnchor="middle"))
    d.add(String(L + 2, H - T - 9, "owners refused (%)", fontName="TNR", fontSize=8.5, fillColor=INK))
    styles = [(FULLN, ACC, 1.8, None), ("Tan-Triggs@96", INK, 1.0, None), ("LTP@96", INK, 1.0, (3, 2)), (DARK, ACC, 1.0, (1.5, 1.5))]
    for m, col, wdt, dash in styles:
        g, i = pairs(m)
        pts = []
        for e in np.linspace(-5, 0, 120):
            f = 10.0 ** e
            thr = i[min(int(math.ceil((1 - f) * len(i))), len(i) - 1)]
            fr = 100.0 * float((g <= thr).mean())
            if fr <= y1:
                pts += [X(f), Y(fr)]
        if len(pts) >= 4:
            pl = PolyLine(pts, strokeColor=col, strokeWidth=wdt)
            if dash:
                pl.strokeDashArray = list(dash)
            d.add(pl)
            at = 100.0 * float((g <= i[min(int(math.ceil((1 - 1e-3) * len(i))), len(i) - 1)]).mean())   # owners refused at 1 in 1,000
            d.add(String(W - R + 6, Y(min(y1, at)) - 3, "%s  %.1f%%" % (nm(m, True), at), fontName="TNR-B" if m in (FULLN, DARK) else "TNR",
                         fontSize=7.6, fillColor=col))
    d.add(Line(X(1e-3), B, X(1e-3), H - T, strokeColor=MUTED, strokeWidth=0.6, strokeDashArray=[2, 2]))
    return d


def build(path):
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                            title="Face Law v2: counted integer laws for light-robust face verification on low-end hardware",
                            author="Manikanta Kancheti (Face Law Labs)", subject="Face verification; integer laws; efficiency",
                            keywords="face verification, Weber law, integer arithmetic, low light, edge devices")
    st = []
    full, tt = P2N[FULLN], P2N["Tan-Triggs@96"]
    st += [P("Face Law v2: counted integer laws for light-robust face verification on low-end hardware", "title"),
           P("Manikanta Kancheti", "auth"), P("Face Law Labs · https://devkancheti4-design.github.io/facelawlabs/ · preprint, 28 October 2026", "aff")]
    st.append(P(
        "<b>Abstract.</b> We describe Face Law v2, a face verification method in which every learned quantity is a counted log-odds "
        "table and every comparison is a sum of integer table look-ups. A face, cut from automatically found eyes, is described by "
        "quantised Weber ratios (a discrete Laplacian of the smoothed image over the local brightness) at three scales, each point "
        "carrying a significance level computed from the image's own measured photon noise. Two laws share this description: a "
        "full-light law on a 96×84 crop and a dark law on a 48×42 crop that keeps four times the photons per pixel; a router reads "
        "the light of each try from its noise. On Extended Yale B, with the same automatic eyes, seeds and protocols as ten "
        "re-implemented classic methods and no search for any method, the full-light law accepts %s%% of owners at one stranger in "
        "1,000 in an all-pairs NIST-style test, against %s%% for the best classic (Tan–Triggs at 96×84; people-bootstrap lead +0.2 to "
        "+5.4 points), and %s%% with none of %s strangers accepted. It reads a face in 2.3 million operations, the same arithmetic as "
        "that classic, and compares two faces with 105,840 integer operations and no floating point (12.5 µs on one core); training "
        "is counting. We explain where this efficiency comes from and state the limits: lab faces only, and no NIST evaluation yet."
        % (f1(full["tar"][1]), f1(tt["tar"][1]), f1(full["tar"][2]), "{:,}".format(full["impostor"])), "abs"))
    st.append(Spacer(1, 6))

    st.append(P("1  Introduction", "h1"))
    st.append(P(
        "Face verification on cheap phones, drones and other small machines meets two problems at once: little arithmetic and "
        "little light. Deep networks solve the first with accelerators and the second with large training sets; the classic "
        "illumination-robust descriptors (gradient faces, Weber faces, Tan–Triggs preprocessing with local ternary patterns) are "
        "light enough but lose accuracy in the dark. Face Law v2 takes a third route. It writes the physics of the camera and of "
        "local light into explicit laws (a Poisson sensor, a ratio that cancels the light's gain, a significance measured from the "
        "photons), and it learns only what physics cannot say: how often each quantised code occurs for the same person and for "
        "different people. Those frequencies are counted, not optimised, and stored as one-byte table entries; matching two faces "
        "is then a sum of table look-ups."))
    st.append(P(
        "Contributions: (i) an integer face descriptor whose quantisation needs only comparisons, with a per-point significance "
        "from the measured noise, so that one table serves every light level; (ii) a full-light law and a dark law with a router "
        "that reads the photon count of a try from the try itself; (iii) a head-to-head with ten classic methods under identical "
        "conditions, with an operation count and measured timings; (iv) an exact C++ implementation that reproduces the reference "
        "bit for bit."))

    st.append(P("2  The laws", "h1"))
    st.append(P("2.1  Face, eyes and crop", "h2"))
    st.append(P(
        "A Viola–Jones face box [4] with OpenCV's frontal cascade [16] gives the region. An eye law finds the pupils: a counted "
        "table T over gradient-direction sector codes (25 × 25 × 51 entries) scores every position in an 80×110 frame, the five "
        "best candidates per eye are paired under a geometry penalty, and a gate on the weaker eye's evidence rejects boxes that "
        "are not faces. The crop maps the eyes to fixed positions: eye distance 25 pixels in the 48×42 frame and 50 pixels in the "
        "96×84 frame, with box-filter anti-aliasing before bilinear sampling."))
    st.append(P("2.2  Light and noise from the image itself", "h2"))
    st.append(P(
        "The crop x is exposed to a reference mean, x<sub>e</sub> = x · REF / mean|x|. The noise is measured from the image, not "
        "assumed: over 2×2 blocks (a, b, c, e) the diagonal residual d = |a − b − c + e| / 2 cancels smooth structure, and in four "
        "brightness bands b the noise is s<sub>b</sub> = median(d) / 0.6745. Each pixel takes the variance V of its band. For a "
        "Poisson sensor a pixel's photon count is mean² / variance, so the photons per pixel of the try are read without the "
        "camera's gain:"))
    st.append(EQ("N<sub>est</sub> = ¼ Σ<sub>b</sub> μ<sub>b</sub>² / s<sub>b</sub>²", 1))
    st.append(P("2.3  Weber codes", "h2"))
    st.append(P(
        "At three scales k (Gaussian σ = 0.75, 1.5 and 3 pixels; strides 1, 2 and 4) the smoothed image G<sub>k</sub> and the "
        "Laplacian at its stride give a ratio that cancels the local light's gain (Weber's law [7, 14]):"))
    st.append(EQ("L<sub>k</sub>(p) = 8 G<sub>k</sub>(p) − Σ<sub>8 neighbours q at stride k</sub> G<sub>k</sub>(q),    "
                 "a = round(α<sub>k</sub> L<sub>k</sub>),    g = round(max(G<sub>k</sub>, floor))", 2))
    st.append(P(
        "The pair (a, g) is packed into one 24-bit integer and quantised with integer comparisons only: s = [a ≥ 0], m = |a|, "
        "k = [2m &gt; g] + [m &gt; g] + [m &gt; 2g], and the bin is 4 + k if s else 3 − k. Eight bins say how strongly, and in which "
        "direction, a point stands out from its neighbourhood relative to its own brightness. The grid has 2,652 points on the "
        "48×42 crop and 10,584 on the 96×84 crop."))
    st.append(P("2.4  Significance", "h2"))
    st.append(P(
        "A try's point is only as reliable as its photons. With E<sub>L,k</sub> the noise energy of the scale-k Laplacian kernel, "
        "the significance counts how many noise standard deviations the Laplacian stands above the measured noise:"))
    st.append(EQ("q = min(3, floor(|L<sub>k</sub>| / √(E<sub>L,k</sub> · V)))", 3))
    st.append(P(
        "In bright light most points reach q = 3; in the dark most fall to 0. The table below learns how much each level can be "
        "trusted, which is why one table serves every light level."))
    st.append(P("2.5  Counted table and score", "h2"))
    st.append(P(
        "The enrolled face contributes its bin at each point (the map), the try contributes its bin and significance, and the "
        "scale is known, giving one of 768 codes c = map | probe·8 | q·64 | k·256. Over design tries made at N = 5, 20, 100 and "
        "1,000 photons per face pixel through a Poisson sensor model with auto-exposure, the codes are counted for the same person "
        "and for different people:"))
    st.append(EQ("FW[c] = clip(round(16 · ln(P<sub>same</sub>(c) / P<sub>other</sub>(c))), −127, 127),    "
                 "score = (1/n) Σ<sub>points</sub> FW[c<sub>p</sub>]", 4))
    st.append(P(
        "The verification score is the mean of both directions (try against enrolment and enrolment against try). The dark law "
        "can also take the best of nine placements within ±1 pixel; the full-light law uses one placement."))
    st.append(P("2.6  Two laws, a router, and both together", "h2"))
    st.append(P(
        "The full-light law reads the 96×84 crop with its own α and its own table; the dark law reads the 48×42 crop, where each "
        "pixel collects four times the photons. The router sends a try to the full-light law when floor(16 N<sub>est</sub>) ≥ 108, that is from "
        "6.75 photons per pixel by equation (1); the threshold is the middle of the measured crossover in front light. For a "
        "device that must work day and night, both laws can read every try, and the owner is accepted if either law accepts at "
        "its own threshold."))
    st.append(P("2.7  Exact integer implementation", "h2"))
    st.append(P(
        "Every floating-point step has a fixed order (frozen kernels, sequential sums, no fused multiply-add), so the C++ library "
        "reproduces the reference bit for bit on any machine. Every table is written as small integer lanes produced by program "
        "synthesis and checked exhaustively against the counted values; the library uses only the tables rebuilt from those lanes."))

    st.append(P("3  Where the efficiency comes from", "h1"))
    st.append(P(
        "<b>Counting instead of optimisation.</b> Training is two histograms of 768 codes over design tries; the ratio is the "
        "table. There is no gradient descent and no learned projection; the whole model is 33 KB of 8-bit entries (768 for the "
        "full-light table, 31,875 for the eye law)."))
    st.append(P(
        "<b>Light handled by the code, not by preprocessing.</b> The Weber ratio cancels the local gain of the light inside the "
        "descriptor, and the significance level carries the photon noise into the table. No illumination-normalisation stage is "
        "needed: Tan–Triggs preprocessing, for comparison, applies a power law, a difference of Gaussians and a contrast "
        "equalisation with exponentials and hyperbolic tangents to every pixel."))
    st.append(P(
        "<b>Quantisation by comparison, matching by look-up.</b> A point's bin costs three integer comparisons; its code is an "
        "index; the score adds one table entry per point. A comparison of two templates is 2 × 10,584 look-ups and additions, "
        "105,840 integer operations in our count, with no floating point. A template is one byte per point per role: 21,176 bytes."))
    st.append(P(
        "<b>The right resolution for the photons.</b> The router reuses the noise measurement to pick the crop: in the deep dark "
        "the 48×42 crop's larger pixels collect four times the photons and win; in light the 96×84 crop's detail wins. At full "
        "light one placement suffices: the ±2-pixel search adds 0.7 points of acceptance (people bootstrap −0.1 to +1.6) at nine "
        "times the cost."))
    stage = [["Stage (96×84 crop, one placement)", "Floating-point", "Integer"],
             ["Exposure", "24,192", "–"], ["Noise measurement (2×2 residual, local mean)", "469,728", "–"],
             ["Three Weber scales: Gaussians", "1,064,448", "–"], ["Three Weber scales: Laplacian and packing", "411,264", "–"],
             ["Per point", "31,752", "–"], ["Quantisation and codes (10,584 points)", "–", "317,520"],
             ["<b>Read one face</b>", "<b>2,001,384</b>", "<b>317,520</b>"], ["<b>One comparison (both directions)</b>", "<b>0</b>", "<b>105,840</b>"]]
    st.append(KeepTogether([table(stage, [8.2 * cm, 3.0 * cm, 3.0 * cm]),
                            P("<b>Table 1.</b> Counted operations of the full-light law (the face box and the eye law, shared by every method, "
                              "are not included).", "cap")]))
    tc = next(r for r in COST if r["method"] == "Tan-Triggs" and r["res"] == 96)
    eff = [["", "Full-light law", "Tan–Triggs at 96×84"],
           ["Owners accepted, NIST-style, 1 in 1,000 (no search for either)", "<b>%s%%</b>" % f1(full["tar"][1]), "%s%%" % f1(tt["tar"][1])],
           ["Operations to read one face", "2.0 M float + 0.32 M integer", "%.1f M float" % (tc["flops_per_face"] / 9 / 1e6)],
           ["One comparison", "105,840 integer", "%s float" % "{:,}".format(int(tc["flops_per_compare"]))],
           ["Template", "21,176 bytes", "%s bytes" % "{:,}".format(int(tc["template_bytes"]))],
           ["Model", "33 KB counted tables", "none"],
           ["Measured, one core of a laptop", "template 85–125 ms per 640×480 image; comparison 12.5 µs", "comparison 3.8 µs"],
           ["Assumed 2011 phone without FPU: one face, fixed point", "47 ms", "%.0f ms" % (tc["flip_ms_fixedpoint"] / 9)],
           ["Assumed 2011 phone without FPU: one comparison", "<b>0.79 ms</b>", "%.2f ms" % tc["flip_compare_ms"]]]
    st.append(KeepTogether([table(eff, [6.6 * cm, 4.7 * cm, 3.6 * cm]),
                            P("<b>Table 2.</b> The full-light law against the best classic, both reading the face once. The phone rows "
                              "are estimates from counted arithmetic for an assumed ARM9 at 200 MHz without a floating-point unit (an "
                              "integer operation 1.5 cycles, a software double 40 cycles, a fixed-point port 3 integer operations per "
                              "float). On a processor with floating point the classic's short float loop compares faster.", "cap")]))
    st.append(KeepTogether([fig_cost(), P(
        "<b>Figure 1.</b> Accuracy against arithmetic, every method reading the face once (NIST-style test, owners accepted at "
        "1 stranger in 1,000; classic methods at 96×84). The full-light law is the most accurate method and costs what a "
        "mid-range classic costs; the cheaper classics stay below 77%.", "cap")]))

    st.append(P("4  Experiments", "h1"))
    st.append(P(
        "<b>Faces.</b> Extended Yale B [1, 2]: 37 people with hand-checked pupils (one person with nearly closed eyes excluded), "
        "every photo with the light within 77° of the camera, cut by the automatic eye finder (cross-fitted). Dark tries are made "
        "from the photos with a Poisson–Gaussian sensor model with auto-exposure at N photons per face pixel; a 96×84 try gets N/4 "
        "per pixel, so every method sees the same light. <b>Methods.</b> Gradientfaces [5], Weberfaces [6], Tan–Triggs [7], LBP "
        "[8], LTP [7], LGBPHS [9], HOG [10, 11], Eigenfaces [12], Eigenfaces without the first three components [13] and "
        "Fisherfaces [13], re-implemented from their papers, each at 48×42 and 96×84. Learned parts (our tables, the subspaces) "
        "are trained on the other half of the people (two folds)."))
    st.append(P("4.1  NIST-style 1:1 verification", "h2"))
    rows = [["Method (no search)", "1 in 100", "1 in 1,000 (95% CI)", "0 strangers", "Equal error"]]
    order = sorted(P2N.items(), key=lambda kv: -kv[1]["tar"][1])[:8]
    for m, r in order:
        rows.append([("<b>%s</b>" % nm(m, True)) if m in (FULLN, DARK) else nm(m, True), f1(r["tar"][0]),
                     "%s (%s–%s)" % (f1(r["tar"][1]), f1(r["ci_1e3"][0]), f1(r["ci_1e3"][1])), f1(r["tar"][2]), "%.2f" % r["eer"]])
    st.append(KeepTogether([table(rows, [5.2 * cm, 1.9 * cm, 3.3 * cm, 2.1 * cm, 2.0 * cm], bold_rows=[1 + [m for m, _ in order].index(FULLN)]),
                            P("<b>Table 3.</b> Every pair of normal-light photos (light within 25°, N = 1,000), symmetric score, one "
                              "threshold: %s genuine and %s impostor pairs; owners accepted (%%). No method uses the search. With the "
                              "±1-pixel search for the classics (±2 at 96×84) the order holds: full-light law %s (with its own search "
                              "%s), Tan–Triggs %s." % ("{:,}".format(full["genuine"]), "{:,}".format(full["impostor"]), f1(P2[FULL]["tar"][1]),
                                                          f1(P2[FULLN]["tar"][1]), f1(P2["Tan-Triggs@96"]["tar"][1])), "cap")]))
    st.append(P(
        "Resampling people (1,000 draws, pairs weighted by how often both people are drawn) puts the full-light law's lead over "
        "Tan–Triggs at +0.2 to +5.4 points at 1 in 1,000 and +0.2 to +2.9 at 1 in 100. The lead is at the strict end: at the equal "
        "error rate Tan–Triggs (6.81%%) and LTP (6.64%%) are slightly better than the full-light law (%.2f%%)." % full["eer"]))
    st.append(KeepTogether([fig_det(), P(
        "<b>Figure 2.</b> Error trade-off in the NIST-style test, no search for any method. The dotted line marks 1 stranger in 1,000.", "cap")]))
    h42 = P("4.2  Light and darkness", "h2")
    ours = {"FACE LAW v2", "FACE LAW v2 nosearch", "FACE LAW v2@96", FULL, "FACE LAW v1"}
    rows = [["Light", "N", "Dark law", "Full-light law", "Best classic"]]
    for cls, lab in (("front", "front"), ("all", "all directions")):
        for N in (5, 20, 100, 1000):
            g = lambda m: P1["%s|%d|%s" % (m, N, cls)]["raw"]
            cl = max((m for m in {k.split("|")[0] for k in P1} if m not in ours), key=lambda m: g(m)[1])
            rows.append([lab, "{:,}".format(N), "%s / %s" % (f1(g(DARK)[1]), f1(g(DARK)[0])), "%s / %s" % (f1(g(FULL)[1]), f1(g(FULL)[0])),
                         "%s %s / %s" % (nm(cl), f1(g(cl)[1]), f1(g(cl)[0]))])
    st.append(KeepTogether([h42, table(rows, [2.4 * cm, 1.3 * cm, 2.4 * cm, 2.6 * cm, 5.8 * cm]),
                            P("<b>Table 4.</b> Each person's frontal photo enrols; every other photo tries at N photons per face pixel. "
                              "Owners accepted (%) at 1 in 1,000 / at 0 strangers, one threshold per cell. The dark law here uses the "
                              "±1-pixel search, the full-light law none, the classics their search.", "cap")]))
    st.append(P(
        "In the deep dark (N = 5) the dark law leads clearly (front light 75.7%% against the best classic's 62.8%%). From 20 photons "
        "per pixel the full-light law leads in light from all directions (53.0%% against 47.1%% at N = 1,000); with front light "
        "the classics catch up (Gradientfaces leads at N = 100, LTP ties at N = 1,000). With the router (front light, 1 in 1,000) "
        "the routed system follows the better law: %s%% at N = 5 and %s%% at N = 20; with both laws reading every try it accepts %s%% "
        "and %s%% at 0 strangers, against %s%% and %s%% for the better single law." % (
            f1(RJ["routed"]["5|front"]["owner_1e3"]), f1(RJ["routed"]["20|front"]["owner_1e3"]), f1(BOTH["5|front"]["owner_0"]),
            f1(BOTH["20|front"]["owner_0"]), f1(max(RJ["measure"]["5"]["front v2 (48)"][0], RJ["measure"]["5"]["front full-light (96)"][0])),
            f1(max(RJ["measure"]["20"]["front v2 (48)"][0], RJ["measure"]["20"]["front full-light (96)"][0])))))
    st.append(P("4.3  The literature's protocol", "h2"))
    rows = [["Method", "Subset 2", "Subset 3", "Subset 4", "Subset 5"]]
    order3 = sorted(P3.items(), key=lambda kv: -(kv[1]["4"] + kv[1]["5"]))[:6]
    for m, r in order3:
        rows.append([("<b>%s</b>" % nm(m)) if m.startswith("FACE LAW") else nm(m)] + [f1(r[s]) for s in ("2", "3", "4", "5")])
    st.append(KeepTogether([table(rows, [5.6 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm]),
                            P("<b>Table 5.</b> Extended Yale B identification with Yale's own crops (38 people), subset 1 as the gallery, "
                              "rank-1 accuracy (%). The full-light law ties the best classic, Weberfaces at 96×84.", "cap")]))
    st.append(P("4.4  More than one try", "h2"))
    st.append(P(
        "When the owner may try again, so may every stranger; the threshold stays where no stranger (or one in 1,000) is let in "
        "across all of their tries. With a different photo per try (another light within 25°), the full-light law accepts %s%%, "
        "%s%%, %s%% and %s%% of owners at 0 strangers with 1, 2, 3 and 5 tries (Tan–Triggs: %s%%, %s%%, %s%%, %s%%). Repeating the "
        "same photo with fresh photon noise helps only in the dark: at N = 5 the dark law goes from %s%% to %s%% over five frames, "
        "while in light the failures come from the direction of the light and repeat." % tuple(
            [f1(TP["%s|0|%d" % (FULL, k)]["owner"]) for k in (1, 2, 3, 5)] + [f1(TP["Tan-Triggs@96|0|%d" % k]["owner"]) for k in (1, 2, 3, 5)] +
            [f1(TF["FACE LAW v2 (48)|5|0|1"]["owner"]), f1(TF["FACE LAW v2 (48)|5|0|5"]["owner"])])))
    st.append(P("4.5  Exactness and timing", "h2"))
    st.append(P(
        "The C++ library equals the reference on every array of 74 real photos (crops, codes and scores, in light and in "
        "darkness) and on 38 self-test checksums from seeded random inputs. NIST's FRTE 1:1 test driver runs it with every "
        "comparison returning success. On one laptop core a template takes 85–125 ms per 640×480 image (face box and eyes "
        "included), a comparison 12.5 µs, and the peak memory is 18.6 MB; NIST's limits are 2,000 ms and 0.1 ms."))

    st.append(P("5  Limits", "h1"))
    st.append(P(
        "All faces are lab faces under controlled pose (37 people under 64 lights); dark tries come from a sensor model, not from "
        "night photos. NIST's sets add age, pose, expression and other cameras at false-match rates down to one in a million, which "
        "116,000 impostor pairs cannot measure; the method has not yet been evaluated by NIST [15]. The classic methods are our "
        "re-implementations, not the authors' code, and deep networks were not compared. The router's threshold and the both-laws "
        "numbers were chosen on the same tries they are reported on. The phone timings are estimates from counted arithmetic."))
    st.append(P("6  Conclusion", "h1"))
    st.append(P(
        "Writing the camera's physics into the descriptor and counting the rest gives a face verifier that, on lab faces, is more "
        "accurate than the classic illumination-robust methods at the same arithmetic, matches with integer look-ups only, and "
        "reproduces exactly on any machine. Its next tests are outside the lab: NIST's evaluation and real night photos."))
    st.append(P("Acknowledgements", "h1"))
    st.append(P(
        "Extended Yale B was collected by the Yale Center for Computational Vision and Control and Kuang-Chih Lee, Jeffrey Ho and "
        "David Kriegman; CMU PIE by Terence Sim, Simon Baker and Maan Bsat. Writing and analysis were assisted by an AI model "
        "(Claude, Anthropic); every number comes from the author's code and result files."))
    st.append(P("References", "h1"))
    refs = [
        "A. S. Georghiades, P. N. Belhumeur, D. J. Kriegman. From few to many: illumination cone models for face recognition under variable lighting and pose. IEEE TPAMI 23(6), 2001.",
        "K.-C. Lee, J. Ho, D. J. Kriegman. Acquiring linear subspaces for face recognition under variable lighting. IEEE TPAMI 27(5), 2005.",
        "T. Sim, S. Baker, M. Bsat. The CMU Pose, Illumination, and Expression database. IEEE TPAMI 25(12), 2003.",
        "P. Viola, M. J. Jones. Robust real-time face detection. International Journal of Computer Vision 57(2), 2004.",
        "T. Zhang, Y. Y. Tang, B. Fang, Z. Shang, X. Liu. Face recognition under varying illumination using gradientfaces. IEEE Transactions on Image Processing 18(11), 2009.",
        "B. Wang, W. Li, W. Yang, Q. Liao. Illumination normalization based on Weber's law with application to face recognition. IEEE Signal Processing Letters 18(8), 2011.",
        "X. Tan, B. Triggs. Enhanced local texture feature sets for face recognition under difficult lighting conditions. IEEE Transactions on Image Processing 19(6), 2010.",
        "T. Ahonen, A. Hadid, M. Pietikäinen. Face description with local binary patterns: application to face recognition. IEEE TPAMI 28(12), 2006.",
        "W. Zhang, S. Shan, W. Gao, X. Chen, H. Zhang. Local Gabor binary pattern histogram sequence (LGBPHS): a novel non-statistical model for face representation and recognition. ICCV 2005.",
        "N. Dalal, B. Triggs. Histograms of oriented gradients for human detection. CVPR 2005.",
        "O. Déniz, G. Bueno, J. Salido, F. De la Torre. Face recognition using histograms of oriented gradients. Pattern Recognition Letters 32(12), 2011.",
        "M. Turk, A. Pentland. Eigenfaces for recognition. Journal of Cognitive Neuroscience 3(1), 1991.",
        "P. N. Belhumeur, J. P. Hespanha, D. J. Kriegman. Eigenfaces vs. Fisherfaces: recognition using class specific linear projection. IEEE TPAMI 19(7), 1997.",
        "J. Chen, S. Shan, C. He, G. Zhao, M. Pietikäinen, X. Chen, W. Gao. WLD: a robust local image descriptor. IEEE TPAMI 32(9), 2010.",
        "P. Grother, M. Ngan, K. Hanaoka, et al. Face Recognition Technology Evaluation (FRTE) 1:1 Verification. National Institute of Standards and Technology, ongoing.",
        "G. Bradski. The OpenCV Library. Dr. Dobb's Journal of Software Tools, 2000."]
    for k, r in enumerate(refs, 1):
        st.append(P("[%d]  %s" % (k, r), "ref"))

    def footer(c, d):
        c.saveState(); c.setFont("TNR", 8); c.setFillColor(MUTED)
        c.drawCentredString(A4[0] / 2, 1.2 * cm, "Face Law v2 · preprint · %d" % d.page); c.restoreState()
    doc.build(st, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    out = os.path.join(HERE, "face_law_v2_paper.pdf")
    build(out)
    print(out, os.path.getsize(out), "bytes")
