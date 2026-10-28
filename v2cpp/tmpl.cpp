// FACE LAW v2 templates: one format for every role (-1 no face box, -2 no eyes, -3 eyes below the GATE: no template).
// 8-byte header "FL2T", version 2, flags; the FULL-LIGHT law (v2_full.py): the map (10,584 Weber bins of the 96 x 84
// crop) and the probe codes (bin | significance << 3, one placement).  Matching: the score of one template's probe
// codes against the other's map with the synthesis's FW96; SYMMETRIC = the mean of both directions.  21,176 bytes.
#include "facelaw2.h"
#include "tmpl.h"
#include <cstring>
#include <limits>

namespace fl2 {

static const uint8_t MAGIC[4] = {'F', 'L', '2', 'T'};
static const int HDR = 8, BODY = 2 * NPTS96, SIZE = HDR + BODY, VERSION = 2;

static const int SIZE_FAILED = HDR;

void failed_template(std::vector<uint8_t> &out) {           // "template creation failed", encoded: header, flags 0
    out.assign(SIZE_FAILED, 0);
    std::memcpy(out.data(), MAGIC, 4);
    out[4] = VERSION; out[5] = 0;
}

int template_from_box(const Mat &lin, const Mat &disp, const Box &box, std::vector<uint8_t> &out, Eyes &eyes) {
    out.clear();
    reset_state();
    eyes = find_eyes(disp, box);
    if (!eyes.ok) return -2;
    if (!eyes_pass_gate(eyes)) return -3;                      // no eyes here: the face box was not a face
    Mat c = crop96(lin, eyes);
    out.assign(SIZE, 0);
    std::memcpy(out.data(), MAGIC, 4);
    out[4] = VERSION; out[5] = 3;
    uint8_t *map = out.data() + HDR;
    map_bins96(c, map);
    static uint8_t bins[NPTS96], sig[NPTS96];
    probe_codes96(c, bins, sig);
    uint8_t *pc = map + NPTS96;
    for (int i = 0; i < NPTS96; i++) pc[i] = (uint8_t)(bins[i] | (sig[i] << 3));
    return 0;
}

int make_template(const uint8_t *data, int w, int h, int channels, std::vector<uint8_t> &out, Eyes &eyes, Box &box) {
    out.clear();
    reset_state();
    eyes = Eyes{0, 0, 0, 0, false};
    Mat lin = linear_of(data, w, h, channels);
    box = face_box(lin);
    if (!box.ok) return -1;
    Mat disp = display_u8(lin);
    return template_from_box(lin, disp, box, out, eyes);
}

int make_templates_all(const uint8_t *data, int w, int h, int channels, std::vector<std::vector<uint8_t>> &outs, std::vector<Eyes> &eyes) {
    outs.clear(); eyes.clear();
    reset_state();
    Mat lin = linear_of(data, w, h, channels);
    std::vector<Box> boxes = face_boxes(lin);
    if (boxes.empty()) return -1;
    Mat disp = display_u8(lin);
    for (const Box &b : boxes) {
        std::vector<uint8_t> t; Eyes e;
        if (template_from_box(lin, disp, b, t, e) == 0) { outs.push_back(t); eyes.push_back(e); }
    }
    return outs.empty() ? -3 : 0;
}

bool valid_template(const std::vector<uint8_t> &t) {
    return t.size() == (size_t)SIZE && std::memcmp(t.data(), MAGIC, 4) == 0 && t[4] == VERSION && t[5] == 3;
}

static double one_way(const std::vector<uint8_t> &probe_t, const std::vector<uint8_t> &map_t) {
    static uint8_t bins[NPTS96], sig[NPTS96];
    const uint8_t *pc = probe_t.data() + HDR + NPTS96;
    for (int i = 0; i < NPTS96; i++) { uint8_t v = pc[i]; bins[i] = v & 7; sig[i] = v >> 3; }
    return score96(bins, sig, map_t.data() + HDR);
}

double match_templates(const std::vector<uint8_t> &verif, const std::vector<uint8_t> &enrol, bool symmetric, bool &ok) {
    ok = valid_template(verif) && valid_template(enrol);
    if (!ok) return -std::numeric_limits<double>::infinity();
    double a = one_way(verif, enrol);
    return symmetric ? (a + one_way(enrol, verif)) / 2.0 : a;
}

}  // namespace fl2
