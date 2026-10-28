// A plain C interface to the v2 law, for the exactness tests (Python ctypes) and the face lock.
#include "facelaw2.h"
#include <cstring>

using namespace fl2;

static Mat mat_of(const double *a, int h, int w) {
    Mat m(h, w);
    std::memcpy(m.a.data(), a, sizeof(double) * (size_t)h * w);
    return m;
}

extern "C" {

void fl2_linear_of(const uint8_t *data, int w, int h, int ch, double *out) {
    Mat m = linear_of(data, w, h, ch);
    std::memcpy(out, m.a.data(), sizeof(double) * m.a.size());
}

void fl2_gauss(const double *in, int h, int w, int k8, double *out) {
    Mat m = gauss(mat_of(in, h, w), k8);
    std::memcpy(out, m.a.data(), sizeof(double) * m.a.size());
}

void fl2_u8(const double *in, int h, int w, double *out) {
    Mat m = display_u8(mat_of(in, h, w));
    std::memcpy(out, m.a.data(), sizeof(double) * m.a.size());
}

void fl2_eye_codes(const double *F, int h, int w, int *out) {
    std::vector<int> c = eye_codes(mat_of(F, h, w));
    std::memcpy(out, c.data(), sizeof(int) * c.size());
}

int fl2_find_eyes(const double *disp, int h, int w, const double *box, double *out) {
    Eyes e = find_eyes(mat_of(disp, h, w), Box{box[0], box[1], box[2], box[3], true});
    out[0] = e.lx; out[1] = e.ly; out[2] = e.rx; out[3] = e.ry; out[4] = e.ev;
    return e.ok ? 1 : 0;
}

void fl2_crop(const double *img, int h, int w, const double *eyes, double *out) {
    Mat c = crop(mat_of(img, h, w), Eyes{eyes[0], eyes[1], eyes[2], eyes[3], true});
    std::memcpy(out, c.a.data(), sizeof(double) * c.a.size());
}

void fl2_map_bins(const double *c, uint8_t *out) { map_bins(mat_of(c, 48, 42), out); }

void fl2_probe_all(const double *c, uint8_t *bins, uint8_t *sig) {
    probe_all(mat_of(c, 48, 42), reinterpret_cast<uint8_t(*)[NPTS]>(bins), reinterpret_cast<uint8_t(*)[NPTS]>(sig));
}

double fl2_score(const uint8_t *bins, const uint8_t *sig, const uint8_t *map) {
    return score(reinterpret_cast<const uint8_t(*)[NPTS]>(bins), reinterpret_cast<const uint8_t(*)[NPTS]>(sig), map);
}

void fl2_crop96(const double *img, int h, int w, const double *eyes, double *out) {
    Mat c = crop96(mat_of(img, h, w), Eyes{eyes[0], eyes[1], eyes[2], eyes[3], true});
    std::memcpy(out, c.a.data(), sizeof(double) * c.a.size());
}

void fl2_map_bins96(const double *c, uint8_t *out) { map_bins96(mat_of(c, 96, 84), out); }

void fl2_probe_codes96(const double *c, uint8_t *bins, uint8_t *sig) { probe_codes96(mat_of(c, 96, 84), bins, sig); }

double fl2_score96(const uint8_t *bins, const uint8_t *sig, const uint8_t *map) { return score96(bins, sig, map); }

int fl2_face_box(const double *lin, int h, int w, double *out) {
    Box b = face_box(mat_of(lin, h, w));
    out[0] = b.x; out[1] = b.y; out[2] = b.w; out[3] = b.h;
    return b.ok ? 1 : 0;
}

int fl2_vj(const uint8_t *gray, int h, int w, double *out, int maxn) {
    std::vector<Box> v = vj_detect(gray, w, h, 1.1, 4, 24);
    int n = 0;
    for (const Box &b : v) {
        if (n >= maxn) break;
        out[4 * n] = b.x; out[4 * n + 1] = b.y; out[4 * n + 2] = b.w; out[4 * n + 3] = b.h; n++;
    }
    return (int)v.size();
}

// the whole template from an 8-bit image: -1 no face, -2 no eyes, 0 ok; eyes, the map, the probe codes
int fl2_template(const uint8_t *data, int w, int h, int ch, double *eyes_out, double *box_out, uint8_t *map, uint8_t *bins, uint8_t *sig) {
    Mat lin = linear_of(data, w, h, ch);
    Box b = face_box(lin);
    box_out[0] = b.x; box_out[1] = b.y; box_out[2] = b.w; box_out[3] = b.h;
    if (!b.ok) return -1;
    Mat disp = display_u8(lin);
    Eyes e = find_eyes(disp, b);
    if (!e.ok) return -2;
    eyes_out[0] = e.lx; eyes_out[1] = e.ly; eyes_out[2] = e.rx; eyes_out[3] = e.ry;
    Mat c = crop(lin, e);
    map_bins(c, map);
    probe_all(c, reinterpret_cast<uint8_t(*)[NPTS]>(bins), reinterpret_cast<uint8_t(*)[NPTS]>(sig));
    return 0;
}

}  // extern "C"

#include "tmpl.h"
extern "C" {
// a whole template (21,176 bytes: the full-light law) from an 8-bit image: 0 ok, -1 no face, -2 no eyes
int fl2_make_template(const uint8_t *data, int w, int h, int ch, uint8_t *out, double *eyes, double *box) {
    std::vector<uint8_t> t; Eyes e; Box b;
    int r = make_template(data, w, h, ch, t, e, b);
    eyes[0] = e.lx; eyes[1] = e.ly; eyes[2] = e.rx; eyes[3] = e.ry; box[0] = b.x; box[1] = b.y; box[2] = b.w; box[3] = b.h;
    if (r == 0) std::memcpy(out, t.data(), t.size());
    return r;
}
double fl2_match(const uint8_t *verif, const uint8_t *enrol, int n, int symmetric) {
    std::vector<uint8_t> a(verif, verif + n), b(enrol, enrol + n);
    bool ok;
    return match_templates(a, b, symmetric != 0, ok);
}
}
