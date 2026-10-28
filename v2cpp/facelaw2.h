// FACE LAW v2 -- the exact C++ port of v2ref.py (bit for bit), plus a Viola-Jones face box.  No dependencies.
// Build with -O2 -ffp-contract=off -fwrapv (no fused multiply-add: every product and sum rounds as in Python).
#pragma once
#include <cstdint>
#include <vector>

namespace fl2 {

struct Mat {
    int h = 0, w = 0;
    std::vector<double> a;
    Mat() {}
    Mat(int h_, int w_, double v = 0.0) : h(h_), w(w_), a((size_t)h_ * w_, v) {}
    double &operator()(int y, int x) { return a[(size_t)y * w + x]; }
    double operator()(int y, int x) const { return a[(size_t)y * w + x]; }
};

struct Eyes { double lx, ly, rx, ry; bool ok; double ev = 0.0; };   // ev: the weaker eye's eye-law score
struct Box { double x, y, w, h; bool ok; };

static const int NPTS = 2652;                           // the Weber codes' points (3 scales)
static const int NPTS96 = 10584;                        // the full-light law's points: the same codes on the 96 x 84 frame

// image
Mat linear_of(const uint8_t *data, int w, int h, int channels);  // 8-bit sRGB (3) or grey (1) -> linear light
Mat display_u8(const Mat &lin);                                   // -> display bytes (as doubles)
Mat gauss(const Mat &x, int k8);
Mat warp(const Mat &img, double ox, double oy, double ux, double uy, double s, const Mat &U, const Mat &V);
void reset_state();                                               // drops the smoothing cache (called at every entry)

// the face box (Viola-Jones, OpenCV's frontal-face cascade), the eye law, the crop
Box face_box(const Mat &lin);                                    // the largest face box
std::vector<Box> face_boxes(const Mat &lin);                     // every face box, largest first
std::vector<Box> vj_detect(const uint8_t *gray, int w, int h, double scale_factor, int min_neighbors, int min_size);
std::vector<int> eye_codes(const Mat &F);
std::vector<int64_t> score_map(const std::vector<int> &C, int h, int w);
Eyes find_eyes(const Mat &disp, const Box &box);
Mat crop(const Mat &img, const Eyes &e);
bool eyes_pass_gate(const Eyes &e);                           // ev >= GATE (the synthesis's lane)

// the Weber codes and the score
void map_bins(const Mat &c, uint8_t out[NPTS]);
void probe_codes(const Mat &x, uint8_t bins[NPTS], uint8_t sig[NPTS]);
void probe_all(const Mat &x, uint8_t bins[9][NPTS], uint8_t sig[9][NPTS]);
double score(const uint8_t bins[9][NPTS], const uint8_t sig[9][NPTS], const uint8_t map[NPTS]);

// the full-light law (v2_full.py): the same face at twice the sampling (eye distance 50 px), its own ALPHA96 and the
// synthesis's FW96, one placement (no search)
Mat crop96(const Mat &img, const Eyes &e);
void map_bins96(const Mat &c, uint8_t out[NPTS96]);
void probe_codes96(const Mat &x, uint8_t bins[NPTS96], uint8_t sig[NPTS96]);
double score96(const uint8_t bins[NPTS96], const uint8_t sig[NPTS96], const uint8_t map[NPTS96]);

// the phone app (wasm_api.cpp): the dark law at one placement, the router's photons per pixel (router.py nhat), and the
// measured noise variance per pixel (v2's Law 1)
double score1(const uint8_t bins[NPTS], const uint8_t sig[NPTS], const uint8_t map[NPTS]);
double photons_per_pixel(const Mat &c48);
std::vector<double> noise_var(const Mat &x);

}  // namespace fl2
