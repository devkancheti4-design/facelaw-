// Viola-Jones face detection with OpenCV's frontal-face cascade (haarcascade_frontalface_default.xml, as arrays in
// cascade_ff.inc), following OpenCV 4's CascadeClassifier::detectMultiScale for stump cascades: an image pyramid
// (factor 1.1), windows every 2 px below scale 2 and every 1 px above, the variance-normalised window, stage sums of
// stumps, a skipped position after a first-stage reject, then groupRectangles (eps 0.2, more than min_neighbors).
// The pyramid uses its own bilinear resize, so the boxes agree with OpenCV's closely, not bit for bit.
#include "facelaw2.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

namespace fl2 {
#include "cascade_ff.inc"

namespace {
struct R { int x, y, w, h; };

void resize_bilinear(const uint8_t *src, int W, int H, std::vector<uint8_t> &dst, int w, int h) {
    dst.resize((size_t)w * h);
    double sx = (double)W / w, sy = (double)H / h;
    for (int y = 0; y < h; y++) {
        double fy = (y + 0.5) * sy - 0.5;
        if (fy < 0) fy = 0;
        int y0 = (int)fy; double ay = fy - y0; int y1 = std::min(y0 + 1, H - 1); y0 = std::min(y0, H - 1);
        for (int x = 0; x < w; x++) {
            double fx = (x + 0.5) * sx - 0.5;
            if (fx < 0) fx = 0;
            int x0 = (int)fx; double ax = fx - x0; int x1 = std::min(x0 + 1, W - 1); x0 = std::min(x0, W - 1);
            double v = (1 - ay) * ((1 - ax) * src[(size_t)y0 * W + x0] + ax * src[(size_t)y0 * W + x1]) +
                       ay * ((1 - ax) * src[(size_t)y1 * W + x0] + ax * src[(size_t)y1 * W + x1]);
            int iv = (int)std::lrint(v);
            dst[(size_t)y * w + x] = (uint8_t)std::min(255, std::max(0, iv));
        }
    }
}

inline long long rsum(const std::vector<long long> &I, int stride, int x, int y, int w, int h) {
    return I[(size_t)(y + h) * stride + (x + w)] - I[(size_t)y * stride + (x + w)] - I[(size_t)(y + h) * stride + x] + I[(size_t)y * stride + x];
}

int run_at(const std::vector<long long> &S, const std::vector<long long> &Q, int stride, int x, int y) {
    const int nx = 1, ny = 1, nw = VJ_W - 2, nh = VJ_H - 2;
    long long vs = rsum(S, stride, x + nx, y + ny, nw, nh), vq = rsum(Q, stride, x + nx, y + ny, nw, nh);
    double nf = (double)(nw * nh) * (double)vq - (double)vs * (double)vs;
    nf = nf > 0. ? std::sqrt(nf) : 1.;
    double vnf = 1. / nf;
    for (int si = 0; si < VJ_NSTAGES; si++) {
        double tmp = 0;
        for (int k = VJ_STAGE_FIRST[si]; k < VJ_STAGE_FIRST[si] + VJ_STAGE_N[si]; k++) {
            int fi = VJ_WEAK_FEAT[k];
            const int *r = &VJ_RECT[fi * 12];
            const float *wt = &VJ_RW[fi * 3];
            float ret = wt[0] * (float)rsum(S, stride, x + r[0], y + r[1], r[2], r[3]) + wt[1] * (float)rsum(S, stride, x + r[4], y + r[5], r[6], r[7]);
            if (wt[2] != 0.0f) ret += wt[2] * (float)rsum(S, stride, x + r[8], y + r[9], r[10], r[11]);
            double val = (double)ret * vnf;
            tmp += val < VJ_WEAK_THR[k] ? VJ_WEAK_L[k] : VJ_WEAK_R[k];
        }
        if (tmp < VJ_STAGE_THR[si]) return -si;
    }
    return 1;
}

bool similar(const R &a, const R &b, double eps) {
    double delta = eps * (std::min(a.w, b.w) + std::min(a.h, b.h)) * 0.5;
    return std::abs(a.x - b.x) <= delta && std::abs(a.y - b.y) <= delta && std::abs(a.x + a.w - b.x - b.w) <= delta &&
           std::abs(a.y + a.h - b.y - b.h) <= delta;
}

std::vector<R> group(const std::vector<R> &rl, int thr, double eps) {
    int n = (int)rl.size();
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int i) { while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (similar(rl[i], rl[j], eps)) { int a = find(i), b = find(j); if (a != b) parent[b] = a; }
    std::vector<int> label(n, -1), root_label(n, -1);
    int nc = 0;
    for (int i = 0; i < n; i++) { int r = find(i); if (root_label[r] < 0) root_label[r] = nc++; label[i] = root_label[r]; }
    std::vector<long long> sx(nc, 0), sy(nc, 0), sw(nc, 0), sh(nc, 0);
    std::vector<int> cnt(nc, 0);
    for (int i = 0; i < n; i++) { int c = label[i]; sx[c] += rl[i].x; sy[c] += rl[i].y; sw[c] += rl[i].w; sh[c] += rl[i].h; cnt[c]++; }
    std::vector<R> rr(nc);
    for (int c = 0; c < nc; c++) {
        float s = 1.f / cnt[c];
        rr[c] = R{(int)std::lrint(sx[c] * s), (int)std::lrint(sy[c] * s), (int)std::lrint(sw[c] * s), (int)std::lrint(sh[c] * s)};
    }
    std::vector<R> out;
    for (int i = 0; i < nc; i++) {
        if (cnt[i] <= thr) continue;
        int j;
        for (j = 0; j < nc; j++) {
            if (j == i || cnt[j] <= thr) continue;
            int dx = (int)std::lrint(rr[j].w * eps), dy = (int)std::lrint(rr[j].h * eps);
            if (rr[i].x >= rr[j].x - dx && rr[i].y >= rr[j].y - dy && rr[i].x + rr[i].w <= rr[j].x + rr[j].w + dx &&
                rr[i].y + rr[i].h <= rr[j].y + rr[j].h + dy && (cnt[j] > std::max(3, cnt[i]) || cnt[i] < 3))
                break;
        }
        if (j == nc) out.push_back(rr[i]);
    }
    return out;
}
}  // namespace

std::vector<Box> vj_detect(const uint8_t *gray, int W, int H, double sf, int min_neighbors, int min_size) {
    std::vector<R> found;
    std::vector<uint8_t> img;
    std::vector<long long> S, Q;
    for (double factor = 1.0;; factor *= sf) {
        int ww = (int)std::lrint(VJ_W * factor), wh = (int)std::lrint(VJ_H * factor);
        int sw = (int)std::lrint(W / factor), sh = (int)std::lrint(H / factor);
        if (sw < VJ_W || sh < VJ_H) break;
        if (ww < min_size || wh < min_size) continue;
        resize_bilinear(gray, W, H, img, sw, sh);
        int st = sw + 1;
        S.assign((size_t)(sh + 1) * st, 0); Q.assign((size_t)(sh + 1) * st, 0);
        for (int y = 0; y < sh; y++) {
            long long rs = 0, rq = 0;
            for (int x = 0; x < sw; x++) {
                long long v = img[(size_t)y * sw + x];
                rs += v; rq += v * v;
                S[(size_t)(y + 1) * st + x + 1] = S[(size_t)y * st + x + 1] + rs;
                Q[(size_t)(y + 1) * st + x + 1] = Q[(size_t)y * st + x + 1] + rq;
            }
        }
        int step = factor >= 2 ? 1 : 2;
        int xmax = sw + 1 - VJ_W, ymax = sh + 1 - VJ_H;
        for (int y = 0; y < ymax; y += step)
            for (int x = 0; x < xmax; x += step) {
                int r = run_at(S, Q, st, x, y);
                if (r > 0) found.push_back(R{(int)std::lrint(x * factor), (int)std::lrint(y * factor), ww, wh});
                if (r == 0) x += step;
            }
    }
    std::vector<Box> out;
    for (const R &r : group(found, min_neighbors, 0.2)) out.push_back(Box{(double)r.x, (double)r.y, (double)r.w, (double)r.h, true});
    return out;
}

}  // namespace fl2
