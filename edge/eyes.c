/* DARK FACE edge build: the EYES (LAWS 1-2) and the pyramid (LAW 3) in C, mirroring darkface_law.py step for step.
   The sector lanes are the synthesis's own expressions (lanes_sector.json, lane_cast.json), proven by it on all 2^24 inputs.
   Build: cc -O2 -ffp-contract=off -fwrapv -shared -fPIC -o libeyes.so eyes.c */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define T 16
#define PAD 8
#define K_SIG 2.5

static inline int32_t lane_O_sy(int32_t x){ return (1 - (x >> 23)); }
static inline int32_t lane_O_sx(int32_t x){ return (1 - (1 & (x >> 11))); }
static inline int32_t lane_O_ax(int32_t x){ return (((((x) & 4095) - 2048) >> 31) ^ ((((x) & 4095) - 2048) + ((((x) & 4095) - 2048) >> 31))); }
static inline int32_t lane_O_ay(int32_t x){ return ((2048 ^ ((2048 - (x >> 12)) >> 12)) - ((x ^ (2048 - (x >> 12))) >> 12)); }
static inline int32_t lane_O_perm(int32_t x){ return ((x >> 2) ^ (x ^ (x >> 1))); }
static inline int32_t lane_cast(int32_t x){ return (((4095 - x) >> 12) - (((x >> 12) - x) >> 12)); }

/* scipy.ndimage gaussian kernel: radius int(4 sigma + 0.5), exp(-x^2 / (2 sigma^2)), normalised */
static int gkernel(double sigma, double *w){
    int r = (int)(4.0 * sigma + 0.5); double s = 0;
    for (int i = -r; i <= r; i++){ w[i + r] = exp(-0.5 / (sigma * sigma) * (double)(i * i)); s += w[i + r]; }
    for (int i = 0; i <= 2 * r; i++) w[i] /= s;
    return r;
}
static inline int refl(int i, int n){            /* scipy 'reflect': d c b a | a b c d | d c b a */
    while (i < 0 || i >= n){ if (i < 0) i = -i - 1; if (i >= n) i = 2 * n - i - 1; }
    return i;
}
/* separable gaussian, axis 0 then axis 1 (scipy's order); out may not alias in.  Row-wise passes, reflection only at
   the borders; every element keeps the same sum in the same order (j = -r .. r), so the values are unchanged. */
static void gauss2(const double *in, double *out, int H, int W, double sigma){
    double w[256]; int r = gkernel(sigma, w);
    double *tmp = malloc(sizeof(double) * H * W);
    for (int y = 0; y < H; y++){
        double *acc = tmp + (size_t)y * W;
        for (int x = 0; x < W; x++) acc[x] = 0;
        for (int j = -r; j <= r; j++){
            const double *src = in + (size_t)refl(y + j, H) * W; double wj = w[j + r];
            for (int x = 0; x < W; x++) acc[x] += wj * src[x];
        }
    }
    for (int y = 0; y < H; y++){
        const double *row = tmp + (size_t)y * W; double *o = out + (size_t)y * W;
        int lo = r < W ? r : W, hi = W - r > lo ? W - r : lo;
        for (int x = 0; x < lo; x++){ double a = 0; for (int j = -r; j <= r; j++) a += w[j + r] * row[refl(x + j, W)]; o[x] = a; }
        for (int x = lo; x < hi; x++){ double a = 0; const double *p = row + x - r; for (int j = 0; j <= 2 * r; j++) a += w[j] * p[j]; o[x] = a; }
        for (int x = hi; x < W; x++){ double a = 0; for (int j = -r; j <= r; j++) a += w[j + r] * row[refl(x + j, W)]; o[x] = a; }
    }
    free(tmp);
}
/* scipy.ndimage.zoom, order 1, output round(n * z), grid (n_in - 1) / (n_out - 1) */
static void zoom1(const double *in, int H, int W, double z, double *out, int *Ho, int *Wo){
    int h = (int)rint(H * z), w = (int)rint(W * z);       /* python round(): half to even (rint, default mode) */
    double zy = h > 1 ? (double)(H - 1) / (h - 1) : 0, zx = w > 1 ? (double)(W - 1) / (w - 1) : 0;
    for (int i = 0; i < h; i++){
        double cy = i * zy; int y0 = (int)floor(cy); double fy = cy - y0; int y1 = y0 + 1 < H ? y0 + 1 : H - 1;
        for (int j = 0; j < w; j++){
            double cx = j * zx; int x0 = (int)floor(cx); double fx = cx - x0; int x1 = x0 + 1 < W ? x0 + 1 : W - 1;
            /* scipy 'constant' mode: a sample mapped past the last input pixel (even by one ulp) is outside: cval 0 */
            if (cy > H - 1 || cx > W - 1){ out[i * w + j] = 0; continue; }
            out[i * w + j] = in[y0 * W + x0] * (1 - fy) * (1 - fx) + in[y0 * W + x1] * (1 - fy) * fx
                           + in[y1 * W + x0] * fy * (1 - fx) + in[y1 * W + x1] * fy * fx;
        }
    }
    *Ho = h; *Wo = w;
}
/* the k-th smallest (Hoare selection): the same value a full sort would put at index k */
static double select_k(double *a, int n, int k){
    int lo = 0, hi = n - 1;
    while (lo < hi){
        double p = a[lo + (hi - lo) / 2]; int i = lo, j = hi;
        while (i <= j){
            while (a[i] < p) i++;
            while (a[j] > p) j--;
            if (i <= j){ double t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (k <= j) hi = j; else if (k >= i) lo = i; else return a[k];
    }
    return a[k];
}
static double median(const double *v, int n){            /* numpy median: middle value, or the mean of the two middles */
    double *c = malloc(sizeof(double) * n); memcpy(c, v, sizeof(double) * n);
    double m;
    if (n & 1) m = select_k(c, n, n / 2);
    else {
        double hi = select_k(c, n, n / 2), lo = c[0];
        for (int i = 1; i < n / 2; i++) if (c[i] > lo) lo = c[i];   /* after selection, c[0 .. n/2-1] <= c[n/2] */
        m = (lo + hi) / 2;
    }
    free(c); return m;
}
static void grad(const double *G, int H, int W, double *gx, double *gy){   /* np.gradient: centre, one-sided at edges */
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++){
        int i = y * W + x;
        gx[i] = W == 1 ? 0 : x == 0 ? G[i + 1] - G[i] : x == W - 1 ? G[i] - G[i - 1] : (G[i + 1] - G[i - 1]) / 2;
        gy[i] = H == 1 ? 0 : y == 0 ? G[i + W] - G[i] : y == H - 1 ? G[i] - G[i - W] : (G[i + W] - G[i - W]) / 2;
    }
}

/* linear-light luminance of sRGB bytes (darkface_law.to_gray) */
void eyes_gray(const uint8_t *rgb, int n, double *out){
    for (int i = 0; i < n; i++){
        double s = 0;
        for (int c = 0; c < 3; c++){ double a = rgb[3 * i + c] / 255.0; s += a <= 0.04045 ? a / 12.92 : pow((a + 0.055) / 1.055, 2.4); }
        out[i] = 255 * (s / 3);
    }
}
/* one pyramid level: resample, then LAWS 1-2 -> padded direction codes cd and centre-surround codes cg.
   Returns the padded size via Hp, Wp; cd / cg must hold (H / s + 2 PAD + 1)^2-ish (caller sizes generously). */
void eyes_level(const double *gray, int H, int W, double s, double GE, double DE, double QSIG,
                int8_t *cd, int8_t *cg, int *Hp, int *Wp, double *noise){
    int h = H, w = W; double *x;
    if (s > 1){
        double *g2 = malloc(sizeof(double) * H * W); gauss2(gray, g2, H, W, 0.5 * sqrt(s * s - 1));
        x = malloc(sizeof(double) * ((size_t)(H / s + 2) * (size_t)(W / s + 2))); zoom1(g2, H, W, 1 / s, x, &h, &w); free(g2);
    } else if (s < 1){
        x = malloc(sizeof(double) * ((size_t)(H / s + 2) * (size_t)(W / s + 2))); zoom1(gray, H, W, 1 / s, x, &h, &w);
    } else { x = malloc(sizeof(double) * H * W); memcpy(x, gray, sizeof(double) * H * W); }
    int n = h * w;
    double *G = malloc(sizeof(double) * n), *gx = malloc(sizeof(double) * n), *gy = malloc(sizeof(double) * n), *m = malloc(sizeof(double) * n);
    gauss2(x, G, h, w, 0.75); grad(G, h, w, gx, gy);
    for (int i = 0; i < n; i++) m[i] = hypot(gx[i], gy[i]);
    double sc = median(m, n) / 1.1774, fl = QSIG * GE / (s > 1 ? s : 1); if (fl > sc) sc = fl;
    /* sectors(x * (2 / sc)): the authored lanes over the packed, quantised gradient */
    double k = 2.0 / sc;
    double *I = malloc(sizeof(double) * n); for (int i = 0; i < n; i++) I[i] = x[i] * k;
    gauss2(I, G, h, w, 0.75); grad(G, h, w, gx, gy);
    int Wp_ = w + 2 * PAD, Hp_ = h + 2 * PAD;
    memset(cd, 0, (size_t)Hp_ * Wp_); memset(cg, 0, (size_t)Hp_ * Wp_);
    for (int y = 0; y < h; y++) for (int xx = 0; xx < w; xx++){
        int i = y * w + xx;
        double qx = rint(16 * gx[i]), qy = rint(16 * -gy[i]);
        if (qx < -2048) qx = -2048; if (qx > 2047) qx = 2047; if (qy < -2048) qy = -2048; if (qy > 2047) qy = 2047;
        int32_t v = ((int32_t)qx + 2048) | (((int32_t)qy + 2048) << 12);
        int32_t sy = lane_O_sy(v), sx = lane_O_sx(v), ax = lane_O_ax(v), ay = lane_O_ay(v);
        int32_t d = lane_cast(ay | (ax << 12));
        int32_t sec = lane_O_perm(4 * sy + 2 * sx + d);
        cd[(y + PAD) * Wp_ + xx + PAD] = m[i] >= K_SIG * sc ? (int8_t)(1 + sec) : 0;
    }
    /* centre-surround: G1 - G2.5 against its own MAD noise */
    double *G1 = malloc(sizeof(double) * n), *G25 = malloc(sizeof(double) * n);
    gauss2(x, G1, h, w, 1.0); gauss2(x, G25, h, w, 2.5);
    for (int i = 0; i < n; i++){ G1[i] -= G25[i]; I[i] = fabs(G1[i]); }
    double sd = median(I, n) / 0.6745, fd = QSIG * DE / (s > 1 ? s : 1); if (fd > sd) sd = fd;
    for (int y = 0; y < h; y++) for (int xx = 0; xx < w; xx++){
        double dv = G1[y * w + xx];
        cg[(y + PAD) * Wp_ + xx + PAD] = dv >= K_SIG * sd ? 1 : (dv <= -K_SIG * sd ? 2 : 0);
    }
    *Hp = Hp_; *Wp = Wp_; noise[0] = sc; noise[1] = sd;
    free(x); free(G); free(gx); free(gy); free(m); free(I); free(G1); free(G25);
}

static inline int nrefl(int i, int n){           /* numpy.pad 'reflect': c b | a b c d | c b (edge not repeated) */
    if (n == 1) return 0;
    int p = 2 * (n - 1);
    i %= p; if (i < 0) i += p;
    return i < n ? i : p - i;
}
/* eyes_level plus the description's inputs: xp (pixels, numpy-reflect padded), mp (|grad| / noise, zero padded), spix */
void eyes_level2(const double *gray, int H, int W, double s, double GE, double DE, double QSIG,
                 int8_t *cd, int8_t *cg, double *xp, double *mp, int *Hp, int *Wp, double *noise){
    int h = H, w = W; double *x;
    if (s > 1){
        double *g2 = malloc(sizeof(double) * H * W); gauss2(gray, g2, H, W, 0.5 * sqrt(s * s - 1));
        x = malloc(sizeof(double) * ((size_t)(H / s + 2) * (size_t)(W / s + 2))); zoom1(g2, H, W, 1 / s, x, &h, &w); free(g2);
    } else if (s < 1){
        x = malloc(sizeof(double) * ((size_t)(H / s + 2) * (size_t)(W / s + 2))); zoom1(gray, H, W, 1 / s, x, &h, &w);
    } else { x = malloc(sizeof(double) * H * W); memcpy(x, gray, sizeof(double) * H * W); }
    int n = h * w;
    double *G = malloc(sizeof(double) * n), *gx = malloc(sizeof(double) * n), *gy = malloc(sizeof(double) * n), *m = malloc(sizeof(double) * n);
    gauss2(x, G, h, w, 0.75); grad(G, h, w, gx, gy);
    for (int i = 0; i < n; i++) m[i] = hypot(gx[i], gy[i]);
    double sc = median(m, n) / 1.1774, fl = QSIG * GE / (s > 1 ? s : 1); if (fl > sc) sc = fl;
    int Wp_ = w + 2 * PAD, Hp_ = h + 2 * PAD;
    for (int y = 0; y < Hp_; y++) for (int xx = 0; xx < Wp_; xx++){
        int yi = y - PAD, xi = xx - PAD;
        xp[(size_t)y * Wp_ + xx] = x[(size_t)nrefl(yi, h) * w + nrefl(xi, w)];
        mp[(size_t)y * Wp_ + xx] = (yi >= 0 && yi < h && xi >= 0 && xi < w) ? m[(size_t)yi * w + xi] / sc : 0.0;
    }
    double k = 2.0 / sc;
    double *I = malloc(sizeof(double) * n); for (int i = 0; i < n; i++) I[i] = x[i] * k;
    gauss2(I, G, h, w, 0.75); grad(G, h, w, gx, gy);
    memset(cd, 0, (size_t)Hp_ * Wp_); memset(cg, 0, (size_t)Hp_ * Wp_);
    for (int y = 0; y < h; y++) for (int xx = 0; xx < w; xx++){
        int i = y * w + xx;
        double qx = rint(16 * gx[i]), qy = rint(16 * -gy[i]);
        if (qx < -2048) qx = -2048; if (qx > 2047) qx = 2047; if (qy < -2048) qy = -2048; if (qy > 2047) qy = 2047;
        int32_t v = ((int32_t)qx + 2048) | (((int32_t)qy + 2048) << 12);
        int32_t sy = lane_O_sy(v), sx = lane_O_sx(v), ax = lane_O_ax(v), ay = lane_O_ay(v);
        int32_t d = lane_cast(ay | (ax << 12));
        int32_t sec = lane_O_perm(4 * sy + 2 * sx + d);
        cd[(y + PAD) * Wp_ + xx + PAD] = m[i] >= K_SIG * sc ? (int8_t)(1 + sec) : 0;
    }
    double *G1 = malloc(sizeof(double) * n), *G25 = malloc(sizeof(double) * n);
    gauss2(x, G1, h, w, 1.0); gauss2(x, G25, h, w, 2.5);
    for (int i = 0; i < n; i++){ G1[i] -= G25[i]; I[i] = fabs(G1[i]); }
    double sd = median(I, n) / 0.6745, fd = QSIG * DE / (s > 1 ? s : 1); if (fd > sd) sd = fd;
    for (int y = 0; y < h; y++) for (int xx = 0; xx < w; xx++){
        double dv = G1[y * w + xx];
        cg[(y + PAD) * Wp_ + xx + PAD] = dv >= K_SIG * sd ? 1 : (dv <= -K_SIG * sd ? 2 : 0);
    }
    *Hp = Hp_; *Wp = Wp_; noise[0] = sc; noise[1] = sd; noise[2] = sc / GE;
    free(x); free(G); free(gx); free(gy); free(m); free(I); free(G1); free(G25);
}

/* test hooks: the two resampling pieces, for comparison with scipy */
void eyes_gauss(const double *in, double *out, int H, int W, double sigma){ gauss2(in, out, H, W, sigma); }
void eyes_zoom(const double *in, int H, int W, double z, double *out, int *Ho, int *Wo){ zoom1(in, H, W, z, out, Ho, Wo); }
