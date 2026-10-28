/* DARK FACE edge build: LAW 7's description of a candidate window in C, read from the window itself (no full-level box
   maps).  Mirrors darkface_v2.features, darkface_v3.fine_feats / context_feats / at_level and darkface_v4.describe:
   the same sums in the same order (numpy's pairwise sum for the 256-pixel mean and variance; box sums as
   ((a00 + a10) + a01) + a11), the same rint / log2 / clip -- checked feature for feature against Python.
   Build: cc -O3 -ffp-contract=off -fwrapv -shared -fPIC -o libdesc.so desc.c */
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define T 16
#define PAD 8
#define ZQ 24.0
#define NBASE 896
#define NB 1856

typedef struct { const int8_t *cd, *cg; const double *xp, *mp; int Hp, Wp; double spix, s; } Level;

static double pw(const double *a, int n){                 /* numpy pairwise_sum */
    if (n < 8){ double r = 0.; for (int i = 0; i < n; i++) r += a[i]; return r; }
    if (n <= 128){
        double r[8]; int i;
        for (int j = 0; j < 8; j++) r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    int n2 = n / 2; n2 -= n2 % 8;
    return pw(a, n2) + pw(a + n2, n - n2);
}
static inline double clipd(double v, double lo, double hi){ return v < lo ? lo : (v > hi ? hi : v); }

/* the window at padded (r, c): mean, std (numpy order), z = (x - mean) / max(std, 2 spix) */
static void window_z(const Level *L, int r, int c, double *z, double *mu_out, double *sd_out){
    double w[T * T], d[T * T];
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) w[i * T + j] = L->xp[(size_t)(r + i) * L->Wp + c + j];
    double mu = pw(w, T * T) / (T * T);
    for (int i = 0; i < T * T; i++){ double t = w[i] - mu; d[i] = t * t; }
    double sd = sqrt(pw(d, T * T) / (T * T));
    double den = sd > 2 * L->spix ? sd : 2 * L->spix;
    for (int i = 0; i < T * T; i++) z[i] = (w[i] - mu) / den;
    *mu_out = mu; *sd_out = sd;
}
/* box2 of the sector-masked gradient energy at (y, x): ((a00 + a10) + a01) + a11 */
static inline double hog2(const Level *L, int y, int x, int sct){
    const int8_t *cd = L->cd; const double *mp = L->mp; int W = L->Wp;
    double a00 = cd[(size_t)y * W + x] == sct ? mp[(size_t)y * W + x] : 0.0;
    double a10 = cd[(size_t)(y + 1) * W + x] == sct ? mp[(size_t)(y + 1) * W + x] : 0.0;
    double a01 = cd[(size_t)y * W + x + 1] == sct ? mp[(size_t)y * W + x + 1] : 0.0;
    double a11 = cd[(size_t)(y + 1) * W + x + 1] == sct ? mp[(size_t)(y + 1) * W + x + 1] : 0.0;
    return ((a00 + a10) + a01) + a11;
}
static inline int dog2(const Level *L, int y, int x, int code){
    const int8_t *cg = L->cg; int W = L->Wp;
    return (cg[(size_t)y * W + x] == code) + (cg[(size_t)(y + 1) * W + x] == code) + (cg[(size_t)y * W + x + 1] == code) + (cg[(size_t)(y + 1) * W + x + 1] == code);
}
static void at_level(int r, int c, double s, const Level *M, int *r2, int *c2){
    double cy = (r - PAD + T / 2.0 - 0.5) * s, cx = (c - PAD + T / 2.0 - 0.5) * s;
    int a = (int)rint(cy / M->s - T / 2.0 + 0.5) + PAD, b = (int)rint(cx / M->s - T / 2.0 + 0.5) + PAD;
    if (a < 0) a = 0; if (a > M->Hp - T) a = M->Hp - T; if (b < 0) b = 0; if (b > M->Wp - T) b = M->Wp - T;
    *r2 = a; *c2 = b;
}
static void fine(const Level *M, int r2, int c2, int8_t *o){                 /* 512: 8 sectors x 8 x 8 cells */
    for (int sct = 0; sct < 8; sct++) for (int a = 0; a < 8; a++) for (int b = 0; b < 8; b++)
        o[sct * 64 + a * 8 + b] = (int8_t)clipd(rint(16 * log2(1 + hog2(M, r2 + 2 * a, c2 + 2 * b, sct + 1))), 0, 127);
}
static void context(const Level *M, int r2, int c2, int8_t *o, int pool_order){  /* 224 = hog 128 + dog 32 + z 64 */
    for (int sct = 0; sct < 8; sct++) for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++){
        int y = r2 + 4 * a, x = c2 + 4 * b;
        double v = ((hog2(M, y, x, sct + 1) + hog2(M, y + 2, x, sct + 1)) + hog2(M, y, x + 2, sct + 1)) + hog2(M, y + 2, x + 2, sct + 1);
        o[sct * 16 + a * 4 + b] = (int8_t)clipd(rint(12 * log2(1 + v)), 0, 127);
    }
    for (int code = 1; code <= 2; code++) for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++){
        int y = r2 + 4 * a, x = c2 + 4 * b;
        o[128 + (code - 1) * 16 + a * 4 + b] = (int8_t)(((dog2(M, y, x, code) + dog2(M, y + 2, x, code)) + dog2(M, y, x + 2, code)) + dog2(M, y + 2, x + 2, code));
    }
    double z[T * T], mu, sd; window_z(M, r2, c2, z, &mu, &sd);
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++){
        double p00 = z[(2 * i) * T + 2 * j], p01 = z[(2 * i) * T + 2 * j + 1], p10 = z[(2 * i + 1) * T + 2 * j], p11 = z[(2 * i + 1) * T + 2 * j + 1];
        double m = pool_order ? (((p00 + p10) + p01) + p11) / 4 : (((p00 + p01) + p10) + p11) / 4;
        o[160 + i * 8 + j] = (int8_t)clipd(rint(ZQ * m), -127, 127);
    }
}

/* describe n candidates of level k (padded r, c): X8 (n x 1856), Xe (n x 8) */
void describe(const Level *Lk, const Level *Lf, const Level *Lc1, const Level *Lc2, int k, int n, const int64_t *r, const int64_t *c,
              const int64_t *S1, const int64_t *sig, int8_t *X8, int32_t *Xe, int pool_order){
    for (int q = 0; q < n; q++){
        int8_t *o = X8 + (size_t)q * NB; int32_t *e = Xe + (size_t)q * 8;
        double z[T * T], mu, sd; window_z(Lk, (int)r[q], (int)c[q], z, &mu, &sd);
        for (int i = 0; i < T * T; i++) o[i] = (int8_t)clipd(rint(ZQ * z[i]), -127, 127);
        for (int sct = 0; sct < 8; sct++) for (int a = 0; a < 8; a++) for (int b = 0; b < 8; b++)
            o[256 + sct * 64 + a * 8 + b] = (int8_t)clipd(rint(16 * log2(1 + hog2(Lk, (int)r[q] + 2 * a, (int)c[q] + 2 * b, sct + 1))), 0, 127);
        for (int code = 1; code <= 2; code++) for (int a = 0; a < 8; a++) for (int b = 0; b < 8; b++)
            o[768 + (code - 1) * 64 + a * 8 + b] = (int8_t)dog2(Lk, (int)r[q] + 2 * a, (int)c[q] + 2 * b, code);
        int r2, c2;
        if (Lf){ at_level((int)r[q], (int)c[q], Lk->s, Lf, &r2, &c2); fine(Lf, r2, c2, o + NBASE); }
        else for (int i = 0; i < 512; i++) o[NBASE + i] = 0;
        if (Lc1){ at_level((int)r[q], (int)c[q], Lk->s, Lc1, &r2, &c2); context(Lc1, r2, c2, o + NBASE + 512, pool_order); }
        else for (int i = 0; i < 224; i++) o[NBASE + 512 + i] = 0;
        if (Lc2){ at_level((int)r[q], (int)c[q], Lk->s, Lc2, &r2, &c2); context(Lc2, r2, c2, o + NBASE + 736, pool_order); }
        else for (int i = 0; i < 224; i++) o[NBASE + 736 + i] = 0;
        e[0] = (int32_t)S1[q]; e[1] = k; e[2] = (int32_t)sig[q];
        e[3] = (int32_t)rint(16 * log2(1 + sd / Lk->spix));
        e[4] = (int32_t)rint(16 * log2(1 + (mu > 0 ? mu : 0) / Lk->spix));
        e[5] = Lf != 0; e[6] = Lc1 != 0; e[7] = Lc2 != 0;
    }
}
