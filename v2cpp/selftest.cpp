// SELF-TEST of the v2 law on any machine, with no face image: every stage run on seeded random arrays (a 64-bit LCG,
// the same in v2_selftest_ref.py), each output reduced to an FNV-1a 64 checksum of its bytes.  The expected checksums
// come from the reference on the Mac; a machine whose arithmetic differs anywhere prints MISMATCH.
//     selftest            print the checksums      selftest expected.txt    compare with a file of them
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "facelaw2.h"

using namespace fl2;

static uint64_t g_state = 20261028ULL;
static double rnd() {
    g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(g_state >> 11) * (1.0 / 9007199254740992.0);
}
static uint64_t fnv(const void *p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
static Mat rmat(int h, int w, double scale) { Mat m(h, w); for (double &v : m.a) v = rnd() * scale; return m; }

int main(int argc, char **argv) {
    std::map<std::string, uint64_t> got;
    std::vector<std::string> order;
    auto put = [&](const std::string &k, uint64_t v) { got[k] = v; order.push_back(k); };
    for (int k = 0; k <= 24; k++) { Mat x = rmat(37, 29, 255.0); Mat g = gauss(x, k); put("gauss_" + std::to_string(k), fnv(g.a.data(), g.a.size() * 8)); }
    { Mat x = rmat(50, 61, 300.0); Mat u = display_u8(x); put("display_u8", fnv(u.a.data(), u.a.size() * 8)); }
    { std::vector<uint8_t> rgb(31 * 27 * 3); for (auto &b : rgb) b = (uint8_t)(rnd() * 256.0); Mat l = linear_of(rgb.data(), 27, 31, 3); put("linear_of", fnv(l.a.data(), l.a.size() * 8)); }
    { Mat F = rmat(80, 110, 255.0); std::vector<int> c = eye_codes(F); put("eye_codes", fnv(c.data(), c.size() * 4));
      std::vector<int64_t> s = score_map(c, 80, 110); put("score_map", fnv(s.data(), s.size() * 8)); }
    { Mat img = rmat(300, 260, 255.0); Box b{40.0, 50.0, 180.0, 180.0, true}; Eyes e = find_eyes(img, b);
      double v[5] = {e.lx, e.ly, e.rx, e.ry, e.ev}; put("find_eyes", fnv(v, sizeof v));
      Mat c = crop(img, Eyes{101.25, 120.5, 158.75, 123.0, true}); put("crop", fnv(c.a.data(), c.a.size() * 8));
      uint8_t m[NPTS]; map_bins(c, m); put("map_bins", fnv(m, NPTS));
      static uint8_t bins[9][NPTS], sig[9][NPTS]; probe_all(c, bins, sig); put("probe_all", fnv(sig, sizeof sig, fnv(bins, sizeof bins)));
      double s = score(bins, sig, m); put("score", fnv(&s, 8));
      Mat c96 = crop96(img, Eyes{101.25, 120.5, 158.75, 123.0, true}); put("crop96", fnv(c96.a.data(), c96.a.size() * 8));
      static uint8_t m96[NPTS96], b96[NPTS96], s96[NPTS96]; map_bins96(c96, m96); put("map_bins96", fnv(m96, NPTS96));
      probe_codes96(c96, b96, s96); put("probe_codes96", fnv(s96, NPTS96, fnv(b96, NPTS96)));
      double t = score96(b96, s96, m96); put("score96", fnv(&t, 8)); }
    int bad = 0;
    std::map<std::string, uint64_t> exp;
    if (argc > 1) {
        FILE *f = fopen(argv[1], "r");
        char k[64]; unsigned long long v;
        while (f && fscanf(f, "%63s %llx", k, &v) == 2) exp[k] = v;
        if (f) fclose(f);
    }
    for (const std::string &k : order) {
        if (argc > 1) {
            bool ok = exp.count(k) && exp[k] == got[k];
            bad += !ok;
            printf("%-12s %016llx %s\n", k.c_str(), (unsigned long long)got[k], ok ? "match" : "MISMATCH");
        } else printf("%s %016llx\n", k.c_str(), (unsigned long long)got[k]);
    }
    if (argc > 1) printf("%s: %d of %zu checksums differ from the reference\n", bad ? "FAILED" : "PASSED", bad, order.size());
    return bad ? 1 : 0;
}
