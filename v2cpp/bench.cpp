// TIMING of FACE LAW v2 on one core: template creation per image (and per 640 x 480 pixels, NIST's unit) and one
// symmetric comparison.  NIST's limits (API 6.0.1): 2000 ms per 640 x 480 image to make a template, 0.1 ms to compare.
//     bench image.ppm ...        (binary PPM or PGM)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "facelaw2.h"
#include "tmpl.h"

static bool read_pnm(const char *p, std::vector<uint8_t> &d, int &w, int &h, int &ch) {
    std::ifstream f(p, std::ios::binary);
    std::string m; int mx;
    if (!(f >> m >> w >> h >> mx) || (m != "P6" && m != "P5")) return false;
    f.get();
    ch = m == "P6" ? 3 : 1;
    d.resize((size_t)w * h * ch);
    return (bool)f.read((char *)d.data(), d.size());
}

int main(int argc, char **argv) {
    using clk = std::chrono::steady_clock;
    std::vector<std::vector<uint8_t>> T;
    std::vector<double> ms, per;
    int fail = 0;
    for (int i = 1; i < argc; i++) {
        std::vector<uint8_t> d; int w, h, ch;
        if (!read_pnm(argv[i], d, w, h, ch)) continue;
        std::vector<uint8_t> t; fl2::Eyes e; fl2::Box b;
        auto t0 = clk::now();
        int r = fl2::make_template(d.data(), w, h, ch, t, e, b);
        double dt = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        ms.push_back(dt); per.push_back(dt * (640.0 * 480.0) / ((double)w * h));
        if (r == 0) T.push_back(t); else fail++;
    }
    if (ms.empty()) { printf("no images\n"); return 1; }
    std::sort(ms.begin(), ms.end()); std::sort(per.begin(), per.end());
    printf("template creation: %zu images, %d without a template; median %.1f ms per image (max %.1f), median %.1f ms per 640x480 pixels  [limit 2000]\n",
           ms.size(), fail, ms[ms.size() / 2], ms.back(), per[per.size() / 2]);
    bool ok; double acc = 0; long n = 0;
    auto t0 = clk::now();
    for (int rep = 0; rep < 20; rep++)
        for (size_t i = 0; i < T.size(); i++)
            for (size_t j = 0; j < T.size(); j++) { acc += fl2::match_templates(T[i], T[j], true, ok); n++; }
    double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / std::max(n, 1L);
    printf("comparison: %ld symmetric comparisons, %.2f us each  [limit 100 us]  (checksum %.3f)\n", n, us, acc);
    return 0;
}
