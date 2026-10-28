# Face Law v2 — 1:1 face verification (C++)

Standard C++17, no dependencies. Build with `-ffp-contract=off -fwrapv` so every machine produces the same bytes.

## Build and check

```bash
c++ -std=c++17 -O2 -ffp-contract=off -fwrapv -o selftest src/selftest.cpp src/facelaw2.cpp src/vj.cpp src/tmpl.cpp
./selftest src/selftest_expected.txt
```

The self-test must end with `PASSED: 0 of 38 checksums differ from the reference`.

```bash
c++ -std=c++17 -O2 -ffp-contract=off -fwrapv -shared -fPIC -o libfacelaw2.so src/facelaw2.cpp src/vj.cpp src/tmpl.cpp src/capi.cpp
```

`bench.cpp` times template creation and comparison on your own PPM/PGM images.

## Use

- `tmpl.h`: `make_template(image, w, h, channels, template, eyes, box)` makes a 21,176-byte template with the full-light law
  (status -1: no face, -2: no eyes, -3: the eyes fail the gate). `match_templates(a, b, symmetric, ok)` gives the score.
- `facelaw2.h`: the two laws. `crop`, `map_bins`, `probe_all`, `score` (dark law, 48×42, ±1 px search) and `crop96`,
  `map_bins96`, `probe_codes96`, `score96` (full-light law, 96×84, one placement).
- `capi.cpp`: a plain C interface for apps.
- `nist/facelaw11.cpp`: NIST's FRTE 1:1 interface.

## Which law for which light

Owner accepted % at 0 strangers / at 1 in 1,000 (Extended Yale B, raw score). The full-light law wins from about 10 photons
per face pixel up; the dark law keeps four times the photons per pixel and wins below.

**The router** reads the light from the try itself: in each of four brightness bands of the 48×42 crop, a pixel's photon count
is mean² / variance (the variance from the 2×2 diagonal residual, so no camera gain is needed); Nhat is their mean. The try
goes to the full-light law when `min(4095, floor(16 Nhat)) >= 108`, that is from 6.75 photons per pixel.

| Photons per pixel | Nhat (median) | Routed, front light | Routed, all light | Read by the full-light law |
|---|---|---|---|---|
{ROUTER_ROWS}

**Both laws** (for a phone that unlocks day and night): both laws read every try, and the owner is in if either says yes, each
law at its own threshold. At 0 strangers this beat each law alone and the router in every light tested.

Status: the router and the both-laws rule are specified here and measured; their C versions are next. The functions for both
laws are in this library and bit-exact.
