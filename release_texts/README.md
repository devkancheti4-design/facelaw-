# Face Law Labs — release 2026-10-28

Face recognition built from counted integer laws: no neural network, no GPU, no training beyond counting. Every table is
written as integer lanes by program synthesis and proven exact; the C++ reproduces the reference bit for bit.

| Product | What it does | Status |
|---|---|---|
| [`face_law_v2/`](face_law_v2/README.md) | 1:1 face verification. Two laws: **full-light** (96×84) and **dark** (48×42), plus a light router. C++, no dependencies. | Bit-exact with the reference on 74 real photos; 38 of 38 self-test checksums |
| [`nist_frte11/`](nist_frte11/SUBMISSION.md) | The full-light law behind NIST's FRTE 1:1 API 6.0.1, packaged for NIST's validation run | NIST's own test driver: 124 of 124 comparisons Success. The Ubuntu validation run and the submission are the owner's steps |
| [`dark_face_detector/`](dark_face_detector/README.md) | Finding faces in night photos (DARK FACE) | The frozen law and its compiled trees; 28.99% AP on the sealed test; runtime port to C in progress |
| [`face_unlock_phone/`](face_unlock_phone/README.md) | Face unlock for any phone: the owner maps the angles they unlock at; the phone's own camera and screen, no depth camera, no infrared | **No stranger got in, in any test** (lab faces and real lab photographs); a photograph of the owner can |
| [`reports/`](reports/) | The head-to-head with ten classic recognisers; the full-light law report | |

## Results (Extended Yale B lab faces; cross-fitted: tables counted on the other half of the people)

| Test | Result |
|---|---|
| NIST-style 1:1, normal light, every pair | **86.5%** of owners accepted at 1 in 1,000 strangers (best classic, Tan-Triggs at 96×84: 84.0%); **72.7%** at 0 of 115,977 strangers (70.7%) |
| Yale B identification, hardest subsets 4 / 5 | 99.0 / 98.6 (tied first with Weberfaces at 96×84: 99.6 / 98.3) |
| Deep dark, 5 photons per face pixel, front light | dark law 76.6% at 1 in 1,000 (full-light law 61.5%) |
| Both laws, in if either says yes | best at 0 strangers in every light tested (front light, 5 photons: 48.4% vs 32.8% and 35.6% alone) |
| Two tries, different photos, strangers also get two | 72.7% → 88.6% at 0 strangers |

## Night detection against an Android phone

{NIGHT_ROWS}

## Efficiency

The full-light law reads a face in **2.3 million operations** (2.0 million floating-point, 0.3 million integer): the same
arithmetic as the best classic method read the same way (Tan–Triggs at 96×84, one placement: 2.3 million, much of it powers
and exponentials), while accepting more owners (86.5% vs 83.8% at 1 in 1,000, no search for either). Its comparison is
**integer table look-ups and sums only** (105,840 operations, 12.5 µs on one core): on a chip without floating point that is
about six times cheaper than the classic's 24,192-flop comparison; on a CPU with floating point the classic's comparison is
the faster one (3.8 µs). Template 21 KB, tables 33 KB, training by counting in seconds. Details: [`EFFICIENCY.md`](EFFICIENCY.md).

## Limits

- Lab faces only: 37 people under 64 lights (Yale B); the 48×42 law also passed a sealed CMU PIE check. NIST's sets add age,
  pose, expression and other cameras, at false-match rates down to one in a million.
- The router's threshold and the both-laws numbers were chosen on the same tries they are reported on.
- The dark-face detector's runtime is not yet standalone (see its README).
- Face unlock: photographs of the owner can open it; live strangers have not yet tried it on the phone; Apple Face ID opens far more often for the owner (see its README).

[`MANIFEST.txt`](MANIFEST.txt) lists every file with its sha256.
