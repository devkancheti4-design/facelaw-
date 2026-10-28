# Efficiency

Both methods read the face once (one placement, no search), the setting of the NIST product. Times are measured on one core
of an Apple Silicon Mac. The flip-phone rows are estimates from counted arithmetic for an **assumed 2011 handset**: an ARM9 at
200 MHz with no floating-point unit (an integer operation 1.5 cycles, a software double 40 cycles, a fixed-point port 3
integer operations per float); they are not measurements on a phone.

| | Full-light law (the NIST product) | Tan–Triggs at 96×84 (the best classic) |
|---|---|---|
| Owners accepted, NIST-style test, 1 stranger in 1,000 | **86.5%** | 83.8% |
| Arithmetic to read one face | 2.0 M floating-point + 0.32 M integer | 2.3 M floating-point, much of it powers and exponentials |
| One comparison | 105,840 integer operations (both directions), **no floating point** | 24,192 floating-point operations (L1 distance) |
| Template | 21,176 bytes (one byte per point) | 32,256 bytes (one float per pixel) |
| Model | 33 KB of counted tables (768 + 31,875 entries, 8-bit values) | none |
| Training | counting, seconds; no gradient descent | none |
| This Mac, one core | template from a 640×480 image 85–125 ms; comparison 12.5 µs (C++) | comparison 3.8 µs (numpy) |
| Flip phone: read one face, fixed point / software float | 47 ms / 403 ms | 52 ms / 461 ms |
| Flip phone: one comparison | **0.79 ms** | 4.84 ms |

What this means:

- **Same arithmetic, more accuracy.** Per face, the law costs what the best classic costs and accepts 2.7 more owners in 100.
- **Integer matching.** A comparison never touches floating point, so on chips without a floating-point unit it is about six
  times cheaper than the classic's; on a CPU with one, the classic's short float loop is faster (3.8 µs against 12.5 µs).
- **Small and fixed.** One byte per point; 33 KB of tables; every run gives the same bytes on any machine
  (`-ffp-contract=off -fwrapv`).

Also measured: peak memory 18.6 MB for a 640×480 image; the NIST library is about 480 KB on arm64 (mostly OpenCV's
frontal-face cascade compiled in as arrays). NIST's limits are 2,000 ms per template and 0.1 ms per comparison.

The dark law (48×42, ±1-pixel search, as in the head-to-head report) reads a face in 4.5 M floating-point operations and
compares with 238,680 integer operations; without the search it is 0.5 M and 26,520.

The dark-face detector: 2,122 integer trees (129,442 nodes); about 3.9 s per image on one core in the current pipeline.
Not yet phone-ready.
