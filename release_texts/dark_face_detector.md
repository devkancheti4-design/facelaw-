# Dark-face detector (DARK FACE)

Finds faces in night photos. Measured on DARK FACE (6,000 labelled night images; a fixed split: 4,700 design, 300
design-validation, 1,000 sealed test with 8,143 faces, median face 11 pixels).

| File | What it is |
|---|---|
| `law.json` | The integer law: ranking (922 trees) and box placement dx, dy, size (3 × 400 trees), integer leaves in units of 1/4096; 129,442 nodes. Every number is written as integer lanes by program synthesis and proven exact; it is identical to the sealed build (tag b3b1680ec7d8a145) |
| `trees.c` | The same trees as C: int32 arrays and one `ct_<part>(X, n, nf, out)` function per part |
| `results.json` | Sealed test: AP 28.99% at IoU 0.5, max recall 78.6%. Design-validation: AP 31.42% |

## What limits it

- 21.4% of the faces never get a candidate box, so even a perfect ranking would score at most 78.6% AP.
- Below about 8 pixels, a face in the dark carries little that separates it from night clutter; precision is 93% at the top
  100 detections, 60% at recall 23%, 40% at 30%.

## Status

The candidate stage and the description run in the private pipeline (about 3.9 s per image, one core). Their C port is in
progress, so this folder holds the law, not yet a runtime.
