# Dark Unlock — the specification (v4, 2026-10-27)

**What it is:** a scanner that maps the owner's face once. After that, only that map opens the device, and everything else cancels out: darkness, noise, exposure, gain and tone. It runs on any front camera, uses the screen as the light, and needs no depth or infrared hardware. PRIVATE: it names program synthesis, so scrub it before any public use.

## 1. The scan (makes the map)
| Item | Spec |
|---|---|
| Light | the screen at full white, as the main light; a dark room is fine |
| Pose | frontal, at arm's length, inside an on-screen face guide (places the face within ±1 px of the grid) |
| Alignment | both pupils mapped onto the canonical eye points (8, 11) and (33, 11) of a 42 × 48 frame (a similarity transform, anti-aliased) |
| **The map** | at each of 3 scales (σ = 0.75 / 1.5 / 3 px; grids at strides 1 / 2 / 4, so 2,016 + 504 + 132 = **2,652 points**): the **gradient direction sector (0–7)**, written by the synthesis's sector lanes and proven on all 2²⁴ inputs; plus a **flat bit** (\|∇\| < 1) |
| Light signature | a_scan = ((left − right), (top − bottom)) / mean brightness |
| Size | about **1.3 KB**, stored on the device only |

## 2. The unlock try (only the map opens)
| Step | Spec |
|---|---|
| a. Capture | the screen flashes white; the face is aligned as in the scan |
| b. **Noise, measured from the frame itself** (Law 1) | σ² per brightness band (4 bands) = (median\|d\| / 0.6745)², with d = (a − b − c + e)/2 over 2×2 blocks. No sensor constants, so any camera works. |
| c. **Codes** (Law 2: gain and any monotone tone curve cancel exactly) | per point: the sector from the authored lanes, and the significance c = min(7, ⌊\|∇\| / √(2·E_k·V)⌋) in noise units |
| d. Code against the map | code = sector gap (0–4) \| c≪3 \| flat≪6 \| scale≪7, giving **512 codes** |
| e. **Score** | S = the best of **9 placements (±1 px)** of the mean over the 2,652 points of **F[code]**. F is a 512-entry table of 16·ln P(code \| same) / P(code \| other), measured, then **authored as integer lanes** (256 lanes, 0 wrong) |
| f. **Cancel** | margin m = (S(owner) − mean S(cohort)) / spread S(cohort), against a **fixed cohort of stranger maps** shipped with the device. Darkness, blur and exposure shift every score together, so they cancel. |
| g. **State, measured from the frame** | light class (imbalance < 0.6 front / 0.6–1.2 oblique / ≥ 1.2 side) × evidence band (share of significant points) × **light differs from the scan** (\|a_try − a_scan\|₁ ≥ **0.345**, authored as integer lanes) × margin band (steps of 0.25) |
| h. **Decision** | **ACT4[state]**: 744 entries, **authored as integer lanes** (372 lanes, 0 wrong, every lane alive). Answers: **OPEN**, **NOT NOW** ("face the screen" / light differs / too dark) or **NOT YOU**. OPEN only in cells where measured strangers opened ≤ 1 in 10,000 (zero seen in ≥ 30,000 tries per cell, rule of three). Side light always gives NOT NOW. |
| i. Attempts | up to 5 tries, then the PIN |

## 3. Liveness: a photo or screen must not open it (SPEC, NOT YET BUILT)
- Capture two frames, screen white and screen dark, and take the difference: the face's response to the screen's own light.
- A **3D face** shades according to the light subspace (nose, cheeks and brow at different angles). A **photo or screen is flat**: it gives a uniform ratio and screen glare.
- The decision is a residual from the 9-term light subspace against a measured threshold, authored as integer lanes.
- **Target: spoofs open ≤ 7%** (the Android strong class).

## 4. Exactness
- Every table (sector lanes, F, ACT4, D1) is authored as integer lanes on independent runs, and proven by Python and compiled C.
- The runtime is rebuilt from the lanes only. Every test runs under a sealed protocol with SHA-256 hashes, once.

## 5. Measured (sealed, lab images, darkness simulated by Law 1)
| | Result |
|---|---|
| Strangers who opened | **3 of 1,109,760** (all light); about **1 in 476,000** (front light). This passes Android's 1 in 50,000. |
| Owner, front light, ≥ 200 photons per face pixel | OPEN **81–84%**, NOT NOW 11–12%, NOT YOU **4–7%** |
| Owner, light within 12° of the scan (design data) | OPEN **93–95.5%**, 0 strangers of about 12,400 |
| The user's own 2 photos | owner: never NOT YOU; side-lit photo: NOT NOW. Strangers opened 12 / 10,240, with a room-lit scan and a lab cohort. This shows the requirements below. |

## 6. Requirements (each one measured)
1. The scan **must be screen-lit from the front**. With oblique-lit scans, strangers' shared shadow patterns match.
2. The cohort **must come from the same camera domain** as the scan (phone captures of consenting volunteers).
3. Darkness: it needs the screen's light, because there is no infrared.

## 7. Not yet proven
Liveness (≤ 7% spoofs); real-device FAR with ≥ 150,000 stranger tries; owner first-try OPEN ≥ 95% on a real phone; twins and siblings; population scale.

## Files
- `unlock_law.py`, `unlock_lanes.json` (v3 sealed: protocol `14323f0d`, result `43ac8890`)
- `unlock_v4.py`, `unlock_v4_lanes.json` (v4 sealed: protocol `233518f2`, result `5d39d755`)
- `phone_unlock.py` (real photos)
