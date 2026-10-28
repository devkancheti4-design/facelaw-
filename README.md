# facelaw: face recognition laws for one dark phone image

The public copy of Face Law Labs' face recognition laws: the C++ laws, every law table as plain numbers, and every
test and result. Every table entry is written as integer lanes by program synthesis and proven exact; the tools that
write and prove them stay private (see *Not in this public copy*).

Six exact laws for recognising a face from one dim, noisy phone image. Nuisances cancel algebraically:
- Law 1, **sensor**;
- Law 2, **light**;
- Law 3, **geometry**;
- Law 4, **identity**;
- Law 5, **decision**;
- Law 6, **limit** ("too dark to tell").

The physics equations are the specification. Every integer runtime law is **authored as integer lanes**:
- from rows that are true by construction or counted from design data;
- each lane independently;
- each lane **proven exhaustively** (Python and compiled C).

The full derivations, every check and every result are in **[LAWS.md](LAWS.md)**.

## Verdict

| test | result |
|---|---|
| synthetic world: Laws 1–6 predictions vs Monte Carlo | match; FAR to about 1 point; Law 6 bound held 96 of 96 |
| Extended Yale B, standard split (train 7 near-frontal photos, 38 people) | superset 100 / 100 / **99.05** / **98.46**, ahead of every method I ran and of VGG-Face. **Not sealed**: variants were explored on it |
| **sealed** CMU PIE illumination (68 people, 1 training photo) | 100%, tied with Gradientfaces, Weberfaces and Tan–Triggs: the test is saturated |
| **sealed** dark CMU PIE (1–10 photons per pixel) | 64.73 / 94.39 / 99.58 / 99.98. **Not No. 1**: Weberfaces is ahead at every level. **Never guesses**: 0 wrong among answers, but the gate was over-cautious |
| synthetic optimal (Chow) law with strangers | **not calibrated** yet: 4.5–14% wrong among answers, against a promise of 1% or less |

## Files

The six laws as computation:

| file | role |
|---|---|
| `world.py` | exact synthetic dark-face world: Poisson sensor, Lambertian light, perspective, area-integrating pixels |
| `laws.py` | the laws: sensor inverse, light subspace fit, Law 5 exact moments, Law 6 exact Bhattacharyya |
| `check_static.py`, `check_noise.py` | Laws 1–4 and 5–6, predicted vs measured |

Program synthesis writes each lane, and a matching script proves it:

| authoring script | output | proof |
|---|---|---|

Yale B:

| file | role |
|---|---|
| `yale.py` | loader, photometric-stereo enrolment, Laws 2–6 on real faces |
| `yale_std.py` | standard protocols vs published numbers |
| `yale_cast.py`, `yale_joint.py` | cast shadows; the joint light law |
| `yale_baselines.py` | Gradientfaces, Weberfaces, Tan–Triggs (my reimplementations) |
| `yale_local.py` | the local census scan |
| `yale_law.py`, `yale_cast_law.py` | the per-pixel data laws |


The design of the dark law (on Yale B only) is in `dark_law.py`.

Other folders:
- `explore/` holds the exploration scripts that informed choices, disclosed: the light convention, the sector scan, the lit-only fit.

## Running

- **Requirements:** Python 3 with numpy and scipy, and a C compiler for the proofs.

### Data (not included: research licences)

Put the data in `data/`, each in its own folder, and run the scripts with `python3 -I`.

- **Extended Yale B, cropped** (`data/yaleb/CroppedYale`)
  - The official UCSD links are dead. Use the Internet Archive's copy of `vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip`.
  - Size: 58,493,820 bytes.
  - SHA-1 (base32): `5Y2B5Y44MGKBHLPLOKF5366WSQYKRQM3`, which matches the archive's digest.
- **CMU PIE, processed by Deng Cai** (`data/pie/lights=27.mat`)
  - Internet Archive copy of `www.cad.zju.edu.cn/home/dengcai/Data/PIE/lights=27.mat`.
  - Size: 1,303,519 bytes.
  - SHA-1 (base32): `OULOPDBCGH27BQNIILMJ7LQ5F43Q7M2L`.

## Everything tested on 2026-10-28 (v2 for NIST, the phone face unlock v7–v9, the head-to-heads)

Every number below was measured on this date; the phone trials ran under hash-locked protocols written before them
(`phone_faceid/protocol10.txt` … `protocol12_addendum2.txt`, each with its `.sha256`). The owner's photos, map
(`phone_faceid/vault.json`) and trial log (`phone_faceid/trials.jsonl`) stay on the Mac and are not in this repo.

### The laws (each authored as integer lanes as proven lanes, rebuilt from the lanes and checked at every load)

| law | files | size | proof and held-out result |
|---|---|---|---|

### v2 for NIST FRTE 1:1 (`v2cpp/nist/`, `nist_submission/`, `dist/facelaw_v2_nist.zip`)

| test | result |
|---|---|
| NIST-style 1:1 on Extended Yale B | 86.5% of owners accepted at 1 in 1,000 strangers (best classic, Tan–Triggs: 84.0%); 72.7% at 0 of 115,977 |
| NIST's own test driver | 124 of 124 comparisons Success; multi-face works |
| cost | 87–125 ms per 640×480 template, 12.5–19 µs per comparison, 21,176-byte template |
| status | kit ready; waiting on the organization email (Face Law Labs domain), the Kaggle validation run, the key, the agreement |

### The phone face unlock (phone over USB to this Mac: `phone_faceid/server.py`, `index.html`)

| version | protocol | the change | what it measured |
|---|---|---|---|
| v7 pose grid, NEAR4 | 10 (+ addendum 1) | the pose law reads every frame's angle; a try is compared with the 4 enrolled views nearest it | the owner's face matched in 7 of 8 attempts, but the flash check refused 8 of 11; 3 refusals were mislabelled NOT YOU (page bug, fixed). The angle read on the phone sat at the grid's edge on 71–100% of NOT YOU frames |
| v8 mapped angles | 11 | the owner maps the angles they choose; the password adds an angle after a NOT YOU (PBKDF2 hash, 5-try lockout); every try against every mapped view; margins 0.536 / 1.531 (`mapped_match.json`) | 48 owner tries: 20 opened, 35 recognised; renders end to end 12/12 at mapped angles, 18/24 at 5–10° off; an added angle 0/10 → 9/9; strangers 0/252. **Real photographs of strangers: 74 of 17,492 frames cleared a bar** (`tests/phone/v8_strangers_real.py`) |
| v9 all laws | 12 (+ addenda 1, 2) | the eye-check law in front of every law: candidate pairs from the tracked and detected boxes, the best checked pair or the frame is dropped; bars from the checked stranger bundle; margins 0.625 / 1.876 (`mapped_match9.json`) | renders end to end 12/12, 19/24 at 5–10°, 0/10 at 20°+; an added angle 0/10 → 9/9; strangers 0/252. **Real photographs of strangers: 3 of 14,420 frames** (one photo, yaleB09 lit 40° from below; eyes right) (`tests/phone/v9_strangers_real.py`). Pitch dark on the phone: recognised 7 of 10 tries once the screen was bright; opened in 1.8 s with a dark angle |

**Why v8 leaked, and what fixed it.** Replaying one owner's run exactly (`tests/phone/replay_hits.py`, `hits_yaleB01.json`):
all 12 stranger hits were crops at WRONG eyes (the pair 10–26 px apart where the true eyes are 94–102 px apart): shadows
taken for eyes, a blown-up patch scoring up to 19 above a bar. The eye-check law removed them (74 → 3). The 3 left are a
hard-light near miss (light from below, which the rendered stranger set never had); next: real photographs in the
stranger set (v9.1).

### The flash check (liveness)

| test | result |
|---|---|
| a flat print moved ~1 camera pixel per flash (renders, `tests/phone/check_test.py`, `check_v1.py`) | passes as live: phone v7.0 check 21–23 of 24; the public challenge's check 23 of 24 (11 of 24 at 0.5 px). **Not photo-proof** |
| printed real photographs of the owner (`print_identity.py`, `print_attack.py`) | the map matched front-lit prints 14 of 14; moving prints then opened 39 of 39 |
| FOLLOW (eyes found again on every flash frame) | refused: opened still prints 12 of 24 |
| blink (`blink_probe.py`) | probed, then removed at the owner's request |
| the public challenge | paused (22 tries, 0 opened) |

### Head-to-heads

| test | result |
|---|---|
| Apple Face ID, the owner, same light | opened 11 of 11; refused the owner's mother. Face Law well behind |
| the phone's own Android face unlock (vivo I2301; registered as a CONVENIENCE-class biometric), photos on the Mac screen (`phone_h2h/unlock_screen_h2h.py`) | opened for a stranger's photo (yaleB23) 2 of 4; refused the owner's photos. Face Law: one stranger's photo (yaleB21, lit from below) matched the owner's angle 1 in 2 of 3, the flash check stopped it; 0 of 38 opened before the run was stopped |
| eye finding, 2,354 real Yale photos, Android's own face engine (Chrome FaceDetector) vs Face Law (`phone_h2h/eyes_h2h.py`, `eyes_h2h.json`) | normal light: eyes right 96.2% (Face Law v9) vs 94.9% (Android); harsh light: Android 40.5% vs 6.3%; **wrong eyes: 0 (v9) vs 19 (Android) vs 104 (v8)** |
| night faces, 300 DARK FACE photos (`phone_h2h/`) | Android's detector found 1.6% of the faces; Face Law's night detector AP 31.4% (sealed test 28.99%) |

### Not done yet

v9.1 (real photographs in the stranger set); a liveness law that stops moving prints; an attention check; the phone build
without the Mac; the NIST submission (the owner's steps).


## Not in this public copy

- the private authoring tools and the lane records they write (the laws themselves are here as tables and C++);
- the Python reference files that call those tools: `darkface_cscore.py`, `darkface_lite.py`, `darkface_speed.py`, `darkface_v4.py`, `eye_check.py`, `eye_law.py`, `optimal.py`, `personal.py`, `pie_dark.py`, `pose_law.py`, `recognize.py`, `router.py`, `unlock_law.py`, `unlock_mf.py`, `unlock_v4.py`, `v2_full.py`, `v2ref.py`, `yale_gate.py`;
- the face shapes reconstructed from Extended Yale B (`pose_shapes.npz`, `pose_mean_depth.npz`) and every dataset;
- the phone owner's photos, map and trial log.
