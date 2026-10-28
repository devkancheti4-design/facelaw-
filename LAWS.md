# The face laws: one dark phone image → who, or "too dark to tell"

**The core idea:** for the right person, every nuisance cancels to exactly zero. Light, exposure, gain, pose and expression all drop out (10⁻¹⁵ measured, with no noise). For anyone else, a remainder λ survives. The decision compares those remainders against the photon noise, and nothing else.

**Who did what:**
- **Program synthesis** authored every integer runtime lane, 12 of them (`lanes_face.json`). Each came independently. Each is proven over its whole input domain twice: by the engine's own evaluator and by compiled C (`proof_face.json`), with zero wrong.
- **The equations** are the specification: physics plus statistics, with every parameter measurable.
- **The predictions** below were derived before they were measured, then checked on synthetic dark images rendered from an independent surface sampling (`world.py`). The recogniser therefore never shares the world's pixelation.

**The world:**
- **Face:** 64 × 64 sensor; the face is about 40 px wide.
- **Sensor:** b = 64 DN, g = 2.7 DN/e⁻, s_r = 1.5 e⁻, W = 1023.
- **Light:** a distant point source with ambient light and attached shadows.
- **Nuisances:** pose up to ±17° yaw, ±10° pitch and ±6° roll, with translation and scale; random expression.
- **Gallery:** 10 enrolled people; strangers are drawn from the same population law.
- **Brightness:** N̄ means photons per face pixel.

---

## LAW 1: sensor

1. **Holds when:**
   - the image is RAW: linear, before demosaic (one channel here);
   - photon arrivals are Poisson and read noise is Gaussian, independent of signal;
   - gain is linear, and the black level and white level are fixed.
2. **Equation:**
   - x(p) = clip(round(b + g·(N(p) + r(p))), 0, W), with N ~ Poisson(Φ(p)) and r ~ N(0, s_r²).
   - **Inverse:** N̂(p) = (x(p) − b)/g, with Var N̂ = Φ + s_r² + 1/(12g²).
   - **Measured parameters:**
     - b is the mean of the optically black pixels (or the DNG BlackLevel tag);
     - s_r is their standard deviation divided by g;
     - g is the slope of Var x against E[x] − b (the photon-transfer law; or the DNG NoiseProfile tag).
3. **Cancels:** E[x] = b + gΦ, so (E[x] − b)/g = Φ for every g and b. Gain and black level drop out exactly.
4. **Remains:** Φ, plus noise of known variance.
5. **Breaks:**
   - **Saturation:** x = W is only a lower bound, so the pixel carries no number (lane `L1_sat`).
   - **Black clipping** when b is small.
   - **Hot pixels and fixed-pattern noise**, unless calibrated.
   - **JPEG.** A known, strictly increasing tone curve can be inverted only to an interval per code. Denoising, sharpening and DCT quantisation map many images to one, so no inverse exists. Pixels are no longer independent Poisson, so the Law 5/6 derivations do not apply. The JPEG path keeps only the order of codes (lane `L2_census`). Unless the phone's processing is known, the laws cannot certify an identity from a JPEG.
6. **Prediction:** bias 0 and variance ratio 1.000.
   - **Measured (400,000 pixels each):**

| Φ (photons) | bias | variance ratio |
|---|---|---|
| 1 | +0.006 ± 0.006 | 1.004 ± 0.005 |
| 3 | +0.009 ± 0.007 | 1.002 |
| 10 | −0.003 ± 0.011 | 1.000 |

**Synthesis lanes:**
- `L1_photon` = `(x - 64)`. The division by g happens once, in the final step, so the law never rounds twice.
- `L1_sat` = `((x + 1) >> 10)`.

## LAW 2: light

1. **Holds when:**
   - skin is diffuse (Lambertian);
   - the light is distant;
   - there are no cast shadows or interreflections;
   - light enters through its 9 low-frequency spherical-harmonic terms. This is exact for such fields; attached shadows keep about 98–99% of their energy in those 9 terms (Basri–Jacobs, Ramamoorthi).
2. **Equation:**
   - Φ(p) = κ·ρ(q)·Σⱼ ℓⱼ Yⱼ(n(q)), so y = B c, with c = κℓ ∈ ℝ⁹ and Bⱼ(p) = ρ(q(p)) Yⱼ(n(q(p))).
   - **Transform:** T(y) = (I − P_B) y, with P_B = B(BᵀWB)⁻¹BᵀW.
3. **Cancels:** light ℓ and exposure κ enter only through c, and T(Bc) = Bc − B(BᵀWB)⁻¹(BᵀWB)c = Bc − Bc = 0 for every c. The exact step is (BᵀWB)⁻¹(BᵀWB) = I.
4. **Remains:** identity, through B (albedo ρ and shape via the normals n), and noise. For the wrong person B′, (I − P_{B′})Bc ≠ 0. That remainder is the identity signal.
5. **Breaks:**
   - **Not exact, with a bound:** attached shadows beyond the 9 terms. The bound is about 8.8% relative for a single point source; measured below.
   - **Not covered by the 9-term model:**
     - cast shadows (the nose's shadow);
     - specular shine;
     - glasses;
     - a beard (not Lambertian, and it occludes);
     - interreflection.
   - **The outlier gate** catches these: more than 2% of pixels with whitened residual above 4.
6. **Prediction:** T(y) = 0 to machine precision for any light and any exposure. The point-light leftover stays within 8.8%.
   - **Measured (60 faces, no noise):**

| case | median | 95th percentile | max |
|---|---|---|---|
| 9-term light, same pixelation | 5.4·10⁻¹⁶ | 1.8·10⁻¹⁵ | 4.5·10⁻¹⁵ |
| exposure × 1000 | 5.4·10⁻¹⁶ | 2.4·10⁻¹⁵ | 6.1·10⁻¹⁵ |
| 9-term light, independent pixelation | 1.1·10⁻³ | 2.3·10⁻³ | 3.7·10⁻³ |
| point light with attached shadows | 2.3·10⁻³ | 3.6·10⁻² | 5.7·10⁻² |

**JPEG path:** the census lane `L2_census` = `(((255 - x) >> 8) - (((x >> 8) - x) >> 8))` computes [a > b] on two 8-bit codes, with no sign extraction and no mask.
- **Proven:** on all 65,536 code pairs.
- **Tone and exposure:** any strictly increasing tone curve and any exposure k > 0 cancel exactly. Order is lost only on ties, and every lost pair was a tie: 0.3–3.5% of pairs at normal exposure, 22–24% when clipping (4 curves × 3 exposures, 400,000 pairs each).

## LAW 3: geometry

1. **Holds when:**
   - the camera is a pinhole (lens distortion removed using the phone's lens profile);
   - the head is rigid;
   - each pixel integrates radiance over its footprint;
   - the face is fully in frame.
2. **Equation:**
   - p = π(R S(q) + t), with π(X, Y, Z) = (fX/Z + c_x, −fY/Z + c_y);
   - visibility: front-facing (n·(−P) > 0) and front-most.
   - **Back-map:** never warp the noisy image. Render the candidate at the pose and compare in canonical coordinates q.
   - **Pose is estimated from the data only**, with no landmarks and no learned detector:
     1. the face's footprint (photons above the black background), which does not depend on light;
     2. a coarse grid on the mean face;
     3. a per-candidate refit that maximises the same likelihood.
   - **Valid pose:** a physical pose (yaw ≤ 34°, pitch ≤ 23°, roll ≤ 17°, scale 0.8–1.25) with the whole face in frame.
3. **Cancels:** pose and scale enter only through which q lands on which pixel. f and t_z appear only as f/t_z (exactly so under weak perspective), so absolute size cancels.
4. **Remains:** the shape, up to the depth ambiguity. One image cannot recover the generalized bas-relief family:
   - z′ = λz + μu + νv;
   - ρ′ = ρ|m′|/(λ|m|);
   - s′ = λA⁻ᵀs.

   This family gives the identical image for any light, attached shadows included. **Handling:** the law never estimates S from the probe. S is each candidate's enrolled reference, the one handmade example, so the ambiguity never enters a decision. It only marks two people as indistinguishable (Law 4).
5. **Breaks:**
   - motion blur (the footprint moves during the exposure);
   - rolling shutter;
   - an uncalibrated lens or a wrong f;
   - extreme pose (self-occlusion);
   - a cluttered or bright background, because pose initialisation assumes a dark one.
6. **Prediction:** depth-ambiguity images are identical under orthographic projection, and perspective breaks the ambiguity.
   - **Measured:**
     - orthographic: difference 1.2·10⁻¹⁶ in two cases (5.2·10⁻⁴ in the third, from the renderer's depth-layer test);
     - perspective at 8 face-widths: 16–17% image difference, so a phone held close does carry weak depth information;
     - whole pipeline: see the end-to-end results below.

## LAW 4: identity

1. **Holds when:** Laws 1–3 hold, and expression moves only the lower face (v < −0.15).
2. **Descriptor:** Dᵢ = span{ρᵢ Yⱼ(nᵢ)} on the rigid region v ≥ −0.15, in canonical coordinates. This is the 9-dimensional set of every image person i can produce.
3. **What drops out:**
   - **Light and exposure:** they are c (Law 2), and the subspace does not change.
   - **Pose and focal length:** they enter only through the render map (Law 3), and D lives in canonical coordinates.
   - **Expression:** its support lies outside the rigid region, so B there does not depend on it. Measured 1.2·10⁻¹⁶ (40 faces).
   - **Gain and black level:** Law 1.
   - **Noise:** it does not cancel. It enters only the test (Law 5).
4. **Remains:** ρ and S on the rigid region, modulo what is lost:
   - the lower face;
   - any albedo pattern that equals a shading change (absorbed by c);
   - the depth-ambiguity family, under orthographic viewing;
   - absolute scale;
   - points occluded at this pose;
   - everything below the photon noise.
5. **Breaks:** expressions that move the brows or cheeks; ageing; makeup; hair over the forehead.
6. **Prediction:** the rigid-region image is unchanged by any expression. Measured 1.2·10⁻¹⁶.

**Synthesis lane:** `L4_rigid` = `((x - x) - ((x - 18) >> 31))`. This is optimal, not minimal: `x - x` folds away in the compiled code.

## LAW 5: decision

### The threshold test

- **Statistic:** z = (R − m)/sd₀, where:
  - R = Σ w (y − P_B y)²;
  - w = 1/(μ̂ + s_r² + 1/(12g²));
  - m = |F| − 9;
  - sd₀² = 2m + Σ w² μ̂.
- **Derivation:**
  - **True person:** E R = tr(I − H) = m exactly. Var R = 2 tr((I − H)²) plus a Poisson fourth-cumulant term, so z ≈ N(0, 1). FRR(τ) = 1 − Φ(τ), independent of N.
  - **Impostor:** R = λ + noise, with λ(N̄) = min_c ‖W^½(μ_A − B_B c)‖² (pose refit too). FAR(τ, N̄) = Φ((τ − λ/sd₀)/√(1 + 4λ/sd₀²)). Both rates come from exact moments, with nothing fitted.
- **Measured, τ = 2.326** (96 trials × 40 noise draws; the impostor is the nearest gallery member, pose and light refitted):

| N̄ | FRR predicted | FRR measured | FAR predicted | FAR measured |
|---|---|---|---|---|
| 1 | 1.0% | 1.7 ± 0.4% | 98.9% | 98.0 ± 0.4% |
| 5 | 1.1% | 1.0 ± 0.3% | 97.5% | 97.2 ± 0.5% |
| 10 | 1.2% | 1.3 ± 0.4% | 92.5% | 92.6 ± 0.8% |
| 20 | 1.8% | 2.3 ± 0.5% | 74.1% | 73.5 ± 1.4% |
| 50 | 5.6% | 5.7 ± 0.7% | 25.6% | 26.2 ± 1.4% |

- **Why FRR rises with N̄:** the attached-shadow leftover grows in proportion to N̄ (Law 2).
- **The threshold test's limit:** it needs λ ≥ λ_open ≈ 5·sd₀ (about 200) before it rejects a look-alike at 1%. Law 6 says an ideal observer needs only λ ≥ 12.9. So the threshold test spends about 15× the photons. **It is not optimal.**

### The optimal law: Bayes with a reject option (Chow)

- **Rule:** name the most probable hypothesis only if its posterior is at least 1 − ε. No rule has lower error at the same abstention rate.
- **Scores:**
  - **Enrolled person i:** log p(x|i) = −Rᵢ/2, with every nuisance profiled.
  - **Stranger:** the Laplace evidence of the population face law. The identity is fitted to the image's own pixels, and the score is −R_s/2 − ψᵀPψ/2 + ½ log det P − ½ log det H, with H = JᵀW(I − P_B)J + P (light profiled out).
- **Common terms cancel:** every hypothesis is scored on one evidence set E, and only score differences enter the posterior. Light, exposure, gain and pose cost cancel between people. Only the remainders λ, set against the noise, decide.

**Synthesis lanes:**
- `L5_conf` = `(((x - 74) >> 31) - (x - (x + 1)))`: the posterior odds are at least 99:1.
- `OUT_abst`: authored after the lane-law cut. The whole lane hit the size wall at ladder 4, so it was posed as `OUT_abst_x` = `(4 ^ (x & 7))` declared as one operation. The result is `(1 & ((-(4 ^ ((x) & 7))) >> 31))`.

**Results:** see the end-to-end section below.

## LAW 6: limit

1. **Holds when:** pixels are independent, and the sensor law (Law 1) is exact.
2. **Equation:** take an ideal observer choosing between A (under its nuisance) and B (under nature's worst choice: light and pose refitted to A's image).
   - D_B = −Σ_p ln Σ_x √(P_A(x) P_B(x)), using the exact Poisson ⊛ Gaussian ⊛ rounding probabilities of each DN value.
   - **Any method's error** (equal priors) is at least ½(1 − √(1 − e^{−2D_B})).
   - **The likelihood-ratio test** achieves at most ½e^{−D_B}.
   - **N_min(ε)** is the smallest N̄ where the lower bound is at most ε. Below it, "too dark to tell", never a guess.
   - **Gaussian limit:** D_B = λ/8, so the gate is λ < 4 ln(1/(4ε(1 − ε))) = 12.92 at ε = 1%.
3. **Cancels:** nothing new. The bound is taken at the nuisances that make A and B most alike.
4. **Remains:** the photon count and the pair's remainder λ per photon.
5. **Breaks:**
   - correlated pixels (JPEG denoising), where the product bound no longer holds;
   - a model leftover larger than the identity signal (Law 2 breaks).
6. **Prediction:** the oracle's error lies between the bounds. **Measured:** in 96 of 96 pairs at N̄ = 1, 2, 5 and 10, it did.

| N̄ | median D_B | lower bound | oracle error | upper bound |
|---|---|---|---|---|
| 1 | 0.09 | 29.5% | 33.3% | 45.2% |
| 2 | 0.27 | 18.2% | 23.4% | 37.6% |
| 5 | 0.96 | 5.3% | 9.6% | 19.9% |
| 10 | 2.28 | 1.2% | 2.9% | 7.6% |

For the nearest pair in this gallery, N_min(1%) ≈ 10 photons per pixel.

**Synthesis lane:** `L6_dark` = `((x - x) - ((x - 207) >> 31))`.

---

## End to end (nothing given: pose, light, expression, exposure all unknown)

### Synthetic world (`recognize.py`, `optimal.py`; 336 probes, half strangers)

- **Threshold law:** never named an enrolled person wrongly, and named one stranger in 24 at 50 photons. It said "too dark" for everyone up to 10 photons per pixel.
- **Optimal law (Chow):** answers much more often, but is **not yet calibrated**. After two fixes (a stranger fit that only accepts improving steps, and pose profiling with the order-4 light terms), 4.5–14% of its answers are wrong, against a promise of 1% or less.

### Real faces: Extended Yale B, protocol A (train on 7 near-frontal photos, 38 people; recognition %)

| method | 20–25° | 35–51° | 60–77° | 85–130° |
|---|---|---|---|---|
| **superset: global light law + lane-authored local law + synthesis gate** | 100 | 100 | **99.05** | **98.46** |
| local law alone (synthesis gradient sectors, 192×168) | 100 | 100 | 96.96 | 95.80 |
| global light law alone (9 terms where lit, ambient in cast shadow; synthesis `L2_cast` lane) | 100 | 100 | 95.06 | 77.73 |
| Gradientfaces (my run, 192×168) | 100 | 100 | 97.34 | 96.36 |
| Tan–Triggs (my run, 192×168) | 99.56 | 100 | 97.15 | 95.94 |
| VGG-Face FC7, published (Ghazi & Ekenel 2016) | 100 | 92.32 | 52.44 | 18.28 |

- **The gate:** it changed 182 answers, fixed 174, and overturned 5 the global law had right.
- **Not sealed:** I explored variants on these same images, so this result may be optimistic.

### Sealed test: CMU PIE illumination (`pie_protocol.txt`, SHA-256 `483ad646…`, written before running)

- **Protocol:** 68 people, 1 frontal training image, 20 test lights, 1,360 probes, 32×32.
- **Ours (synthesis local law, frozen from Yale B):** 100.00.
- **Gradientfaces, Weberfaces, Tan–Triggs:** 100.00 each.
- **VGG-Face, published, on another crop:** 93.16 (FC6) and 92.87 (FC7).
- **Reading:** the sealed rule reads "No. 1", but the test is at its ceiling and cannot separate the leading methods.

## What the lane-authored

12 lanes. Each written independently.000–0.058 s per lane. 

| lane | law | expression | inputs proven |
|---|---|---|---|
| `L1_photon` | 1 | `(x - 64)` | 1,024 |
| `L1_sat` | 1 | `((x + 1) >> 10)` | 1,024 |
| `L2_census` | 2 | `(((255 - x) >> 8) - (((x >> 8) - x) >> 8))` | 65,536 |
| `L4_rigid` | 4 | `((x - x) - ((x - 18) >> 31))` | 32 |
| `L5_accept` | 5 | `((x - x) - ((x - 38) >> 31))` | 2,048 |
| `L5_open` | 5 | `((x - x) - ((x - 256) >> 31))` | 4,096 |
| `L5_conf` | 5 (optimal) | `(((x - 74) >> 31) - (x - (x + 1)))` | 4,096 |
| `L6_dark` | 6 | `((x - x) - ((x - 207) >> 31))` | 4,096 |
| `OUT_any3` | decision | `(1 & ((-(x & 7)) >> 31))` | 32 |
| `OUT_both` | decision | `((x & (x >> 1)) >> 3)` | 32 |
| `OUT_abst_x` | decision (optimal) | `(4 ^ (x & 7))` | 16 |
| `OUT_abst` | decision (optimal) | `(1 & ((-(4 ^ ((x) & 7))) >> 31))` | 16 |

All are correct over their whole domains, in both the engine's evaluator and compiled C. The decision glue (masks and ORs) matches the decision table on all 32 states.

## Limits of this evidence

- The world is synthetic, grayscale and Lambertian, with a black background.
- The gallery has 10 people, drawn from one population law. Strangers are drawn from that same law, so the stranger hypothesis is told the true population.
- Every photon threshold here (N_min ≈ 10) belongs to this world's identity spread. Real faces differ by more, so they would separate at fewer photons. That has not been measured.
- The renderer is my code. The world and the recogniser use independent surface samplings, but the same physics.


### Sealed dark test: CMU PIE at 1–10 photons per pixel (`pie_dark_protocol.txt`, SHA-256 `d74e2eaa…`)

**Design.** The dark law was designed on Yale B only.
- **Votes:** gradient-direction sectors at three scales, each pixel weighted by its significance against Law 1's exact noise.
- **Table:** the data's own likelihood ratios, authored as integer lanes as 48 lanes plus a gate. Proven, with 0 wrong on 248 rows.

**Run.** Frozen, then run once on PIE.
- **Enrolment:** 1 clean frontal photo per person.
- **Probes:** 4,080 per photon level, forced-choice recognition %.

| method | 1 photon | 2 | 5 | 10 |
|---|---|---|---|---|
| Weberfaces (my run) | **64.83** | **97.57** | **100.00** | **100.00** |
| **ours: synthesis dark law** | 64.73 | 94.39 | 99.58 | 99.98 |
| Gradientfaces (my run) | 60.39 | 92.08 | 99.61 | 99.98 |
| local law, clean, frozen | 55.61 | 89.80 | 99.41 | 99.88 |
| Tan–Triggs (my run) | 27.97 | 73.21 | 99.75 | 100.00 |

**Verdict under the sealed rules:**
- **No. 1 in the dark: NO.** Weberfaces wins at every level.
- **Never guesses: YES.** 0 wrong among the answers it gave.
- **But the gate was far too cautious.** It answered 0%, 0%, 7.6% and 25% of the time at 1, 2, 5 and 10 photons. It was calibrated on Yale B, which is much harder.
