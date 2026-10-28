# Face unlock for any phone (mapped angles)

Face unlock with the phone's ordinary front camera and its screen as the light: no depth camera, no infrared, no neural
network. **The owner decides which angles open the phone**, maps each one, and adds more later with a password. Every frame
passes an eye-check law (are these really the face's eyes?) before two identity laws compare it with every mapped angle,
using integer table look-ups and sums (about 13,000 per mapped angle per frame).

**Where it stands: no stranger has got in through a misplaced eye since the eye-check law; {REAL_MATCH} of {REAL_FRAMES} real-photograph stranger frames still cleared a bar (one photograph, hard light from below), which the next version closes. A photograph of the owner can open it.**

## How it works

| Step | What happens |
|---|---|
| Enrol | The owner sets a password (kept only as a salted PBKDF2-SHA256 hash), then holds the phone each way they unlock it and taps *Map this angle* (about 2.5 s of frames per angle). Enrolment closes at *Done*. |
| Try | Each frame: the eye finder's candidate pairs (from the tracked and the detected face box) go to the eye-check law, which keeps the best pair or drops the frame. The kept frame is compared with every mapped angle by both identity laws (96×84 and 48×42); it opens only if it clears that angle's bar, then a short flash check runs. About 6–7 frames a second; the first frame that clears opens it. |
| Add an angle | After a NOT YOU, the owner's password turns that try's frames into one more mapped angle. Five wrong passwords lock this for five minutes. |
| Bars | Each mapped angle's bar is the best of {BUNDLE} rendered stranger tries compared with it, plus a margin (96×84: {M96}, 48×42: {M48}). The margin is the smallest that kept every held-out new person out at all 35 angles at once, so it holds whichever angles the owner maps. |

Each mapped angle holds only a few degrees, so the owner maps every way they really hold the phone, including the
dark: an angle added with the password in pitch darkness, lit by the screen alone, matched within 1–3 frames afterwards.
An angle the attacker does not know is an extra hurdle, though most people map "straight on".

Unlike Face ID it does not require the owner's eyes on the screen: it opened while the owner looked away. That is
convenient, and it is also a risk Face ID's attention check exists to stop (someone holding the phone to a sleeping
owner's face).

## Results

| Test | Result |
|---|---|
| Strangers, rendered lab faces, end to end through the phone server | **0 of 252** opened (closest 0.41 below the bar) |
| Strangers, real lab photographs against v9's bars ({REAL_OWNERS} enrolled owners; each person checked by the eye-check half-law that never counted them) | **{REAL_MATCH} of {REAL_FRAMES}** frames cleared a bar |
| Owner (simulation), at a mapped angle | {AT1}% of frames with 1 angle mapped, {AT9}% with 9 |
| Owner (rendered, end to end), 3 angles mapped | 12 of 12 at them, 19 of 24 at 5–10° off, 0 of 10 at 20° or more |
| An angle that failed, after it was added with the password | 0 of 10 before, 9 of 9 after |
| Owner on the phone (current protocol, real face) | {PHONE_OPEN} of {PHONE_TRIES} attempts opened; in {PHONE_RECOG} more it recognised the owner but the flash check saw movement |
| Apple Face ID, same owner, same light | opened {FID_OPEN} of {FID_TRIES} tries: well ahead of this lock |

## What it does not do

- **Photographs of the owner can open it.** A printed front-lit photo of the owner matched the owner's map 14 of 14 times
  in simulation, and a print moved about one pixel per flash passed the flash check 39 of 39 times. Do not rely on it
  where someone may hold up your photo.
- Live strangers have not yet tried it on the phone; the stranger numbers above are lab faces and lab photographs.
- Beyond about 30° of turn the eye finder loses one eye, so a side profile cannot open it.
- In the dark its only light is the screen: at low screen brightness it cannot see the eyes (Face ID brings its own
  infrared light and works at any brightness).
- It does not check that the owner is looking at the phone.
- The phone pipeline runs on a computer over the USB cable in this release; the phone build is the next step.

## Protocols

Trials are counted under hash-locked protocols written before the trials:
`protocol12.txt` (sha256 `{P12}`), `protocol11.txt` (sha256 `{P11}`) and `protocol10_addendum1.txt` (sha256 `{PA1}`). Protocol 10 itself
(sha256 `{P10}`) stays private.
