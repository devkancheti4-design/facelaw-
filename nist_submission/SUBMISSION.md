# FRTE 1:1 submission: FACE LAW v2 (the full-light law)

Checked against NIST's 1:1 API 6.0.1 (2026-10-06), the General Evaluation Specifications 3.1, the validation kit
(github.com/usnistgov/frvt, commit 18c2f84), the participation agreement (2023-05-19), the encryption procedure and the
submission form, all read on 2026-10-28.

## Done (the software)

| Requirement | Where it stands |
|---|---|
| API: `FRVT_11::Interface`, all five functions, `getImplementation()`; frvt11.h API 6.0, frvt_structs.h 3.1 | `v2cpp/nist/facelaw11.cpp`; NIST's own test driver runs it (124 of 124 comparisons return Success; multi-face works) |
| The law | the full-light law: v2's Weber codes on the 96x84 crop, one placement; 21,176-byte templates; NIST-style test on Extended Yale B: 86.5% of owners accepted at 1 in 1,000 (best classic 84.0%) |
| Failed template still matchable; score -1 plus `VerifTemplateError` when a template failed | an 8-byte "failed" template; tested |
| Similarity never negative | the v2 score + 127 (always 0-254) |
| Eye coordinates per image (the SUBJECT's left eye is image-right) | reported |
| Binary only, Ubuntu 24.04.3, g++ 13.3, C++20, dynamically linkable, 64-bit | built in the Kaggle run, inside Ubuntu 24.04.3 |
| No standard libraries shipped; no image libraries | the library links libstdc++, libm and libc from the OS only |
| Single-threaded; safe across `fork()` | no threads; the validation counts threads |
| Quiet: nothing to stdout or stderr; no files read or written; no network | yes; the law is compiled in, and `config/` holds only a README |
| Fatal errors still return a code | every API call is wrapped (MemoryError / UnknownError / MatchError) |
| Stateless: same input, same output, whatever ran before | 25 of 25 templates identical in reverse order here; Kaggle repeats it on 653 |
| The same output on every machine | bit-exact with the reference; 38 self-test checksums the Kaggle build must reproduce |
| Time limits (median, one core, per 640x480): 2,000 ms per template, 0.1 ms per comparison | this Mac: 125 ms and 12.5 us; Kaggle's Xeon gives the real check |
| Folders `lib/ config/ doc/ validation/`; `doc/version.txt` (NIST publishes it) | written by the Kaggle run |

## Yours, in this order

1. **Organization.** NIST does not take anonymous or personal entries. You need an organization name, its website, its
   country of headquarters, and an email address at the organization's own domain for each person. NIST refuses gmail,
   yahoo, qq and the like, on both the agreement and the form.
2. **Provider name.** One word, e.g. your organization's short name. The library becomes `libfrvt_11_<provider>_000.so`,
   and the public results carry it. The first submission is `000`; each later one adds 1; at most one every 4 months.
3. **Run the validation on Kaggle** with `PROVIDER=<provider>` and `SEQ=000` (see `dist/facelaw_v2_nist.zip`, its
   README). Bring back `run.log`, `selftest.txt`, `timing.txt` and `statelessness.txt`; keep the archive
   `libfrvt_11_<provider>_000.v3.1.tar.gz`. Do not rename it.
4. **Your signing key** (once): `gpg --full-generate-key` (RSA and RSA, 3072 bits, does not expire, your name, the
   organization email, a strong passphrase). Never lose it: a new key needs a new agreement.
5. **The participation agreement**: https://pages.nist.gov/frvt/agreements/frvt_participation_agreement.pdf. Print,
   fill rows 1-10, sign, scan to PDF, email it to frvt@nist.gov from the organization address, with your public key
   (`for_frvt_<name>.asc` from step 6).
   - Rows 1-6, the Responsible Party: someone with authority to commit the organization (signature, date, name,
     title, email, organization name, website, country).
   - Rows 7-9, the Point of Contact: someone who knows the submission. This may be the same person.
   - Row 10: your key's fingerprint (step 6 prints it).
   - Row 11 is NIST's, already printed: `B907 5A04 A0D8 E18C 82AF 0999 5A18 F90B 6224 A80F`.
6. **Encrypt and sign**: `bash nist_submission/gpg_steps.sh <org email> libfrvt_11_<provider>_000.v3.1.tar.gz`.
   - It prints your fingerprint and exports your public key.
   - It downloads NIST's key and refuses to go on if its fingerprint is not the one on the agreement.
   - It writes `...tar.gz.gpg`, encrypted to NIST and signed by you, and prints its sha256.
7. **Upload** the `.gpg` file somewhere NIST can download it without signing in.
8. **The submission form**: https://pages.nist.gov/frvt/html/frvt_submission_form.html (a Google Form).
   - Page 1: email (organization), other developer emails, organization name, organization website, point of contact
     name, first-time participant: Yes.
   - The later pages ask for the submission itself (the download link); they open only after page 1 is filled.
9. **Subscribe** to the FRTE news list: an email to frvt-news+subscribe@list.nist.gov.

## What happens then

NIST re-runs the validation on its machines and compares it with yours. It times the library on a Xeon Gold 6140, runs the
tests, and publishes the results with the organization's name. When you quote them, NIST requires this sentence and its link:
"Results shown from NIST do not constitute an endorsement of any particular system, product, service, or company by NIST."
https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt-ongoing

## Before you send: what to expect

FACE LAW v2 is 33 KB of counted tables with no learned network. Its measurements are on lab faces under changing
light (Extended Yale B, CMU PIE). NIST's sets add age, pose, expression and other cameras, at false-match rates down to
one in a million. Expect it near the bottom of NIST's table. The value of submitting is an independent, public number
for a tiny, integer, CPU-only matcher, under your organization's name. The next attempt can come only four months later.
