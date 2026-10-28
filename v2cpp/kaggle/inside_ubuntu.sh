#!/bin/bash
# Runs in Ubuntu 24.04.3 (NIST's required OS): build the library, the self-test (must PASS), timing on NIST's
# validation images, then NIST's own run_validate_11.sh, which writes the submission archive.
# Needs W (the work folder), PKG (this bundle), PROVIDER, SEQ.
set -u
export DEBIAN_FRONTEND=noninteractive TERM=${TERM:-xterm}
OUT=$W/out
need=0
for c in g++ cmake make bc gawk lsb_release top; do command -v $c >/dev/null 2>&1 || need=1; done
if [ $need = 1 ]; then
    echo "   installing g++ cmake make bc gawk lsb-release procps ..."
    apt-get -o APT::Sandbox::User=root update -qq >/dev/null 2>&1
    apt-get -o APT::Sandbox::User=root install -y -qq --no-install-recommends g++ cmake make bc gawk sed grep lsb-release sudo procps ncurses-bin ca-certificates \
        >/dev/null 2>&1 || { echo "[ERROR] apt-get install failed"; exit 1; }
fi
echo "== inside: $(lsb_release -ds); $(g++ --version | head -1); $(cmake --version | head -1)"

S=$PKG/src
cd "$W/frvt/11" || exit 1
rm -rf lib config doc bin build validation libfrvt_11_*.tar.gz
mkdir -p lib config doc
LIB=lib/libfrvt_11_${PROVIDER}_${SEQ}.so
FLAGS="-O2 -ffp-contract=off -fwrapv -march=x86-64 -mtune=generic"
echo "== 3a. build $LIB"
g++ -std=c++20 $FLAGS -fPIC -shared -Wall -Wextra -I src/include -I ../common/src/include -I "$S" -o "$LIB" \
    "$S/facelaw2.cpp" "$S/vj.cpp" "$S/tmpl.cpp" "$S/facelaw11.cpp" || { echo "[ERROR] the library did not build"; exit 1; }
echo "   $(stat -c %s "$LIB") bytes; links: $(ldd "$LIB" | awk '{print $1}' | tr '\n' ' ')"
cp "$PKG/config_README.txt" config/README.txt
{
    echo "FACE LAW v2.0.0"
    echo "library: libfrvt_11_${PROVIDER}_${SEQ}.so"
    echo "source sha256: $(cat "$S"/facelaw2.h "$S"/facelaw2.cpp "$S"/vj.cpp "$S"/tmpl.h "$S"/tmpl.cpp "$S"/law_v2.inc "$S"/law_v2_full.inc "$S"/cascade_ff.inc "$S"/facelaw11.cpp | sha256sum | cut -c1-64)"
    echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(lsb_release -ds), $(g++ --version | head -1), flags: $FLAGS"
    echo "API: FRTE 1:1 6.0.1 (frvt11.h API 6.0), frvt_structs.h $(grep -o 'FRVT_STRUCTS_MAJOR_VERSION{[0-9]*}' ../common/src/include/frvt_structs.h | grep -o '[0-9]*').$(grep -o 'FRVT_STRUCTS_MINOR_VERSION{[0-9]*}' ../common/src/include/frvt_structs.h | grep -o '[0-9]*')"
    echo "threads: none; GPU: none; configuration files: none (the law is compiled in)"
} > doc/version.txt
cat doc/version.txt | sed 's/^/   /'

echo "== 3b. self-test (every stage on seeded random arrays vs the reference's checksums)"
g++ -std=c++17 $FLAGS -I "$S" -o "$W/selftest" "$S/selftest.cpp" "$S/facelaw2.cpp" "$S/vj.cpp" "$S/tmpl.cpp" || exit 1
"$W/selftest" "$S/selftest_expected.txt" > "$OUT/selftest.txt"; st=$?
tail -1 "$OUT/selftest.txt" | sed 's/^/   /'
[ $st = 0 ] || { echo "[ERROR] this machine's build differs from the reference: see out/selftest.txt"; exit 1; }

echo "== 3c. timing on NIST's validation images (one core)"
g++ -std=c++17 $FLAGS -I "$S" -o "$W/bench" "$S/bench.cpp" "$S/facelaw2.cpp" "$S/vj.cpp" "$S/tmpl.cpp" || exit 1
"$W/bench" $(ls ../common/images/face/*.ppm | head -80) | tee "$OUT/timing.txt" | sed 's/^/   /'

echo "== 4. NIST's run_validate_11.sh"
bash ./run_validate_11.sh 2>&1 | tee "$OUT/validation_console.txt"
cp -f libfrvt_11_*.tar.gz "$OUT/" 2>/dev/null && echo "   archive: $(ls libfrvt_11_*.tar.gz)" || echo "[ERROR] no submission archive was produced"
rm -rf "$OUT/validation"; cp -r validation "$OUT/validation" 2>/dev/null
if [ -f validation/match.log ]; then
    awk 'NR > 1 { n++; if ($4 == 0) ok++; s[n] = $3 } END { printf "   match.log: %d comparisons, %d return 0\n", n, ok }' validation/match.log
fi
echo "== 5. statelessness (General Evaluation Specifications 7.4): the verification templates again, in REVERSE order"
if [ -d validation/templates ] && [ -x bin/validate11 ]; then
    tac input/verif.txt > "$W/verif_rev.txt"; rm -rf "$W/rev_out" "$W/rev_t"; mkdir -p "$W/rev_out" "$W/rev_t"
    LD_LIBRARY_PATH=$(pwd)/lib bin/validate11 createTemplate -x verif -c config -o "$W/rev_out" -h rev -i "$W/verif_rev.txt" -t 1 -j "$W/rev_t" >/dev/null 2>&1
    n=0; d=0
    for f in "$W"/rev_t/*.template; do b=$(basename "$f"); [ -f "validation/templates/$b" ] || continue; n=$((n+1)); cmp -s "$f" "validation/templates/$b" || d=$((d+1)); done
    echo "   $n templates remade in reverse order, $d differ" | tee "$OUT/statelessness.txt"
fi
ls libfrvt_11_*.tar.gz >/dev/null 2>&1
