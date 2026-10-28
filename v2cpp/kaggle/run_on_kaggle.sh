#!/bin/bash
# FACE LAW v2 -> NIST FRTE 1:1 validation package, run from a Kaggle notebook (Settings: Internet ON).
# 1) NIST's 1:1 kit + the 1,319 validation images the 1:1 lists read (about 1.4 GB, from github.com/usnistgov/frvt)
# 2) an Ubuntu 24.04.3 userland (NIST's required OS) in a udocker container, unless the host already is 24.04.3
# 3) inside it: build the library, run the self-test (must PASS), time it, run NIST's run_validate_11.sh
# Everything lands in /kaggle/working/frte11/out.  PROVIDER and SEQ name the library: libfrvt_11_<PROVIDER>_<SEQ>.so
set -u
PROVIDER=${PROVIDER:-facelaw}
SEQ=${SEQ:-000}
PKG=$(cd "$(dirname "$0")" && pwd)
W=/kaggle/working/frte11
mkdir -p "$W/out"
exec > >(tee -a "$W/out/run.log") 2>&1
echo "== FACE LAW v2 -> FRTE 1:1 validation   $(date -u)   provider=$PROVIDER seq=$SEQ"
echo "== host"; (grep -E '^(PRETTY_NAME|VERSION)=' /etc/os-release; uname -m; nproc; grep -m1 'model name' /proc/cpuinfo; df -h /kaggle/working | tail -1) 2>&1

rm -rf "$W/pkg"; cp -r "$PKG" "$W/pkg"

echo "== 1. NIST's kit (sparse: 11/, common/src, common/scripts, the 1:1 validation images)"
cd "$W"
if [ ! -f frvt/11/run_validate_11.sh ]; then
    rm -rf frvt
    git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/usnistgov/frvt.git frvt || { echo "[ERROR] git clone failed (is Internet ON?)"; exit 1; }
    (cd frvt && git sparse-checkout set --no-cone /11/ /common/src/ /common/scripts/)
    (cd frvt/11/input && cat enroll.txt verif.txt enroll_multiface.txt short_enroll.txt verif_multiperson.txt) | tr ' \t' '\n\n' \
        | grep '^\.\./common/images/' | sed 's#^\.\./#/#' | sort -u > "$W/out/images_needed.txt"
    (printf '/11/\n/common/src/\n/common/scripts/\n'; cat "$W/out/images_needed.txt") | (cd frvt && git sparse-checkout set --no-cone --stdin)
fi
echo "   commit $(cd frvt && git rev-parse HEAD);  images present: $(ls frvt/common/images/face 2>/dev/null | wc -l) of $(wc -l < "$W/out/images_needed.txt");  $(du -sh frvt/common/images 2>/dev/null | cut -f1)"

echo "== 1b. the law on this host's own compiler (diagnostic: the self-test must PASS on any x86_64 Linux)"
if command -v g++ >/dev/null 2>&1; then
    S=$W/pkg/src; HF="-O2 -ffp-contract=off -fwrapv -march=x86-64 -mtune=generic"
    g++ -std=c++17 $HF -I "$S" -o "$W/selftest_host" "$S/selftest.cpp" "$S/facelaw2.cpp" "$S/vj.cpp" "$S/tmpl.cpp" \
        && "$W/selftest_host" "$S/selftest_expected.txt" > "$W/out/selftest_host.txt"; echo "   $(g++ --version | head -1): $(tail -1 "$W/out/selftest_host.txt" 2>/dev/null)"
    g++ -std=c++17 $HF -I "$S" -o "$W/bench_host" "$S/bench.cpp" "$S/facelaw2.cpp" "$S/vj.cpp" "$S/tmpl.cpp" \
        && "$W/bench_host" $(ls frvt/common/images/face/*.ppm 2>/dev/null | head -40) | sed 's/^/   /'
fi

echo "== 2. Ubuntu 24.04.3"
HOSTDESC=$(lsb_release -ds 2>/dev/null || (. /etc/os-release; echo "$PRETTY_NAME"))
if [ "$HOSTDESC" == "Ubuntu 24.04.3 LTS" ]; then
    echo "   the host is $HOSTDESC: running directly"
    W=$W PKG=$W/pkg PROVIDER=$PROVIDER SEQ=$SEQ bash "$W/pkg/inside_ubuntu.sh"
    rc=$?
else
    echo "   the host is '$HOSTDESC': using a udocker container"
    export UDOCKER_DIR=$W/.udocker
    pip install -q udocker >/dev/null 2>&1 || pip install udocker
    udocker --allow-root install >/dev/null 2>&1 || udocker --allow-root install
    FOUND=""
    for TAG in noble-20251013 noble-20260113 noble-20251001 noble-20250925 noble-20250910 noble-20260210.1 noble-20250805; do
        udocker --allow-root pull "ubuntu:$TAG" >/dev/null 2>&1 || { echo "   ubuntu:$TAG: pull failed"; continue; }
        udocker --allow-root rm u2404 >/dev/null 2>&1
        udocker --allow-root create --name=u2404 "ubuntu:$TAG" >/dev/null 2>&1 || continue
        V=$(udocker --allow-root run --nobanner u2404 /bin/cat /etc/os-release 2>/dev/null | grep '^VERSION=')
        echo "   ubuntu:$TAG -> $V"
        case "$V" in *24.04.3*) FOUND=$TAG; break ;; esac
    done
    if [ -z "$FOUND" ]; then echo "[ERROR] no Ubuntu 24.04.3 image could be run"; rc=1; else
        udocker --allow-root run --nobanner -v "$W:/w" --workdir=/w -e W=/w -e PKG=/w/pkg -e PROVIDER="$PROVIDER" -e SEQ="$SEQ" \
            -e TERM=xterm -e HOME=/root u2404 /bin/bash /w/pkg/inside_ubuntu.sh
        rc=$?
    fi
fi
echo "== done (exit $rc).  Download /kaggle/working/frte11/out from the notebook's Output panel:"
ls -la "$W/out"
exit $rc
