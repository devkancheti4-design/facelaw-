#!/bin/bash
# NIST FRTE 1:1 submission: the steps only YOU can do (your key, your passphrase), following NIST's "Encrypting Software
# for Transmission to NIST".  The only network use: downloading NIST's public key from nist.gov.  Nothing is sent.
#     bash gpg_steps.sh  you@your-organization.com  libfrvt_11_<provider>_000.v3.1.tar.gz
set -euo pipefail
EMAIL=${1:?"your ORGANIZATION email address (NIST refuses gmail, yahoo, qq and the like)"}
ARCHIVE=${2:?"the archive NIST's validation script produced"}
NIST_FPR=B9075A04A0D8E18C82AF09995A18F90B6224A80F       # printed on the participation agreement, row 11
NIST_KEY_URL=https://www.nist.gov/document/frvtongoinggpgtxt

case "$EMAIL" in
    *@gmail.com|*@googlemail.com|*@yahoo.*|*@qq.com|*@outlook.com|*@hotmail.com|*@icloud.com)
        echo "NIST will not accept $EMAIL. Use an address at your organization's own domain."; exit 1 ;;
esac
[ -f "$ARCHIVE" ] || { echo "no such archive: $ARCHIVE"; exit 1; }

echo "== 1. your signing key"
if ! gpg --list-secret-keys "$EMAIL" >/dev/null 2>&1; then
    echo "There is no secret key for $EMAIL yet. Create it once, interactively:"
    echo "    gpg --full-generate-key"
    echo "  choose: RSA and RSA, 3072 bits, does not expire, your real name, $EMAIL, a strong passphrase."
    echo "  Then run this script again.  Never lose this key: a new key means a new participation agreement."
    exit 1
fi
FPR=$(gpg --with-colons --fingerprint "$EMAIL" | awk -F: '/^fpr:/ {print $10; exit}')
echo "   your fingerprint (participation agreement, row 10):"
echo "   $(echo "$FPR" | sed 's/.\{4\}/& /g')"

echo "== 2. your public key, to email to frvt@nist.gov with the signed agreement"
gpg --armor --output "for_frvt_${EMAIL%%@*}.asc" --export "$EMAIL"
echo "   wrote for_frvt_${EMAIL%%@*}.asc"

echo "== 3. NIST's FRTE public key, checked against the fingerprint printed on the agreement"
curl -fsSL "$NIST_KEY_URL" -o nist_frvt_public_key.asc
gpg --import nist_frvt_public_key.asc 2>&1 | sed 's/^/   /'
if ! gpg --with-colons --fingerprint "$NIST_FPR" >/dev/null 2>&1; then
    echo "[STOP] the key downloaded from $NIST_KEY_URL does not have NIST's fingerprint $NIST_FPR. Do not encrypt: write to frvt@nist.gov."
    exit 1
fi
echo "   NIST's key fingerprint matches the agreement: $NIST_FPR"

echo "== 4. encrypt for NIST and sign with your key (you will be asked for your passphrase)"
# the recipient is named by its verified fingerprint, so trusting it for this one encryption is safe
gpg --default-key "$EMAIL" --trust-model always --output "$ARCHIVE.gpg" --encrypt --recipient "$NIST_FPR" --sign "$ARCHIVE"
echo "   wrote $ARCHIVE.gpg  ($(wc -c < "$ARCHIVE.gpg") bytes)"
echo "   sha256 $(shasum -a 256 "$ARCHIVE.gpg" | cut -d' ' -f1)"
echo "   check: $(gpg --list-packets "$ARCHIVE.gpg" 2>/dev/null | grep -c 'pubkey enc packet') encryption packet(s) for NIST's key"

echo
echo "Next (yours): put $ARCHIVE.gpg where NIST can download it without signing in, then fill the FRTE Submission Form:"
echo "  https://pages.nist.gov/frvt/html/frvt_submission_form.html"
