#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 TAG" >&2; exit 2; }
tag="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository="${FRIDAY_RELEASE_REPOSITORY:-debpalash/friday}"
dist="$root/dist"

cd "$root"
scripts/check-release.sh --release "$tag"
mkdir -p "$dist"
find "$dist" -mindepth 1 -delete

source_name="friday-source-$tag.tar.gz"
source_path="$dist/$source_name"
auth=()
[[ -z "${GH_TOKEN:-}" ]] || auth=(-H "Authorization: Bearer $GH_TOKEN")
curl --fail --location --retry 5 --retry-all-errors \
  "${auth[@]}" --output "$source_path.part" \
  "https://codeload.github.com/$repository/tar.gz/$tag"
mv "$source_path.part" "$source_path"
source_sha256="$(sha256sum "$source_path" | cut -d' ' -f1)"

installer_name="friday-installer-$tag.sh"
installer_path="$dist/$installer_name"
cp install.sh "$installer_path"
python3 - "$installer_path" "$tag" "$source_sha256" <<'PY'
from pathlib import Path
import sys

path, tag, digest = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
body = path.read_text(encoding="utf-8")
replacements = {
    'local install_ref="${FRIDAY_INSTALL_REF:-main}"':
        f'local install_ref="${{FRIDAY_INSTALL_REF:-{tag}}}"',
    'local source_sha256="${FRIDAY_SOURCE_SHA256:-}"':
        f'local source_sha256="${{FRIDAY_SOURCE_SHA256:-{digest}}}"',
}
for old, new in replacements.items():
    if body.count(old) != 1:
        raise SystemExit(f"release installer marker changed: {old}")
    body = body.replace(old, new)
path.write_text(body, encoding="utf-8")
PY
chmod 755 "$installer_path"

(
  cd "$dist"
  sha256sum "$installer_name" "$source_name" > SHA256SUMS
)
printf 'built %s with source digest %s\n' "$installer_name" "$source_sha256"
