#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="8.30.1"
ARCHIVE="gitleaks_${VERSION}_linux_x64.tar.gz"
EXPECTED="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
staging="$(mktemp -d)"

cleanup() {
  find "$staging" -depth -delete 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM

curl --fail --location --retry 5 --retry-all-errors \
  --output "$staging/$ARCHIVE" \
  "https://github.com/gitleaks/gitleaks/releases/download/v$VERSION/$ARCHIVE"
printf '%s  %s\n' "$EXPECTED" "$staging/$ARCHIVE" | sha256sum --check
tar -xzf "$staging/$ARCHIVE" -C "$staging" gitleaks
"$staging/gitleaks" git --redact --no-banner \
  --config "$ROOT/.gitleaks.toml" --log-opts=--all "$ROOT"
