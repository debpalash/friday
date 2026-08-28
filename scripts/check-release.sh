#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

release_tag=""
if [[ "${1:-}" == "--release" ]]; then
  [[ $# -eq 2 ]] || { echo "usage: $0 [--release TAG]" >&2; exit 2; }
  release_tag="$2"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--release TAG]" >&2
  exit 2
fi

failures=0
fail() {
  printf 'release check failed: %s\n' "$*" >&2
  failures=$((failures + 1))
}

required=(
  README.md CHANGELOG.md CONTRIBUTING.md SECURITY.md SUPPORT.md
  CODE_OF_CONDUCT.md THIRD_PARTY.md VERSION .gitleaks.toml
  docs/README.md docs/architecture.md docs/privacy.md docs/releasing.md
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || fail "missing required file: $path"
done

version="$(tr -d '\r\n' < VERSION)"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]] \
  || fail "VERSION is not a semantic version: $version"

if [[ -n "$release_tag" ]]; then
  [[ -s LICENSE ]] || fail "LICENSE must exist before a tag can publish"
  [[ "$release_tag" == "v$version" ]] \
    || fail "tag $release_tag does not match VERSION v$version"
  [[ "$(git rev-parse "$release_tag^{commit}" 2>/dev/null || true)" == \
      "$(git rev-parse HEAD)" ]] \
    || fail "release tag does not resolve to HEAD"
fi

if git ls-files | grep -Eq '(^|/)(state|models|venv|capabilities|backups)/'; then
  fail "runtime state, models, environments, or generated capabilities are tracked"
fi
if git ls-files | grep -Eq '^skills/workflow-'; then
  fail "learned workflow skills are tracked"
fi
if git ls-files | grep -Eiq '\.(wav|mp3|flac|ogg|m4a|pt|pth|onnx|gguf|safetensors)$'; then
  fail "model, voice, or user media is tracked"
fi
if git ls-files | grep -Eiq '(^|/)(\.env($|\.)|id_rsa|credentials?|secrets?|tokens?|private[-_.]?key)'; then
  fail "a secret-shaped filename is tracked"
fi
maintainer_path='/home/'"pal/"
if git grep -n "$maintainer_path" -- . ':!docs/agi-roadmap.md' \
    >/dev/null 2>&1; then
  fail "a maintainer-specific home path remains in the release tree"
fi
if git grep -nE 'raw\.githubusercontent\.com/.*/main/install\.sh.*\|[[:space:]]*(ba)?sh' \
    -- '*.md' '*.sh' >/dev/null 2>&1; then
  fail "an unversioned curl-to-shell install command remains"
fi

while IFS= read -r reference; do
  [[ -z "$reference" ]] && continue
  value="${reference#*@}"
  value="${value%%[[:space:]#]*}"
  [[ "$value" =~ ^[0-9a-f]{40}$ ]] \
    || fail "GitHub Action is not pinned to a full commit: $reference"
done < <(git grep -hE '^[[:space:]]*(-[[:space:]]+)?uses:' -- .github/workflows)

bash -n install.sh ops/fridayctl ops/provision_qwen_runtime.sh \
  scripts/uninstall.sh scripts/build-release.sh scripts/check-release.sh \
  scripts/scan-secrets.sh \
  || fail "a shell entrypoint does not parse"
git diff --check || fail "Git reports whitespace errors"

if (( failures )); then
  printf '%d release-tree check(s) failed\n' "$failures" >&2
  exit 1
fi
printf 'release tree ready for %s\n' "${release_tag:-development}"
