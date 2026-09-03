#!/usr/bin/env bash
set -Eeuo pipefail

# Point GitHub Pages at the Site workflow and attach the custom domain that
# serves https://friday.palash.dev/install and /install.ps1.
#
# Run once after the repository is public and .github/workflows/pages.yml has
# deployed at least once. Before running it, create the DNS record:
#
#   friday.palash.dev  CNAME  debpalash.github.io   (DNS only, not proxied)
#
# Requires an authenticated `gh` with admin rights on the repository.

usage() {
  echo "usage: $0 [--domain DOMAIN] [OWNER/REPO]" >&2
  exit 2
}

domain="friday.palash.dev"
repository=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) [[ $# -ge 2 ]] || usage; domain="$2"; shift ;;
    -h|--help) usage ;;
    -*) usage ;;
    *) [[ -z "$repository" ]] || usage; repository="$1" ;;
  esac
  shift
done
repository="${repository:-${FRIDAY_REPOSITORY:-debpalash/friday}}"
[[ "$domain" =~ ^[a-z0-9.-]+$ ]] || { echo "invalid domain: $domain" >&2; exit 2; }

gh auth status >/dev/null

visibility="$(gh repo view "$repository" --json visibility --jq .visibility)"
if [[ "$visibility" != "PUBLIC" ]]; then
  echo "$repository is $visibility; GitHub Pages on a free plan needs a public repository" >&2
  exit 1
fi

step() { printf '\n==> %s\n' "$*"; }

step "Pages site built by the Site workflow"
if gh api "repos/$repository/pages" >/dev/null 2>&1; then
  printf '{"build_type":"workflow"}' \
    | gh api --silent -X PUT "repos/$repository/pages" --input -
  echo "updated existing Pages site"
else
  printf '{"build_type":"workflow"}' \
    | gh api --silent -X POST "repos/$repository/pages" --input -
  echo "created Pages site"
fi

step "Custom domain $domain"
printf '{"cname":"%s"}' "$domain" \
  | gh api --silent -X PUT "repos/$repository/pages" --input -
if printf '{"https_enforced":true}' \
    | gh api --silent -X PUT "repos/$repository/pages" --input - 2>/dev/null; then
  echo "HTTPS enforced"
else
  echo "HTTPS enforcement is not available yet; GitHub issues the certificate" \
       "after DNS resolves. Rerun this script or enable it in Settings > Pages."
fi

step "Repository homepage"
gh api --silent -X PATCH "repos/$repository" -f homepage="https://$domain"

step "Verify"
gh api "repos/$repository/pages" --jq '
  "url: " + (.html_url // ""),
  "cname: " + (.cname // ""),
  "build: " + (.build_type // ""),
  "https enforced: " + ((.https_enforced // false) | tostring),
  "dns state: " + ((.protected_domain_state // "unknown") | tostring)'
echo
echo "Expected DNS: $domain CNAME ${repository%%/*}.github.io"
echo "Then check:   curl -fsSL https://$domain/install | head -n 3"
