#!/usr/bin/env bash
set -Eeuo pipefail

# Apply the public-repository protections listed in docs/releasing.md.
#
# GitHub Free cannot enforce these rules on a private repository, so the
# script refuses to run until the repository is public. It is idempotent:
# rerunning it updates the existing ruleset instead of creating a second one.
#
# Requires an authenticated `gh` with admin rights on the repository.

usage() {
  echo "usage: $0 [--require-signatures] [OWNER/REPO]" >&2
  exit 2
}

require_signatures=0
repository=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --require-signatures) require_signatures=1 ;;
    -h|--help) usage ;;
    -*) usage ;;
    *) [[ -z "$repository" ]] || usage; repository="$1" ;;
  esac
  shift
done
repository="${repository:-${FRIDAY_REPOSITORY:-debpalash/friday}}"

gh auth status >/dev/null

visibility="$(gh repo view "$repository" --json visibility --jq .visibility)"
if [[ "$visibility" != "PUBLIC" ]]; then
  echo "$repository is $visibility; make it public before applying protections" >&2
  exit 1
fi
default_branch="$(gh repo view "$repository" \
  --json defaultBranchRef --jq .defaultBranchRef.name)"

step() { printf '\n==> %s\n' "$*"; }

step "Dependabot alerts and security updates"
gh api --silent -X PUT "repos/$repository/vulnerability-alerts"
gh api --silent -X PUT "repos/$repository/automated-security-fixes"

step "Private vulnerability reporting"
gh api --silent -X PUT "repos/$repository/private-vulnerability-reporting"

step "Secret scanning and push protection"
gh api --silent -X PATCH "repos/$repository" --input - <<'JSON'
{
  "security_and_analysis": {
    "secret_scanning": {"status": "enabled"},
    "secret_scanning_push_protection": {"status": "enabled"}
  }
}
JSON

step "Branch ruleset for $default_branch"
# Required checks are the reusable verify workflow's Linux jobs and CodeQL;
# the macOS and Windows columns report without blocking until qualified.
ruleset="$(python3 - "$require_signatures" <<'PY'
import json
import sys

rules = [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {
        "type": "pull_request",
        "parameters": {
            "required_approving_review_count": 0,
            "dismiss_stale_reviews_on_push": True,
            "require_code_owner_review": False,
            "require_last_push_approval": False,
            "required_review_thread_resolution": False,
        },
    },
    {
        "type": "required_status_checks",
        "parameters": {
            "strict_required_status_checks_policy": True,
            "required_status_checks": [
                {"context": "verify / release-tree"},
                {"context": "verify / tests-linux-x86_64"},
                {"context": "analyze"},
            ],
        },
    },
]
if sys.argv[1] == "1":
    rules.append({"type": "required_signatures"})
print(json.dumps({
    "name": "protected-default-branch",
    "target": "branch",
    "enforcement": "active",
    "bypass_actors": [],
    "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
    "rules": rules,
}))
PY
)"
existing_id="$(gh api "repos/$repository/rulesets" \
  --jq '.[] | select(.name == "protected-default-branch") | .id' | head -n 1)"
if [[ -n "$existing_id" ]]; then
  printf '%s' "$ruleset" \
    | gh api --silent -X PUT "repos/$repository/rulesets/$existing_id" --input -
  echo "updated ruleset $existing_id"
else
  printf '%s' "$ruleset" \
    | gh api --silent -X POST "repos/$repository/rulesets" --input -
  echo "created ruleset"
fi

step "Verify"
gh api "repos/$repository" --jq '
  "secret scanning: " + .security_and_analysis.secret_scanning.status,
  "push protection: " + .security_and_analysis.secret_scanning_push_protection.status'
gh api "repos/$repository/rulesets" \
  --jq '.[] | "ruleset: \(.name) (\(.enforcement))"'
echo "vulnerability alerts: $(gh api "repos/$repository/vulnerability-alerts" \
  >/dev/null 2>&1 && echo enabled || echo disabled)"
