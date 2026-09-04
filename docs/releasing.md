# Release process

A public release changes the privacy, legal, and supply-chain exposure of the
whole repository. A green unit test run is necessary but not sufficient.

## Owner decisions before the first public release

1. Confirm the license. Friday ships under Apache-2.0 in `LICENSE`. The
   `piper-tts` runtime is GPL-3.0-or-later; the recorded disposition in
   `compliance/dependency-review-v1.json` and `THIRD_PARTY.md` must still hold
   for the shipped dependency graph.
2. Confirm Git history contains no voice clips, learned workflow data, or
   private state. `scripts/scan-secrets.sh` scans every commit for secrets;
   run `git log --all --name-only --pretty=format:` and review the unique
   paths for media, `skills/workflow-*`, `state/`, or `persona/voices/`
   entries. If any exist, purge them and force-push while the repository is
   still private.
3. Confirm the product name, icon, and public screenshots do not use third-party
   marks, private conversations, or unlicensed voice likenesses.

Do not change repository visibility until these three decisions are closed.

## Release gates

Run the accounted test suite first; it fails on any platform-conditional
skip that is not declared for the host, and Linux must skip nothing:

```bash
venv/bin/python scripts/run_tests.py --require-no-platform-skips
```

Run from a clean checkout with the reviewed Qwen environment available:

```bash
venv/bin/python ops/friday_release_candidate.py \
  --qwen-python "$FRIDAY_LLM_REPO/venv/bin/python" \
  --output "$XDG_STATE_HOME/friday/release-candidate.json"
```

The command refuses a dirty worktree, builds an exact HEAD source archive,
runs the release tree, full tests, full-history secret scan, all local
scorecards including live conversation, the 322-package license inventory, and
the clean synthetic installer lifecycle. It retains only hashes, counts,
timings, declared blockers, and privacy-safe summaries in a new mode-0600
report. It does not tag, publish, or change repository settings.

Then validate a clean install on a supported Linux x86_64 machine with at least
22 GiB VRAM. The test must cover install, first boot, voice and text turns,
approval rejection, an approved bounded action, stop, restart, update rollback,
uninstall, and reinstall with preserved state.

Review every changed dependency and model pin against `THIRD_PARTY.md`. Confirm
that release notes describe migrations, new network access, new permissions,
known regressions, and minimum hardware changes.

## Publish

The version in `VERSION` must match the tag with a `v` prefix.

```bash
VERSION=$(tr -d '\n' < VERSION)
git tag -s "v$VERSION" -m "Friday v$VERSION"
git push origin "v$VERSION"
```

Pushing the tag runs `.github/workflows/release.yml`. It:

1. runs the release-tree and installer lifecycle gates;
2. downloads the exact GitHub source archive for the tag;
3. embeds that tag and archive SHA-256 into a release installer;
4. publishes the installer, source archive, and `SHA256SUMS`;
5. creates GitHub build-provenance attestations;
6. creates a prerelease when the semantic version has a prerelease suffix.

Verify the attestation and perform a fresh install from the published assets
before announcing the release.

## Repository settings after visibility changes

- Require pull requests and the `CI` and `CodeQL` checks on `main`.
- Require signed commits or vigilant mode if that is the chosen contribution
  policy.
- Require immutable full-SHA action references.
- Enable Dependabot alerts, secret scanning, push protection, and private
  vulnerability reporting.
- Disable force pushes and branch deletion on `main`.
- Keep release creation limited to maintainers.

Private repositories on GitHub Free cannot enforce every branch rule. Apply and
verify the rules immediately after the repository becomes public:

```bash
scripts/apply-repo-protections.sh   # add --require-signatures if chosen
```

The script refuses to run against a private repository, enables Dependabot
alerts and security updates, private vulnerability reporting, secret scanning
with push protection, and creates or updates a ruleset on the default branch
that blocks force pushes and deletion, requires pull requests, and requires the
`verify / release-tree`, `verify / tests-linux-x86_64`, and `analyze` checks.
The macOS and Windows test columns run on every pull request but do not block
merges until their platforms are qualified. Review the result in the
repository settings afterwards.

## Install site

`.github/workflows/pages.yml` builds `site/` with Astro and deploys it to
GitHub Pages on every push to `main` that touches the site, assets, or
`VERSION`. It only runs on a public repository, and the workflow token cannot
create the Pages site itself. Once the repository is public:

1. Run `scripts/configure-install-site.sh`. It creates the Pages site in
   workflow mode, attaches the domain, and sets the repository homepage.
2. Trigger the `Site` workflow (`gh workflow run pages.yml --ref main`) or
   push a change under `site/`.
3. Create the DNS record `friday.palash.dev CNAME debpalash.github.io`
   (DNS only, not proxied, so GitHub can issue the certificate), then rerun
   the script to enforce HTTPS once the certificate exists.
4. Confirm both bootstraps are served as text:

   ```bash
   curl -fsSLI https://friday.palash.dev/install | grep -i content-type
   curl -fsSL https://friday.palash.dev/install | head -n 3
   curl -fsSL https://friday.palash.dev/install.ps1 | head -n 3
   ```

The bootstraps resolve the newest published release, so publishing a tag is
enough to update what they install. Nothing on the site needs to change per
release except the version shown on the page, which is read from `VERSION`.

## Incident response

If an installer or release asset is wrong, mark the release as withdrawn, stop
announcing its install command, publish the affected hashes, and cut a new tag.
Do not move an existing tag. The embedded source digest will deliberately reject
a moved tag whose archive bytes change.
