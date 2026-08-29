# Release process

A public release changes the privacy, legal, and supply-chain exposure of the
whole repository. A green unit test run is necessary but not sufficient.

## Owner decisions before the first public release

1. Select a license for Friday. The `piper-tts` runtime is GPL-3.0-or-later, so
   compatibility must be reviewed before choosing a permissive Friday license.
2. Purge previously committed voice clips and learned workflow data from Git
   history, then force-push while the repository is still private.
3. Confirm the product name, icon, and public screenshots do not use third-party
   marks, private conversations, or unlicensed voice likenesses.

Do not change repository visibility until these three decisions are closed.

## Release gates

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
verify the rules immediately after the repository becomes public.

## Incident response

If an installer or release asset is wrong, mark the release as withdrawn, stop
announcing its install command, publish the affected hashes, and cut a new tag.
Do not move an existing tag. The embedded source digest will deliberately reject
a moved tag whose archive bytes change.
