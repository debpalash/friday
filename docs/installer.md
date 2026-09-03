# Installer contract

Friday's installer owns application lifecycle, not personal data.

## Transaction

1. Reject unsupported platform, GPU, disk, user-service, and unsafe-path states.
2. Lock the install root so install, repair, and update cannot overlap.
3. Stop the installed and legacy development services.
4. Materialize a new release without changing the active `current` link.
5. Build or reflink-copy an isolated Python environment.
6. Verify pinned ASR, Piper, OmniVoice, embedding, Qwen, and vLLM assets.
7. Write private per-user configuration, service, CLI, icon, and launcher.
8. Atomically replace `current` with the new release.
9. Run `friday doctor`; if starting, require authenticated model and HTTPS health.
10. On any failure or signal, restore the prior release, configuration, launcher,
    CLI, and service state.

State and large assets are persistent siblings of releases. Updating code does
not copy, replace, or delete the user's database, skills, credentials, voice
profiles, browser profile, or downloaded models.

## Supply-chain pins

- uv `0.12.1` standalone archive is SHA-256 pinned when a system uv is absent.
- Application and Qwen runtime dependencies are fully version locked and
  hash-required for Python 3.12 on Linux x86_64.
- The Qwen virtual environment is created relocatable and its vLLM launcher is
  executed after the final-path switch before the previous runtime is removed.
- Parakeet's upstream release archive and extracted inference files are SHA-256
  pinned and extracted with traversal/link/device rejection.
- Piper, OmniVoice, and multilingual embedding assets are revision, size, and
  SHA-256 pinned.
- The Qwen runtime checkout and model snapshot are exact-commit pinned. vLLM and
  its complete Python environment use a generated lock, then the runtime's own
  verifier must pass.

GitHub source archives can additionally be bound with `FRIDAY_SOURCE_SHA256`.
The published release installer embeds the exact tag and archive digest. An
already downloaded release-style tarball can be installed with `--archive` and
the mandatory `--source-sha256`; the installer verifies it before listing or
extracting any member and performs no source download.

## Recovery

An incomplete release is never referenced by `current`. After the switch, any
failed diagnostic or readiness check restores every installer-managed file and
the previous release. The failed release is removed, while personal data and
shared model assets remain.

`friday repair` repeats the transaction from the installed source. `friday
update` fetches the configured upstream ref. Both use the same lock and rollback
path as the first installation.

## Install bootstrap

`https://friday.palash.dev/install` (bash) and `/install.ps1` (PowerShell) are
static files in `site/public/`, published by `.github/workflows/pages.yml`.
They are conveniences, not a third install path:

1. The Linux bootstrap lists published releases through the GitHub API,
   takes the newest tag, downloads `friday-installer-<tag>.sh` and
   `SHA256SUMS` from that release, and refuses to run unless the checksum
   matches. `FRIDAY_VERSION` pins a tag. Arguments after `bash -s --` reach
   the installer unchanged. The whole script is wrapped in a function so a
   truncated download cannot execute.
2. The Windows bootstrap never installs on Windows. It locates an existing
   WSL 2 distribution, checks for x86_64 and `curl`, asks for confirmation,
   and runs the Linux bootstrap inside it. WSL 2 is not a qualified target;
   the installer's own platform checks decide.

The bootstraps inherit the trust of the site deployment rather than of a tag.
The manual steps in the README remain the fully verifiable path. Both files
are covered by `tests/test_bootstrap.py`.
