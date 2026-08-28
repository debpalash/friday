# Contributing

Friday is a single-user Linux application with unusually sharp privacy and
execution boundaries. A useful contribution preserves those boundaries and
arrives with a focused test.

## Before opening a change

- Use an issue or discussion for changes that alter stored data, permissions,
  supported hardware, model choice, or installer behavior.
- Never commit credentials, conversation state, browser profiles, voice clips,
  generated skills, downloaded models, or real user documents.
- Keep model-controlled text out of shell commands, process identifiers,
  executable paths, and authorization decisions.
- Treat external pages, model output, generated capabilities, and imported
  skills as untrusted input.

## Local setup

Friday currently develops and tests against Python 3.12 on Linux x86_64.

```bash
uv venv --python 3.12 venv
uv pip sync --python venv/bin/python --require-hashes requirements/runtime.lock
venv/bin/python -m unittest discover -v
```

The full suite does not require a running Friday service, but some real-model,
audio, GPU, compositor, or browser checks skip when their local dependencies are
absent.

Run the release-tree checks too:

```bash
scripts/check-release.sh
scripts/scan-secrets.sh
bash -n install.sh ops/fridayctl ops/provision_qwen_runtime.sh \
  scripts/uninstall.sh scripts/build-release.sh scripts/check-release.sh \
  scripts/scan-secrets.sh
git diff --check
```

## Pull requests

A pull request should contain one coherent change. Include:

- the user-visible problem and the boundary it affects;
- tests for success, rejection, restart, and failure paths where relevant;
- migration behavior for any stored-state change;
- documentation updates for new environment variables, downloads, permissions,
  or network access;
- measured latency or memory evidence for runtime-profile changes.

Do not weaken an existing test to make a change pass. Do not include generated
model output as proof that an effect happened.

By submitting a contribution, you state that you have the right to submit it
under the repository's selected license.
