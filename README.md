<p align="center">
  <img src="assets/friday.svg" width="88" alt="Friday">
</p>

# Friday

Friday is a local-first voice and text assistant for one Linux workstation. It
runs speech recognition, a 27B Qwen checkpoint, speech synthesis, memory, and
approved desktop actions on your hardware.

> Friday is a developer preview. It supports a narrow Linux and NVIDIA setup,
> has not received an independent security audit, and should not be trusted with
> safety-critical work.

## What works

- Voice and text conversations through a loopback HTTPS interface
- Local Parakeet speech recognition and Piper or OmniVoice speech synthesis
- Durable tasks, corrections, reminders, preferences, and evidence in SQLite
- Exact approval prompts before state-changing browser or desktop actions
- Bounded file, process, browser, web research, and document tools
- Transactional install, update, repair, rollback, stop, and uninstall commands

Friday does not include user conversations, learned skills, credentials, model
weights, or cloned voices in this repository.

## Requirements

The supported public target is intentionally narrow:

| Requirement | Current support |
|---|---|
| Operating system | Linux x86_64 with a working systemd user session |
| GPU | NVIDIA CUDA GPU with at least 22 GiB VRAM |
| Disk | About 50 GiB free for the default runtime and model assets |
| Python | Python 3.12, provisioned into isolated environments |
| Host tools | `bash`, `bwrap`, `curl`, `git`, `openssl`, `patch`, `tar`, `systemctl` |

Other GPUs, macOS, Windows, containers, and multi-user hosts are not supported
by the installer yet.

## Install a release

Use a versioned installer asset. Do not pipe the mutable `main` branch into a
shell.

```bash
VERSION=v0.1.0-alpha.1
BASE="https://github.com/debpalash/friday/releases/download/$VERSION"
mkdir -p /tmp/friday-install
curl -fL "$BASE/friday-installer-$VERSION.sh" \
  -o "/tmp/friday-install/friday-installer-$VERSION.sh"
curl -fL "$BASE/SHA256SUMS" -o /tmp/friday-install/SHA256SUMS
(
  cd /tmp/friday-install
  sha256sum --check --ignore-missing SHA256SUMS
  bash "friday-installer-$VERSION.sh"
)
```

The release installer contains the exact source tag and source archive digest.
It stops before changing the active installation if platform, hardware, disk,
service, path, or asset checks fail.

For development from a trusted local checkout:

```bash
git clone https://github.com/debpalash/friday.git
cd friday
./install.sh --local . --build-venv
```

The first install downloads several pinned model and runtime assets. It can take
a while even on a fast connection.

## Operate Friday

```text
friday open       Open the local interface
friday status     Show service and model state
friday doctor     Run actionable diagnostics
friday logs       Follow local logs
friday update     Install the configured upstream ref transactionally
friday repair     Rebuild the active release transactionally
friday stop       Stop Friday and unload its model
friday uninstall Preserve personal data and downloaded models
```

`friday uninstall --purge` removes Friday-owned state and shared assets too.
The command prints the exact roots before deleting them.

## Data and network boundaries

The installer binds the interface to `127.0.0.1:8500`, stores private state
outside versioned releases, and does not enable LAN or public access.

| Activity | Network behavior |
|---|---|
| Conversation, ASR, model inference, TTS, memory | Local by default |
| Installation and update | Downloads pinned code, packages, and model assets |
| News, search, page reading, managed browser | Connects only after the user asks for that capability |
| Remote reasoning | Disabled unless explicitly configured and approved |
| Telemetry | None implemented |

Voice transcripts and task history persist locally. Raw microphone audio stays
in memory for up to ten minutes so a user can correct a transcript. Audio is
written only after a correction, encrypted with a desktop-keyring key, and can
be deleted separately. See [Privacy](docs/privacy.md) for the complete boundary.

## Architecture

```text
browser microphone and text
            |
            v
loopback HTTPS controller
            |
     ASR -> planner -> local Qwen -> response -> TTS
               |             |
               v             v
        policy and approval   SQLite graph and private state
               |
               v
       bounded tools and supervised processes
```

The supervisor owns the model lifecycle. The application owns conversation and
task state. Effectful tools require policy checks, an exact approved action, and
a verification receipt. Read [Architecture](docs/architecture.md) and
[Runtime profiles](docs/runtime-profiles.md) for details and current limits.

## Project documents

- [Install and rollback contract](docs/installer.md)
- [Privacy and data retention](docs/privacy.md)
- [Security policy](SECURITY.md)
- [Third-party components and model provenance](THIRD_PARTY.md)
- [Contributing](CONTRIBUTING.md)
- [Release process](docs/releasing.md)
- [Documentation index](docs/README.md)

Questions and setup problems belong in GitHub Discussions. Reproducible bugs
belong in Issues. Security reports must follow [SECURITY.md](SECURITY.md).
