# Friday

Friday is a private local voice and text assistant that can reason over your
work, operate an explicitly approved desktop boundary, and keep its state on
your machine.

## Install

The current public target is Linux x86_64 with systemd and an NVIDIA GPU with
at least 22 GiB VRAM. The default model installation needs about 45 GiB free
disk and can take a while on the first run.

```bash
curl -fsSL https://raw.githubusercontent.com/debpalash/friday/main/install.sh | bash
```

For a local checkout with an already verified Qwen runtime:

```bash
./install.sh --local . --llm-root /path/to/qwen-runtime
```

The installer binds Friday to loopback, creates a private user service, installs
a desktop launcher, verifies every pinned assistant asset, and starts Friday
only after preflight succeeds. Code releases are versioned; state, models,
skills, voices, and credentials persist outside the release so a failed update
can roll back without touching personal data.

## Use

```text
friday open
friday status
friday doctor
friday logs
friday update
friday repair
friday stop
friday uninstall
```

`friday stop` stops both the interface and the supervised model process, freeing
GPU memory. `friday uninstall` preserves personal state and downloaded models;
`friday uninstall --purge` removes those exact Friday-owned roots as well.

## Data boundaries

| Data | Default location |
|---|---|
| Versioned application releases | `~/.local/share/friday/releases` |
| Models, skills, voices | `~/.local/share/friday/shared` |
| Private database, TLS identity, browser profile | `~/.local/state/friday` |
| Private configuration | `~/.config/friday/friday.env` |
| CLI | `~/.local/bin/friday` |

The UI listens on `127.0.0.1:8500` by default. LAN and public exposure are not
enabled by the installer.

See [docs/installer.md](docs/installer.md) for the transaction and recovery
contract, and [docs/runtime-profiles.md](docs/runtime-profiles.md) for hardware
selection.
