# Privacy and data retention

Friday is local-first, not network-free. Conversation and inference stay on the
workstation by default. Installation, explicit web tools, and an optional remote
reasoning provider use the network under the conditions below.

## Stored data

| Data | Storage | Retention |
|---|---|---|
| Transcripts, tasks, plans, receipts, approvals, corrections, reminders, and memories | `~/.local/state/friday/friday.db` | Until removed or the state root is purged |
| Current text session | `~/.local/state/friday/session.json` | Until replaced or removed |
| Corrected raw audio | Encrypted files under the private state root | Written only after a transcript correction; deletable separately |
| Voice reference audio | `~/.local/share/friday/shared/persona/voices` | Owner-managed |
| Browser profile | Private state root | Until removed or purged |
| Model and embedding weights | Shared data root | Reused across code updates |
| API keys, local model key, TLS keys | Private config, model runtime, desktop keyring, or private state | Until rotated, removed, or purged |
| Logs | Private state root | Owner-managed |

Raw microphone samples are buffered in memory for up to ten minutes to support
transcript correction. If the user corrects that transcript during the window,
Friday encrypts the samples with AES-256-CTR plus HMAC-SHA256 using key material
stored in the desktop keyring. Otherwise the samples are dropped without being
written. The correction endpoint exposes a separate delete operation for the
encrypted artifact.

Transcripts and model responses are stored as conversation and task history.
They are not equivalent to ephemeral audio.

## Network access

Friday implements no telemetry or analytics sender.

The installer contacts GitHub, PyPI, Hugging Face, and model release hosts to
fetch pinned code, packages, and model assets. These services receive ordinary
connection metadata such as IP address and request headers.

At runtime, the local model endpoint is restricted to explicit loopback HTTP.
The following capabilities can reach public hosts when the user asks for them:

- news retrieval from Google News RSS;
- search through DuckDuckGo and reads of selected public pages;
- the dedicated managed browser;
- Skills.sh search and approved skill import;
- a configured remote reasoning endpoint after disclosure approval.

Public HTTP helpers reject local, private, link-local, and special network
destinations before and after redirects. The managed browser uses a constrained
public proxy. These checks reduce SSRF risk but do not make external content
trustworthy.

## Local controllers

The installer binds Friday to loopback HTTPS. A browser controller receives a
short-lived pairing challenge and keeps its private signing key in browser
storage. Do not expose the service to a LAN or the public internet without a
separate deployment review.

## Removing data

`friday uninstall` removes the application and services while preserving state,
models, skills, voices, and credentials. `friday uninstall --purge` removes the
exact Friday-owned roots too. Back up wanted data before purging.

Friday does not currently provide selective deletion for every graph record.
That is a known product limitation for the alpha release.
