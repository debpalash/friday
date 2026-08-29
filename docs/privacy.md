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

## Local browser access

The installer binds Friday to loopback HTTPS. The browser UI has no account,
token, pairing step, or persistent authentication key. The local OS account is
the access boundary. Friday rejects non-loopback bind hosts and must not be
exposed directly to a LAN or the public internet.

## Removing data

Create a consistent, owner-only export without stopping Friday:

```text
friday export /path/to/new-export-directory
friday verify-export /path/to/new-export-directory
```

The export contains a SQLite snapshot and a canonical manifest with the format
version, graph schema version, byte size, SHA-256 digest, and row count for
every logical table. Export directories are mode `0700`; their files are mode
`0600`. Verification rejects symlinks, public permissions, malformed manifests,
hash or inventory differences, and failed SQLite integrity checks. The export
contains private user content and should be stored accordingly.

Preview deletion by conversation, task, artifact node, memory claim, or bounded
UTC time range. A preview returns counts and a selector hash, never the selected
content:

```text
friday delete conversation SESSION_ID
friday delete task TASK_ID
friday delete artifact NODE_ID
friday delete memory_claim CLAIM_ID
friday delete time_range --start 2026-08-01T00:00:00Z --end 2026-09-01T00:00:00Z
```

Stop Friday, repeat the command with `--confirm`, and inspect the receipt. The
confirmed operation builds a private replacement database, removes dependent
graph and projection records, retains a content-free hashed tombstone, checks
foreign keys and integrity, compacts free pages, and then atomically replaces
the database. A failed rewrite leaves the source database unchanged.

This is logical deletion at Friday's application and SQLite layers. Flash
translation layers, storage snapshots, backups, and forensic media recovery are
outside Friday's guarantees and must be handled by the owner or platform.

`friday uninstall` removes the application and services while preserving state,
models, skills, voices, and credentials. `friday uninstall --purge` removes the
exact Friday-owned roots too. Back up wanted data before purging.

Selective deletion is offline by design so a running projector cannot race the
replacement. Run `friday start` after a confirmed deletion succeeds.
