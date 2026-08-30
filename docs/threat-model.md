# Threat model

This checklist covers Friday `0.1.x-alpha` as a single-user local assistant. It
is an engineering threat model, not an independent penetration test.

## Security assumptions

The local OS user, kernel, filesystem permission model, systemd user manager,
desktop compositor, GPU driver, and browser cryptography are trusted. Root, a
compromised kernel, a malicious process with the same user's unrestricted file
access, or physical access to an unlocked session can defeat these boundaries.

The model, web pages, documents, skills, generated code, browser messages,
archives, and network responses are untrusted. Model text is never evidence of
an effect. Friday is not designed for multiple mutually distrusting users.

## Boundary checklist

| Boundary | Assets and attacks | Preventive controls | Verification and detection | Incident response |
|---|---|---|---|---|
| HTTPS and WebSocket control plane | Local process access, DNS rebinding, hostile Host/Origin, accidental network exposure | Hard loopback bind restriction, local CA, exact Host and Origin allowlists, no application credential to leak | Local control-plane server tests, frontend integration tests, architecture check | Stop Friday; inspect same-user processes and browser extensions; rotate TLS material only if transport keys were exposed |
| Browser UI | XSS, injected transcript/Markdown, background-speech admission, self-listening | Text-node rendering, no `innerHTML`, exact local approvals, addressed-command gate, browser voice isolation, microphone playback and 1.5-second tail gates | Frontend integration tests, CSP headers, voice scorecard, architecture check | Close the browser, stop Friday, inspect only redacted logs |
| Public network egress | SSRF, localhost/cloud metadata access, DNS rebinding, redirect escape, peer substitution | Public-only resolver, complete-answer validation, one DNS read, numeric pinned connect, peer check, bounded redirects and responses, SOCKS policy | `web_proxy.py`, `public_http.py`, DNS and adversarial tests, aggregate proxy status | Stop managed browser/proxy, revoke affected task, block domain externally, retain hashed receipt |
| Model endpoint | Credential disclosure, wrong model/listener, redirect/proxy capture, prompt injection, false completion | Loopback-only URL normalization, private key file, no ambient proxy trust, exact PID/listener/model/profile binding, typed tools and receipt verification | Supervisor calibration, wrong-key and listener tests, runtime receipt, process identity status | `friday stop`; rotate model API key; restore last-known-good profile; quarantine uncertain tasks |
| ASR and TTS | Background speech, voice replay, self-transcription loop, unlicensed likeness, malformed audio, GPU pressure | Pre-journal loudness and addressed-command gates, in-memory raw-audio window, correction-only encrypted persistence, playback echo gate, bounded utterance, pinned assets, resource admission | Voice transport and endpointing tests, voice v1 WER/latency/privacy scorecard, runtime backend receipt | Stop speech, delete corrected-audio artifact, remove voice profile, require rights review before distribution |
| Durable graph and private files | Data theft, corruption, schema confusion, partial deletion, unsafe downgrade | Mode-0700/0600 roots, SQLite transactions and foreign keys, encrypted exact payloads, forward-only schema, atomic export/deletion/compaction, no-symlink opens | Migration/restart tests, export manifest hashes, selective deletion tests, schema-15 compatibility contract | Stop service, copy private state, verify export/backup, restore through documented migration path; never force downgrade |
| Task, approval, and receipt authority | Stale approval, cross-task binding, duplicate effect, false completion, crash after dispatch | Exact task/step/argument/executor hashes, one-time local decisions, durable leases, idempotency class, independent verifier, explicit unknown-outcome reconciliation | Durable worker and SIGKILL tests, project eval, recovery eval | Cancel dependent suffix, retain unknown outcome, reconcile read-only postcondition, never replay consequential work blindly |
| Filesystem operator | Path traversal, symlink swap, special files, overwrite race, rollback over human edit | Descriptor-relative granted roots, no symlinks/special files, bounded I/O, atomic replacement, post-write hash, encrypted rollback journal | Machine-operator tests and recovery scorecard | Revoke grant, stop worker, restore only when current hash still matches Friday's write |
| Managed processes and desktop | PID reuse, process/browser replacement, cgroup escape, wrong window, stale compositor or Omarchy state, arbitrary desktop command execution | Curated specs, executable/start-time/cgroup identity, systemd limits, sandbox defaults, opaque window IDs, fixed Omarchy routes, exact command fingerprints, before/after runtime verification | Process, desktop, Omarchy, cleanup, boot, and adversarial replacement tests; readiness fails closed | Stop managed unit and Friday, quarantine capability, reconcile retained workload before releasing reservation |
| Documents and images | Prompt injection, XXE, path traversal, parser execution, zip/pixel bomb, TOCTOU | Non-executing bounded extraction, DTD/entity rejection, archive/member/ratio limits, predecode image limits, source hash before and after granted read | Document/image tests, adversarial malicious-document and archive-bomb scorecard | Revoke grant, delete fixture, preserve only bounded hashes and failure code |
| Skills and generated capabilities | Hostile instructions, permission widening, code substitution, supply-chain replacement | Bounded snapshots, static scan, independent audit requirement, immutable hash-bound versions, empty default permissions, sandbox, quarantine and explicit promotion | Skill registry, capability, evolution, and adversarial hostile-skill tests | Quarantine version, revoke permission/grant, retain provenance, inspect upstream pin before replacement |
| Installer and update | Moved tag, archive traversal/link/bomb, dependency substitution, partial switch, destructive uninstall | Exact source SHA-256, structural tar validation, hash-required locks, pinned assets, private transaction, atomic `current`, managed-file snapshot and rollback, safe-root uninstall | Installer lifecycle and verified-archive rehearsal, release check, Gitleaks, dependency review | Stop service; restore previous release/config/service; withdraw bad hash; never move a tag; preserve state unless explicit purge |
| Release repository | Secret/history exposure, unreviewed action changes, compromised workflow, mutable build input | Full-SHA Actions, full-history secret scan, required review files, source digest, tag/version check, provenance attestations when public | CI, CodeQL when available, release-candidate report, repository settings checklist | Keep private or withdraw release, rotate exposed credentials, purge history before public visibility, publish affected hashes |

## Abuse-case invariants

- A rejected, expired, replayed, wrong-origin, or wrong-key request executes no
  state-changing effect.
- An action with an uncertain post-dispatch outcome is not retried as though it
  had failed.
- Untrusted content can influence a proposal but cannot grant authority,
  approve itself, activate a skill, or satisfy its own verifier.
- Secrets and raw private payloads do not enter public status, evaluation
  output, or release reports.
- Shutdown drains durable settlement barriers and leaves the model unloaded.

## Residual risk and release gates

Same-user malware, compromised upstream artifacts whose pinned bytes were
approved, model behavior outside measured tasks, and implementation bugs remain
possible. The first public release still requires the owner license/asset
decisions, cross-device qualification for every advertised target, and an
independent penetration test. See [Roadmap closure ledger](roadmap-closure.md)
and [Security policy](../SECURITY.md).
