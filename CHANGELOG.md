# Changelog

Friday uses [Semantic Versioning](https://semver.org/). The project is still in
an alpha series, so interfaces and stored schemas can change between minor
releases. Migrations must preserve user state.

## Unreleased

### Changed

- Voice input now requires `Friday, <request>` in one utterance. Unaddressed or
  low-level audio is discarded before journaling and model inference, browser
  voice isolation is requested when available, and reconnect noise is bounded.
- Removed browser tokens, pairing, bearer sessions, and signing keys. Friday now
  opens directly and rejects any non-loopback bind host.
- Pinned and installer-verified the WebSocket runtime backend required by live
  voice and text sessions.
- Split the control plane into explicit frontend, transport, conversation,
  voice-session, speech, and recovered-task boundaries.
- Added a forward-only alpha compatibility contract for graph schema 15,
  receipts, runtime manifests, skills, extensions, and the local UI protocol.
- Added verified local source-archive installation with pre-mutation digest
  checks and bounded no-link tar extraction.
- Made the provisioned Qwen environment relocatable and verify its launcher
  after the atomic runtime switch, with rollback on failure.

### Added

- Typed Omarchy desktop control with live status, exact approvals, packaged
  command identity binding, postcondition receipts, and reconciliation for
  themes, fonts, night light, idle policy, brightness, screenshots, and lock.
- Private export and selective deletion across conversations, tasks, artifacts,
  memory claims, and time ranges.
- Artifact-backed voice, semantic-scale, long-horizon project,
  controller-browser, injected-recovery, and adversarial scorecards.
- A private release-candidate command covering full tests, full-history secret
  scanning, live scorecards, exact dependency review, and installer rehearsal.
- A complete threat-model checklist and 322-package plus model engineering
  license inventory.

### Fixed

- Broken conversational outputs are now withheld and repaired once when the
  model returns an empty response, a lone fragment, an unfinished code fence,
  a token-limited completion, or an unverified external-action claim. Repeated
  short reply loops no longer pollute active context, ambiguous requests are
  clarified, and an empty reminder list verifies as a valid observation.
- Reconnecting browsers now fast-forward to the live progress cursor instead
  of replaying old task events with a new timestamp. Task progress renders once,
  diagnostic events use their recorded time, and log entries use separate lines.

## 0.1.0-alpha.1

Initial public developer preview.

### Included

- Local voice and text sessions
- Transactional per-user installation and rollback
- Supervised Qwen model lifecycle
- Parakeet ASR with Piper and OmniVoice speech backends
- Durable task planning, approval, execution, and receipt verification
- Local memory, transcript correction, reminders, and learned extensions
- Bounded desktop, process, browser, web, document, and image operations
