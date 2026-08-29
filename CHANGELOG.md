# Changelog

Friday uses [Semantic Versioning](https://semver.org/). The project is still in
an alpha series, so interfaces and stored schemas can change between minor
releases. Migrations must preserve user state.

## Unreleased

### Changed

- Removed browser tokens, pairing, bearer sessions, and signing keys. Friday now
  opens directly and rejects any non-loopback bind host.
- Split the control plane into explicit frontend, transport, conversation,
  voice-session, speech, and recovered-task boundaries.
- Added a forward-only alpha compatibility contract for graph schema 15,
  receipts, runtime manifests, skills, extensions, and the local UI protocol.
- Added verified local source-archive installation with pre-mutation digest
  checks and bounded no-link tar extraction.
- Made the provisioned Qwen environment relocatable and verify its launcher
  after the atomic runtime switch, with rollback on failure.

### Added

- Private export and selective deletion across conversations, tasks, artifacts,
  memory claims, and time ranges.
- Artifact-backed voice, semantic-scale, long-horizon project,
  controller-browser, injected-recovery, and adversarial scorecards.
- A private release-candidate command covering full tests, full-history secret
  scanning, live scorecards, exact dependency review, and installer rehearsal.
- A complete threat-model checklist and 321-package plus model engineering
  license inventory.

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
