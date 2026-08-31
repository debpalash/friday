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

- A thirteen-family installed general-assistant benchmark for deductive and
  quantitative reasoning, contradiction detection, uncertainty, instruction
  resistance, constraints, planning, grounded inspection and research, action
  integrity, and real voice control.
- An installed-server assistant scorecard that exercises the verified local
  TLS WebSocket, conversation continuity, ambiguity handling, runtime truth,
  receipt-backed reads, live evidence follow-ups, and quiet task progress.
- A stateful conversation scorecard for referents, corrections, temporary
  constraints, chronological decisions, and ambiguity across related turns.
- A recorded turn contract that separates answering, clarification, live
  observation, external action, and durable memory before generation begins.
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

- Stored voice profiles now survive restart as the audible voice. A requested
  profile can replace Piper with local CPU OmniVoice after a synthesis test,
  without taking GPU memory from the language model. CPU NumPy verification
  output is accepted correctly, and reselecting the active profile no longer
  repeats the expensive synthesis check.
- Installed assistant evaluations now use a fresh nonpersistent conversation
  context while retaining real TLS transport, model, tools, receipts, and
  progress evidence. Repeated runs cannot inherit or overwrite the owner's
  saved chat history.
- Thin evidence and planning replies are repaired with their missing basis,
  explicit project inspections begin with bounded ranked source search, multi-source follow-ups
  remain bound to the URLs shown to the user, and capability answers name the
  unavailable native-scene boundary.
- The text conversation lane now defaults to 60 words and four sentences, and
  receipt-grounded two-source comparisons are repaired above a 190-word bound.
- Explicit requests to claim an external action without tools are now refused
  deterministically, without invoking the model or creating a task.
- Follow-up rewrites such as "make that shorter" now stay in the conversation
  lane when a prior answer supplies the target. Context-only updates cannot
  trigger unsolicited essays and are repaired against a strict word limit.
- Article-level follow-ups now resolve an ordinal or uniquely named recent
  source, read that exact URL, and answer from the page receipt. Ambiguous source
  references ask which result to open, and redirect shells are reported as
  insufficient evidence instead of being padded with headline inference.
- Repeated vacuous replies are removed from model context without deleting
  meaningful short user constraints. Read-only observations retain durable
  receipts and verification while their internal task lifecycle stays out of
  the conversation, progress poll, and reconnect UI, including model-selected
  project, machine, browser, clipboard, document, and desktop reads.
- Questions about Friday's active runtime and operational capabilities now use
  live receipts instead of model recall. Natural combined ASR/TTS wording is
  recognized, and Omarchy control is reported only when its verified broker is
  live.
- Selective deletion now checkpoints committed WAL pages before its physical
  source-integrity guard, avoiding false concurrent-write failures.
- Broken conversational outputs are now withheld and repaired once when the
  model returns an empty response, a lone fragment, an unfinished code fence,
  a token-limited completion, or an unverified external-action claim. Repeated
  short reply loops no longer pollute active context, ambiguous requests are
  clarified, exact headline-and-link requests render directly from verified
  receipts, and an empty reminder list verifies as a valid observation.
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
