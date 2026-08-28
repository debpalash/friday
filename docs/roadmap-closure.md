# Roadmap closure ledger

This ledger turns Friday's roadmap into release evidence. An item is closed
only when a repeatable test, scorecard, runtime receipt, or owner decision
exists. A passing narrow score is never treated as proof of a broader claim.

Status values:

- `closed`: the repository or qualified runtime contains repeatable evidence.
- `open`: Friday can close the item through local engineering and verification.
- `external`: closure requires another machine, independent reviewer, or legal
  decision.
- `owner`: a product decision must be made by the repository owner.

Publishing a release, adding a public tag, or changing repository visibility is
not authorized by this ledger.

## Verified capability baseline

| Area | Status | Evidence |
|---|---|---|
| Receipt-grounded model, ASR, speech, voice, and device identity | closed | Deterministic runtime receipts and server integration tests |
| Ordinary conversation fast lane | closed | No tool schema, task ceremony, or tokenization preflight; durable actions retain the full worker path |
| Voice and text output quality | closed | Three consecutive 8/8 live runs on one exact Qwen runtime fingerprint, with identical answer hashes |
| Capability-core governance | closed | 13/13 deterministic cases |
| Lexical memory retrieval | closed | 7/7 isolated cases |
| Multilingual semantic fallback | closed | 8/8 isolated cases on the pinned CPU model |
| Semantic scale and correction lift | closed | At 5,000 active claims: precision 1.000, recall 0.867, abstention 1.000, warm p50 66.6 ms, warm p95 178.7 ms, corrected-task lift 1.000, complete oldest-claim recall, and no retained superseded vectors |
| Document and OCR reasoning | closed | 5/5 live artifact-backed cases |
| Installed evaluation authority | closed | Commands resolve owner-only installer config, state, assets, loopback authority, and credentials; unsafe inputs fail closed |
| Runtime performance | closed | Five samples: 59.3 ms median first token, 106.2 decode tokens/s, and 22,546 MiB Qwen VRAM on the qualified RTX 4090 profile |
| Planned shutdown accounting | closed | Real installed stop leaves recovery `ready`, zero failures, and `expected_running=false`; Qwen is unloaded |
| Source-history privacy | closed | Voice clips, learned workflow snapshots, backup refs, and the synthetic credential finding were purged from every reachable branch; full-history Gitleaks passes |
| Private data lifecycle | closed | Owner-only complete export and offline deletion by conversation, task, artifact, memory claim, and time range; manifests, hashes, tombstones, dependency closure, compaction, and failure atomicity are tested |

The performance numbers above describe one local machine and one exact runtime
profile. They are not a cross-device claim.

## First-release acceptance scenarios

| Scenario | Status | Evidence |
|---|---|---|
| No fake progress | closed | Conversation and server integration tests reject workflow narration and require graph-backed progress |
| Verified completion | closed | Capability-core completion tests and task receipt verification |
| Crash recovery without duplicate effects | closed | Durable worker, SIGKILL, reconciliation, process, and desktop restart tests |
| Unsupported assistant text cannot become memory | closed | Memory provenance scorecard and source-author tests |
| User corrections supersede prior values | closed | Memory correction and semantic projection tests |
| Contradictory evidence remains disputed | closed | Memory governance tests |
| Skills require repeated verified success | closed | Evolution and skill quarantine tests |
| File prompt injection remains content | closed | Document, capability, and skill-registry injection tests |
| Broken self-edit rolls back | closed | Upgrade harness, deployment, and rollback tests |
| Status reports durable state | closed | Task, admission, supervisor, and server status tests |
| Service recovery preserves ownership and startup order | closed | Boot calibration, process binding, orphan, and planned-stop tests |
| Ordinary chat avoids background consolidation | closed | Bounded fast-lane integration tests and live conversation scorecard |

## Open local engineering

### Capability and runtime

- Add an artifact-backed ASR, speech synthesis, echo rejection, interruption,
  and end-to-end voice latency scorecard. Report p50 and p95 without retaining
  microphone content.
- Add long-horizon project scorecards that grade files, tests, recovery, and
  user-visible outcomes from receipts rather than assistant claims.
- Exercise the signed paired-controller and managed-browser workflow end to
  end, including approval rejection, exact approval use, reconnect, and
  controller revocation.
- Measure rollback rate and recovery time under injected model, worker,
  browser, and filesystem failures.

### Architecture and compatibility

- Split `server.py` into transport, conversation, speech, task orchestration,
  controller, and frontend boundaries without changing the security model.
- Separate frontend assets from the control-plane module so UI changes do not
  require reviewing unrelated execution code.
- Publish an alpha compatibility policy for graph schemas, tool receipts,
  runtime manifests, skills, and extension APIs.

### Installer and release rehearsal

- Run the complete install, first boot, voice turn, text turn, rejection,
  approved action, stop, restart, failed update rollback, uninstall, and
  state-preserving reinstall sequence from published-style assets on a clean
  supported host.
- Record installer duration, download size, disk use, rollback time, and every
  contacted host from that clean rehearsal.
- Add a release-candidate command that runs the exact local gates and emits one
  private, hash-bound rehearsal report without user content.

### Security hardening

- Add adversarial end-to-end tests for paired-controller theft, browser process
  replacement, DNS rebinding, malicious documents, archive bombs, hostile
  skills, and stale approval replay.
- Produce a threat-model checklist that maps every network, process, storage,
  key, model, and UI boundary to code, tests, and incident response.
- Complete dependency and model-license review against `THIRD_PARTY.md` for the
  exact release candidate.

## External and owner gates

| Gate | Status | Closure condition |
|---|---|---|
| Friday license and Piper GPL compatibility | owner | Written license decision reviewed for the shipped dependency graph |
| Product name, icon, screenshot, and voice-likeness rights | owner | Written approval for every public asset and mark |
| Independent penetration test | external | Review by a party independent of the implementation, with high-severity findings closed |
| Cross-device hardware matrix | external | Clean qualification on each advertised GPU, driver, and Linux target |
| Native-vision qualification | external | Exact 5/5 score and VRAM proof on a supported higher-memory profile |
| Public repository protections | external | After visibility changes: protected `main`, required CI and CodeQL, secret scanning, push protection, and private vulnerability reporting |
| Public release and announcement | owner | Explicit approval after every applicable gate above is closed |

## Closure order

1. Voice and long-horizon capability evidence.
2. Server boundary split and compatibility policy.
3. Paired-controller, browser, installer, and adversarial release rehearsal.
4. External hardware, legal, and independent security gates.
5. Owner-approved public release.
