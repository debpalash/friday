# Friday intelligent assistant architecture

Friday is a supervised, local-first personal operator. The language model proposes
structured work; deterministic services own authorization, execution state, evidence,
completion, memory promotion, and rollback.

## Cognitive loop

1. Interpret conversation, questions, actions, corrections, and cancellations.
2. Build a versioned task contract with measurable criteria, required tools,
   permissions, freshness, and risk.
3. Build typed plan steps with expected observations and verifier names.
4. Apply supervised policy before each atomic action.
5. Execute the action and record its structured effects.
6. Independently verify the receipt and then the complete task contract.
7. Report only grounded results; failed or uncertain work cannot become `completed`.
8. Reflect and evolve only from tasks whose verification remains passed and undisputed.

The append-only event graph remains the audit source of truth. Mutable SQLite tables are
rebuildable projections, upgraded with forward-only migrations in
`friday_core/db_migrations.py`.

## Operator surface

- Live news plus public web search and page reading with attributable URLs.
- A visible Chromium profile isolated under owner-only `state/browser-profile`. Chromium is
  an exact persistent singleton process spec: startup uses the normal approval, resource,
  systemd-cgroup, compositor, and reconciliation path. Browser actions are accepted only while
  its sole loopback debugging listener is proven inside that same managed execution.
- Project-scoped files, local clipboard, desktop notifications, and local file opening.
- Exact machine-directory grants stored encrypted with independent inspect/list/read/write
  authority. Machine text writes are separately approved, atomic, hash-verified, idempotent,
  and rollbackable without overwriting a later human edit.
- Format-aware document reading for exact read-granted PDF, DOCX, ODT, EPUB, PPTX, and XLSX
  files. Every receipt binds the canonical path, grant, source byte count and hash, normalized
  text count and hash, extractor, format, and truncation state. This is extraction-only: Friday
  does not run macros, embedded programs, active content, or archive members.
- Exact-grant English OCR for PNG and JPEG images. Friday reports bounded dimensions, pixels,
  source/text hashes, truncation, and whether any text was detected. The receipt explicitly says
  `ocr_only`: it must not be used as evidence that Friday recognized objects, people, layouts,
  or other non-text visual content.
- Exact-grant single-image question answering is exposed only when the active hardware profile
  enables native vision and that exact model/runtime fingerprint has passed Friday's five-scene
  artifact-backed scorecard. The source image is converted to a bounded canonical PNG inside a
  networkless sandbox; only that ephemeral PNG reaches local Qwen. Durable records contain
  hashes and bounded provenance, never the image or model answer.
- Persistent one-shot and recurring reminders. The server owns the only delivery worker;
  transport failure leaves a reminder scheduled for retry, and successful delivery is
  confirmed before it is marked fired.
- Voice and text input, editable transcripts, task cards, cancellation, approval prompts,
  feedback, and encrypted corrected-audio evidence.

## External skill discovery

Friday can search Skills.sh only when no relevant active local skill supplies the needed
procedural knowledge. Search is read-only. A selected `owner/repository/skill` snapshot is
downloaded through the registry API, bounded by file and byte limits, assigned a local
SHA-256 content pin, and linked to the requesting task in the graph.

Import is never silent: it requires a durable approval, grants no executable permissions,
and must pass both Friday's prompt-injection/static checks and every available upstream
security audit at `SAFE`, `NONE`, or `LOW` risk. A missing, warning, or failed audit sends
the immutable candidate to quarantine instead of runtime context. Registry popularity is
discovery metadata, not evidence of safety or quality.

Install Friday's complete hash-locked application environment with:

```bash
uv pip sync --python venv/bin/python --require-hashes requirements/runtime.lock
```

Install Friday's fast local CPU speech backend and exact hash-pinned voice with:

```bash
venv/bin/python ops/install_piper_voice.py
venv/bin/python ops/install_omnivoice_model.py
```

The automatic 24 GiB reasoning profile uses this persistent Piper voice so spoken replies are
generated faster than playback without reducing Qwen's 200K context. CUDA speech profiles retain
the exact locally installed OmniVoice snapshot and reference-based voice support.

Install the single supervisor unit; it owns both Friday and its embedded reminder worker:

```bash
systemctl --user link "$PWD/ops/friday-supervisor.service"
systemctl --user enable --now friday-supervisor.service
```

## Privacy and autonomy

- Reversible local work runs automatically. Project writes, browser mutations,
  external-skill imports, remote reasoning, and core upgrades require a durable approval
  receipt; file-write approval displays the exact ephemeral content while persisting only
  its hash and metadata.
- Browser input and remote prompts are redacted from action and approval events.
- Corrected audio is retained only after a transcript edit, encrypted locally, and keyed
  through Secret Service (`secret-tool`). It can be inspected or deleted through the
  artifact API.
- Long-term preferences are promoted only from graph-linked, user-authored utterances; a node
  labeled as an utterance but authored by Friday cannot forge user evidence. Candidate text,
  JSON values, provenance sets, confidence, validity windows, and retrieval queries are bounded
  and finite. Expired claims are excluded, and a correction or duplicate refresh leaves one
  active value while retaining immutable supersession provenance.
- Memory retrieval uses normalized lexical terms, field-aware deterministic ranking, and a
  relevance-band filter for multi-term queries. When lexical retrieval is empty or weak, a
  bounded local semantic fallback may retrieve an active claim only if its similarity and
  nearest-neighbor margin both pass calibrated thresholds; ambiguous queries abstain. Raw FTS
  operators cannot enter the query grammar.
- Semantic memory uses the exact pinned `intfloat/multilingual-e5-small` revision on CPU with
  local-files-only loading. It performs no runtime model download, network request, GPU/vLLM
  allocation, or remote disclosure. Its hardware-adaptive batch size preserves GPU capacity for
  Qwen and speech; a missing or failed encoder safely leaves lexical retrieval available.
- Embeddings are a rebuildable private SQLite projection, bound to both the verified claim
  content hash and embedding-model fingerprint. Expiry and lifecycle gates run before indexing
  or retrieval, corrections remove superseded projections, and malformed cached vectors are
  repaired rather than trusted. These boundaries establish the scored fallback cases, not
  arbitrary semantic recall or general memory intelligence.
- Public web tools reject local, private, link-local, multicast, and special IP ranges.
- Dynamic capabilities run in Bubblewrap with read-only mounts, no network by default,
  private data storage, time limits, and output limits.
- Document archives are parsed in-process without unpacking to disk and are bounded by source,
  member, expanded-byte, XML, spine, output, and compression-ratio limits. Unsafe paths,
  duplicate members, encryption, unsupported compression, DTD/entity declarations, malformed
  XML, extension/signature mismatches, and source mutation fail closed. PDF extraction uses the
  pinned system Poppler tool inside Bubblewrap with no network, an exact descriptor-pinned
  input, one writable tmpfs output file, and CPU, address-space, process, descriptor, output,
  and wall-time limits. PDF support therefore requires `/usr/bin/pdftotext` and
  `/usr/bin/bwrap`; Friday reports the capability unavailable if either boundary is absent.
- PNG/JPEG headers are parsed before decoder dispatch to reject extension/signature disagreement,
  malformed JPEG marker inventories, dimensions above 20,000 per side, more than 40 million
  pixels, and sources above 32 MiB. Tesseract then runs with an exact descriptor-pinned input,
  no network, read-only system files, private tmpfs output, and wall-clock, CPU, address-space,
  process, descriptor, and output limits. A source mutation invalidates the receipt.
- Native scene understanding uses a separate 16 MiB source/output ceiling and the same
  descriptor-pinned, no-network image boundary. ImageMagick strips metadata, auto-orients, and
  downsizes to the profile's maximum side under CPU, memory, file, process, descriptor, and wall
  limits. It accepts one granted PNG or JPEG and cannot itself perform external actions.
- The browser control plane never trusts loopback reachability alone. Its exact listener inode,
  same-UID ownership, cgroup membership, durable singleton instance, systemd invocation, and
  process identities are revalidated around control operations; attachment and direct-spawn
  fallbacks are disabled in the server.
- Browser, research, news, Skills.sh, and optional remote-reasoning egress are connect-time
  public-only. A bounded local SOCKS5 proxy owns browser DNS, while every built-in HTTP client
  rejects mixed/private/special answer sets, pins and rechecks a numeric peer, leaves TLS
  end-to-end, and revalidates every redirect. HTTPS downgrade, credential-bearing redirects,
  POST redirects, and unbounded messages fail closed. Chromium also has direct DNS, proxy
  bypass, QUIC, and non-proxied WebRTC UDP disabled.
- Local Qwen traffic is constrained to exact numeric loopback HTTP under `/v1`. Its OpenAI SDK,
  authenticated tokenization probe, and control-plane health probes ignore ambient proxies and
  reject redirects; supervisor probes additionally prove the expected listener/process owner.
- Hardware/model/TTS assumptions are resolved once by the supervisor and published as a
  non-secret runtime manifest. Explicit homogeneous multi-GPU profiles bind every physical
  model rank, tensor-parallel degree, per-rank memory envelope, and isolated speech placement;
  see `docs/runtime-profiles.md`.
- Durable actions are admitted against profile-derived CPU, RAM, concurrency, network, and
  per-GPU budgets before an action receipt starts. Reservations share the step transaction,
  heartbeat, completion, and crash-recovery fence.

## Remote reasoning

Remote reasoning is absent from the model's tool schema until all three variables exist:

```bash
FRIDAY_REMOTE_BASE_URL=https://provider.example/v1
FRIDAY_REMOTE_MODEL=model-name
FRIDAY_REMOTE_API_KEY=secret
```

The local Qwen model remains the default. A remote prompt is redacted, shown in an approval
card, hashed into a disclosure receipt, and sent only after approval through the shared pinned
public-HTTPS transport. Remote HTTP and redirects are not accepted.

## Verification

Live and stateful evaluation commands resolve the installer-generated private
configuration before selecting the active state root, model credential, and
application assets. Symlinked, non-owner, group-readable, world-readable,
oversized, malformed, non-loopback, or non-finite runtime inputs fail closed.

Run unit/integration tests and the versioned capability suite:

```bash
venv/bin/python -m unittest discover -s tests -v
venv/bin/python ops/run_cognitive_evals.py
```

Run the separate deterministic long-term-memory scorecard with:

```bash
venv/bin/python ops/run_memory_evals.py
```

Its seven synthetic cases use disposable databases for all candidate memories and journal only
aggregate results plus observation hashes. They cover shared-term distractor precision at one,
correction propagation, stale rejection, source-author forgery, hostile-query containment,
duplicate refresh, and singular/plural recall. The result does not measure open-domain semantic
retrieval, multilingual morphology, or long-horizon usefulness.

Install the exact hash-pinned local embedding checkpoint, then run the separate semantic-memory
scorecard with:

```bash
venv/bin/python ops/install_embedding_model.py
venv/bin/python ops/run_semantic_memory_evals.py
```

Its eight disposable-database cases cover lexically disjoint English recall, Spanish, Hindi, and
German retrieval, unrelated-query abstention, expiry, correction propagation with stale-vector
removal, and a bounded 129-claim corpus. Passing them does not establish arbitrary paraphrase
quality, production-scale nearest-neighbor latency, or downstream long-horizon task lift.

Run the larger held-out scale and correction qualification separately:

```bash
venv/bin/python ops/run_semantic_scale_evals.py
```

The scorecard grows a disposable graph through 128, 1,024, and 5,000 active
claims, then measures paraphrased and multilingual precision, recall,
irrelevant-query abstention, warm p50 and p95, oldest-claim recall, projection
completeness, and corrected-memory task lift. The real pinned CPU model passed
the version 1 gates with precision 1.000, recall 0.867, abstention 1.000, warm
p95 178.7 ms, and corrected-task lift 1.000. Those figures describe this exact
model and machine profile, not every future corpus or host.

Run the artifact-backed voice qualification while Friday is stopped:

```bash
venv/bin/python ops/run_voice_evals.py
```

It creates owner-only, disposable Piper WAV artifacts, sends them through the
real Parakeet path, and exercises the production echo gate and interruption
buffer. The version 1 run measured 3.9% word error rate, 7/8 exact utterances,
91.4 ms ASR p95, 68.9 ms TTS p95, and 95.6 ms first-response-audio p95. Every
playback frame and the 650 ms tail were rejected before VAD; barge-in fired at
220 ms and retained the speech prefix. The first-audio figure uses a declared
deterministic local reply stage, so it isolates the voice path and is not a
claim about model response latency. Generated audio is deleted after the run;
only aggregate measurements, backend identity, and hashes enter the graph.

Run the receipt-grounded recovered-project qualification with:

```bash
venv/bin/python ops/run_project_evals.py
```

The version 1 suite dispatches a multi-file project, abandons the first durable
attempt before execution, resumes it with a replacement worker, runs the
project's tests, and re-hashes every file. The measured run matched 3/3 files,
passed 3/3 tests, verified 5/5 action receipts, resumed the interrupted step as
attempt 2 in 55.8 ms, and observed no duplicate file effect. Its structured
user-visible completion is derived from the task state, receipts, attempt
journal, file hashes, and a fresh test run. Project content and test output live
only in disposable directories and do not enter the aggregate graph record.

Run the signed controller and managed-browser control-path scorecard with:

```bash
venv/bin/python ops/run_controller_browser_evals.py
```

It generates an ephemeral P-256 controller key, pairs it, reconstructs the auth
service, reconnects with a new signed challenge, rejects one exact browser
action, and separately authorizes one exact browser input. The approved action
passes through `WebOperator` with the managed-runtime identity checked before
and after the mutation. The approval is consumed by one effect, both sessions
fail immediately after controller revocation, and raw input plus bearer secrets
stay out of durable state. Browser transport is a deterministic CDP-shaped
fixture in this scorecard; an actual Chromium launch and stop are retained as a
separate clean-host installer rehearsal gate.

Run the four-family injected recovery scorecard with:

```bash
venv/bin/python ops/run_recovery_evals.py
```

It injects an early model loss, an abandoned dispatched worker attempt, a
replaced managed-browser runtime, and a deployment verification failure. The
version 1 run recovered 4/4 with 12.7 ms control-path p95. Model recovery used
the bounded 15-second backoff and returned to stable; the worker used attempt 2
with one effect; the browser rejected the replaced runtime before mutation;
and the deployment restored the original file bytes. Fixtures and raw command
output are removed before the aggregate result is recorded.

Run the separate live document/OCR reasoning scorecard while the local model is healthy:

```bash
venv/bin/python ops/run_document_reasoning_evals.py
```

On a qualifying native-vision profile, run the exact-fingerprint five-scene scorecard with:

```bash
venv/bin/python ops/run_native_vision_evals.py
```

Native-vision boot performs the same artifact-backed score automatically. The user-facing
scene-understanding tool remains absent from the model schema until a current 5/5 result exists.
The five deterministic, text-free cases cover count, color/shape binding, left/right position,
containment, and relative size. They prove only those cases and the bounded image path.

This scorecard materializes actual DOCX, XLSX, PDF, and PNG artifacts, runs Friday's bounded
archive, Poppler, or Tesseract path, gives only the resulting extracted context to the active
local Qwen runtime, and grades fixed answer terms plus verbatim evidence. It includes table
arithmetic, untrusted-document instruction resistance, and missing-evidence abstention. Raw
model answers are hashed rather than journaled, and each score binds the artifact/source/context
hashes, extractor, model identity, and active runtime fingerprint. It is a separate capability
family and never changes the capability-core denominator.

Run the held-out conversation scorecard only while the local model runtime is
already loaded. It grades voice and text output with exact constraints for
brevity, factual minimums, Markdown policy, repetition, and workflow ceremony;
it does not use a model as a judge or retain raw answers.

```bash
venv/bin/python ops/run_conversation_evals.py
```

Run the installed-server scenario scorecard while Friday is running:

```bash
~/.local/share/friday/current/venv/bin/python \
  ~/.local/share/friday/current/ops/run_live_assistant_evals.py
```

It connects to the real loopback-only TLS WebSocket with Friday's private local
CA and grades response content, visible event order, and hidden progress-cursor
movement. Its seven cases cover capability and runtime truth, stateful
correction, ambiguous requests, refusal to invent an unverified action,
receipt-backed project inspection, and live news source follow-up. The news
case depends on public network access. Raw answers are hashed rather than
stored in the result.

The v2 held-out suite has thirteen deterministic cases across intent routing, task contracts,
approval policy, receipt verification, false-completion prevention, exact read-only restart,
nonrepeatable reconciliation, memory provenance, public-network isolation, and hardware
scaling. Stateful scenarios use a fresh disposable database for each case; only the bounded
result record enters Friday's graph. Suite files are bounded regular files and reject symlinks,
non-finite data, duplicate cases, unknown scenario names, malformed inputs, and scenario
exceptions rather than converting them into a pass.

This score is evidence for those named boundaries, not a general-intelligence score. Live
paired-controller browser execution, broad real-world visual competence, and injected-failure
rollback rates remain separate measured gaps; they must not be inferred from a green core suite.
The document/OCR, voice, memory, project, and native-vision scorecards add narrow grounded
evidence, not a general-intelligence, general-vision, or universal speech claim.
