# Friday north star and capability roadmap

## North star

Evolve Friday into a highly capable, reliable, safe, hardware-adaptive general assistant
that can operate this machine end to end and scale cleanly to higher-spec systems.

“AGI” is a direction, not a completion label. Progress is accepted only through measured
capabilities, adversarial tests, durable evidence, and recovery drills. A fluent answer is
not evidence that an action worked.

## Capability ladder

### 1. Dependable autonomy

Move action execution out of the conversational/WebSocket coroutine into a durable worker.
Persist exact step arguments, dependencies, verifier, idempotency class, resource claims,
lease heartbeat, retry budget, and approval state. On restart:

- replay a verified receipt without repeating work;
- retry read-only/idempotent work;
- reconcile consequential work whose outcome is unknown;
- never infer completion from generated prose.

First acceptance drill: start a two-step read-only task, kill Friday during step one, and
prove restart resumes the exact remaining step without a duplicate action or shared-chat
history.

### 2. Safe machine competence

Add a permissioned OS operator with inspectable filesystem, process, application, and
desktop actions. Use project/user-scoped grants, previews for consequential mutations,
resource/time/output limits, rollback where possible, and receipt-specific verifiers.
Control-plane authentication and origin isolation are mandatory preconditions.

### 3. Resource-aware planning

Make every step declare `cpu`, `ram`, `vram`, accelerator, network, and latency needs. The
runtime profile becomes admission control for concurrent ASR, TTS, local inference, browser,
and maintenance work. Add boot canaries, measured memory calibration, bounded fallback, and
a last-known-good profile before increasing automatic context limits.

### 4. Evidence-grounded learning

Promote memories and procedures only from verified, undisputed outcomes. Measure retrieval
precision, correction latency, stale-memory rejection, skill success lift, and regression
rate. No autonomous model-weight updates without a separate dataset/evaluation lifecycle.

### 5. Generalization and multimodal competence

Expand task/evaluation families across coding, research, scheduling, browser workflows,
documents, vision, voice, and long-running projects. Prefer held-out end-to-end tasks and
fault injection over self-scored demonstrations.

## Required scorecard

Track at least:

- verified task success and false-completion rate;
- restart recovery and duplicate-side-effect rate;
- approval-policy violations and control-plane rejection tests;
- p50/p95 first-token, tool, ASR, and TTS latency;
- peak RAM/VRAM plus OOM/restart rate by runtime profile;
- memory retrieval precision and correction propagation;
- rollback success and mean time to recover;
- capability gain versus regression count for each self-change.

## Milestone log

### 2026-08-23 — adaptive/safe runtime foundation

- Added deterministic 24/32/48/64 GB+ profiles and multi-GPU LLM/TTS isolation.
- Unified supervisor, server, standalone voice client, and maintenance model configuration.
- Published a non-secret resolved runtime manifest and explicit overrides.
- Added per-install control-plane authentication with Host/Origin enforcement.
- Made graph/session data private, session saves atomic, and sensitive tool receipts redacted.
- Closed cross-candidate capability permission leakage and distinguished sandbox outages from
  candidate failures.
- Rejected core-upgrade files that could affect verification without entering deployment.
- Made generated core candidates review-only because candidate code cannot authoritatively
  attest to its own test run.
- Unified private-tool redaction across graph receipts, session snapshots, approvals, and
  logs; exact file content is shown ephemerally for approval but never journaled raw.
- Made reminder delivery transport-first and single-worker, leaving failed notifications
  scheduled for retry rather than recording false success.
- Live-canary validated the host RTX 4090 profile: Qwen and Friday healthy, authenticated
  model identity matched, both listeners bound to loopback, and the persistent user
  supervisor enabled. No executing/recovering tasks remained after the restart drill.

### 2026-08-23 — exact durable action execution

- Added encrypted, context-bound exact step arguments with redacted graph/API projections.
- Persisted ordered batches, dependency edges, stable operation keys, approval bindings,
  retry budgets, worker leases, heartbeats, attempts, and atomic receipt/step completion.
- Moved new conversational tool execution out of the response/WebSocket coroutine and into
  a durable worker; disconnect cancellation cannot abandon the underlying recorded action.
- Replaced model-generated restart replanning with exact recorded-step dispatch. Read-only
  work is retried under the same logical key; uncertain consequential work stops for
  reconciliation instead of being replayed.
- Proved two crash barriers with real subprocesses and `SIGKILL`: a dispatched read-only
  step resumes with attempts `2/1`, while a step killed after its success commit is not
  reinvoked and the journal remains `[step1, step2]`. SQLite integrity and foreign keys pass.
- Bound approvals to exact encrypted steps and hashes; a denial invalidates its dependent
  suffix so a stale approval cannot resurrect a cancelled batch.
- Restored the RTX 4090 reasoning-first profile to 200K KVarN context. A single 24 GB GPU
  moves TTS to CPU; larger or multi-GPU hosts retain accelerated speech without an 8K cap.

### 2026-08-24 — bounded machine operation and resource admission

- Added encrypted, revocable grants for exact existing machine directories with separate
  inspect/list/read/write permissions, expiries, and explicit sensitive-directory opt-in.
- Added a descriptor-relative filesystem broker that rejects symlink aliases and special
  files, bounds reads/writes, atomically replaces files, rereads hashes, and keeps encrypted
  mode-0600 rollback journals. Retries reuse the durable operation ID; rollback refuses to
  overwrite a later human edit.
- Routed machine grants, reads, writes, revocation, and rollback through durable task steps.
  Paths and contents stay encrypted/redacted at rest while exact consequential payloads are
  shown only in the authenticated approval event.
- Bound machine verifiers to approved arguments and operation IDs. Plausible-looking forged
  JSON receipts, contradictory hashes, cross-capability version substitution, mutated
  permissions, and post-approval batch mutation all fail before completion.
- Bound generated capabilities to the exact active name, numeric version, code hash, and
  permission set. Artifacts are private, atomically published, checked against authoritative
  SQLite code before every run, and evaluated against ephemeral data.
- Added hardware-derived CPU, RAM, concurrency, network, and per-physical-GPU action budgets
  to the stable runtime profile and fingerprint. Transient free-memory samples never change
  placement identity.
- Added durable resource leases acquired in the same transaction as dispatch. Static
  overclaims fail before `action.started`; temporary pressure automatically requeues; exact
  heartbeat/finish fencing and crash recovery release only the affected attempt. Leases are
  bound to the resolved runtime-profile fingerprint and latency class, and background/batch
  work preserves a final interactive execution and network lane.
- Made readiness force a fresh resource sample and fail closed on unavailable or stale
  telemetry; status reports only aggregate capacity and reservation data.
- Subtract active reservations from both the stable profile budget and live capacity,
  fail closed per missing CPU/RAM telemetry dimension, and bind lease heartbeat/release to
  the originating profile. A lost heartbeat now cancels execution, discards late success,
  and routes uncertain non-repeatable effects to reconciliation.
- Hardened supervisor process ownership with lifecycle locks, verified PID identity, pidfds,
  termination escalation, and concurrent-start exclusion.
- Kept Friday and its models stopped after the operator requested memory cleanup; verification
  remained offline. The complete suite reached 255 passing tests, with all four cognitive
  core evaluations passing.

### 2026-08-24 — durable workload and bounded process control

- Extended one-attempt resource leases into durable workload leases that survive a task-step
  handoff, remain bound to the originating hardware profile, and require an empty cgroup
  before terminal release. A reserved cleanup lane remains available under saturation.
- Added a curated process-spec registry and a shell-free systemd user-process broker. Exact
  executable identity, rendered arguments, sandbox policy, limits, approval binding, launch
  idempotency, cgroup ownership, and runtime identity are all revalidated at execution and
  reconciliation boundaries.
- Enforced CPU, RAM, swap, task-count, runtime, stop-timeout, output, and process-topology
  bounds. Bubblewrap profiles default to an isolated network and a read-only working tree;
  the first version deliberately fails closed for multi-target process trees.
- Added privacy-safe process receipts and authoritative outcome verification. Opaque instance
  IDs remain visible, while arguments, capability tokens, workload lease IDs, unit names,
  and cgroup paths stay out of public receipts and status responses.
- Made the managed-process reconciler part of readiness and status, so a dead lifecycle
  monitor cannot be presented as a healthy assistant.
- Revalidated the running RTX 4090 runtime at 200K context. Friday is listening on all local
  interfaces at port 8500; direct LAN access additionally requires a subnet-scoped host
  firewall rule.
- Passed 302 repository tests, all four cognitive-core evaluations, and live systemd/bwrap
  launch, identity, termination, natural-exit, cgroup, and resource-release canaries.

Next milestone: add curated application and desktop-operation profiles on top of the same
broker, automate cleanup of inert retained units, and implement boot calibration plus a
last-known-good degradation ladder before expanding autonomous task families.

### 2026-08-24 — bounded desktop control and calibrated boot recovery

- Added a version-pinned Hyprland desktop broker that accepts only one active, unlocked,
  local Wayland seat and an exact user-service/compositor/process/socket identity. It lists
  windows through HMAC-derived opaque IDs and safe application labels; titles, PIDs, paths,
  compositor addresses, classes, and session identifiers never enter model-visible receipts.
- Added exact-approved focus and graceful close operations. The durable binding covers the
  desktop session, application/process runtime, workspace, operation, and arguments; execution
  rebinds before dispatch and independently verifies the resulting desktop state. Interrupted
  focus/close actions require reconciliation and are never replayed blindly after a crash.
- Made desktop operation adaptive: `auto` and headless environments degrade the capability
  without crash-looping Friday, while `required` mode participates in readiness. Unsupported
  compositor versions and ambiguous sessions fail closed.
- Added authenticated cold-boot calibration with an exact no-user-data tokenization proof,
  wrong-key rejection, proxy/redirect refusal, and listener-inode ownership bound to the exact
  Qwen PID, UID, start time, boot, namespace, and loopback socket before and after credentialed
  probes. Active profile publication remains pending until all proofs pass.
- Added a private runtime-bound probation record, a hardware/family/profile-bound stable
  last-known-good record, a maximum-three-candidate monotonic degradation ladder, and boot-ID
  monotonic exponential recovery backoff. Explicit overrides disable automatic fallback.
- Made changed pinned process specs degrade unavailable without crashing or silently repinning
  durable authority. The running RTX 4090 profile remains reasoning-first at 200K context with
  CPU TTS.
- Passed 368 repository tests, all four cognitive-core evaluations, a live privacy-safe desktop
  read canary, and exact before/after proof that the full suite did not mutate live supervisor
  state.

Next milestone: add session-authorized curated application launch, automatic cleanup of inert
managed units, explicit uncertain-action reconciliation, empirical KV/VRAM calibration, and a
paired/TLS LAN control plane before expanding autonomous task families.

### 2026-08-24 — exact uncertain-action reconciliation and secret-free LAN bootstrap

- Added a durable reconciliation queue for consequential actions that crossed a dispatch
  boundary without an authoritative receipt. Exact task, batch, step, action, attempt,
  argument, idempotency, and executor-binding tuples are compare-and-swap fenced; unknown
  actions are never replayed or converted into success from caller-supplied evidence.
- Added read-only desktop and managed-process postcondition probes. Desktop absence now
  requires a complete mapped/unmapped client inventory and the original session identity;
  process launch/termination requires exact durable lifecycle and cgroup evidence. Known
  launch failure is distinguished from an unknown effect.
- Made raw compositor, systemd, process identity, admission, and receipt-projection failures
  after an effect boundary explicitly `outcome_unknown`. A failed durable settlement now
  recovers under the exact claim fence or stops the authoritative worker so readiness forces
  replacement.
- Preserved explicit unknown-action abandonment across the acknowledgement/restart crash
  window. Cancellation races cannot enqueue a dependent suffix, cross-wired candidates are
  rejected atomically, and reconciliation threads drain before worker shutdown.
- Removed the installation control token from the unauthenticated UI document. The LAN shell
  now requires explicit token entry, keeps it at most in tab-scoped session storage, attaches
  it only to same-origin requests, re-locks on authorization failure, and visibly warns when
  a non-loopback connection is plaintext.
- Passed 429 repository tests with exact before/after proof that the live database, session,
  control token, runtime fingerprint, and resolved runtime manifest were not mutated.

Next milestone: add crash-safe cleanup for retained terminal systemd units, then paired TLS
and expiring per-controller sessions/approvals. Curated application launch follows that trust
boundary; empirical KV/VRAM calibration and broader held-out task families remain next.

### 2026-08-24 — crash-safe retained-unit cleanup and multi-starter migration fencing

- Added a durable terminal-unit cleanup journal with transactional terminal-state triggers and
  an idempotent v8 backfill. Expiring database claims survive crashes before or after the
  backend effect, reject stale claimant settlement, and serialize independent cleaners.
- Required the exact workload reservation to be released or fenced both before and after
  retirement. Missing reservations, identity disagreement, nonempty cgroups, and projection
  contradictions block without sending a stop; retryable user-bus failures remain pending
  under capped backoff.
- Fenced systemd retirement by unit token, invocation, deterministic cgroup, optional boot,
  raw ActiveState/SubState, empty job queue, and authoritative cgroup-v2 membership. Cleanup
  uses `--job-mode=fail`, treats absent unit plus empty exact cgroup as idempotent success, and
  never exposes unit, invocation, cgroup, PID, argv, environment, or tokens in events/status.
- Quarantined new managed-process launch/termination while cleanup is blocked or retrying,
  without crash-looping the rest of Friday. Status exposes only aggregate pending, retrying,
  blocked, and last-pass completion counts.
- Made forward migrations savepoint-atomic, reject future or gapped histories, acquire a
  database writer lock before version discovery, and serialize complete schema initialization
  through a mode-0600 cross-process lock. Sixteen-way fresh and v8-upgrade races converge on
  one valid schema and backfill.
- Deployed schema v10 with a consistent v9 checkpoint, restarted Friday only, and preserved
  the Qwen PID. The RTX 4090 runtime remains at 200K context; `/healthz` and authenticated
  status are ready with zero pending/retrying/blocked cleanups.
- Passed 454 repository tests and all four cognitive-core evaluations. The full suite left the
  live database, control token, resolved runtime manifest, runtime fingerprint, and managed
  process counts unchanged byte-for-byte/count-for-count.

Next milestone: bind every terminate effect to an exact durable process-operation record so a
natural or unrelated exit cannot settle the wrong action. Then add paired TLS, expiring
controller-bound sessions and approvals, and only afterward enable curated application launch.

### 2026-08-24 — exact process-termination provenance

- Added a schema-v11 process-operation journal bound by composite foreign keys to the exact
  task step, receipt, attempt, lease, worker, arguments, executor binding, and target runtime
  boundary. Provenance is immutable and lifecycle transitions are monotonic.
- Split termination into durable `prepared`, `dispatching`, `effect_acknowledged`, and
  `completed` barriers, with explicit `known_failed` and `outcome_unknown` branches. Backend
  acceptance is committed before terminal process projection, so projection recovery never
  repeats the signal.
- Required a live exact claim and its exact approved durable approval record at both preparation
  and dispatch. Completed operations replay without a backend call; ambiguous dispatches never
  signal again; a force escalation must be a separately approved action rather than a changed
  replay.
- Removed state-only causality inference. A natural or unrelated exit is recorded as `exited`;
  only an acknowledged operation against the same runtime boundary may produce `terminated`
  and satisfy the exact action verifier or reconciler.
- Made systemd termination check the stop command result before accepting a coincident terminal
  observation. Explicit no-effect rejection is authoritative failure evidence, while transport
  ambiguity remains unresolved.
- Deployed the migration from a consistent v10 backup and passed 471 repository tests plus all
  four cognitive-core evaluations, including crash injection at every dispatch barrier,
  concurrent replay, forged provenance/events, migration rollback/races, natural exits, and
  nonzero systemd-stop results. SQLite integrity and foreign-key checks pass with no inferred
  legacy operation rows.
- Restarted the supervised RTX 4090 runtime at the exact 200K profile. Qwen passed authenticated
  identity, wrong-key, tokenization, and generation canaries; Friday is ready on `0.0.0.0:8500`
  with schema v11, stable boot calibration, and zero pending process cleanups or action
  reconciliations. Remote LAN access still requires the documented client-scoped UFW rule.

Next milestone: add paired TLS, expiring controller-bound sessions and approval grants, then
enable curated application launch behind that trust boundary. Empirical KV/VRAM calibration,
long-context throughput measurement, and broader held-out task families remain in parallel.

### 2026-08-24 — paired HTTPS controllers and exact approval authority

- Added a stable private P-256 installation CA and SAN-specific leaf generations. Startup
  verifies the CA/key relationship, leaf/key relationship, exact SAN set, ownership, modes,
  link counts, digests, and OpenSSL chain before serving HTTPS; trust-material damage fails
  closed instead of rotating identity.
- Replaced reusable browser authority with P-256 paired controllers. The browser keeps a
  non-exportable private key in IndexedDB, while short-lived operational sessions remain only
  in memory and are bound to the exact HTTPS origin and stable CA fingerprint.
- Restricted the installation control token to one-use pairing creation. Added fresh signed
  challenges for returning sessions, immediate controller/session revocation, strict Host and
  Origin checks, replay resistance, durable expiry, and secret-free controller inventory.
- Bound approvals and first effects to the authenticated controller and exact signed decision.
  Cross-controller decisions, expired challenges, revoked identities, malformed signatures,
  and legacy approval reuse fail before the effect boundary.
- Added journaled, idempotent retirement for unsigned pre-controller approvals so stale legacy
  prompts cannot masquerade as actionable paired authority or leave empty tasks permanently
  waiting for input.
- Added schema-v13 atomic migrations plus concurrent-upgrader, TLS handshake/tamper, endpoint,
  signature, replay, expiry-boundary, revocation-race, and forged-effect trigger tests.
- Passed 526 maintained repository tests and all four cognitive-core evaluations in isolated
  state before and after the live HTTPS migration.
- Added authenticated, fixed-canary runtime measurement bound to the exact active process,
  listener, profile, and hardware. Private tamper-detecting records now track first-token
  latency, decode throughput, and exact process-group VRAM without retaining generated text.
- Added in-place watcher reloads so source/config changes do not restart a proven 200K Qwen
  runtime, while normal service stops retain full control-group shutdown semantics.
- Deployed HTTPS from a consistent schema-v13 backup, retired five unsigned/no-step legacy
  prompts with journal evidence, and preserved Qwen PID 2135113 at 200K context and eight
  sequences. Live readiness, SQLite quick/integrity checks, foreign keys, authenticated model
  identity, wrong-key rejection, SAN validation, and plaintext-HTTP refusal pass.
- Measured the live RTX 4090 profile at 56.2 ms median first token, 107.9 decode tokens/second,
  and 22,550 MiB Qwen process-group VRAM across three fixed 256-token samples.

Next milestone: complete remote-client CA trust and the client-scoped firewall check, then use
the new evidence format for bounded multi-profile KV/context comparisons and curated
application launch on top of the paired-controller authority boundary.

### 2026-08-24 — first session-bound curated application profile

- Added explicit Wayland/user-bus capability metadata to immutable process specs. The backend,
  not the model, derives the exact same-UID runtime socket; malformed names, missing sockets,
  aliases, ownership changes, and environment collisions fail before launch.
- Added a versioned Friday Terminal profile for the installed Foot 1.27.0 binary. Its fixed
  application ID/title, exact executable identity, working directory, CPU/RAM/task/runtime
  limits, host-network boundary, and Wayland access are approval-visible and fingerprinted.
- Kept browsers, editors, and file managers out of the registry because their reuse/daemon
  behavior cannot yet be attributed to Friday's exact cgroup. An unsafe executable is omitted
  instead of weakening validation or crash-looping the assistant.
- Added adversarial tests for sandbox/session mixing, legacy fingerprint stability, socket-name
  traversal, backend-derived transport values, and privacy-safe inventory.
- Replaced the launch-only v1 profile with a compound v2 binding. Approval now covers the exact
  process contract, compositor session, presentation fingerprint, and hashed application ID.
  Success requires exactly one mapped window owned by the cgroup's exact leader PID/start time
  and executable device/inode/hash; receipts expose only an opaque window ID and safe label.
- Added non-replaying reconciliation for crashes between process launch and compositor proof.
  Missing, wrong-session, wrong-PID, forged-ID, and ambiguous-window observations remain
  unresolved rather than being mistaken for launch success.
- Added journaled retirement for durable specs removed from the code-curated registry. An
  obsolete spec is revoked once it has no nonterminal instances; a still-owned instance keeps
  the contract retained and explicitly degraded until reconciliation can finish safely.
- Passed 536 maintained repository tests and all four cognitive-core evaluations in isolated
  state while the live 200K Qwen process remained running.

Next milestone: prove the v2 compound receipt through the paired-controller action path, then
generalize activation receipts for portal/DBus applications without treating daemon reuse as
process ownership.

### 2026-08-24 — bounded hardware-performance portfolio

- Added a private, atomic performance portfolio containing at most twelve exact profiles for
  one hardware/model family. Every entry carries bounded fixed-canary samples, recomputed
  medians, VRAM, exact tuning, and a runtime-identity binding; malformed, insecure, expired,
  future-dated, cross-machine, and candidate-mismatched evidence fails closed.
- Added separate advisory reasoning and throughput recommendations. Reasoning preserves the
  greatest measured context/concurrency before speed, while throughput ranks measured decode
  speed and latency. Neither view silently changes or promotes the active runtime.
- Added an explicit `benchmark-profiles` lifecycle operation for the launcher's huge, long,
  and fast KV/context modes. It derives candidates from the exact active profile, contains a
  mode-specific failure, and revalidates restoration of the exact original Qwen profile before
  restarting Friday. Invalid bounds fail before any lifecycle effect.
- Removed cross-generation status races by deriving both recommendations from one validated
  portfolio snapshot. A new authenticated measurement repairs a damaged portfolio from that
  measurement alone rather than carrying forward partially trusted entries.
- Passed 545 maintained repository tests and all four cognitive-core evaluations in isolated
  state. Live deployment preserved Qwen PID 2135113 at 200K context/eight sequences and added
  a fresh RTX 4090 portfolio entry: 55.9 ms median first-token latency, 106.3 decode tokens/s,
  and 22,550 MiB process-group VRAM.
- Restored Friday's strict allowlisted HTTPS LAN listener on `0.0.0.0:8500`; loopback and the
  host's `192.168.1.158` address both pass readiness. The remaining remote-client step is the
  client-scoped UFW rule, which requires local administrator authentication.

Next milestone: complete controller pairing and the client-scoped firewall rule, exercise the
v2 terminal launch through a signed approval, then extend attributable activation receipts to
portal/DBus applications. Run the disruptive three-mode benchmark only during an operator-
approved maintenance window; current single-profile evidence remains advisory.

### 2026-08-24 — attributable child-process GUI surfaces

- Generalized compound GUI launch proof without weakening the original terminal contract.
  Each presentation now explicitly chooses leader-only or managed-cgroup window ownership;
  that authority is approval-visible and fingerprinted, while legacy leader fingerprints stay
  byte-for-byte compatible.
- Added a private process-broker proof for child surfaces. It binds the durable instance to the
  exact live systemd token, boot, invocation, cgroup, leader PID/start/executable identity, then
  checks the child PID's cgroup membership before and after hashing its executable and
  re-observes the managed execution.
- Reused desktop daemons, processes outside the managed cgroup, PID/start-time replacement,
  executable disagreement, session changes, and multiple qualifying windows cannot produce a
  launch receipt. Reconciliation and independent receipt verification use the same authority
  and never redispatch the application.
- Passed 549 maintained repository tests and all four cognitive-core evaluations in isolated
  state, including server-path callback wiring, cgroup/execution replacement, PID reuse,
  ambiguity, missing verifier, privacy, and legacy fingerprint tests.
- Deployed the verifier without restarting Qwen. The durable Foot v2 fingerprint remained
  compatible and active, no managed instances were in flight, Friday/LAN readiness passed,
  SQLite quick/foreign-key checks passed, and Qwen PID 2135113 remained at the proven 200K
  profile.

Next milestone: add a singleton, isolated-profile lifecycle for one multi-process browser or
editor and curate it with managed-cgroup presentation proof. Do not admit an application that
can hand its window to a preexisting process outside Friday's exact managed boundary.

### 2026-08-24 — managed singleton browser boundary

- Replaced the server browser operator's blind loopback attach/direct-spawn behavior with a
  durable managed-only contract. Browser tools now require one exact active singleton and
  double-fence its systemd token, invocation, leader identity, cgroup, and unique same-UID
  loopback listener inode around every Playwright operation.
- Curated Chromium 151.0.7922.173 with an owner-only fixed profile, fixed blank initial page,
  managed-cgroup window proof, bounded resources, and no model-controlled arguments. Friday
  pins the package's final engine instead of its distinct launcher and uses the compositor's
  exact `chromium-browser` application ID.
- Intentionally withheld the user D-Bus transport. A live adversarial canary showed that
  D-Bus-enabled Chromium moves its leader into a sibling XDG application scope; the Wayland-
  only profile kept all 13 observed processes, the visible window PID, and the debug listener
  inside Friday's exact service cgroup. Cleanup removed the unit and listener completely.
- Added atomic singleton admission, restart-after-terminal behavior, exact listener ownership,
  pre/post browser-operation fences, owner-only profile validation, no-fallback behavior, and
  privacy-safe receipts. Passed all 560 maintained repository tests and all four cognitive-
  core evaluations.
- Deployed Friday alone. Qwen remained PID 2135113 at 200K context/eight sequences; Friday
  became PID 3138535 on the strict HTTPS LAN listener; loopback/LAN readiness, SQLite quick and
  foreign-key checks, active spec fingerprints, zero nonterminal instances, and absence of an
  unapproved port-9223 listener all passed.

Next milestone: pair the first signed controller, install the client-scoped UFW rule, then
exercise the managed browser through the full approval path and verify its durable launch,
browser action, reconciliation, and termination receipts without bypassing controller consent.

### 2026-08-24 — connect-time public-web boundary

- Closed DNS rebinding in both research fetches and the visible managed browser. Public URL
  normalization now rejects credentials, control characters, invalid ports, local names,
  special literals, and any DNS set containing even one non-global address. Each connection
  uses a numeric address from that validated set and rechecks its actual peer.
- Added a bounded loopback SOCKS5 service owned by Friday. Chromium v2 sends HTTP(S) and
  WebSockets through it, disables direct hostname resolution, explicit loopback bypass, QUIC,
  DNS prefetch, background networking, and non-proxied WebRTC UDP. TLS remains end-to-end and
  proxy status exposes only aggregate counters. Shutdown actively closes exact tunnels.
- Replaced browser-open's optimistic CDP tab acknowledgement with a Playwright navigation
  postcondition. The final public URL, HTTP status, and title are observed before success;
  snapshot, click, and type reject non-public pages before/after model-visible interaction.
- Rejected a weaker design with live evidence: per-user systemd `IPAddressDeny=localhost` still
  reached Friday and systemd warned that IP firewalling was not running as root. No fail-open
  property is counted as protection.
- Live canaries proved pinned HTTPS research, public TLS through SOCKS, zero requests to a
  temporary loopback server, blocked private attempts, exact cgroup/window/CDP ownership for
  14 Chromium processes, and complete cleanup. All 577 maintained tests and all four cognitive-
  core evaluations passed.
- Deployed Friday alone as PID 3241723. The old browser v1 contract was revoked and immutable
  v2 activated with zero managed instances; Friday owns both `0.0.0.0:8500` and the loopback-
  only proxy listener. Live public proxy TLS returned 200, loopback was denied, readiness
  reported `browser_network: true`, database checks passed, and Qwen remained PID 2135113 at
  the measured 200K context/eight-sequence profile.

Next milestone: pair the first signed controller, install the client-scoped UFW rule, and
exercise a fully approved browser workflow with durable launch/action/termination receipts.

### 2026-08-24 — classified built-in HTTP egress

- Unified public web research, Google News, Skills.sh discovery, and optional remote reasoning
  behind one bounded transport. It rejects mixed/private/special DNS sets, connects to a
  validated numeric address, rechecks the peer, preserves TLS hostname verification, and
  repeats the complete proof for every permitted redirect.
- Closed redirect and message-boundary gaps: HTTPS downgrade, credential-bearing redirects,
  POST redirects, unsafe caller headers, ambiguous lengths, oversized requests/responses, and
  unsupported content encodings fail before data can cross the wrong authority boundary.
- Constrained local Qwen configuration to exact numeric loopback HTTP under `/v1`. Both the
  OpenAI SDK and authenticated tokenization path ignore ambient proxies and reject redirects;
  CDP health probes use the same proxy-free loopback discipline.
- Passed all 591 maintained repository tests and all four cognitive-core evaluations. Live
  canaries fetched pinned public TLS, returned attributed Google News and Skills.sh results,
  and reached the authenticated 200K Qwen runtime even with every ambient proxy variable aimed
  at a dead listener.
- Deployed Friday alone as PID 3346064. Qwen remained PID 2135113 at the measured 200K/eight-
  sequence profile; loopback and LAN HTTPS readiness passed, Friday alone owned ports 8500 and
  9224, no CDP listener was exposed, public SOCKS TLS returned 200, and a loopback SOCKS attempt
  was denied. Schema-13 integrity/foreign-key checks, active durable spec fingerprints, and zero
  nonterminal process instances all remained clean. Post-deploy pinned fetches returned public
  TLS, two attributed news headlines, and two Skills.sh results.

Next milestone: pair the first signed controller, install the client-scoped UFW rule, and
exercise a fully approved managed-browser workflow through durable launch/action/termination
receipts.

### 2026-08-24 — explicit tensor-parallel scaling contract

- Closed the gap between multi-GPU inventory and model scaling. A higher-spec operator may now
  select a canonical set of 1–16 physical CUDA devices; multi-rank profiles require unique,
  detected, equal-capacity devices and bind the complete set plus tensor-parallel degree into
  the operational and calibration family fingerprints.
- Made the launcher topology exact. Ambient device/parallel variables are stripped,
  `CUDA_VISIBLE_DEVICES` names the canonical physical set, and a mandatory final
  `--tensor-parallel-size` agrees with it. The installed vLLM 0.27 runtime was inspected and
  confirms native support for this argument.
- Extended resource admission to reserve the per-rank LLM envelope on every selected GPU and
  publish both per-rank and aggregate budgets. An unselected eligible GPU is used for speech;
  otherwise automatic TTS moves to CPU instead of silently stealing a model rank's headroom.
  Explicit shared placement remains possible and retains its reserve checks.
- Preserved the proven single-GPU operational and family fingerprints byte-for-byte, so the
  live RTX 4090 does not require a Qwen restart. Synthetic homogeneous, heterogeneous,
  duplicate, undetected, ordering, admission, launcher-override, and speech-placement cases
  pass alongside all 596 repository tests and all four cognitive-core evaluations.
- Deployed Friday alone as PID 3404962. Qwen remained PID 2135113 with the exact active
  `9bf3d272…` fingerprint, 200K context, eight sequences, and measured 106.3 decode tokens/s.
  Loopback/LAN readiness, public-proxy TLS, private-proxy denial, schema-13 integrity and
  foreign keys, durable process-spec fingerprints, zero nonterminal instances, and absence of
  a CDP listener all passed.

Next milestone: continue the paired-controller/browser workflow when administrator authority
is available, while expanding held-out end-to-end capability and recovery scorecards.

### 2026-08-24 — held-out capability-core v2

- Replaced the four-case cognitive smoke check with a thirteen-case versioned capability suite
  spanning ten named areas: intent, contracts, approval policy, receipt verification, false-
  completion fencing, exact read-only restart, nonrepeatable reconciliation, memory provenance,
  public-network isolation, and tensor-parallel hardware scaling.
- Added six fixed stateful scenarios. Each receives a fresh disposable schema-13 graph and
  returns only deterministic aggregate observations, so evaluation cannot create live tasks,
  claims, memories, approvals, or leases. The live graph records only the bounded suite result.
- Hardened the evaluator itself: fixtures must be bounded non-symlink regular files with finite
  JSON, valid metadata, unique cases, typed inputs, and allowlisted scenarios. An implementation
  exception becomes an explicit failed case; it cannot abort the rest of the suite or become a
  pass. Tests prove invalid fixtures leave no evaluation record.
- All 13 capability cases and all 600 maintained repository tests pass. The score is explicitly
  scoped: paired-controller browser workflows, documents/vision, live speech latency, memory
  precision, rollback rates, and long-horizon success remain unmeasured rather than being
  implied by this result. Qwen and the live frontend were not restarted for this offline-only
  evaluator change.

Next milestone: execute the first signed-controller managed-browser workflow when local
administrator authority is available; independently add measured document/vision and memory-
retrieval task families without inflating the capability-core score.

### 2026-08-24 — permissioned bounded document extraction

- Added one format-aware `machine_read_document` action behind the existing exact encrypted
  read grants. It supports PDF, DOCX, ODT, EPUB, PPTX, and XLSX while preserving the same
  descriptor-relative, no-symlink authority boundary as machine text reads.
- Added source-bound receipts with canonical grant/path identity, source size and SHA-256,
  normalized extracted-text size and SHA-256, format, extractor, truncation state, and PDF page
  count. The outcome verifier recomputes text evidence and rejects forged hashes, invalid
  formats/extractors, impossible sizes, or malformed page evidence.
- Made archive handling non-executing and extraction-free. Member count, member size, aggregate
  expansion, compression ratio, XML size, EPUB spine, returned bytes, and returned characters
  are bounded; traversal, duplicate/encrypted members, unsupported compression, malformed XML,
  DTD/entities (including alternate-width evasion), and extension/signature disagreement fail
  closed.
- Sandboxed the pinned Poppler PDF extractor in Bubblewrap with no network, an exact already-open
  input descriptor, a single private tmpfs output file, read-only system files, and CPU,
  address-space, process, descriptor, file-size, and wall-clock limits. A real three-page PDF
  canary produced a source/text-hashed bounded receipt.
- Routed document reads through durable contracts and claimed-step execution. Paths and raw text
  use the established private-payload redaction path; task progress exposes only a safe label.
  Symlink, mutation, archive-bomb, malicious-path, entity, malformed-signature, all-format,
  truncation, execution, redaction, and forged-receipt tests pass.
- All 609 maintained repository tests and all 13 capability-core v2 cases pass. These tests prove
  the extraction and authority boundaries; held-out semantic document reasoning and vision are
  still unmeasured and are not claimed as AGI evidence.

Next milestone: deploy this Friday-only change without restarting Qwen, then add held-out
document question-answering and vision cases. The first paired-controller browser workflow and
client-scoped UFW rule remain pending on local administrator authority.

### 2026-08-24 — artifact-backed document reasoning and bounded image OCR

- Added exact-grant PNG/JPEG OCR with pre-decoder source, signature, JPEG-marker, dimension,
  pixel, and byte bounds. Tesseract runs in Bubblewrap with no network, a descriptor-pinned
  input, read-only system files, private tmpfs output, and CPU, memory, task, descriptor, output,
  and wall-time limits. Blank images return verified no-text evidence; this capability is
  explicitly `ocr_only` and is not labeled general vision.
- Routed OCR through the durable read-only policy, task contract, claimed-step executor, private
  argument/result redaction, safe progress projection, and source/text-hashed outcome verifier.
  Symlinks, source mutation, pixel bombs, malformed/mismatched headers, forged dimensions,
  boolean/count substitutions, hashes, and capability overclaims fail closed.
- Added a separate five-case `friday-document-reasoning` v1 scorecard rather than inflating
  capability-core. Every case materializes a real DOCX, XLSX, PDF, or PNG; uses the bounded
  archive, Poppler, or Tesseract extractor; sends only extracted context to local Qwen; and uses
  exact required answer terms plus verbatim in-context evidence instead of an LLM judge.
- The live runtime scored 5/5 on superseding-window fact retrieval, spreadsheet maximum/difference
  arithmetic, OCR notice retrieval, untrusted-document prompt-injection resistance, and
  missing-evidence abstention. Raw answers are not journaled; records contain hashes, grading
  flags, artifact/extractor provenance, model identity, and the exact active runtime fingerprint.
- All 623 maintained tests and all 13 capability-core v2 cases pass. The score proves only these
  narrow grounded tasks. The checkpoint contains vision-tower weights, but the measured 24 GB,
  200K runtime intentionally remains language-only; general non-text scene understanding is
  still unmeasured.
- Deployed Friday alone as PID 3644775. Qwen remained PID 2135113 with the exact active
  `9bf3d272…` fingerprint, 200K context, eight sequences, measured 55.9 ms first-token latency,
  and 106.3 decode tokens/s. Loopback/LAN HTTPS returned 200; public proxy TLS returned 200;
  private proxy access was denied; no CDP listener was exposed; schema-13 integrity/foreign keys,
  zero nonterminal managed processes, and zero controllers/pairings remained clean.

Next milestone: add a hardware-calibrated native-vision profile for higher-memory GPUs, with
VRAM/context/latency boot canaries and held-out scene-understanding cases before automatic
activation. Controller pairing, the client-scoped UFW rule, and the first durable browser
workflow remain pending administrator and controller action.

### 2026-08-24 — hardware-adaptive native-vision launch gate (source verified)

- Added fingerprinted native-vision profiles for the pinned Qwen checkpoint on homogeneous model
  ranks with at least 30 GiB. Image count, maximum side, per-rank GPU reserve, and host reserve
  scale through bounded 30/46/60 GiB tiers; the 24 GiB 200K/eight-sequence profile remains
  byte-for-byte compatible and language-only.
- Kept explicit resource choices authoritative. Automatic vision moves default shared speech to
  CPU on the smallest tier when necessary, but never rewrites explicit speech placement or an
  explicit model-memory ceiling. Incompatible explicit native-vision requests fail closed.
- Reasserted language-only or exact multimodal limits after all launcher and user arguments.
  Native activation also requires a bounded exact local checkpoint directory containing the
  pinned conditional-generation architecture, Qwen3VL processor metadata, and visual weights.
- Replaced the provisional OCR-like canary with a generated text-free scene that requires color,
  shape, and left/right binding. The authenticated loopback request is bound to the exact
  listener/process identity, rejects redirects, and is rechecked after the response.
- Native boot calibration now records scene-probe latency and exact Qwen process-group VRAM;
  activation and last-known-good reuse require both scene and bounded-VRAM evidence in addition
  to the existing identity, credential-rejection, context, and startup proof.
- All 630 maintained repository tests pass, including the hardware/supervisor/calibration and
  capability-core evaluation coverage. The current live process
  was not restarted or relabeled: selected and active fingerprints still match
  `9bf3d272…`, with 200K context, eight sequences, and native vision disabled.

Next milestone: add the separate multi-case artifact-backed native-vision scorecard and durable
exact-grant scene-understanding tool, then execute those cases on a qualifying GPU before making
any general-vision claim. The current 24 GiB deployment should remain unchanged.

### 2026-08-24 — score-qualified exact-grant native scene understanding

- Added a durable read-only `machine_understand_image` action using the existing encrypted exact
  path grants and no-symlink descriptor boundary. A networkless, resource-limited ImageMagick
  sandbox strips metadata, auto-orients, bounds dimensions, and emits one canonical PNG; only
  those ephemeral bytes are sent to the profile-bound local model.
- Kept private data out of durable state. Tool arguments use the private-payload redaction path,
  the image wrapper hides bytes from representations, and receipts contain only path/grant,
  source/image, question/answer, model, and runtime hashes plus bounded image provenance. The
  independent verifier rejects malformed or overclaimed receipts.
- Added five deterministic text-free artifact cases for counting, color/shape binding, spatial
  position, containment, and relative size. Exact grading records answer hashes rather than raw
  answers, contains individual case failures, and binds every passing run to the exact model,
  runtime fingerprint, sanitizer, and profile maximum side.
- Made qualification fail closed. A native-vision boot must pass the original listener-bound
  canary, the full 5/5 scorecard, and the process-group VRAM proof before activation or last-known-
  good reuse. The tool remains absent from the model schema unless the append-only graph contains
  a valid matching 5/5 record.
- Focused supervisor, calibration, image, task, cognition, hardware, and integration verification
  passed, followed by all 642 maintained repository tests. The live 24 GiB Qwen/Friday deployment
  was not restarted: it remains language-only at
  the existing 200K/eight-sequence fingerprint, so no live native-vision score or general-vision
  claim is made on this machine.

Next milestone: execute and retain the native scorecard on qualifying hardware, expand held-out
real-image coverage without weakening privacy or authority boundaries, and measure false-positive
and abstention behavior before broadening any visual capability claim.

### 2026-08-24 — provenance-bound, stale-safe memory retrieval

- Closed an authority gap in memory promotion. `user_explicit` evidence now requires every
  derived utterance to be authored by the user in its immutable creation event; assigning an
  utterance node type from an assistant event no longer promotes a preference.
- Bounded subjects, predicates, scopes, evidence labels, retention reasons, finite JSON values,
  unique source sets, confidence, UTC validity windows, and retrieval inputs. Expired candidates
  cannot promote, and active claims disappear from retrieval at their validity boundary even if
  their auditable projection remains present.
- Made correction propagation exact: a newly verified value or duplicate refresh supersedes all
  older active values for the same subject/predicate/scope, removes them from FTS, and keeps their
  graph nodes plus `supersedes` edges. Repeated preferences no longer create duplicate prompt
  context.
- Replaced raw OR-token retrieval with a normalized, FTS-syntax-safe lexical pipeline, light
  morphology, field-aware deterministic scores, and a relative relevance gate for verbose
  queries. Tests include distractors sharing `progress` or `notifications`, so precision at one
  is not established by unique vocabulary alone.
- Added the separate seven-case `friday-memory-retrieval` v1 scorecard. Each scenario receives a
  disposable graph; the durable result contains suite identity, aggregate score, case names, and
  observation hashes rather than synthetic facts or randomized claim IDs. Fixture tampering,
  symlinks, duplicate coverage, non-finite JSON, and scenario exceptions fail closed.
- The scorecard passes 7/7 in isolated verification and its live aggregate audit run also scored
  7/7 without retaining scenario claims. In addition, 77 focused memory/server/evolution tests
  and all 650 maintained repository tests pass. This proves only the named deterministic lexical
  and governance boundaries; arbitrary paraphrase retrieval, multilingual quality, large-corpus
  latency, and measured downstream task lift remain open.
- The live Friday and Qwen processes were not restarted or reconfigured, so their 24 GiB,
  200K-context/eight-sequence deployment remains unchanged.

Next milestone: add privacy-preserving semantic retrieval with a hardware-adaptive embedding
index, then measure precision/recall and downstream task lift on paraphrased, multilingual, and
large-corpus held-out sets before allowing it to influence normal prompt context.

### 2026-08-24 — pinned private multilingual semantic-memory fallback

- Added an atomic installer for the exact `intfloat/multilingual-e5-small` revision. Every model
  asset has an expected size and SHA-256 digest, and an invalid existing directory is rejected
  rather than overwritten or trusted.
- Added a lazy, local-files-only CPU encoder with bounded inputs and a hardware-adaptive batch
  size. It uses no network, GPU, vLLM, or remote provider path, so it does not reduce Qwen's VRAM
  budget. Missing or failed embedding support degrades safely to lexical retrieval.
- Advanced durable storage to schema 14 with a rebuildable embedding projection bound to the
  claim-content hash, model fingerprint, and vector dimension. Only active, unexpired claims may
  influence retrieval; corrections remove superseded vectors, and invalid cached vectors are
  recomputed.
- Kept deterministic lexical retrieval primary and added semantic retrieval only as a fallback
  for empty or weak lexical matches. An absolute score threshold plus nearest-neighbor margin
  rejects low-confidence or ambiguous results; calibration included related distractors and
  unrelated weather, cuisine, and hardware questions.
- The real pinned model passed all 8 isolated semantic-memory cases: lexically disjoint English,
  Spanish, Hindi, and German recall; unrelated-query abstention; expiry; correction propagation;
  and top-one retrieval with projection reuse in a 129-claim corpus. No semantic score was
  written to the live aggregate audit graph.
- All 97 focused tests and all 656 maintained repository tests passed. The live Friday and Qwen
  processes were not restarted, so the verified source is not yet deployed and the running
  24 GiB, 200K-context/eight-sequence service remains unchanged.
- This milestone does not prove arbitrary paraphrase recall, long-horizon task lift, or
  production-scale vector search. The semantic scan is bounded to 4,096 active claims, and the
  measured 129-claim case is not evidence for an ANN-scale corpus.

Next milestone: deploy the change in an intentional maintenance window, retain the live semantic
scorecard after restart, measure retrieval p50/p95 and real corrected-memory task lift, and add
ANN or sharded indexing beyond 4,096 active claims only when measurements justify it.

### 2026-08-28: measured semantic scale and correction lift

- Added a versioned held-out scorecard spanning English paraphrases, Spanish,
  Hindi, German, irrelevant requests, and four correction tasks. Synthetic
  claims live only in disposable graphs; the durable run contains aggregate
  metrics, suite identity, and model identity.
- The first 5,000-claim run exposed ranking degradation from embedding hubness
  and the fixed 4,096-claim cap. That run remained failed evidence.
- Replaced the cap with a complete 65,536-claim bounded index. Missing vectors
  are encoded in requests of at most 512, and exact scoring runs in 1,024-vector
  shards. Corpus-centred ranking reduces embedding hubness. Static local null
  passages and weak-lexical rejection preserve abstention.
- The final pinned-model run passed every gate. At 5,000 active claims,
  precision was 1.000, recall 0.867, irrelevant-query abstention 1.000, warm
  p50 66.6 ms, warm p95 178.7 ms, and corrected-memory task lift 1.000. Every
  checkpoint indexed all claims and recovered the oldest target. Superseded
  claims and vectors did not reappear.
- Sharded exact search meets the current 1,000 ms p95 gate, so an approximate
  nearest-neighbour dependency is not justified by this measurement. Revisit
  that decision if a qualified corpus or machine breaches the gate.

### 2026-08-28: artifact-backed private voice qualification

- Added a versioned eight-utterance voice scorecard using the exact pinned CPU
  Piper and Parakeet runtimes. It writes real WAV artifacts into an owner-only
  temporary directory, verifies their mode and hashes, then removes them before
  recording a result.
- The real run passed at 3.9% word error rate and 7/8 exact utterances. ASR p50
  was 59.6 ms and p95 was 91.4 ms; TTS p50 was 46.8 ms and p95 was 68.9 ms.
  Both backends stayed far below real time.
- Speech-to-first-response-audio measured 72.7 ms p50 and 95.6 ms p95 with an
  explicitly declared deterministic local reply stage. Language-model latency
  remains separately measured by the live conversation scorecard.
- The production playback gate rejected every synthesized playback frame before
  VAD, held the full 650 ms acoustic tail, and reopened at the boundary. The
  production utterance buffer triggered barge-in at 220 ms and retained the
  complete speech prefix.
- Durable evidence contains aggregate metrics, hashes, and backend identity. It
  contains no phrases, transcripts, microphone content, or audio artifacts.

### 2026-08-28: receipt-grounded recovered project

- Added a versioned long-horizon project scorecard that stages exact file,
  test, and verification steps in the production durable task journal.
- The evaluator abandons the first dispatched attempt before execution. A new
  worker recovers the same logical action under attempt 2, completes the ordered
  batch, and proves that each file effect occurred exactly once.
- The measured run matched 3/3 files, passed 3/3 tests, verified all 5 action
  receipts, and recovered in 55.8 ms. A fresh test run and fresh file hashes
  independently agree with the recorded receipts.
- The user-visible result is structured from durable task state, receipts,
  action attempts, and fresh probes. Assistant prose cannot turn a failed or
  incomplete run into success.
- Source fixtures, the temporary project, and raw test output are removed after
  the run. Only counts, durations, hashes, gate decisions, and the structured
  outcome enter Friday's graph.
