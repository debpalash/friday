# Hardware-aware runtime profiles

Friday resolves one non-secret runtime profile before starting either vLLM or the
assistant. The supervisor publishes `state/runtime-resolved.json` only after vLLM passes
health plus an authenticated `/v1/models` identity check; it therefore describes the
active runtime, not merely a proposal. The server consumes the same model endpoint,
context budget, ASR thread count, and TTS device through environment variables.
`/api/status` exposes that manifest to the local UI.

The policy is reasoning-first. A single 24 GB card cannot simultaneously hold the
27B model's proven 200K KV profile and resident CUDA OmniVoice, so automatic placement
keeps long-context reasoning on the GPU and uses the pinned local Piper voice on CPU. Larger or
multi-GPU systems retain accelerated OmniVoice without shrinking the reasoning window.

| Detected layout | Automatic vLLM profile | Speech placement |
|---|---|---|
| Single 22–28 GB CUDA | 0.93 utilization, KVarN `huge`, 200K, 8 sequences | Pinned Piper on CPU |
| Shared 32 GB CUDA | about 22.5 GiB, KVarN `huge`, 200K, 8 sequences | OmniVoice on the same GPU |
| Shared 48 GB+ CUDA | about 22.5 GiB, KVarN `huge`, 200K, 8 sequences | OmniVoice on the same GPU |
| Second CUDA GPU with at least 8 GB | Up to 0.93 on the LLM GPU, KVarN `huge`, 200K | OmniVoice on the smaller eligible GPU |
| NVIDIA device exists but probe fails | Known-safe 24 GB settings plus a visible warning | Resolved CUDA default |
| No NVIDIA CUDA device | Local Qwen runtime is reported unsupported and is not crash-looped | Disabled |

VRAM is never summed automatically. On multi-GPU hosts, Friday first isolates the model and
TTS processes with `CUDA_VISIBLE_DEVICES`; automatic selection retains one proven LLM device
and prefers a separate eligible speech GPU. Placement uses physical identity and total VRAM
rather than transient free memory, so restarting TTS cannot silently change profiles. This
keeps a personal assistant's latency and failure modes simpler.

Larger checkpoints can explicitly use homogeneous tensor parallelism by setting
`FRIDAY_LLM_CUDA_DEVICES` to a comma-separated physical set such as `0,1`. Friday canonicalizes
and validates 1–16 unique detected indices, rejects mixed-capacity ranks, fingerprints the
complete set and tensor-parallel degree, exposes the topology in the non-secret manifest, and
reserves the per-rank LLM envelope on every selected GPU. `CUDA_VISIBLE_DEVICES` and a mandatory
late `--tensor-parallel-size` argument then describe the same exact set. An unselected eligible
GPU remains available for speech; if every selected GPU is a model rank, automatic speech moves
to CPU rather than silently consuming model headroom. Sharing speech with a model rank requires
an explicit TTS placement. The existing authenticated boot/tokenization canary must still prove
the selected checkpoint and exact context before the profile becomes active. Since a device-set
override is an explicit operator choice, it never inherits an automatic fallback or calibration
from a different topology.

`llm_memory_budget_gib` and `unallocated_gpu_gib` are per LLM rank for tensor-parallel profiles;
`llm_total_memory_budget_gib` reports their aggregate. This distinction keeps admission control
correct on every physical accelerator instead of treating aggregate VRAM as one fictitious GPU.

`tts_reserve_gib` is the minimum shared-GPU speech/headroom requirement;
`unallocated_gpu_gib` reports the actual capacity left outside vLLM. They are deliberately
separate: a larger card does not donate all otherwise-free memory to the same 27B model.
On the automatic single-24-GB reasoning profile the reserve is zero because TTS is on CPU.
Setting `FRIDAY_TTS_DEVICE=cuda` explicitly selects the reversible voice-balanced escape
hatch (0.724 utilization and 8K context) when low speech latency matters more than context.

When the resolved speech device is CPU, `FRIDAY_TTS_BACKEND=auto` verifies and selects the exact
pinned Kristin Piper checkpoint. It loads once, performs no runtime network access, produces
bounded mono audio locally, rejects silent/malformed output, and resamples to Friday's 24 kHz
browser stream. If the pinned voice is absent, automatic mode preserves the older OmniVoice CPU
fallback; explicit `piper` fails closed instead. Piper prioritizes immediate speech and does not
offer reference-based voice activation. CUDA speech automatically retains OmniVoice and its voice-profile
lifecycle. The local status response reports the active backend and runtime voice.

The 200K ceiling is capacity, not a promise of constant decode speed. KVarN's own measured
cost is small on short prompts but material beyond 100K active tokens; long context should
be used when the evidence actually requires it, with retrieval and compaction keeping normal
turns short.

## Action admission budget

The resolved profile also publishes a stable `admission_budget`. It reserves at least two
CPU cores (or ten percent), at least 2 GiB of RAM (or ten percent), bounded global/network
slots, and independent action VRAM on each physical CUDA device after LLM, TTS, and safety
headroom. These values participate in the profile fingerprint; transient load, available
RAM, and free VRAM do not.

Immediately before durable dispatch, Friday combines that static envelope with a short-lived
live sample and active SQLite resource leases. A claim larger than the profile is rejected
without creating an action receipt. Temporary contention remains pending and is requeued.
The resource lease is acquired atomically with the action attempt, renewed by the same fenced
heartbeat, released with receipt completion, and fenced during crash recovery. Each lease is
bound to the resolved profile fingerprint and requested latency class. Background and batch
claims preserve the final concurrency and network lanes for interactive work. `/healthz`
forces a fresh telemetry sample, while `/api/status` reports budgets and aggregate
reservations without exposing tool arguments.

Long-running managed processes atomically transfer the step reservation into a durable
workload lease. Reconciliation can adopt only the exact recorded runtime identity; terminal
release requires authoritative process state and an empty cgroup. A zero-resource control
lane is reserved for bounded inspect, terminate, and reconciliation work so saturation does
not prevent cleanup. The initial process sandbox supports one target executable and fails
closed if the observed topology expands beyond that contract.

Curated GUI specs may explicitly request a local user-session transport. Friday derives the
Wayland socket and optional D-Bus address from its pinned user session; model arguments cannot
set or override either value. The socket must be a direct, same-UID Unix socket under the exact
`/run/user/<uid>` runtime directory. V1 exposes session transport only to an exact unsandboxed
application spec because the non-desktop bubblewrap profile does not yet have a portal/socket
policy. That broader host boundary is shown in inventory and approval metadata and is never
silently inferred. On this installation, the first such profile is the exact Foot 1.27.0
binary with fixed application ID/title and no model-controlled arguments. Its executable,
argv, limits, session access, and package file identity are part of the durable spec binding;
an update degrades the old version unavailable instead of repinning it.

GUI launch approval additionally binds the immutable process contract to the exact current
Hyprland session, a hash of the expected application ID, and an explicit window-ownership
mode. The original leader mode requires one and only one mapped window whose compositor PID,
process start time, executable device, inode, and SHA-256 match the cgroup-owned process
leader. This mode and all existing Foot bindings retain their original durable fingerprints.

An explicitly fingerprinted `managed_cgroup` mode supports applications whose mapped surface
belongs to a child process. The process broker revalidates the durable instance and complete
systemd execution, then checks exact cgroup membership on both sides of the child's PID/start/
executable identity read and re-observes the unit afterwards. A preexisting or reused desktop
daemon remains outside the dedicated cgroup and cannot satisfy the receipt. PID reuse,
execution replacement, membership loss, wrong application ID, and multiple qualifying windows
all fail closed. The model still sees only an opaque window ID and safe application label. If
the process effect crossed its boundary before presentation could be confirmed, the step is
quarantined for read-only reconciliation and is never launched again.

The managed browser is an explicitly singleton, persistent Chromium 151.0.7922.173 profile.
Friday pins the package's final `/usr/lib/chromium/chromium` engine rather than the distinct
`/usr/bin/chromium` launcher, because the launcher's executable replacement must not be
mistaken for the approved runtime. It receives a private owner-only profile, fixed loopback
debug address/port, fixed blank initial page, and Wayland access, but intentionally receives
no user D-Bus address. Chromium otherwise asks systemd over D-Bus to move itself into a
sibling application scope, escaping the service cgroup that owns its approval and lifecycle.
Browser tools never spawn Chromium or attach merely because a loopback endpoint answers.
Before and after each control operation, Friday revalidates the one durable singleton
execution and proves that the unique same-UID loopback listener's socket inode is held by a
member of its exact cgroup. A foreign listener, duplicate live instance, executable change,
profile substitution, missing window, or cgroup escape fails closed.

The v2 browser profile also routes every HTTP(S)/WebSocket connection through Friday's
loopback-only, bounded SOCKS5 proxy. Chromium cannot bypass loopback, perform its own hostname
resolution, use QUIC, or send non-proxied WebRTC UDP. The proxy validates the complete DNS
answer set, rejects it if any result is not globally routable, connects to one numeric address
from that same set, and rechecks the connected peer before tunneling bytes unchanged. TLS
therefore remains end-to-end while DNS rebinding, private redirects, and private subresources
fail at connect time. Friday's non-browser research fetcher uses the same numeric-IP pin and
manually revalidates every bounded redirect. The proxy stores only aggregate accepted/blocked/
failed counts and closes active tunnels during restart. Per-user systemd `IPAddressDeny=` is
not treated as an enforcement layer on this host: a live canary reached an explicitly denied
loopback endpoint and systemd reported that the filter was not running as root.

All built-in HTTP traffic is classified before it opens a socket. Public web research, news,
Skills.sh discovery, and optional remote reasoning share one public-only transport: it rejects
mixed or special DNS answers, connects to a validated numeric address, verifies the actual peer,
and preserves the original hostname for TLS certificate verification. Every redirect repeats
that proof. Redirects cannot downgrade HTTPS to HTTP, forward credentials to another request,
or replay a POST; caller headers, request bodies, redirect counts, and response bodies are
strictly bounded. Browser traffic uses the equivalent connect-time policy through the SOCKS5
boundary above.

Local model traffic has the inverse contract. The configured Qwen endpoint must be exactly
numeric loopback HTTP at `http://127.0.0.1:<port>/v1`; hostnames, LAN addresses, TLS endpoints,
credentials, queries, fragments, and alternate paths fail during startup. The OpenAI SDK client
ignores ambient proxy variables and refuses redirects. The authenticated tokenization probe and
CDP health probes likewise use proxy-free, redirect-rejecting loopback openers. Supervisor
readiness and calibration additionally bind Qwen responses to the expected listener inode and
process identity, so neither a desktop proxy nor an unrelated loopback service can silently
become the model authority.

Terminal transitions also journal a retained-unit cleanup intent in the same SQLite
transaction. Cleanup uses an expiring database claim, requires the workload lease to be
released or fenced both before and after the backend effect, and retires only the exact empty
systemd unit behind its token, invocation, deterministic cgroup, raw terminal state, and empty
job queue. Missing units with empty/missing exact cgroups are idempotent success. Transport
failures retry with capped backoff; identity, projection, or membership contradictions
quarantine new managed-process effects while Friday remains available for status and other
work. The same-UID user-systemd namespace is a trusted boundary: systemd has no atomic
compare-by-InvocationID stop operation, although `--job-mode=fail` prevents cleanup from
replacing an already queued start/restart job.

Task-facing process termination also has a separate durable operation journal. A termination
must match the exact running task step, receipt, attempt, lease, worker, approved arguments,
the step-bound approved approval record, executor binding, and target runtime boundary before
dispatch. The journal is committed before the backend call, and an exact backend acknowledgement
is committed before terminal process projection. Replays of completed operations do not signal
again; a crash after dispatch with no authoritative acknowledgement remains `outcome_unknown`.
Monitoring records an empty cgroup as `terminated` only when the same boundary has an
acknowledged termination operation; otherwise it is an ordinary `exited` process and cannot
settle a terminate action. Administrative untracked termination remains available for local
recovery but cannot satisfy a durable task receipt.

## Configuration overrides

Every automatic choice can be overridden without editing source:

| Variable | Purpose |
|---|---|
| `FRIDAY_LLM_REPO` | Qwen/vLLM checkout containing `single-user/start_qwen.sh` |
| `FRIDAY_LLM_EXTRA_ARGS` | Explicit launcher arguments; disables automatic boot fallback |
| `FRIDAY_QWEN_MODEL` | Checkpoint path or model identifier passed to the launcher |
| `FRIDAY_LOCAL_MODEL` | OpenAI-compatible served model name |
| `FRIDAY_LLM_HOST`, `FRIDAY_LLM_PORT` | Loopback inference endpoint |
| `FRIDAY_MODEL_CONTEXT_TOKENS` | Shared server/client context ceiling |
| `FRIDAY_GPU_UTIL` | vLLM GPU-memory fraction, from `0.30` through `0.98` |
| `FRIDAY_KV_MODE` | External launcher profile: `fast`, `long`, or `huge` |
| `FRIDAY_MAX_SEQS` | vLLM request slots |
| `FRIDAY_CUDAGRAPH_CAPTURE` | Maximum CUDA graph capture size |
| `FRIDAY_LLM_CUDA_DEVICES` | One physical CUDA index, or 1–16 comma-separated homogeneous indices for explicit tensor parallelism |
| `FRIDAY_TTS_DEVICE` | OmniVoice device map, normally `cuda` or `cpu` |
| `FRIDAY_TTS_BACKEND` | `auto` (Piper for a verified CPU voice, OmniVoice for CUDA), `piper`, or `omnivoice` |
| `FRIDAY_TTS_CUDA_DEVICES` | Explicit single physical device index for the assistant/TTS process |
| `FRIDAY_TTS_RESERVE_GIB` | Minimum shared-GPU reserve; it may reduce but never expand the tier budget |
| `FRIDAY_NATIVE_VISION` | `auto`, `enabled`, or `disabled`; automatic activation is limited to the pinned vision-capable checkpoint and model ranks with at least 30 GiB |
| `FRIDAY_ALLOW_UNPROBED_CUDA` | Explicitly opt into the conservative fallback when CUDA is hidden from probes |
| `FRIDAY_ASR_THREADS` | Sherpa-ONNX CPU thread count |
| `FRIDAY_EMBEDDING_MODEL` | `auto` (default), a disabled value, or the exact local pinned embedding-model directory |
| `FRIDAY_EMBEDDING_BATCH_SIZE` | Optional semantic-memory CPU batch override from 1 through 64 |
| `FRIDAY_LOCAL_API_KEY` | In-memory local provider credential |
| `FRIDAY_LOCAL_API_KEY_FILE` | Local provider credential file; never copied into status |
| `FRIDAY_BIND_HOST`, `FRIDAY_PORT` | Friday UI listener; the host must be `localhost`, `127.0.0.1`, or `::1` |
| `FRIDAY_DESKTOP_MODE` | Hyprland operator mode: `auto`, `required`, or `disabled` |

The live 24 GB profile remains explicitly language-only and retains its proven 200K/eight-sequence
fingerprint. PNG/JPEG text remains available through the separately sandboxed CPU OCR path.

Semantic-memory embeddings are deliberately CPU-first and choose a batch size from stable CPU/RAM
capacity unless `FRIDAY_EMBEDDING_BATCH_SIZE` overrides it. The default `auto` mode uses only the
exact locally installed, hash-pinned checkpoint; if it is absent or cannot load, Friday retains
lexical memory retrieval without attempting a runtime download. This leaves GPU/VRAM available
for Qwen and speech. Embedding projections carry their own model fingerprint and are not part of
the Qwen runtime-profile fingerprint. The source implementation described here is not active in
an already-running Friday process until an intentional restart.

For the pinned vision-capable checkpoint, larger model ranks select bounded native-vision modes:
30–45 GiB permits one image at 1024px with 3 GiB of additional model budget, 46–59 GiB permits
two images at 1536px with 4 GiB, and 60 GiB or more permits four images at 2048px with 8 GiB.
On the smallest tier, automatically placed speech moves to CPU if necessary; explicit speech or
model-memory settings always win, disabling automatic vision or rejecting an incompatible
explicit vision request.

The supervisor reasserts either `--language-model-only` or the exact multimodal bounds after all
launcher and operator arguments. Before a native-vision process can become active it validates
the exact local checkpoint architecture, processor, and visual-weight index; proves the bound
authenticated listener can answer a text-free color/shape/spatial image canary; measures that
probe; runs a separate five-case artifact-backed native-vision scorecard; and verifies process-
group VRAM remains inside the profile envelope. Promotion and later last-known-good reuse require
all of that exact-profile evidence. The durable exact-grant scene-understanding tool is exposed
only while a matching 5/5 score record remains available; its sandboxed canonical image is
ephemeral and its receipt stores hashes rather than pixels or answer text. These checks prove a
bounded launch and five deterministic scene tasks, not general visual competence. The current
24 GiB language-only profile neither runs nor advertises this tool.

Friday's browser and model listeners are loopback-only. `FRIDAY_BIND_HOST` accepts only
`localhost`, `127.0.0.1`, or `::1`; startup rejects wildcard, LAN, and public addresses. Direct
LAN access is unsupported because the local browser API has no application authentication.

Friday still serves HTTPS and creates a stable local P-256 CA plus leaf certificate under the
owner-only `state/tls/` directory. HTTPS preserves browser microphone access and transport
integrity. CA/key tampering, symlinks, unsafe permissions, or certificate verification failures
abort startup instead of silently rotating trust. Remote use requires a separately reviewed
authenticated tunnel that terminates at loopback.

Invalid numeric ranges or KV modes fail at startup with an explicit error. Explicit values
always beat the automatic profile and are listed by variable name in the resolved manifest;
secret values are never serialized.

The external launcher is constrained at the final argument position to
`--host 127.0.0.1`, the resolved served-model alias, and—when selected—the exact tensor-
parallel degree. Ambient launcher variables cannot replace the resolved physical device set.
The same local API credential is given to vLLM and Friday. `localhost` model overrides are
canonicalized to numeric `127.0.0.1`. Before and after every credential-bearing probe, the supervisor requires one
unambiguous loopback LISTEN socket whose inode is open by the exact verified vLLM PID and
whose effective user, process start time, and Linux user/network namespaces remain stable.
Credential probes disable proxies and reject every redirect. A healthy process with an
unknown profile, wrong listener owner, wrong credential, or wrong model alias is rejected
instead of being relabeled; use `restart-all` to adopt a new profile. `supervisor.py status`
reports selected and active profiles separately without exposing socket or process identity.

Inspect the decision without starting either service:

```bash
venv/bin/python supervisor.py status
```

When no supported local CUDA runtime is visible, `watch` backs off for 60 seconds and
re-probes instead of repeatedly launching doomed processes. `FRIDAY_ALLOW_UNPROBED_CUDA`
exists for containers or unusual device isolation where the operator knows CUDA is present.

Cold boots now pass health, authenticated model identity, rejection of a deliberately invalid
credential, and a fixed authenticated tokenization canary before the candidate is published
as active. Credentialed probes disable proxies and redirects and prove that the exact
loopback-listener inode belongs to the expected same-user Qwen PID, process start, namespace,
and socket set before and after the request. The canary measures startup and probe latency and
requires vLLM's observed `max_model_len` to equal the candidate's exact context ceiling. It
contains no user data. A proposal that fails this proof is never relabeled as active.

Successful automatic boots first enter a separate owner-only (`0600`) probation record.
Only the exact process identity that supplied the authenticated startup evidence may promote
that record after two uninterrupted minutes; a new canary therefore cannot overwrite the
prior stable baseline. Promotion and the stable last-known-good record use private atomic,
fsynced replacements. They are bound to the exact profile fingerprint, model/device family,
machine capacity topology, and stable GPU UUID. The private process binding, UUID, and binding
hashes never enter the UI-visible manifest. Stable records expire after 90 days and fail closed
if malformed, insecure, from different hardware, or not a monotonic memory/capacity reduction
from the current proposal.

An automatic cold boot attempts at most three candidates: the proposal, a usable
last-known-good profile when present, then deterministic reductions of context, concurrent
sequences, and CUDA-graph capture. Explicit environment overrides disable all automatic
fallback, so an operator's value is either proven exactly or rejected. The active manifest
always describes the candidate that actually booted; `supervisor.py status` separately exposes
the proposal, active manifest, and privacy-safe aggregate calibration state.

Boot failures and early post-start exits are remembered in another private atomic record.
Retries use exponential backoff from 15 seconds through 15 minutes, count a missing runtime
only once per observed loss, and clear failure history after two minutes of stable service.
Hardware or proposal changes start a fresh recovery epoch rather than inheriting stale
backoff. Backoff uses the kernel's monotonic clock and is bound to a hashed OS boot identity,
so a reboot or wall-clock correction cannot inherit an invalid retry deadline. Calibration
metadata failures are retried by the watch loop without stopping an otherwise healthy Friday
or Qwen process.

Measure the active profile without a restart or user data:

```bash
venv/bin/python supervisor.py calibrate-performance --samples 3 --tokens 256
```

The command holds the lifecycle fence, revalidates the authenticated model identity and
wrong-key rejection, warms up with a fixed canary, then measures exact first-token latency and
decode throughput. Listener/process identity is checked before and after every sample. VRAM is
summed only for processes in the exact Qwen process group. Generated canary text is discarded;
the owner-only `state/runtime-performance.json` stores only profile/hardware bindings, bounded
numeric samples, and aggregates. The same authenticated samples also update the owner-only
`state/runtime-performance-portfolio.json`, which retains at most twelve exact profile records
for the current hardware/model family. `supervisor.py status` exposes only sample count, age,
median latency/throughput, model VRAM, and privacy-safe reasoning/throughput recommendations.
Tampered, mismatched, insecure, future-dated, or older-than-30-day records are not trusted.

An operator can explicitly compare all three launcher-supported KV/context modes:

```bash
venv/bin/python supervisor.py benchmark-profiles --samples 3 --tokens 256
```

This command is intentionally disruptive: under the global lifecycle fence it stops Friday,
restarts Qwen through at most three exact active-profile-derived candidates, records only
authenticated fixed-canary evidence, restores the exact Qwen profile that was active on entry,
then restores Friday if it had been healthy. A failure in one mode is reported without silently
promoting another mode, and restoration is revalidated before success is returned. The
portfolio keeps separate recommendations: reasoning preserves the greatest measured context
and concurrency, while throughput ranks measured decode speed and latency. Recommendations are
advisory; neither calibration command changes the selected runtime automatically.

The user service supports `systemctl --user reload friday-supervisor.service`; reload execs the
watcher in place while preserving the independently identity-fenced Qwen and Friday processes.
A normal stop still uses `KillMode=control-group` and terminates the entire service tree.

## Local control-plane protection

Friday has no browser token, account, pairing flow, bearer session, or signing key. The UI opens
directly and every HTTP and WebSocket route is available to the local OS user. Foreign Host and
Origin values are rejected, the server cannot bind outside loopback, responses disable caching
and framing, and HTTPS remains enabled.

Consequential tools still require an explicit in-UI approval bound to the exact durable task
step and arguments. That approval is not cryptographically tied to a browser. Any process with
access to the same OS account can call the local API, so OS login, process isolation, browser
extensions, and local malware are outside Friday's application-level protection.
