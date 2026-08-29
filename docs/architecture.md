# Architecture

Friday is a single-user local service with four primary process boundaries.

```text
Browser interface
  microphone, text, local approvals
             |
             v
Friday application service
  ASR, conversation, planner, policy, tools, TTS
             |                 |
             v                 v
SQLite graph and files      systemd user process broker
             |
             v
Supervisor service -> pinned loopback Qwen/vLLM runtime
```

## Runtime ownership

`supervisor.py` selects a hardware profile, starts the pinned Qwen runtime,
checks its process and listener identity, performs health and calibration gates,
and stops or replaces it when the binding is no longer valid.

`server.py` is the FastAPI composition root. Pure Host, Origin, and WebSocket
parsing lives in `friday_core/transport.py`; history compilation lives in
`conversation_runtime.py`; per-connection echo and VAD state lives in
`voice_transport.py`; recovered-batch finalization lives in
`task_orchestration.py`; and `frontend/index.html` is loaded through a bounded,
no-symlink asset reader. Voice commands must start with `Friday` and contain the
request in the same utterance. Low-level or unaddressed speech is discarded
before durable graph writes or model inference. The browser requests voice
isolation, disables automatic gain control, mutes the microphone while TTS
plays, and holds a 1.5-second playback tail before reopening input.

`friday_core/` contains durable graph, task, policy, process, desktop, browser,
memory, speech, and evaluation services. SQLite is authoritative for durable
task state. Model output is a proposal, not proof that a side effect occurred.

The installer uses versioned release directories and a single atomic `current`
link. Personal state and large shared assets live beside those releases, so a
code rollback does not roll back or delete user data.

## Trust boundaries

Friday assumes one trusted local operating-system user. The kernel, filesystem
permissions, user systemd manager, desktop compositor, GPU stack, local browser,
and desktop keyring are part of the trusted computing base.

External pages, model output, imported skills, generated capabilities, and
maintenance-worker changes are untrusted. State-changing actions pass through:

1. a typed tool contract;
2. policy and resource admission;
3. an exact user approval when required;
4. a bounded executor;
5. an independent receipt or postcondition check.

Unknown outcomes remain unknown until reconciled. A transport success alone is
not an action receipt.

## Enforced composition boundaries

`scripts/check-architecture.py` prevents the frontend from returning to an
embedded Python string, requires every named boundary, and caps growth of the
composition root. Unit and integration tests exercise each extracted boundary
and the retained server-facing compatibility wrappers.

The compatibility contract for graph migrations, receipts, runtime manifests,
skills, extensions, and the local UI protocol is published in
[Alpha compatibility policy](compatibility.md).

## Remaining release constraints

- The supported installer target is one Linux and NVIDIA family. Hardware
  selection code covers more cases than the public install matrix proves.
- Browser and desktop boundaries rely on Linux user-session facilities and have
  not received an independent penetration test.
- Model and speech quality are measured on the maintainer's hardware. There is
  no public cross-device benchmark matrix yet.
- The repository has one maintainer. Alpha extension APIs intentionally have no
  compatibility window yet; durable data and authority have the narrower
  guarantees documented in the policy.

These are release constraints, not hidden roadmap items. Changes that widen a
network, execution, or storage boundary require tests and documentation in the
same pull request.
