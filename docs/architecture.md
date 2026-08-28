# Architecture

Friday is a single-user local service with four primary process boundaries.

```text
Browser controller
  microphone, text, signed approvals
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

`server.py` owns the HTTPS and WebSocket interface, speech pipeline, task
orchestration, approvals, tools, and the current HTML/CSS/JavaScript client. The
browser microphone is muted while TTS plays, and a playback tail gate prevents
Friday from transcribing its own output.

`friday_core/` contains durable graph, task, policy, process, desktop, browser,
memory, speech, and evaluation services. SQLite is authoritative for durable
task state. Model output is a proposal, not proof that a side effect occurred.

The installer uses versioned release directories and a single atomic `current`
link. Personal state and large shared assets live beside those releases, so a
code rollback does not roll back or delete user data.

## Trust boundaries

Friday assumes one trusted local operating-system user. The kernel, filesystem
permissions, user systemd manager, desktop compositor, GPU stack, browser
controller, and desktop keyring are part of the trusted computing base.

External pages, model output, imported skills, generated capabilities, and
maintenance-worker changes are untrusted. State-changing actions pass through:

1. a typed tool contract;
2. policy and resource admission;
3. an exact user approval when required;
4. a bounded executor;
5. an independent receipt or postcondition check.

Unknown outcomes remain unknown until reconciled. A transport success alone is
not an action receipt.

## Current architectural debt

- `server.py` contains the API, orchestration wiring, and embedded frontend. Its
  size raises review cost and couples UI release cadence to the control plane.
- The supported installer target is one Linux and NVIDIA family. Hardware
  selection code covers more cases than the public install matrix proves.
- The SQLite event graph has migrations and restart tests, but no supported
  selective export or deletion tool for every record type.
- Browser and desktop boundaries rely on Linux user-session facilities and have
  not received an independent penetration test.
- Model and speech quality are measured on the maintainer's hardware. There is
  no public cross-device benchmark matrix yet.
- The repository has one maintainer and no published compatibility window for
  alpha schemas or extension APIs.

These are release constraints, not hidden roadmap items. Changes that widen a
network, execution, or storage boundary require tests and documentation in the
same pull request.
