# Friday core-upgrade harness

Friday uses two different extension paths because reusable behavior and core code have
different risk profiles.

## Extension layers

1. **Skills** are versioned instructions. An active skill is injected only when every
   tool named in its permission manifest is currently available.
2. **Capabilities** are executable tools. Candidate Python handlers receive static AST
   checks, a narrow permission manifest, at least two executable tests, immutable
   versions, and quarantine on any failure.
3. **Core upgrades** use the installed Pi agent as an untrusted maintenance worker.
   Pi never edits the live checkout.

## Core upgrade flow

```text
user-approved objective
        |
        v
append-only upgrade/task records
        |
        v
copy allowlisted source -> state/upgrades/<job>/workspace
        |
        v
Pi in Bubblewrap
  - workspace is the only writable project mount
  - no home directory or credentials
  - isolated network namespace
  - Unix-socket bridge reaches only local Qwen
        |
        v
reject deletion or modification of existing tests
        |
        v
independent full test suite in staging
        |
        v
preserve candidate + test output
        |
        v
explicit, diff-specific human review
```

Pi's JSON event stream is translated into Friday progress events, so file edits and
commands remain visible while the worker runs. Pi output is evidence, never authority.
Candidate code participates in its own test process and could fake a successful exit, so
the harness records `awaiting_review` and never modifies the live checkout automatically.
Promotion requires a separate, diff-specific human review workflow; only the external
supervisor can restart the live service.

## Protected state

The worker does not receive `state/`, `session.json`, `persona/`, model API-key files,
the user's home directory, or the live repository. Existing tests cannot be changed or
deleted. New focused tests are allowed, but they run alongside the protected suite.

The local Qwen API credential exists only in the maintenance-specific Pi configuration
inside the isolated job directory. The worker has no Internet or LAN route; a host-side
Unix socket and an inner loopback proxy expose only `127.0.0.1:18021`.
