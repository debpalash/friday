# Alpha compatibility policy

This policy applies to the `0.1.x` alpha line. It protects durable user data and
security authority without pretending that every experimental extension is
stable. The machine-readable counterpart is
[`compatibility/v1.json`](../compatibility/v1.json).

## Release changes

Every release records user-visible and migration changes in `CHANGELOG.md`.
Patch releases may add fields and capabilities. An alpha release may still
change an experimental interface, but a change cannot silently reinterpret an
existing approval, receipt, permission, or runtime identity.

Friday supports upgrading data through the migration path shipped in the new
release. Downgrading a database after its schema advances is unsupported. The
installer must preserve state, take its private pre-update backup, and restore
the prior code and configuration if the new release fails before schema
activation. Operators should also create a private export before an update.

## Durable graph

- The current SQLite graph schema is version 15.
- Migrations are ordered, transactional, restart-safe, and forward-only.
- A runtime rejects a database newer than the schema it understands.
- Existing identifiers, evidence hashes, and tombstones are never reused with
  a different meaning.
- Selective export and deletion formats declare their own format and graph
  schema versions. Import or deletion rejects mismatched manifests.

No promise is made that old Friday code can open a newly migrated database.
Rollback therefore restores code only when the database remains compatible;
otherwise recovery uses the pre-update backup or export.

## Tool contracts and receipts

Task contracts use version 1. Authority-bearing request bodies reject unknown
fields. An approved action remains bound to its exact task, durable step, tool,
arguments, executor identity, policy decision, and one-time local approval.
Receipt verification does not infer compatibility from similar prose or a
transport success.

Additive display metadata may appear in a release. A change to execution
meaning, required arguments, authority, idempotency, or receipt verification
requires a new contract or tool version and a migration for nonterminal work.
Unknown versions fail closed.

## Runtime manifests

Resolved runtime, TLS, calibration, and performance records currently use
schema version 1. Readers reject unknown schema versions. Hardware placement,
model identity, context, speech backend, and admission budgets are bound to a
runtime fingerprint; a changed fingerprint must pass calibration before it can
replace last-known-good state.

Environment overrides are deployment inputs, not a stable extension API.

## Skills and generated capabilities

Skill and capability versions are immutable. Code, manifest, permissions, and
evaluation evidence are hash-bound. Modified instructions or widened
permissions create a new version. Promotion requires explicit activation or
the verified-success policy; quarantined or superseded versions do not regain
authority through name reuse.

The skill manifest shape and generated-capability handler interface are
experimental during alpha. There is no cross-release compatibility window for
third-party extensions unless a later document names one. Unsupported fields,
versions, or permissions fail closed rather than being guessed.

## Local UI protocol

The current WebSocket subprotocol is `friday.v1`. It carries no bearer token or
paired identity. The server accepts it only on a loopback listener with an exact
allowed Host and HTTPS Origin. A breaking wire change uses a new subprotocol or
endpoint version.

## Support window

Until a stable release, only the current source release is supported for bug
fixes. The immediately previous installer release is exercised only as the
rollback source in release rehearsal. Security fixes may require an immediate
upgrade. A stable compatibility window will be declared before `1.0.0`, based
on real migration and extension usage rather than an untested promise.
