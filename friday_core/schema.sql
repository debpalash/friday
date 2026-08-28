PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS graph_events (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL UNIQUE,
    occurred_at     TEXT NOT NULL,
    actor           TEXT NOT NULL,
    session_id      TEXT,
    turn_id         TEXT,
    task_id         TEXT,
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    payload_sha256  TEXT NOT NULL,
    idempotency_key TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id          TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    created_event_id TEXT NOT NULL REFERENCES graph_events(event_id),
    body_json        TEXT NOT NULL,
    body_sha256      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id          TEXT PRIMARY KEY,
    from_node_id     TEXT NOT NULL REFERENCES nodes(node_id),
    relation         TEXT NOT NULL,
    to_node_id       TEXT NOT NULL REFERENCES nodes(node_id),
    created_event_id TEXT NOT NULL REFERENCES graph_events(event_id),
    attributes_json  TEXT NOT NULL DEFAULT '{}',
    UNIQUE(from_node_id, relation, to_node_id, created_event_id)
);

CREATE INDEX IF NOT EXISTS edges_from_relation
    ON edges(from_node_id, relation);
CREATE INDEX IF NOT EXISTS edges_to_relation
    ON edges(to_node_id, relation);
CREATE INDEX IF NOT EXISTS events_task_seq
    ON graph_events(task_id, seq);
CREATE INDEX IF NOT EXISTS events_turn_seq
    ON graph_events(turn_id, seq);

CREATE TABLE IF NOT EXISTS task_state (
    task_id                  TEXT PRIMARY KEY REFERENCES nodes(node_id),
    objective                TEXT NOT NULL,
    completion_contract_json TEXT NOT NULL,
    status                   TEXT NOT NULL,
    plan_json                TEXT NOT NULL DEFAULT '[]',
    active_step              TEXT,
    lease_id                 TEXT,
    lease_expires_at         TEXT,
    last_error               TEXT,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    last_event_seq           INTEGER NOT NULL REFERENCES graph_events(seq)
);

CREATE INDEX IF NOT EXISTS task_state_status ON task_state(status);

CREATE TABLE IF NOT EXISTS action_receipts (
    idempotency_key TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES task_state(task_id),
    step_id         TEXT,
    action_id       TEXT NOT NULL REFERENCES nodes(node_id),
    tool_name       TEXT NOT NULL,
    args_sha256     TEXT NOT NULL,
    status          TEXT NOT NULL,
    observation_id  TEXT REFERENCES nodes(node_id),
    result_json     TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_step_batches (
    batch_id        TEXT PRIMARY KEY REFERENCES nodes(node_id),
    task_id         TEXT NOT NULL REFERENCES task_state(task_id),
    round_index     INTEGER NOT NULL,
    status          TEXT NOT NULL,
    context_json    TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_event_seq  INTEGER NOT NULL REFERENCES graph_events(seq),
    UNIQUE(task_id, round_index)
);

CREATE TABLE IF NOT EXISTS task_steps (
    step_id             TEXT PRIMARY KEY REFERENCES nodes(node_id),
    task_id             TEXT NOT NULL REFERENCES task_state(task_id),
    batch_id            TEXT NOT NULL,
    round_index         INTEGER NOT NULL,
    ordinal             INTEGER NOT NULL,
    tool_call_id        TEXT NOT NULL,
    tool_name           TEXT NOT NULL,
    args_ciphertext     TEXT NOT NULL,
    args_redacted_json  TEXT NOT NULL,
    args_sha256         TEXT NOT NULL,
    context_json        TEXT NOT NULL DEFAULT '{}',
    idempotency_key     TEXT NOT NULL UNIQUE,
    idempotency_class   TEXT NOT NULL,
    recovery_policy     TEXT NOT NULL,
    status              TEXT NOT NULL,
    depends_on_json     TEXT NOT NULL DEFAULT '[]',
    verifier            TEXT NOT NULL,
    risk                TEXT NOT NULL,
    approval_status     TEXT NOT NULL DEFAULT 'not_required',
    approval_id         TEXT,
    action_id           TEXT REFERENCES nodes(node_id),
    lease_id            TEXT,
    worker_id           TEXT,
    lease_expires_at    TEXT,
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    max_attempts        INTEGER NOT NULL DEFAULT 3,
    last_error          TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    last_event_seq      INTEGER NOT NULL REFERENCES graph_events(seq),
    UNIQUE(task_id, ordinal)
);

CREATE INDEX IF NOT EXISTS task_steps_task_status
    ON task_steps(task_id, status, ordinal);
CREATE INDEX IF NOT EXISTS task_steps_lease
    ON task_steps(status, lease_expires_at);

CREATE TABLE IF NOT EXISTS action_attempts (
    attempt_id       TEXT PRIMARY KEY REFERENCES nodes(node_id),
    idempotency_key  TEXT NOT NULL REFERENCES action_receipts(idempotency_key),
    step_id          TEXT NOT NULL REFERENCES task_steps(step_id),
    attempt_number   INTEGER NOT NULL,
    lease_id         TEXT NOT NULL,
    worker_id        TEXT NOT NULL,
    status           TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    last_error       TEXT,
    last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq),
    UNIQUE(step_id, attempt_number)
);

CREATE INDEX IF NOT EXISTS action_attempts_step
    ON action_attempts(step_id, attempt_number);

CREATE TABLE IF NOT EXISTS claim_state (
    claim_id         TEXT PRIMARY KEY REFERENCES nodes(node_id),
    subject          TEXT NOT NULL,
    predicate        TEXT NOT NULL,
    object_json      TEXT NOT NULL,
    scope            TEXT NOT NULL,
    lifecycle        TEXT NOT NULL,
    confidence       REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    evidence_class   TEXT NOT NULL,
    retention_reason TEXT NOT NULL,
    valid_until      TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq)
);

CREATE INDEX IF NOT EXISTS claims_lifecycle_scope
    ON claim_state(lifecycle, scope);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    claim_id UNINDEXED,
    text
);

CREATE TABLE IF NOT EXISTS memory_embedding_index (
    claim_id          TEXT NOT NULL REFERENCES nodes(node_id),
    model_fingerprint TEXT NOT NULL CHECK(length(model_fingerprint) = 64),
    content_sha256    TEXT NOT NULL CHECK(length(content_sha256) = 64),
    dimension         INTEGER NOT NULL CHECK(dimension > 0 AND dimension <= 4096),
    vector            BLOB NOT NULL CHECK(length(vector) = dimension * 4),
    indexed_at        TEXT NOT NULL,
    PRIMARY KEY (claim_id, model_fingerprint)
);

CREATE INDEX IF NOT EXISTS memory_embedding_model
    ON memory_embedding_index(model_fingerprint, claim_id);

CREATE TABLE IF NOT EXISTS progress_outbox (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT NOT NULL UNIQUE REFERENCES graph_events(event_id),
    task_id      TEXT NOT NULL REFERENCES task_state(task_id),
    payload_json TEXT NOT NULL,
    occurred_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_state (
    skill_id          TEXT PRIMARY KEY REFERENCES nodes(node_id),
    name              TEXT NOT NULL UNIQUE,
    status            TEXT NOT NULL,
    active_version_id TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    last_event_seq    INTEGER NOT NULL REFERENCES graph_events(seq)
);

CREATE TABLE IF NOT EXISTS skill_versions (
    version_id       TEXT PRIMARY KEY REFERENCES nodes(node_id),
    skill_id         TEXT NOT NULL REFERENCES skill_state(skill_id),
    version          INTEGER NOT NULL,
    instructions     TEXT NOT NULL,
    manifest_json    TEXT NOT NULL,
    tests_json       TEXT NOT NULL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq),
    UNIQUE(skill_id, version)
);

CREATE TABLE IF NOT EXISTS deployment_state (
    deployment_id    TEXT PRIMARY KEY REFERENCES nodes(node_id),
    target_path      TEXT NOT NULL,
    checkpoint_path  TEXT,
    before_sha256    TEXT,
    after_sha256     TEXT NOT NULL,
    status           TEXT NOT NULL,
    test_output      TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq)
);

CREATE TABLE IF NOT EXISTS capability_state (
    capability_id    TEXT PRIMARY KEY REFERENCES nodes(node_id),
    name             TEXT NOT NULL UNIQUE,
    status           TEXT NOT NULL,
    active_version_id TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq)
);

CREATE TABLE IF NOT EXISTS capability_versions (
    version_id       TEXT PRIMARY KEY REFERENCES nodes(node_id),
    capability_id    TEXT NOT NULL REFERENCES capability_state(capability_id),
    version          INTEGER NOT NULL,
    description      TEXT NOT NULL,
    parameters_json  TEXT NOT NULL,
    code             TEXT NOT NULL,
    permissions_json TEXT NOT NULL,
    tests_json       TEXT NOT NULL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq),
    UNIQUE(capability_id, version)
);

CREATE TABLE IF NOT EXISTS voice_profiles (
    voice_id         TEXT PRIMARY KEY REFERENCES nodes(node_id),
    name             TEXT NOT NULL UNIQUE,
    kind             TEXT NOT NULL,
    config_json      TEXT NOT NULL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq)
);

CREATE TABLE IF NOT EXISTS voice_runtime (
    singleton        INTEGER PRIMARY KEY CHECK(singleton = 1),
    active_voice_id  TEXT REFERENCES voice_profiles(voice_id),
    previous_voice_id TEXT REFERENCES voice_profiles(voice_id),
    updated_at       TEXT NOT NULL,
    last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq)
);

CREATE TABLE IF NOT EXISTS core_upgrade_state (
    upgrade_id       TEXT PRIMARY KEY REFERENCES nodes(node_id),
    task_id          TEXT NOT NULL REFERENCES task_state(task_id),
    objective        TEXT NOT NULL,
    backend          TEXT NOT NULL,
    workspace_path   TEXT NOT NULL,
    status           TEXT NOT NULL,
    changed_json     TEXT NOT NULL DEFAULT '[]',
    deployment_id    TEXT,
    last_error       TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq)
);

CREATE TABLE IF NOT EXISTS deletion_tombstones (
    deletion_id      TEXT PRIMARY KEY,
    scope            TEXT NOT NULL CHECK(scope IN (
                           'conversation', 'task', 'artifact', 'memory_claim',
                           'time_range')),
    selector_sha256  TEXT NOT NULL CHECK(length(selector_sha256) = 64),
    deleted_at       TEXT NOT NULL,
    rows_json        TEXT NOT NULL,
    UNIQUE(scope, selector_sha256)
);

CREATE TRIGGER IF NOT EXISTS deletion_tombstones_no_update
BEFORE UPDATE ON deletion_tombstones BEGIN
    SELECT RAISE(ABORT, 'deletion tombstones are immutable');
END;

CREATE TRIGGER IF NOT EXISTS deletion_tombstones_no_delete
BEFORE DELETE ON deletion_tombstones BEGIN
    SELECT RAISE(ABORT, 'deletion tombstones are retained');
END;

CREATE TRIGGER IF NOT EXISTS graph_events_no_update
BEFORE UPDATE ON graph_events BEGIN
    SELECT RAISE(ABORT, 'graph_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS graph_events_no_delete
BEFORE DELETE ON graph_events BEGIN
    SELECT RAISE(ABORT, 'graph_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS nodes_no_update
BEFORE UPDATE ON nodes BEGIN
    SELECT RAISE(ABORT, 'nodes is append-only');
END;

CREATE TRIGGER IF NOT EXISTS nodes_no_delete
BEFORE DELETE ON nodes BEGIN
    SELECT RAISE(ABORT, 'nodes is append-only');
END;

CREATE TRIGGER IF NOT EXISTS edges_no_update
BEFORE UPDATE ON edges BEGIN
    SELECT RAISE(ABORT, 'edges is append-only');
END;

CREATE TRIGGER IF NOT EXISTS edges_no_delete
BEFORE DELETE ON edges BEGIN
    SELECT RAISE(ABORT, 'edges is append-only');
END;
