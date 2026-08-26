"""Forward-only SQLite schema migrations for Friday's mutable projections."""

from __future__ import annotations

import sqlite3


LATEST_SCHEMA_VERSION = 14


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_column(conn: sqlite3.Connection, table: str, declaration: str) -> None:
    name = declaration.split()[0]
    if name not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {declaration}")


def _migration_1(conn: sqlite3.Connection) -> None:
    _add_column(conn, "task_state", "contract_version INTEGER NOT NULL DEFAULT 0")
    _add_column(conn, "task_state", "intent_type TEXT NOT NULL DEFAULT 'action'")
    _add_column(conn, "task_state", "risk TEXT NOT NULL DEFAULT 'low'")
    _add_column(conn, "task_state", "verification_status TEXT")
    _add_column(conn, "task_state", "verification_json TEXT")
    _add_column(conn, "task_state", "cancellation_requested INTEGER NOT NULL DEFAULT 0")
    _add_column(conn, "action_receipts", "effects_json TEXT NOT NULL DEFAULT '[]'")
    _add_column(conn, "action_receipts", "verification_json TEXT")
    _add_column(conn, "action_receipts", "risk TEXT NOT NULL DEFAULT 'low'")
    _add_column(conn, "action_receipts", "approval_status TEXT NOT NULL DEFAULT 'not_required'")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS task_verifications (
            verification_id TEXT PRIMARY KEY REFERENCES nodes(node_id),
            task_id          TEXT NOT NULL REFERENCES task_state(task_id),
            status           TEXT NOT NULL,
            summary          TEXT NOT NULL,
            evidence_json    TEXT NOT NULL,
            missing_json     TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq)
        );
        CREATE INDEX IF NOT EXISTS task_verifications_task
            ON task_verifications(task_id, created_at);

        CREATE TABLE IF NOT EXISTS feedback_state (
            feedback_id      TEXT PRIMARY KEY REFERENCES nodes(node_id),
            task_id          TEXT REFERENCES task_state(task_id),
            turn_id          TEXT,
            kind             TEXT NOT NULL,
            comment          TEXT,
            lifecycle        TEXT NOT NULL,
            supersedes_id    TEXT,
            created_at       TEXT NOT NULL,
            last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq)
        );
        CREATE INDEX IF NOT EXISTS feedback_target
            ON feedback_state(task_id, turn_id, created_at);

        CREATE TABLE IF NOT EXISTS transcript_corrections (
            correction_id    TEXT PRIMARY KEY REFERENCES nodes(node_id),
            utterance_id      TEXT NOT NULL REFERENCES nodes(node_id),
            original_text     TEXT NOT NULL,
            corrected_text    TEXT NOT NULL,
            audio_artifact    TEXT,
            created_at        TEXT NOT NULL,
            last_event_seq    INTEGER NOT NULL REFERENCES graph_events(seq)
        );

        CREATE TABLE IF NOT EXISTS reminder_state (
            reminder_id       TEXT PRIMARY KEY REFERENCES nodes(node_id),
            text              TEXT NOT NULL,
            due_at            TEXT NOT NULL,
            interval_seconds  INTEGER,
            status            TEXT NOT NULL,
            source_task_id    TEXT REFERENCES task_state(task_id),
            last_fired_at     TEXT,
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL,
            last_event_seq    INTEGER NOT NULL REFERENCES graph_events(seq)
        );
        CREATE INDEX IF NOT EXISTS reminders_due
            ON reminder_state(status, due_at);

        CREATE TABLE IF NOT EXISTS approval_state (
            approval_id       TEXT PRIMARY KEY REFERENCES nodes(node_id),
            task_id           TEXT NOT NULL REFERENCES task_state(task_id),
            tool_name         TEXT NOT NULL,
            args_json         TEXT NOT NULL,
            reason            TEXT NOT NULL,
            status            TEXT NOT NULL,
            created_at        TEXT NOT NULL,
            decided_at        TEXT,
            last_event_seq    INTEGER NOT NULL REFERENCES graph_events(seq)
        );

        CREATE TABLE IF NOT EXISTS model_disclosures (
            disclosure_id     TEXT PRIMARY KEY REFERENCES nodes(node_id),
            task_id           TEXT REFERENCES task_state(task_id),
            provider          TEXT NOT NULL,
            model             TEXT NOT NULL,
            payload_sha256    TEXT NOT NULL,
            redaction_json    TEXT NOT NULL,
            approved          INTEGER NOT NULL,
            created_at        TEXT NOT NULL,
            last_event_seq    INTEGER NOT NULL REFERENCES graph_events(seq)
        );
        """
    )


def _migration_2(conn: sqlite3.Connection) -> None:
    _add_column(conn, "action_receipts", "step_id TEXT")
    _add_column(conn, "approval_state", "step_id TEXT")
    conn.executescript(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS action_receipts_step
            ON action_receipts(step_id) WHERE step_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS approval_state_step
            ON approval_state(step_id) WHERE step_id IS NOT NULL;

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
        """
    )


def _migration_3(conn: sqlite3.Connection) -> None:
    _add_column(conn, "task_steps", "worker_id TEXT")
    conn.executescript(
        """
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
        """
    )


def _migration_4(conn: sqlite3.Connection) -> None:
    _add_column(
        conn, "task_steps",
        "executor_binding_json TEXT NOT NULL DEFAULT '{}'")
    _add_column(
        conn, "task_steps",
        "resource_claims_json TEXT NOT NULL DEFAULT '{}'")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS operator_grants (
            grant_id            TEXT PRIMARY KEY REFERENCES nodes(node_id),
            scope_kind          TEXT NOT NULL,
            target_ciphertext   TEXT NOT NULL,
            target_redacted     TEXT NOT NULL,
            target_sha256       TEXT NOT NULL,
            permissions_json    TEXT NOT NULL,
            allow_sensitive     INTEGER NOT NULL DEFAULT 0,
            status              TEXT NOT NULL,
            source_task_id      TEXT REFERENCES task_state(task_id),
            expires_at          TEXT,
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL,
            last_event_seq      INTEGER NOT NULL REFERENCES graph_events(seq)
        );
        CREATE INDEX IF NOT EXISTS operator_grants_active
            ON operator_grants(status, scope_kind, expires_at);
        """
    )


def _migration_5(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS resource_leases (
            lease_id           TEXT PRIMARY KEY REFERENCES nodes(node_id),
            step_id            TEXT NOT NULL,
            attempt_id         TEXT NOT NULL,
            worker_id          TEXT NOT NULL,
            runtime_id         TEXT NOT NULL,
            cpu_millis         INTEGER NOT NULL CHECK(cpu_millis >= 0),
            ram_mib            INTEGER NOT NULL CHECK(ram_mib >= 0),
            concurrency_slots  INTEGER NOT NULL CHECK(concurrency_slots >= 1),
            network_slots      INTEGER NOT NULL CHECK(network_slots >= 0),
            accelerator        TEXT NOT NULL,
            vram_mib           INTEGER NOT NULL CHECK(vram_mib >= 0),
            status             TEXT NOT NULL,
            acquired_at        TEXT NOT NULL,
            heartbeat_at       TEXT NOT NULL,
            expires_at         TEXT NOT NULL,
            released_at        TEXT,
            release_reason     TEXT,
            last_event_seq     INTEGER NOT NULL REFERENCES graph_events(seq),
            UNIQUE(step_id, attempt_id)
        );
        CREATE INDEX IF NOT EXISTS resource_leases_active_expiry
            ON resource_leases(status, expires_at);
        CREATE INDEX IF NOT EXISTS resource_leases_runtime
            ON resource_leases(runtime_id, status);
        """
    )


def _migration_6(conn: sqlite3.Connection) -> None:
    _add_column(conn, "task_steps", "resource_lease_id TEXT")
    _add_column(
        conn, "task_steps",
        "admission_state TEXT NOT NULL DEFAULT 'not_checked'")
    _add_column(conn, "task_steps", "admission_reason TEXT")
    _add_column(conn, "task_steps", "admission_checked_at TEXT")
    _add_column(conn, "task_steps", "next_admission_at TEXT")
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS resource_leases_active_step
           ON resource_leases(step_id) WHERE status='active'""")


def _migration_7(conn: sqlite3.Connection) -> None:
    _add_column(
        conn, "resource_leases",
        "profile_fingerprint TEXT NOT NULL DEFAULT 'legacy'")
    _add_column(
        conn, "resource_leases",
        "latency_class TEXT NOT NULL DEFAULT 'interactive'")


def _migration_8(conn: sqlite3.Connection) -> None:
    """Add durable, privacy-preserving process and workload projections."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS process_specs (
            spec_id               TEXT PRIMARY KEY REFERENCES nodes(node_id),
            name                  TEXT NOT NULL CHECK(length(trim(name)) > 0),
            version               INTEGER NOT NULL CHECK(version >= 1),
            spec_ciphertext       TEXT NOT NULL
                                  CHECK(length(spec_ciphertext) > 0),
            display_json          TEXT NOT NULL
                                  CHECK(CASE WHEN json_valid(display_json)
                                        THEN json_type(display_json) = 'object'
                                        ELSE 0 END),
            spec_sha256           TEXT NOT NULL
                                  CHECK(length(spec_sha256) = 64
                                        AND spec_sha256 NOT GLOB '*[^0-9a-f]*'),
            sandbox_fingerprint   TEXT NOT NULL
                                  CHECK(length(sandbox_fingerprint) = 64
                                        AND sandbox_fingerprint
                                            NOT GLOB '*[^0-9a-f]*'),
            status                TEXT NOT NULL
                                  CHECK(status IN ('active', 'revoked')),
            source_task_id        TEXT REFERENCES task_state(task_id),
            created_at            TEXT NOT NULL CHECK(length(created_at) > 0),
            updated_at            TEXT NOT NULL CHECK(length(updated_at) > 0),
            last_event_seq        INTEGER NOT NULL REFERENCES graph_events(seq),
            UNIQUE(name, version)
        );
        CREATE INDEX IF NOT EXISTS process_specs_status
            ON process_specs(status, name, version);

        CREATE TABLE IF NOT EXISTS process_instances (
            instance_id            TEXT PRIMARY KEY REFERENCES nodes(node_id),
            spec_id                TEXT NOT NULL REFERENCES process_specs(spec_id),
            task_id                TEXT REFERENCES task_state(task_id),
            step_id                TEXT REFERENCES task_steps(step_id),
            action_id              TEXT REFERENCES nodes(node_id),
            launch_idempotency_key TEXT NOT NULL UNIQUE
                                   CHECK(length(launch_idempotency_key) > 0),
            args_ciphertext        TEXT NOT NULL
                                   CHECK(length(args_ciphertext) > 0),
            args_redacted_json     TEXT NOT NULL
                                   CHECK(CASE WHEN json_valid(args_redacted_json)
                                         THEN json_type(args_redacted_json)
                                             = 'object'
                                         ELSE 0 END),
            args_sha256            TEXT NOT NULL
                                   CHECK(length(args_sha256) = 64
                                         AND args_sha256
                                             NOT GLOB '*[^0-9a-f]*'),
            spec_fingerprint       TEXT NOT NULL
                                   CHECK(length(spec_fingerprint) = 64
                                         AND spec_fingerprint
                                             NOT GLOB '*[^0-9a-f]*'),
            sandbox_fingerprint    TEXT NOT NULL
                                   CHECK(length(sandbox_fingerprint) = 64
                                         AND sandbox_fingerprint
                                             NOT GLOB '*[^0-9a-f]*'),
            state                  TEXT NOT NULL CHECK(state IN (
                                       'prepared', 'starting', 'running',
                                       'stop_requested', 'stopping',
                                       'terminated', 'exited', 'launch_failed',
                                       'identity_mismatch',
                                       'reconcile_required')),
            unit_name              TEXT NOT NULL UNIQUE
                                   CHECK(length(trim(unit_name)) > 0),
            boot_id                TEXT,
            invocation_id          TEXT,
            control_group          TEXT,
            leader_pid             INTEGER CHECK(leader_pid > 0),
            start_ticks            INTEGER CHECK(start_ticks >= 0),
            exe_device             INTEGER CHECK(exe_device >= 0),
            exe_inode              INTEGER CHECK(exe_inode > 0),
            exe_sha256             TEXT CHECK(
                                       exe_sha256 IS NULL OR
                                       (length(exe_sha256) = 64 AND
                                        exe_sha256
                                            NOT GLOB '*[^0-9a-f]*')),
            persistent             INTEGER NOT NULL DEFAULT 0
                                   CHECK(persistent IN (0, 1)),
            exit_code              INTEGER,
            exit_signal            INTEGER CHECK(exit_signal IS NULL
                                                    OR exit_signal > 0),
            result_code            TEXT CHECK(result_code IS NULL
                                               OR length(result_code) > 0),
            stdout_bytes           INTEGER NOT NULL DEFAULT 0
                                   CHECK(stdout_bytes >= 0),
            stdout_sha256          TEXT CHECK(
                                       stdout_sha256 IS NULL OR
                                       (length(stdout_sha256) = 64 AND
                                        stdout_sha256
                                            NOT GLOB '*[^0-9a-f]*')),
            stdout_truncated       INTEGER NOT NULL DEFAULT 0
                                   CHECK(stdout_truncated IN (0, 1)),
            stderr_bytes           INTEGER NOT NULL DEFAULT 0
                                   CHECK(stderr_bytes >= 0),
            stderr_sha256          TEXT CHECK(
                                       stderr_sha256 IS NULL OR
                                       (length(stderr_sha256) = 64 AND
                                        stderr_sha256
                                            NOT GLOB '*[^0-9a-f]*')),
            stderr_truncated       INTEGER NOT NULL DEFAULT 0
                                   CHECK(stderr_truncated IN (0, 1)),
            prepared_at            TEXT NOT NULL CHECK(length(prepared_at) > 0),
            started_at             TEXT,
            heartbeat_at           TEXT,
            stop_requested_at      TEXT,
            finished_at            TEXT,
            created_at             TEXT NOT NULL CHECK(length(created_at) > 0),
            updated_at             TEXT NOT NULL CHECK(length(updated_at) > 0),
            last_event_seq         INTEGER NOT NULL REFERENCES graph_events(seq),
            CHECK(
                (boot_id IS NULL AND leader_pid IS NULL AND start_ticks IS NULL)
                OR
                (boot_id IS NOT NULL AND length(boot_id) > 0
                 AND leader_pid IS NOT NULL AND start_ticks IS NOT NULL)
            ),
            CHECK(
                (exe_device IS NULL AND exe_inode IS NULL AND exe_sha256 IS NULL)
                OR
                (exe_device IS NOT NULL AND exe_inode IS NOT NULL
                 AND exe_sha256 IS NOT NULL)
            ),
            CHECK(
                state NOT IN ('running', 'stop_requested', 'stopping')
                OR
                (boot_id IS NOT NULL AND length(boot_id) > 0
                 AND invocation_id IS NOT NULL AND length(invocation_id) > 0
                 AND control_group IS NOT NULL AND length(control_group) > 0
                 AND leader_pid IS NOT NULL AND start_ticks IS NOT NULL
                 AND exe_device IS NOT NULL AND exe_inode IS NOT NULL
                 AND exe_sha256 IS NOT NULL)
            ),
            CHECK(stdout_bytes = 0 OR stdout_sha256 IS NOT NULL),
            CHECK(stderr_bytes = 0 OR stderr_sha256 IS NOT NULL),
            CHECK(state NOT IN ('terminated', 'exited', 'launch_failed')
                  OR finished_at IS NOT NULL)
        );
        CREATE INDEX IF NOT EXISTS process_instances_spec_state
            ON process_instances(spec_id, state, updated_at);
        CREATE INDEX IF NOT EXISTS process_instances_task_state
            ON process_instances(task_id, state, updated_at)
            WHERE task_id IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS process_instances_active_runtime_identity
            ON process_instances(boot_id, leader_pid, start_ticks)
            WHERE boot_id IS NOT NULL
              AND leader_pid IS NOT NULL
              AND start_ticks IS NOT NULL
              AND state IN ('starting', 'running', 'stop_requested', 'stopping',
                            'reconcile_required');

        CREATE TABLE IF NOT EXISTS workload_resource_leases (
            lease_id              TEXT PRIMARY KEY REFERENCES nodes(node_id),
            instance_id           TEXT NOT NULL UNIQUE
                                  REFERENCES process_instances(instance_id),
            source_step_lease_id  TEXT UNIQUE
                                  REFERENCES resource_leases(lease_id),
            source_attempt_id     TEXT,
            source_worker_id      TEXT,
            runtime_id            TEXT NOT NULL CHECK(length(runtime_id) > 0),
            profile_fingerprint   TEXT NOT NULL
                                  CHECK(length(profile_fingerprint) > 0),
            latency_class         TEXT NOT NULL CHECK(latency_class IN (
                                      'interactive', 'background', 'batch')),
            cpu_millis            INTEGER NOT NULL CHECK(cpu_millis >= 0),
            ram_mib               INTEGER NOT NULL CHECK(ram_mib >= 0),
            concurrency_slots     INTEGER NOT NULL
                                  CHECK(concurrency_slots >= 1),
            network_slots         INTEGER NOT NULL CHECK(network_slots >= 0),
            accelerator           TEXT NOT NULL
                                  CHECK(length(trim(accelerator)) > 0),
            vram_mib              INTEGER NOT NULL CHECK(vram_mib >= 0),
            enforcement_json      TEXT NOT NULL DEFAULT '{}'
                                  CHECK(CASE WHEN json_valid(enforcement_json)
                                        THEN json_type(enforcement_json)
                                            = 'object'
                                        ELSE 0 END),
            status                TEXT NOT NULL CHECK(status IN (
                                      'active', 'reconciling', 'released',
                                      'fenced')),
            acquired_at           TEXT NOT NULL CHECK(length(acquired_at) > 0),
            heartbeat_at          TEXT NOT NULL CHECK(length(heartbeat_at) > 0),
            expires_at            TEXT NOT NULL CHECK(length(expires_at) > 0),
            reconcile_started_at  TEXT,
            released_at           TEXT,
            release_reason        TEXT,
            last_event_seq        INTEGER NOT NULL REFERENCES graph_events(seq),
            CHECK(vram_mib = 0 OR accelerator <> 'none'),
            CHECK(
                (source_step_lease_id IS NULL AND source_attempt_id IS NULL
                 AND source_worker_id IS NULL)
                OR
                (source_step_lease_id IS NOT NULL
                 AND source_attempt_id IS NOT NULL
                 AND length(source_attempt_id) > 0
                 AND source_worker_id IS NOT NULL
                 AND length(source_worker_id) > 0)
            ),
            CHECK(status <> 'reconciling' OR reconcile_started_at IS NOT NULL),
            CHECK(
                (status IN ('active', 'reconciling') AND released_at IS NULL)
                OR
                (status IN ('released', 'fenced') AND released_at IS NOT NULL
                 AND release_reason IS NOT NULL
                 AND length(release_reason) > 0)
            )
        );
        CREATE INDEX IF NOT EXISTS workload_resource_leases_reserving
            ON workload_resource_leases(status, expires_at, accelerator)
            WHERE status IN ('active', 'reconciling');
        CREATE INDEX IF NOT EXISTS workload_resource_leases_profile
            ON workload_resource_leases(
                profile_fingerprint, latency_class, status);
        """
    )


def _migration_9(conn: sqlite3.Connection) -> None:
    """Journal crash-recoverable retirement of retained process units."""
    # Keep every statement inside the caller's transaction.  ``executescript``
    # implicitly commits in sqlite3 and would let a partial migration escape if
    # the backfill or version marker failed.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS process_unit_cleanups (
               instance_id      TEXT PRIMARY KEY NOT NULL
                                REFERENCES process_instances(instance_id),
               state            TEXT NOT NULL CHECK(state IN (
                                    'pending','cleaning','complete','blocked')),
               attempt_count    INTEGER NOT NULL DEFAULT 0
                                CHECK(attempt_count >= 0),
               requested_at     TEXT NOT NULL CHECK(length(requested_at) > 0),
               last_attempt_at  TEXT,
               next_attempt_at  TEXT,
               claim_token      TEXT CHECK(
                                    claim_token IS NULL OR
                                    (length(claim_token) = 32 AND
                                     claim_token NOT GLOB '*[^0-9a-f]*')),
               claim_expires_at TEXT,
               completed_at     TEXT,
               last_error_code  TEXT CHECK(
                                    last_error_code IS NULL OR
                                    length(last_error_code) BETWEEN 1 AND 80),
               last_event_seq   INTEGER NOT NULL REFERENCES graph_events(seq),
               CHECK(
                   (state = 'cleaning' AND claim_token IS NOT NULL
                    AND claim_expires_at IS NOT NULL)
                   OR
                   (state <> 'cleaning' AND claim_token IS NULL
                    AND claim_expires_at IS NULL)
               ),
               CHECK(
                   (state = 'complete' AND completed_at IS NOT NULL)
                   OR (state <> 'complete' AND completed_at IS NULL)
               ),
               CHECK(state <> 'blocked' OR last_error_code IS NOT NULL)
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS process_unit_cleanups_due
               ON process_unit_cleanups(
                   state,next_attempt_at,claim_expires_at,requested_at)
               WHERE state IN ('pending','cleaning')"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS process_unit_cleanup_after_insert
               AFTER INSERT ON process_instances
               WHEN NEW.state IN ('terminated','exited','launch_failed')
               BEGIN
                   INSERT OR IGNORE INTO process_unit_cleanups
                       (instance_id,state,attempt_count,requested_at,
                        last_event_seq)
                   VALUES
                       (NEW.instance_id,'pending',0,
                        COALESCE(NEW.finished_at,NEW.updated_at),
                        NEW.last_event_seq);
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS process_unit_cleanup_after_terminal
               AFTER UPDATE OF state ON process_instances
               WHEN OLD.state NOT IN ('terminated','exited','launch_failed')
                AND NEW.state IN ('terminated','exited','launch_failed')
               BEGIN
                   INSERT OR IGNORE INTO process_unit_cleanups
                       (instance_id,state,attempt_count,requested_at,
                        last_event_seq)
                   VALUES
                       (NEW.instance_id,'pending',0,
                        COALESCE(NEW.finished_at,NEW.updated_at),
                        NEW.last_event_seq);
               END"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO process_unit_cleanups
               (instance_id,state,attempt_count,requested_at,last_event_seq)
           SELECT instance_id,'pending',0,
                  COALESCE(finished_at,updated_at),last_event_seq
             FROM process_instances
            WHERE state IN ('terminated','exited','launch_failed')"""
    )


def _migration_10(conn: sqlite3.Connection) -> None:
    """Keep periodic cleanup health proportional to unresolved work."""
    conn.execute(
        """CREATE INDEX IF NOT EXISTS process_unit_cleanups_unresolved
               ON process_unit_cleanups(
                   state,last_error_code,next_attempt_at,claim_expires_at)
               WHERE state <> 'complete'"""
    )


def _migration_11(conn: sqlite3.Connection) -> None:
    """Bind managed-process termination evidence to one exact action attempt."""
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS task_steps_process_operation_identity
               ON task_steps(
                   task_id,step_id,action_id,idempotency_key,tool_name,args_sha256,
                   executor_binding_json
               )"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS action_receipts_exact_identity
               ON action_receipts(
                   task_id,step_id,action_id,idempotency_key,tool_name,args_sha256
               )"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS action_attempts_exact_claim
               ON action_attempts(
                   attempt_id,step_id,idempotency_key,attempt_number,lease_id,
                   worker_id
               )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS process_operations (
               operation_id            TEXT PRIMARY KEY NOT NULL,
               idempotency_key         TEXT NOT NULL UNIQUE,
               task_id                 TEXT NOT NULL,
               step_id                 TEXT NOT NULL UNIQUE,
               action_id               TEXT NOT NULL,
               attempt_id              TEXT NOT NULL UNIQUE,
               attempt_number          INTEGER NOT NULL CHECK(attempt_number >= 1),
               step_lease_id           TEXT NOT NULL,
               worker_id               TEXT NOT NULL,
               tool_name               TEXT NOT NULL
                                        CHECK(tool_name =
                                              'machine_terminate_process'),
               args_sha256             TEXT NOT NULL
                                        CHECK(length(args_sha256) = 64 AND
                                              args_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               instance_id             TEXT NOT NULL
                                        REFERENCES process_instances(instance_id),
               executor_binding_sha256 TEXT NOT NULL
                                        CHECK(length(executor_binding_sha256) = 64
                                              AND executor_binding_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               executor_binding_json   TEXT NOT NULL CHECK(
                                        CASE WHEN json_valid(executor_binding_json)
                                        THEN json_type(executor_binding_json) =
                                             'object'
                                         AND json_extract(
                                               executor_binding_json,'$.kind') =
                                             'process_instance'
                                         AND json_extract(
                                               executor_binding_json,
                                               '$.operation') = 'terminate'
                                         AND json_extract(
                                               executor_binding_json,
                                               '$.instance_id') = instance_id
                                        ELSE 0 END),
               target_boundary_sha256  TEXT NOT NULL
                                        CHECK(length(target_boundary_sha256) = 64
                                              AND target_boundary_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               force                   INTEGER NOT NULL CHECK(force IN (0, 1)),
               status                  TEXT NOT NULL CHECK(status IN (
                                            'prepared','dispatching',
                                            'effect_acknowledged','completed',
                                            'known_failed','outcome_unknown')),
               prepared_event_seq      INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               dispatch_event_seq      INTEGER REFERENCES graph_events(seq),
               outcome_event_seq       INTEGER REFERENCES graph_events(seq),
               postcondition_event_seq INTEGER REFERENCES graph_events(seq),
               error_code              TEXT CHECK(
                                            error_code IS NULL OR
                                            length(error_code) BETWEEN 1 AND 80),
               created_at              TEXT NOT NULL CHECK(length(created_at) > 0),
               updated_at              TEXT NOT NULL CHECK(length(updated_at) > 0),
               completed_at            TEXT,
               last_event_seq          INTEGER NOT NULL REFERENCES graph_events(seq),
               FOREIGN KEY(
                   task_id,step_id,action_id,idempotency_key,tool_name,args_sha256,
                   executor_binding_json
               ) REFERENCES task_steps(
                   task_id,step_id,action_id,idempotency_key,tool_name,args_sha256,
                   executor_binding_json
               ),
               FOREIGN KEY(
                   task_id,step_id,action_id,idempotency_key,tool_name,args_sha256
               ) REFERENCES action_receipts(
                   task_id,step_id,action_id,idempotency_key,tool_name,args_sha256
               ),
               FOREIGN KEY(
                   attempt_id,step_id,idempotency_key,attempt_number,
                   step_lease_id,worker_id
               ) REFERENCES action_attempts(
                   attempt_id,step_id,idempotency_key,attempt_number,lease_id,
                   worker_id
               ),
               CHECK(last_event_seq >= prepared_event_seq),
               CHECK(
                   (status = 'prepared' AND dispatch_event_seq IS NULL
                    AND outcome_event_seq IS NULL
                    AND postcondition_event_seq IS NULL AND error_code IS NULL
                    AND completed_at IS NULL)
                   OR
                   (status = 'dispatching' AND dispatch_event_seq IS NOT NULL
                    AND outcome_event_seq IS NULL
                    AND postcondition_event_seq IS NULL AND error_code IS NULL
                    AND completed_at IS NULL)
                   OR
                   (status = 'effect_acknowledged'
                    AND dispatch_event_seq IS NOT NULL
                    AND outcome_event_seq IS NOT NULL
                    AND postcondition_event_seq IS NULL AND error_code IS NULL
                    AND completed_at IS NULL)
                   OR
                   (status = 'completed' AND dispatch_event_seq IS NOT NULL
                    AND outcome_event_seq IS NOT NULL
                    AND postcondition_event_seq IS NOT NULL
                    AND error_code IS NULL AND completed_at IS NOT NULL)
                   OR
                   (status IN ('known_failed','outcome_unknown')
                    AND dispatch_event_seq IS NOT NULL
                    AND outcome_event_seq IS NOT NULL
                    AND postcondition_event_seq IS NULL
                    AND error_code IS NOT NULL AND completed_at IS NULL)
               )
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS process_operations_instance_status
               ON process_operations(instance_id,status,updated_at)"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS process_operations_action
               ON process_operations(action_id)"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS process_operations_accepted_instance
               ON process_operations(instance_id)
               WHERE status IN ('effect_acknowledged','completed')"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS process_operations_unresolved
               ON process_operations(status,updated_at)
               WHERE status IN (
                   'prepared','dispatching','effect_acknowledged','outcome_unknown'
               )"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS process_operations_identity_immutable
               BEFORE UPDATE ON process_operations
               WHEN OLD.operation_id <> NEW.operation_id
                 OR OLD.idempotency_key <> NEW.idempotency_key
                 OR OLD.task_id <> NEW.task_id
                 OR OLD.step_id <> NEW.step_id
                 OR OLD.action_id <> NEW.action_id
                 OR OLD.attempt_id <> NEW.attempt_id
                 OR OLD.attempt_number <> NEW.attempt_number
                 OR OLD.step_lease_id <> NEW.step_lease_id
                 OR OLD.worker_id <> NEW.worker_id
                 OR OLD.tool_name <> NEW.tool_name
                 OR OLD.args_sha256 <> NEW.args_sha256
                 OR OLD.instance_id <> NEW.instance_id
                 OR OLD.executor_binding_sha256 <>
                    NEW.executor_binding_sha256
                 OR OLD.executor_binding_json <> NEW.executor_binding_json
                 OR OLD.target_boundary_sha256 <> NEW.target_boundary_sha256
                 OR OLD.force <> NEW.force
                 OR OLD.prepared_event_seq <> NEW.prepared_event_seq
                 OR OLD.created_at <> NEW.created_at
               BEGIN
                   SELECT RAISE(ABORT, 'process operation identity is immutable');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS process_operations_state_monotonic
               BEFORE UPDATE OF status ON process_operations
               WHEN NOT (
                   OLD.status = NEW.status
                   OR (OLD.status = 'prepared' AND NEW.status IN (
                       'dispatching','known_failed'))
                   OR (OLD.status = 'dispatching' AND NEW.status IN (
                       'effect_acknowledged','known_failed','outcome_unknown'))
                   OR (OLD.status = 'effect_acknowledged'
                       AND NEW.status = 'completed')
               )
               BEGIN
                   SELECT RAISE(ABORT, 'invalid process operation transition');
               END"""
    )


def _migration_12(conn: sqlite3.Connection) -> None:
    """Add durable paired-controller identities and expiring sessions.

    The installation control token is deliberately absent from this schema.  It
    may authorize creation of a short-lived pairing intent, but it is never a
    controller identity and no bearer secret is stored in SQLite.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS controller_identities (
               controller_id           TEXT PRIMARY KEY
                                        REFERENCES nodes(node_id),
               label                   TEXT NOT NULL
                                        CHECK(length(label) BETWEEN 1 AND 80),
               key_algorithm           TEXT NOT NULL
                                        CHECK(key_algorithm =
                                              'ecdsa-p256-sha256'),
               public_jwk_json         TEXT NOT NULL CHECK(
                                        CASE WHEN json_valid(public_jwk_json)
                                        THEN json_type(public_jwk_json) = 'object'
                                         AND json_extract(public_jwk_json,'$.kty') =
                                             'EC'
                                         AND json_extract(public_jwk_json,'$.crv') =
                                             'P-256'
                                         AND length(json_extract(
                                               public_jwk_json,'$.x')) = 43
                                         AND length(json_extract(
                                               public_jwk_json,'$.y')) = 43
                                        ELSE 0 END),
               public_key_sha256       TEXT NOT NULL UNIQUE
                                        CHECK(length(public_key_sha256) = 64 AND
                                              public_key_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               transport_binding_sha256 TEXT NOT NULL
                                        CHECK(length(transport_binding_sha256) = 64
                                              AND transport_binding_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               status                  TEXT NOT NULL
                                        CHECK(status IN ('active','revoked')),
               auth_epoch              INTEGER NOT NULL DEFAULT 1
                                        CHECK(auth_epoch >= 1),
               paired_at               TEXT NOT NULL CHECK(length(paired_at) > 0),
               last_authenticated_at   TEXT CHECK(
                                        last_authenticated_at IS NULL OR
                                        last_authenticated_at >= paired_at),
               revoked_at              TEXT CHECK(
                                        revoked_at IS NULL OR
                                        revoked_at >= paired_at),
               paired_event_seq        INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               last_event_seq          INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               CHECK(last_event_seq >= paired_event_seq),
               CHECK((status = 'active' AND revoked_at IS NULL) OR
                     (status = 'revoked' AND revoked_at IS NOT NULL))
           )"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS controller_identities_auth_identity
               ON controller_identities(controller_id,public_key_sha256)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS controller_identities_status
               ON controller_identities(status,paired_at)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS controller_pairings (
               pairing_id              TEXT PRIMARY KEY,
               code_digest             TEXT NOT NULL UNIQUE
                                        CHECK(length(code_digest) = 64 AND
                                              code_digest
                                                  NOT GLOB '*[^0-9a-f]*'),
               challenge_sha256        TEXT NOT NULL UNIQUE
                                        CHECK(length(challenge_sha256) = 64 AND
                                              challenge_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               transport_binding_sha256 TEXT NOT NULL
                                        CHECK(length(transport_binding_sha256) = 64
                                              AND transport_binding_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               status                  TEXT NOT NULL CHECK(status IN (
                                            'pending','consumed','expired',
                                            'cancelled')),
               attempts_remaining      INTEGER NOT NULL
                                        CHECK(attempts_remaining BETWEEN 0 AND 8),
               proposed_public_jwk_json TEXT,
               proposed_key_sha256     TEXT CHECK(
                                            proposed_key_sha256 IS NULL OR
                                            (length(proposed_key_sha256) = 64 AND
                                             proposed_key_sha256
                                                 NOT GLOB '*[^0-9a-f]*')),
               proof_signature_sha256  TEXT CHECK(
                                            proof_signature_sha256 IS NULL OR
                                            (length(proof_signature_sha256) = 64 AND
                                             proof_signature_sha256
                                                 NOT GLOB '*[^0-9a-f]*')),
               controller_id           TEXT
                                        REFERENCES controller_identities(
                                            controller_id),
               created_at              TEXT NOT NULL CHECK(length(created_at) > 0),
               expires_at              TEXT NOT NULL CHECK(expires_at > created_at),
               consumed_at             TEXT,
               created_event_seq       INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               last_event_seq          INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               CHECK(last_event_seq >= created_event_seq),
               CHECK((status = 'pending' AND attempts_remaining > 0
                      AND proposed_public_jwk_json IS NULL
                      AND proposed_key_sha256 IS NULL
                      AND proof_signature_sha256 IS NULL
                      AND controller_id IS NULL AND consumed_at IS NULL) OR
                     (status = 'consumed' AND attempts_remaining > 0 AND
                      proposed_public_jwk_json IS NOT NULL
                      AND proposed_key_sha256 IS NOT NULL
                      AND proof_signature_sha256 IS NOT NULL
                      AND controller_id IS NOT NULL
                      AND consumed_at >= created_at
                      AND consumed_at < expires_at) OR
                     (status = 'expired' AND attempts_remaining > 0
                      AND proposed_public_jwk_json IS NULL
                      AND proposed_key_sha256 IS NULL
                      AND proof_signature_sha256 IS NULL
                      AND controller_id IS NULL AND consumed_at IS NULL) OR
                     (status = 'cancelled' AND attempts_remaining = 0
                      AND proposed_public_jwk_json IS NULL
                      AND proposed_key_sha256 IS NULL
                      AND proof_signature_sha256 IS NULL
                      AND controller_id IS NULL AND consumed_at IS NULL)),
               FOREIGN KEY(controller_id,proposed_key_sha256)
                   REFERENCES controller_identities(
                       controller_id,public_key_sha256)
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS controller_pairings_pending
               ON controller_pairings(status,expires_at)
               WHERE status = 'pending'"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS controller_session_challenges (
               challenge_id            TEXT PRIMARY KEY,
               controller_id           TEXT NOT NULL,
               controller_key_sha256   TEXT NOT NULL,
               challenge_sha256        TEXT NOT NULL UNIQUE
                                        CHECK(length(challenge_sha256) = 64 AND
                                              challenge_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               origin_sha256           TEXT NOT NULL
                                        CHECK(length(origin_sha256) = 64 AND
                                              origin_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               transport_binding_sha256 TEXT NOT NULL
                                        CHECK(length(transport_binding_sha256) = 64
                                              AND transport_binding_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               status                  TEXT NOT NULL CHECK(status IN (
                                            'pending','consumed','expired',
                                            'cancelled')),
               attempts_remaining      INTEGER NOT NULL
                                        CHECK(attempts_remaining BETWEEN 0 AND 8),
               created_at              TEXT NOT NULL CHECK(length(created_at) > 0),
               expires_at              TEXT NOT NULL CHECK(expires_at > created_at),
               consumed_at             TEXT,
               created_event_seq       INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               last_event_seq          INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               FOREIGN KEY(controller_id,controller_key_sha256)
                   REFERENCES controller_identities(
                       controller_id,public_key_sha256),
               CHECK(last_event_seq >= created_event_seq),
               CHECK((status = 'pending' AND attempts_remaining > 0
                      AND consumed_at IS NULL) OR
                     (status = 'consumed' AND attempts_remaining > 0
                      AND consumed_at >= created_at
                      AND consumed_at < expires_at) OR
                     (status = 'expired' AND attempts_remaining > 0
                      AND consumed_at IS NULL) OR
                     (status = 'cancelled' AND attempts_remaining = 0
                      AND consumed_at IS NULL))
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS controller_session_challenges_pending
               ON controller_session_challenges(controller_id,status,expires_at)
               WHERE status = 'pending'"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS controller_sessions (
               session_id              TEXT PRIMARY KEY
                                        REFERENCES nodes(node_id),
               controller_id           TEXT NOT NULL,
               controller_key_sha256   TEXT NOT NULL,
               controller_epoch        INTEGER NOT NULL
                                        CHECK(controller_epoch >= 1),
               token_digest            TEXT NOT NULL UNIQUE
                                        CHECK(length(token_digest) = 64 AND
                                              token_digest
                                                  NOT GLOB '*[^0-9a-f]*'),
               origin_sha256           TEXT NOT NULL
                                        CHECK(length(origin_sha256) = 64 AND
                                              origin_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               transport_binding_sha256 TEXT NOT NULL
                                        CHECK(length(transport_binding_sha256) = 64
                                              AND transport_binding_sha256
                                                  NOT GLOB '*[^0-9a-f]*'),
               proof_challenge_sha256  TEXT NOT NULL,
               proof_signature_sha256  TEXT NOT NULL,
               status                  TEXT NOT NULL
                                        CHECK(status IN ('active','revoked','expired')),
               issued_at               TEXT NOT NULL CHECK(length(issued_at) > 0),
               idle_expires_at         TEXT NOT NULL CHECK(idle_expires_at > issued_at),
               absolute_expires_at     TEXT NOT NULL
                                        CHECK(absolute_expires_at >= idle_expires_at),
               last_seen_at            TEXT NOT NULL CHECK(
                                        last_seen_at >= issued_at AND
                                        last_seen_at < idle_expires_at),
               ended_at                TEXT CHECK(
                                        ended_at IS NULL OR ended_at >= issued_at),
               issued_event_seq        INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               last_event_seq          INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               FOREIGN KEY(controller_id,controller_key_sha256)
                   REFERENCES controller_identities(
                       controller_id,public_key_sha256),
               CHECK(length(proof_challenge_sha256) = 64 AND
                     proof_challenge_sha256 NOT GLOB '*[^0-9a-f]*'),
               CHECK(length(proof_signature_sha256) = 64 AND
                     proof_signature_sha256 NOT GLOB '*[^0-9a-f]*'),
               CHECK(last_event_seq >= issued_event_seq),
               CHECK((status = 'active' AND ended_at IS NULL) OR
                     (status IN ('revoked','expired') AND ended_at IS NOT NULL))
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS controller_sessions_active
               ON controller_sessions(controller_id,status,idle_expires_at,
                                      absolute_expires_at)"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS controller_sessions_exact_identity
               ON controller_sessions(session_id,controller_id,
                                      controller_key_sha256,absolute_expires_at,
                                      transport_binding_sha256)"""
    )

    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_identities_identity_immutable
               BEFORE UPDATE ON controller_identities
               WHEN OLD.controller_id <> NEW.controller_id
                 OR OLD.label <> NEW.label
                 OR OLD.key_algorithm <> NEW.key_algorithm
                 OR OLD.public_jwk_json <> NEW.public_jwk_json
                 OR OLD.public_key_sha256 <> NEW.public_key_sha256
                 OR OLD.transport_binding_sha256 <>
                    NEW.transport_binding_sha256
                 OR OLD.paired_at <> NEW.paired_at
                 OR OLD.paired_event_seq <> NEW.paired_event_seq
               BEGIN
                   SELECT RAISE(ABORT, 'controller identity is immutable');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_identities_state_monotonic
               BEFORE UPDATE ON controller_identities
               WHEN NOT (
                   (OLD.status = 'active' AND NEW.status = 'active'
                    AND NEW.auth_epoch = OLD.auth_epoch
                    AND NEW.revoked_at IS OLD.revoked_at
                    AND NEW.last_event_seq = OLD.last_event_seq
                    AND (NEW.last_authenticated_at IS
                         OLD.last_authenticated_at
                         OR (NEW.last_authenticated_at IS NOT NULL
                             AND (OLD.last_authenticated_at IS NULL
                                  OR NEW.last_authenticated_at >=
                                     OLD.last_authenticated_at))))
                   OR (OLD.status = 'active' AND NEW.status = 'revoked'
                       AND NEW.auth_epoch = OLD.auth_epoch + 1
                       AND NEW.revoked_at IS NOT NULL
                       AND NEW.last_authenticated_at IS
                           OLD.last_authenticated_at
                       AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status = 'revoked' AND NEW.status = 'revoked'
                       AND NEW.auth_epoch = OLD.auth_epoch
                       AND NEW.revoked_at IS OLD.revoked_at
                       AND NEW.last_authenticated_at IS
                           OLD.last_authenticated_at
                       AND NEW.last_event_seq = OLD.last_event_seq)
               )
               BEGIN
                   SELECT RAISE(ABORT, 'invalid controller transition');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_pairings_identity_immutable
               BEFORE UPDATE ON controller_pairings
               WHEN OLD.pairing_id <> NEW.pairing_id
                 OR OLD.code_digest <> NEW.code_digest
                 OR OLD.challenge_sha256 <> NEW.challenge_sha256
                 OR OLD.transport_binding_sha256 <>
                    NEW.transport_binding_sha256
                 OR OLD.created_at <> NEW.created_at
                 OR OLD.expires_at <> NEW.expires_at
                 OR OLD.created_event_seq <> NEW.created_event_seq
               BEGIN
                   SELECT RAISE(ABORT, 'controller pairing identity is immutable');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_pairings_state_monotonic
               BEFORE UPDATE ON controller_pairings
               WHEN NOT (
                   (OLD.status = 'pending' AND NEW.status = 'pending'
                    AND NEW.attempts_remaining = OLD.attempts_remaining - 1
                    AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status = 'pending' AND NEW.status = 'consumed'
                       AND NEW.attempts_remaining = OLD.attempts_remaining
                       AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status = 'pending' AND NEW.status = 'expired'
                       AND NEW.attempts_remaining = OLD.attempts_remaining
                       AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status = 'pending' AND NEW.status = 'cancelled'
                       AND OLD.attempts_remaining = 1
                       AND NEW.attempts_remaining = 0
                       AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status <> 'pending' AND NEW.status = OLD.status
                       AND NEW.attempts_remaining = OLD.attempts_remaining
                       AND NEW.proposed_public_jwk_json IS
                           OLD.proposed_public_jwk_json
                       AND NEW.proposed_key_sha256 IS
                           OLD.proposed_key_sha256
                       AND NEW.proof_signature_sha256 IS
                           OLD.proof_signature_sha256
                       AND NEW.controller_id IS OLD.controller_id
                       AND NEW.consumed_at IS OLD.consumed_at
                       AND NEW.last_event_seq = OLD.last_event_seq)
               )
               BEGIN
                   SELECT RAISE(ABORT, 'invalid controller pairing transition');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_challenges_identity_immutable
               BEFORE UPDATE ON controller_session_challenges
               WHEN OLD.challenge_id <> NEW.challenge_id
                 OR OLD.controller_id <> NEW.controller_id
                 OR OLD.controller_key_sha256 <> NEW.controller_key_sha256
                 OR OLD.challenge_sha256 <> NEW.challenge_sha256
                 OR OLD.origin_sha256 <> NEW.origin_sha256
                 OR OLD.transport_binding_sha256 <>
                    NEW.transport_binding_sha256
                 OR OLD.created_at <> NEW.created_at
                 OR OLD.expires_at <> NEW.expires_at
                 OR OLD.created_event_seq <> NEW.created_event_seq
               BEGIN
                   SELECT RAISE(ABORT, 'controller challenge identity is immutable');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_challenges_state_monotonic
               BEFORE UPDATE ON controller_session_challenges
               WHEN NOT (
                   (OLD.status = 'pending' AND NEW.status = 'pending'
                    AND NEW.attempts_remaining = OLD.attempts_remaining - 1
                    AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status = 'pending' AND NEW.status = 'consumed'
                       AND NEW.attempts_remaining = OLD.attempts_remaining
                       AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status = 'pending' AND NEW.status = 'expired'
                       AND NEW.attempts_remaining = OLD.attempts_remaining
                       AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status = 'pending' AND NEW.status = 'cancelled'
                       AND OLD.attempts_remaining = 1
                       AND NEW.attempts_remaining = 0
                       AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status <> 'pending' AND NEW.status = OLD.status
                       AND NEW.attempts_remaining = OLD.attempts_remaining
                       AND NEW.consumed_at IS OLD.consumed_at
                       AND NEW.last_event_seq = OLD.last_event_seq)
               )
               BEGIN
                   SELECT RAISE(ABORT, 'invalid controller challenge transition');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_sessions_identity_immutable
               BEFORE UPDATE ON controller_sessions
               WHEN OLD.session_id <> NEW.session_id
                 OR OLD.controller_id <> NEW.controller_id
                 OR OLD.controller_key_sha256 <> NEW.controller_key_sha256
                 OR OLD.controller_epoch <> NEW.controller_epoch
                 OR OLD.token_digest <> NEW.token_digest
                 OR OLD.origin_sha256 <> NEW.origin_sha256
                 OR OLD.transport_binding_sha256 <>
                    NEW.transport_binding_sha256
                 OR OLD.proof_challenge_sha256 <>
                    NEW.proof_challenge_sha256
                 OR OLD.proof_signature_sha256 <>
                    NEW.proof_signature_sha256
                 OR OLD.issued_at <> NEW.issued_at
                 OR OLD.absolute_expires_at <> NEW.absolute_expires_at
                 OR OLD.issued_event_seq <> NEW.issued_event_seq
               BEGIN
                   SELECT RAISE(ABORT, 'controller session identity is immutable');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_sessions_state_monotonic
               BEFORE UPDATE ON controller_sessions
               WHEN NOT (
                   (OLD.status = 'active' AND NEW.status = 'active'
                    AND NEW.ended_at IS NULL
                    AND NEW.last_seen_at >= OLD.last_seen_at
                    AND NEW.idle_expires_at >= OLD.idle_expires_at
                    AND NEW.idle_expires_at <= NEW.absolute_expires_at
                    AND NEW.last_event_seq = OLD.last_event_seq)
                   OR (OLD.status = 'active' AND NEW.status IN (
                       'revoked','expired')
                       AND NEW.last_seen_at = OLD.last_seen_at
                       AND NEW.idle_expires_at = OLD.idle_expires_at
                       AND NEW.ended_at IS NOT NULL
                       AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status <> 'active' AND NEW.status = OLD.status
                       AND NEW.last_seen_at = OLD.last_seen_at
                       AND NEW.idle_expires_at = OLD.idle_expires_at
                       AND NEW.ended_at IS OLD.ended_at
                       AND NEW.last_event_seq = OLD.last_event_seq)
               )
               BEGIN
                   SELECT RAISE(ABORT, 'invalid controller session transition');
               END"""
    )


def _migration_13(conn: sqlite3.Connection) -> None:
    """Bind interactive tasks and first effect use to controller authority."""
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS
               controller_sessions_authority_identity
               ON controller_sessions(
                   session_id,controller_id,controller_key_sha256,
                   controller_epoch,absolute_expires_at,
                   transport_binding_sha256,origin_sha256)"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS approval_state_exact_request
               ON approval_state(approval_id,task_id,step_id,tool_name)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS controller_task_authorities (
               task_id                  TEXT PRIMARY KEY
                                        REFERENCES task_state(task_id),
               controller_id            TEXT NOT NULL,
               controller_key_sha256    TEXT NOT NULL,
               controller_epoch         INTEGER NOT NULL
                                        CHECK(controller_epoch >= 1),
               session_id               TEXT NOT NULL,
               session_absolute_expires_at TEXT NOT NULL,
               transport_binding_sha256 TEXT NOT NULL CHECK(
                                        length(transport_binding_sha256) = 64 AND
                                        transport_binding_sha256
                                            NOT GLOB '*[^0-9a-f]*'),
               origin_sha256            TEXT NOT NULL CHECK(
                                        length(origin_sha256) = 64 AND
                                        origin_sha256 NOT GLOB '*[^0-9a-f]*'),
               bound_at                 TEXT NOT NULL,
               bound_event_seq          INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               FOREIGN KEY(
                   session_id,controller_id,controller_key_sha256,
                   controller_epoch,session_absolute_expires_at,
                   transport_binding_sha256,origin_sha256)
                   REFERENCES controller_sessions(
                       session_id,controller_id,controller_key_sha256,
                       controller_epoch,absolute_expires_at,
                       transport_binding_sha256,origin_sha256),
               FOREIGN KEY(controller_id,controller_key_sha256)
                   REFERENCES controller_identities(
                       controller_id,public_key_sha256),
               UNIQUE(task_id,controller_id,controller_key_sha256,
                      controller_epoch)
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS controller_approval_requests (
               approval_id              TEXT PRIMARY KEY,
               task_id                  TEXT NOT NULL,
               step_id                  TEXT NOT NULL,
               tool_name                TEXT NOT NULL,
               args_sha256              TEXT NOT NULL CHECK(
                                        length(args_sha256) = 64 AND
                                        args_sha256 NOT GLOB '*[^0-9a-f]*'),
               controller_id            TEXT NOT NULL,
               controller_key_sha256    TEXT NOT NULL,
               controller_epoch         INTEGER NOT NULL
                                        CHECK(controller_epoch >= 1),
               request_session_id       TEXT NOT NULL,
               session_absolute_expires_at TEXT NOT NULL,
               transport_binding_sha256 TEXT NOT NULL CHECK(
                                        length(transport_binding_sha256) = 64 AND
                                        transport_binding_sha256
                                            NOT GLOB '*[^0-9a-f]*'),
               origin_sha256            TEXT NOT NULL CHECK(
                                        length(origin_sha256) = 64 AND
                                        origin_sha256 NOT GLOB '*[^0-9a-f]*'),
               requested_at             TEXT NOT NULL,
               expires_at               TEXT NOT NULL CHECK(
                                        expires_at > requested_at AND
                                        expires_at <=
                                            session_absolute_expires_at),
               requested_event_seq      INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               FOREIGN KEY(approval_id,task_id,step_id,tool_name)
                   REFERENCES approval_state(
                       approval_id,task_id,step_id,tool_name),
               FOREIGN KEY(task_id,controller_id,controller_key_sha256,
                           controller_epoch)
                   REFERENCES controller_task_authorities(
                       task_id,controller_id,controller_key_sha256,
                       controller_epoch),
               FOREIGN KEY(
                   request_session_id,controller_id,controller_key_sha256,
                   controller_epoch,session_absolute_expires_at,
                   transport_binding_sha256,origin_sha256)
                   REFERENCES controller_sessions(
                       session_id,controller_id,controller_key_sha256,
                       controller_epoch,absolute_expires_at,
                       transport_binding_sha256,origin_sha256),
               UNIQUE(approval_id,task_id,step_id,tool_name,args_sha256,
                      controller_id,controller_key_sha256,controller_epoch)
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS controller_approval_requests_expiry
               ON controller_approval_requests(expires_at,approval_id)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS controller_approval_decisions (
               decision_id              TEXT PRIMARY KEY
                                        REFERENCES nodes(node_id),
               approval_id              TEXT NOT NULL UNIQUE,
               task_id                  TEXT NOT NULL,
               step_id                  TEXT NOT NULL,
               tool_name                TEXT NOT NULL,
               args_sha256              TEXT NOT NULL CHECK(
                                        length(args_sha256) = 64 AND
                                        args_sha256 NOT GLOB '*[^0-9a-f]*'),
               controller_id            TEXT NOT NULL,
               controller_key_sha256    TEXT NOT NULL,
               controller_epoch         INTEGER NOT NULL
                                        CHECK(controller_epoch >= 1),
               decision_session_id      TEXT NOT NULL,
               session_absolute_expires_at TEXT NOT NULL,
               transport_binding_sha256 TEXT NOT NULL CHECK(
                                        length(transport_binding_sha256) = 64 AND
                                        transport_binding_sha256
                                            NOT GLOB '*[^0-9a-f]*'),
               origin_sha256            TEXT NOT NULL CHECK(
                                        length(origin_sha256) = 64 AND
                                        origin_sha256 NOT GLOB '*[^0-9a-f]*'),
               decision                 TEXT NOT NULL
                                        CHECK(decision IN ('approved','denied')),
               proof_payload_json       TEXT NOT NULL CHECK(
                                        json_valid(proof_payload_json) AND
                                        json_type(proof_payload_json)='object'),
               proof_payload_sha256     TEXT NOT NULL CHECK(
                                        length(proof_payload_sha256) = 64 AND
                                        proof_payload_sha256
                                            NOT GLOB '*[^0-9a-f]*'),
               signature_b64url         TEXT NOT NULL CHECK(
                                        length(signature_b64url) BETWEEN 80 AND 96
                                        AND signature_b64url
                                            NOT GLOB '*[^A-Za-z0-9_-]*'),
               signature_sha256        TEXT NOT NULL CHECK(
                                        length(signature_sha256) = 64 AND
                                        signature_sha256
                                            NOT GLOB '*[^0-9a-f]*'),
               decided_at              TEXT NOT NULL,
               authorization_expires_at TEXT,
               decided_event_seq       INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               FOREIGN KEY(
                   approval_id,task_id,step_id,tool_name,args_sha256,
                   controller_id,controller_key_sha256,controller_epoch)
                   REFERENCES controller_approval_requests(
                       approval_id,task_id,step_id,tool_name,args_sha256,
                       controller_id,controller_key_sha256,controller_epoch),
               FOREIGN KEY(task_id,controller_id,controller_key_sha256,
                           controller_epoch)
                   REFERENCES controller_task_authorities(
                       task_id,controller_id,controller_key_sha256,
                       controller_epoch),
               FOREIGN KEY(
                   decision_session_id,controller_id,controller_key_sha256,
                   controller_epoch,session_absolute_expires_at,
                   transport_binding_sha256,origin_sha256)
                   REFERENCES controller_sessions(
                       session_id,controller_id,controller_key_sha256,
                       controller_epoch,absolute_expires_at,
                       transport_binding_sha256,origin_sha256),
               CHECK((decision='approved' AND
                      authorization_expires_at > decided_at AND
                      authorization_expires_at <=
                          session_absolute_expires_at) OR
                     (decision='denied' AND
                      authorization_expires_at IS NULL)),
               UNIQUE(decision_id,approval_id,task_id,step_id,tool_name,
                      args_sha256,controller_id,controller_key_sha256,
                      controller_epoch,decision_session_id)
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS controller_approval_decisions_expiry
               ON controller_approval_decisions(
                   decision,authorization_expires_at,approval_id)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS controller_effect_uses (
               action_id                TEXT PRIMARY KEY,
               task_id                  TEXT NOT NULL,
               step_id                  TEXT NOT NULL UNIQUE,
               idempotency_key          TEXT NOT NULL UNIQUE,
               tool_name                TEXT NOT NULL,
               args_sha256              TEXT NOT NULL CHECK(
                                        length(args_sha256) = 64 AND
                                        args_sha256 NOT GLOB '*[^0-9a-f]*'),
               controller_id            TEXT NOT NULL,
               controller_key_sha256    TEXT NOT NULL,
               controller_epoch         INTEGER NOT NULL
                                        CHECK(controller_epoch >= 1),
               authorizing_session_id   TEXT NOT NULL,
               session_absolute_expires_at TEXT NOT NULL,
               transport_binding_sha256 TEXT NOT NULL CHECK(
                                        length(transport_binding_sha256) = 64 AND
                                        transport_binding_sha256
                                            NOT GLOB '*[^0-9a-f]*'),
               origin_sha256            TEXT NOT NULL CHECK(
                                        length(origin_sha256) = 64 AND
                                        origin_sha256 NOT GLOB '*[^0-9a-f]*'),
               approval_id              TEXT,
               decision_id              TEXT UNIQUE,
               committed_at             TEXT NOT NULL,
               committed_event_seq      INTEGER NOT NULL
                                        REFERENCES graph_events(seq),
               FOREIGN KEY(task_id,step_id,action_id,idempotency_key,
                           tool_name,args_sha256)
                   REFERENCES action_receipts(
                       task_id,step_id,action_id,idempotency_key,
                       tool_name,args_sha256),
               FOREIGN KEY(task_id,controller_id,controller_key_sha256,
                           controller_epoch)
                   REFERENCES controller_task_authorities(
                       task_id,controller_id,controller_key_sha256,
                       controller_epoch),
               FOREIGN KEY(
                   authorizing_session_id,controller_id,controller_key_sha256,
                   controller_epoch,session_absolute_expires_at,
                   transport_binding_sha256,origin_sha256)
                   REFERENCES controller_sessions(
                       session_id,controller_id,controller_key_sha256,
                       controller_epoch,absolute_expires_at,
                       transport_binding_sha256,origin_sha256),
               FOREIGN KEY(
                   decision_id,approval_id,task_id,step_id,tool_name,
                   args_sha256,controller_id,controller_key_sha256,
                   controller_epoch,authorizing_session_id)
                   REFERENCES controller_approval_decisions(
                       decision_id,approval_id,task_id,step_id,tool_name,
                       args_sha256,controller_id,controller_key_sha256,
                       controller_epoch,decision_session_id),
               CHECK((approval_id IS NULL AND decision_id IS NULL) OR
                     (approval_id IS NOT NULL AND decision_id IS NOT NULL))
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS controller_effect_uses_controller
               ON controller_effect_uses(controller_id,committed_at)"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS approval_state_identity_immutable
               BEFORE UPDATE ON approval_state
               WHEN OLD.approval_id <> NEW.approval_id
                 OR OLD.task_id <> NEW.task_id
                 OR OLD.step_id IS NOT NEW.step_id
                 OR OLD.tool_name <> NEW.tool_name
                 OR OLD.args_json <> NEW.args_json
                 OR OLD.reason <> NEW.reason
                 OR OLD.created_at <> NEW.created_at
               BEGIN
                   SELECT RAISE(ABORT, 'approval identity is immutable');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS approval_state_lifecycle_monotonic
               BEFORE UPDATE ON approval_state
               WHEN NOT (
                   (OLD.status='pending'
                    AND NEW.status IN ('approved','denied','cancelled','expired')
                    AND OLD.decided_at IS NULL
                    AND NEW.decided_at IS NOT NULL
                    AND NEW.last_event_seq > OLD.last_event_seq)
                   OR (OLD.status<>'pending' AND NEW.status=OLD.status
                       AND NEW.decided_at IS OLD.decided_at
                       AND NEW.last_event_seq=OLD.last_event_seq)
               )
               BEGIN
                   SELECT RAISE(ABORT, 'invalid approval transition');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_task_authority_valid
               BEFORE INSERT ON controller_task_authorities
               WHEN NOT EXISTS (
                   SELECT 1
                     FROM controller_sessions s
                     JOIN controller_identities i
                       ON i.controller_id=s.controller_id
                    WHERE s.session_id=NEW.session_id
                      AND s.controller_id=NEW.controller_id
                      AND s.controller_key_sha256=NEW.controller_key_sha256
                      AND s.controller_epoch=NEW.controller_epoch
                      AND s.absolute_expires_at=
                          NEW.session_absolute_expires_at
                      AND s.transport_binding_sha256=
                          NEW.transport_binding_sha256
                      AND s.origin_sha256=NEW.origin_sha256
                      AND s.status='active' AND i.status='active'
                      AND i.auth_epoch=NEW.controller_epoch
                      AND s.idle_expires_at>NEW.bound_at
                      AND s.absolute_expires_at>NEW.bound_at)
               BEGIN
                   SELECT RAISE(ABORT, 'controller task authority is stale');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_approval_request_valid
               BEFORE INSERT ON controller_approval_requests
               WHEN NOT EXISTS (
                   SELECT 1
                     FROM approval_state a
                     JOIN controller_sessions s
                       ON s.session_id=NEW.request_session_id
                     JOIN controller_identities i
                       ON i.controller_id=NEW.controller_id
                    WHERE a.approval_id=NEW.approval_id
                      AND a.task_id=NEW.task_id
                      AND a.step_id=NEW.step_id
                      AND a.tool_name=NEW.tool_name
                      AND a.status='pending'
                      AND json_extract(a.args_json,'$._args_sha256')=
                          NEW.args_sha256
                      AND s.controller_id=NEW.controller_id
                      AND s.controller_key_sha256=NEW.controller_key_sha256
                      AND s.controller_epoch=NEW.controller_epoch
                      AND s.absolute_expires_at=
                          NEW.session_absolute_expires_at
                      AND s.transport_binding_sha256=
                          NEW.transport_binding_sha256
                      AND s.origin_sha256=NEW.origin_sha256
                      AND s.status='active' AND i.status='active'
                      AND i.auth_epoch=NEW.controller_epoch
                      AND s.idle_expires_at>NEW.requested_at
                      AND s.absolute_expires_at>NEW.requested_at)
               BEGIN
                   SELECT RAISE(ABORT, 'controller approval request is stale');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_approval_decision_valid
               BEFORE INSERT ON controller_approval_decisions
               WHEN NOT EXISTS (
                   SELECT 1
                     FROM controller_approval_requests r
                     JOIN approval_state a
                       ON a.approval_id=r.approval_id
                     JOIN controller_sessions s
                       ON s.session_id=NEW.decision_session_id
                     JOIN controller_identities i
                       ON i.controller_id=NEW.controller_id
                    WHERE r.approval_id=NEW.approval_id
                      AND r.task_id=NEW.task_id
                      AND r.step_id=NEW.step_id
                      AND r.tool_name=NEW.tool_name
                      AND r.args_sha256=NEW.args_sha256
                      AND r.controller_id=NEW.controller_id
                      AND r.controller_key_sha256=NEW.controller_key_sha256
                      AND r.controller_epoch=NEW.controller_epoch
                      AND r.expires_at>NEW.decided_at
                      AND a.status=NEW.decision
                      AND s.controller_id=NEW.controller_id
                      AND s.controller_key_sha256=NEW.controller_key_sha256
                      AND s.controller_epoch=NEW.controller_epoch
                      AND s.absolute_expires_at=
                          NEW.session_absolute_expires_at
                      AND s.transport_binding_sha256=
                          NEW.transport_binding_sha256
                      AND s.origin_sha256=NEW.origin_sha256
                      AND s.status='active' AND i.status='active'
                      AND i.auth_epoch=NEW.controller_epoch
                      AND s.idle_expires_at>NEW.decided_at
                      AND s.absolute_expires_at>NEW.decided_at)
               BEGIN
                   SELECT RAISE(ABORT, 'controller approval decision is stale');
               END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS controller_effect_use_valid
               BEFORE INSERT ON controller_effect_uses
               WHEN NOT EXISTS (
                   SELECT 1
                     FROM action_receipts a
                     JOIN controller_task_authorities t
                       ON t.task_id=NEW.task_id
                     JOIN controller_sessions s
                       ON s.session_id=NEW.authorizing_session_id
                     JOIN controller_identities i
                       ON i.controller_id=NEW.controller_id
                    WHERE a.task_id=NEW.task_id
                      AND a.step_id=NEW.step_id
                      AND a.action_id=NEW.action_id
                      AND a.idempotency_key=NEW.idempotency_key
                      AND a.tool_name=NEW.tool_name
                      AND a.args_sha256=NEW.args_sha256
                      AND t.controller_id=NEW.controller_id
                      AND t.controller_key_sha256=NEW.controller_key_sha256
                      AND t.controller_epoch=NEW.controller_epoch
                      AND s.controller_id=NEW.controller_id
                      AND s.controller_key_sha256=NEW.controller_key_sha256
                      AND s.controller_epoch=NEW.controller_epoch
                      AND s.absolute_expires_at=
                          NEW.session_absolute_expires_at
                      AND s.transport_binding_sha256=
                          NEW.transport_binding_sha256
                      AND s.origin_sha256=NEW.origin_sha256
                      AND s.status='active' AND i.status='active'
                      AND i.auth_epoch=NEW.controller_epoch
                      AND s.idle_expires_at>NEW.committed_at
                      AND s.absolute_expires_at>NEW.committed_at
                      AND ((NEW.decision_id IS NULL
                            AND a.approval_status='not_required'
                            AND NEW.authorizing_session_id=t.session_id)
                           OR (NEW.decision_id IS NOT NULL
                               AND a.approval_status='approved'
                               AND EXISTS (
                                   SELECT 1
                                     FROM controller_approval_decisions d
                                    WHERE d.decision_id=NEW.decision_id
                                      AND d.approval_id=NEW.approval_id
                                      AND d.decision='approved'
                                      AND d.authorization_expires_at>
                                          NEW.committed_at))))
               BEGIN
                   SELECT RAISE(ABORT, 'controller effect authority is stale');
               END"""
    )
    for table in (
        "controller_task_authorities", "controller_approval_requests",
        "controller_approval_decisions", "controller_effect_uses",
    ):
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS {table}_immutable
                   BEFORE UPDATE ON {table}
                   BEGIN
                       SELECT RAISE(ABORT, 'controller authority is immutable');
                   END"""
        )
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS {table}_retain
                   BEFORE DELETE ON {table}
                   BEGIN
                       SELECT RAISE(ABORT, 'controller authority is retained');
                   END"""
        )
    for table in (
        "controller_identities", "controller_pairings",
        "controller_session_challenges", "controller_sessions",
    ):
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS {table}_retain
                   BEFORE DELETE ON {table}
                   BEGIN
                       SELECT RAISE(ABORT, 'controller evidence is retained');
                   END"""
        )


def _migration_14(conn: sqlite3.Connection) -> None:
    """Add a rebuildable, model-pinned local semantic-memory projection."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS memory_embedding_index (
               claim_id          TEXT NOT NULL REFERENCES nodes(node_id),
               model_fingerprint TEXT NOT NULL
                   CHECK(length(model_fingerprint) = 64),
               content_sha256    TEXT NOT NULL CHECK(length(content_sha256) = 64),
               dimension         INTEGER NOT NULL
                   CHECK(dimension > 0 AND dimension <= 4096),
               vector            BLOB NOT NULL CHECK(length(vector) = dimension * 4),
               indexed_at        TEXT NOT NULL,
               PRIMARY KEY (claim_id, model_fingerprint)
           )""")
    conn.execute(
        """CREATE INDEX IF NOT EXISTS memory_embedding_model
           ON memory_embedding_index(model_fingerprint, claim_id)""")


MIGRATIONS = ((1, _migration_1), (2, _migration_2), (3, _migration_3),
              (4, _migration_4), (5, _migration_5), (6, _migration_6),
              (7, _migration_7), (8, _migration_8), (9, _migration_9),
              (10, _migration_10), (11, _migration_11),
              (12, _migration_12), (13, _migration_13),
              (14, _migration_14))


def apply_schema_migrations(conn: sqlite3.Connection) -> int:
    """Apply every missing migration and return the resulting schema version."""
    owns_write_lock = not conn.in_transaction
    if owns_write_lock:
        # Serialize version discovery with the migration itself.  Otherwise
        # two fresh runtimes can both observe v9 as missing and one will lose
        # the marker race after executing the same DDL/backfill.
        conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                   version INTEGER PRIMARY KEY,
                   applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        applied = {int(row[0]) for row in conn.execute(
            "SELECT version FROM schema_migrations")}
        reported_version = int(
            conn.execute("PRAGMA user_version").fetchone()[0])
        if reported_version > LATEST_SCHEMA_VERSION or any(
                version > LATEST_SCHEMA_VERSION for version in applied):
            raise RuntimeError(
                "database schema is newer than this Friday runtime")
        if applied:
            highest = max(applied)
            if applied != set(range(1, highest + 1)):
                raise RuntimeError(
                    "database schema migration history is incomplete")
        for version, migration in MIGRATIONS:
            if version in applied:
                continue
            if version >= 9:
                # Legacy migrations contain ``executescript`` and cannot be
                # wrapped retroactively.  Every new migration is savepoint-
                # atomic so DDL, backfill, and marker land together.
                savepoint = f"friday_schema_migration_{version}"
                conn.execute(f"SAVEPOINT {savepoint}")
                try:
                    migration(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations(version) VALUES (?)",
                        (version,))
                except Exception:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    raise
                else:
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            else:
                migration(conn)
                conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)",
                    (version,))
        conn.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION}")
    except Exception:
        if owns_write_lock and conn.in_transaction:
            conn.rollback()
        raise
    else:
        if owns_write_lock and conn.in_transaction:
            conn.commit()
    return LATEST_SCHEMA_VERSION
