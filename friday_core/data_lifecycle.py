"""Private, verifiable exports of Friday's durable SQLite state."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import urllib.parse
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .db_migrations import apply_schema_migrations


EXPORT_FORMAT = "friday-private-export"
EXPORT_FORMAT_VERSION = 1
DATABASE_NAME = "friday.sqlite3"
MANIFEST_NAME = "manifest.json"
MAX_DATABASE_BYTES = 128 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
DELETION_SCOPES = frozenset({
    "conversation", "task", "artifact", "memory_claim", "time_range",
})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _sha256_private(path: Path, *, maximum: int) -> tuple[str, os.stat_result]:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"private export file is unavailable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or before.st_mode & 0o077
                or not 1 <= before.st_size <= maximum):
            raise RuntimeError(f"private export file is invalid: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns)):
            raise RuntimeError(f"private export file changed while being read: {path}")
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def _private_regular_metadata(
    path: Path,
    *,
    minimum: int,
    maximum: int,
) -> os.stat_result:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"private export file is unavailable: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
                or not minimum <= metadata.st_size <= maximum):
            raise RuntimeError(f"private export file is invalid: {path}")
        return metadata
    finally:
        os.close(descriptor)


def _read_private_regular(path: Path, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"private export file is unavailable: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
                or not 1 <= metadata.st_size <= maximum):
            raise RuntimeError(f"private export file is invalid: {path}")
        encoded = b""
        while len(encoded) <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(encoded)))
            if not chunk:
                break
            encoded += chunk
        if len(encoded) != metadata.st_size:
            raise RuntimeError(f"private export file changed while being read: {path}")
        return encoded
    finally:
        os.close(descriptor)


def _validate_private_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"private export directory is unavailable: {path}") from exc
    if (not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077):
        raise RuntimeError(f"private export directory is invalid: {path}")


def _open_read_only(path: Path) -> sqlite3.Connection:
    quoted = urllib.parse.quote(str(path.resolve()), safe="/")
    connection = sqlite3.connect(f"file:{quoted}?mode=ro", uri=True, timeout=15)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    return connection


def _backup_database(source: Path, destination: Path) -> None:
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
             | getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(destination, flags, 0o600)
    os.close(descriptor)
    with _open_read_only(source) as source_connection:
        destination_connection = sqlite3.connect(destination)
        try:
            source_connection.backup(destination_connection)
            destination_connection.execute("PRAGMA journal_mode = DELETE")
            integrity = destination_connection.execute(
                "PRAGMA integrity_check").fetchall()
            if integrity != [("ok",)]:
                raise RuntimeError("exported database failed SQLite integrity check")
        finally:
            destination_connection.close()
    os.chmod(destination, 0o600)
    descriptor = os.open(destination, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _database_inventory(path: Path) -> tuple[int, list[dict[str, Any]]]:
    with _open_read_only(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise RuntimeError("exported database failed SQLite integrity check")
        schema_row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        schema_version = int(schema_row[0]) if schema_row is not None else 0
        rows = connection.execute("PRAGMA table_list").fetchall()
        tables: list[dict[str, Any]] = []
        for schema, name, kind, _columns, _without_rowid, _strict in rows:
            if (schema != "main" or kind not in {"table", "virtual"}
                    or name.startswith("sqlite_")):
                continue
            count = int(connection.execute(
                f"SELECT COUNT(*) FROM {_quote_identifier(name)}"
            ).fetchone()[0])
            tables.append({"name": name, "type": kind, "rows": count})
    tables.sort(key=lambda item: item["name"])
    return schema_version, tables


def _manifest_for(path: Path) -> dict[str, Any]:
    schema_version, tables = _database_inventory(path)
    digest, metadata = _sha256_private(path, maximum=MAX_DATABASE_BYTES)
    return {
        "format": EXPORT_FORMAT,
        "format_version": EXPORT_FORMAT_VERSION,
        "created_at": _utc_now(),
        "schema_version": schema_version,
        "database": {
            "filename": DATABASE_NAME,
            "bytes": metadata.st_size,
            "sha256": digest,
        },
        "tables": tables,
    }


def _write_private_file(path: Path, body: bytes) -> None:
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
             | getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def export_private_data(database: str | Path, target: str | Path) -> dict[str, Any]:
    """Create an atomic owner-only SQLite snapshot and versioned manifest."""
    source = Path(database).expanduser()
    _private_regular_metadata(
        source, minimum=1, maximum=MAX_DATABASE_BYTES)

    requested = Path(target).expanduser()
    if not requested.name or requested.name in {".", ".."}:
        raise RuntimeError("export target must name a new directory")
    parent = requested.parent.resolve()
    if not parent.is_dir():
        raise RuntimeError("export target parent does not exist")
    target_path = parent / requested.name
    if os.path.lexists(target_path):
        raise RuntimeError("export target already exists")

    temporary = Path(tempfile.mkdtemp(
        prefix=f".{requested.name}.partial-", dir=parent))
    os.chmod(temporary, 0o700)
    try:
        snapshot = temporary / DATABASE_NAME
        _backup_database(source, snapshot)
        manifest = _manifest_for(snapshot)
        _write_private_file(temporary / MANIFEST_NAME, _canonical_json(manifest))
        _fsync_directory(temporary)
        if os.path.lexists(target_path):
            raise RuntimeError("export target appeared while the export was running")
        os.rename(temporary, target_path)
        _fsync_directory(parent)
        return verify_private_export(target_path)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_private_export(target: str | Path) -> dict[str, Any]:
    """Verify privacy, hashes, schema identity, inventory, and SQLite integrity."""
    requested = Path(target).expanduser()
    _validate_private_directory(requested)
    try:
        entries = {entry.name for entry in requested.iterdir()}
    except OSError as exc:
        raise RuntimeError("private export directory cannot be read") from exc
    if entries != {MANIFEST_NAME, DATABASE_NAME}:
        raise RuntimeError("private export directory contains unexpected entries")
    manifest_path = requested / MANIFEST_NAME
    database_path = requested / DATABASE_NAME
    encoded = _read_private_regular(manifest_path, maximum=MAX_MANIFEST_BYTES)
    database_digest, database_metadata = _sha256_private(
        database_path, maximum=MAX_DATABASE_BYTES)
    try:
        manifest = json.loads(
            encoded.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("private export manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("private export manifest is invalid")
    database = manifest.get("database")
    if (manifest.get("format") != EXPORT_FORMAT
            or manifest.get("format_version") != EXPORT_FORMAT_VERSION
            or not isinstance(manifest.get("created_at"), str)
            or not isinstance(manifest.get("schema_version"), int)
            or not isinstance(database, dict)
            or database.get("filename") != DATABASE_NAME
            or not isinstance(database.get("bytes"), int)
            or not isinstance(database.get("sha256"), str)
            or not isinstance(manifest.get("tables"), list)):
        raise RuntimeError("private export manifest is incomplete")
    if database["bytes"] != database_metadata.st_size:
        raise RuntimeError("private export database size does not match its manifest")
    if database["sha256"] != database_digest:
        raise RuntimeError("private export database hash does not match its manifest")
    schema_version, tables = _database_inventory(database_path)
    if manifest["schema_version"] != schema_version:
        raise RuntimeError("private export schema version does not match its database")
    if manifest["tables"] != tables:
        raise RuntimeError("private export table inventory does not match its database")
    return manifest


def _normalize_timestamp(value: str, field: str) -> str:
    if not isinstance(value, str) or not 20 <= len(value) <= 40:
        raise RuntimeError(f"deletion {field} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"deletion {field} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"deletion {field} timestamp must include a timezone")
    return parsed.astimezone(UTC).isoformat(
        timespec="microseconds").replace("+00:00", "Z")


def _normalize_deletion_selector(
    scope: str,
    *,
    value: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, str]:
    if scope not in DELETION_SCOPES:
        raise RuntimeError("deletion scope is invalid")
    if scope == "time_range":
        if value is not None or start is None or end is None:
            raise RuntimeError("time-range deletion requires start and end")
        normalized_start = _normalize_timestamp(start, "start")
        normalized_end = _normalize_timestamp(end, "end")
        if normalized_start >= normalized_end:
            raise RuntimeError("deletion time range must have start before end")
        return {"scope": scope, "start": normalized_start, "end": normalized_end}
    if start is not None or end is not None or not isinstance(value, str):
        raise RuntimeError(f"{scope} deletion requires one identifier")
    identifier = value.strip()
    if (not 1 <= len(identifier) <= 200
            or any(not (character.isalnum() or character in "._:-")
                   for character in identifier)):
        raise RuntimeError("deletion identifier is invalid")
    return {"scope": scope, "value": identifier}


def _selector_sha256(selector: dict[str, str]) -> str:
    return hashlib.sha256(_canonical_json(selector)).hexdigest()


def _ordinary_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[1]) for row in connection.execute("PRAGMA table_list")
        if row[0] == "main" and row[2] == "table"
        and not str(row[1]).startswith("sqlite_")
    }


def _foreign_key_groups(
    connection: sqlite3.Connection,
    tables: set[str],
) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    output: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []
    for child in sorted(tables):
        grouped: dict[int, list[sqlite3.Row | tuple[Any, ...]]] = {}
        for row in connection.execute(
                f"PRAGMA foreign_key_list({_quote_identifier(child)})"):
            grouped.setdefault(int(row[0]), []).append(row)
        for rows in grouped.values():
            rows.sort(key=lambda item: int(item[1]))
            parent = str(rows[0][2])
            if parent not in tables:
                continue
            child_columns = tuple(str(row[3]) for row in rows)
            parent_columns = tuple(str(row[4]) for row in rows)
            if not all(parent_columns):
                primary = sorted(
                    ((int(row[5]), str(row[1])) for row in connection.execute(
                        f"PRAGMA table_info({_quote_identifier(parent)})")
                     if int(row[5]) > 0),
                )
                parent_columns = tuple(column for _ordinal, column in primary)
            if len(child_columns) != len(parent_columns) or not parent_columns:
                raise RuntimeError("database foreign-key metadata is incomplete")
            output.append((child, parent, child_columns, parent_columns))
    return output


def _primary_key_columns(
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:
    return {
        str(row[1]) for row in connection.execute(
            f"PRAGMA table_info({_quote_identifier(table)})")
        if int(row[5]) > 0
    }


def _add_rowids(
    connection: sqlite3.Connection,
    selected: dict[str, set[int]],
    table: str,
    where: str,
    parameters: tuple[Any, ...] | list[Any],
) -> bool:
    before = len(selected[table])
    selected[table].update(int(row[0]) for row in connection.execute(
        f"SELECT rowid FROM {_quote_identifier(table)} WHERE {where}",
        parameters,
    ))
    return len(selected[table]) != before


def _selected_column_values(
    connection: sqlite3.Connection,
    table: str,
    rowids: set[int],
    columns: tuple[str, ...],
) -> set[tuple[Any, ...]]:
    output: set[tuple[Any, ...]] = set()
    ordered = sorted(rowids)
    projection = ",".join(_quote_identifier(column) for column in columns)
    for offset in range(0, len(ordered), 400):
        batch = ordered[offset:offset + 400]
        placeholders = ",".join("?" for _ in batch)
        for row in connection.execute(
            f"SELECT {projection} FROM {_quote_identifier(table)} "
            f"WHERE rowid IN ({placeholders})",
            batch,
        ):
            values = tuple(row)
            if all(value is not None for value in values):
                output.add(values)
    return output


def _add_matching_values(
    connection: sqlite3.Connection,
    selected: dict[str, set[int]],
    table: str,
    columns: tuple[str, ...],
    values: set[tuple[Any, ...]],
) -> bool:
    changed = False
    ordered = sorted(values, key=repr)
    batch_size = max(1, 350 // len(columns))
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset:offset + batch_size]
        clauses: list[str] = []
        parameters: list[Any] = []
        for value_tuple in batch:
            clauses.append("(" + " AND ".join(
                f"{_quote_identifier(column)} = ?" for column in columns
            ) + ")")
            parameters.extend(value_tuple)
        if clauses:
            changed |= _add_rowids(
                connection, selected, table, " OR ".join(clauses), parameters)
    return changed


def _deletion_selection(
    connection: sqlite3.Connection,
    selector: dict[str, str],
) -> dict[str, set[int]]:
    tables = _ordinary_tables(connection) - {
        "schema_migrations", "deletion_tombstones",
    }
    selected = {table: set() for table in tables}
    scope = selector["scope"]
    if scope == "conversation":
        _add_rowids(
            connection, selected, "graph_events", "session_id = ?",
            (selector["value"],),
        )
        _add_rowids(
            connection, selected, "nodes", "node_id = ?",
            (selector["value"],),
        )
    elif scope == "task":
        _add_rowids(
            connection, selected, "graph_events", "task_id = ?",
            (selector["value"],),
        )
        _add_rowids(
            connection, selected, "task_state", "task_id = ?",
            (selector["value"],),
        )
        _add_rowids(
            connection, selected, "nodes", "node_id = ?",
            (selector["value"],),
        )
    elif scope in {"artifact", "memory_claim"}:
        _add_rowids(
            connection, selected, "nodes", "node_id = ?",
            (selector["value"],),
        )
        if scope == "memory_claim":
            _add_rowids(
                connection, selected, "claim_state", "claim_id = ?",
                (selector["value"],),
            )
    else:
        _add_rowids(
            connection, selected, "graph_events",
            "occurred_at >= ? AND occurred_at < ?",
            (selector["start"], selector["end"]),
        )

    foreign_keys = _foreign_key_groups(connection, tables)
    primary_keys = {
        table: _primary_key_columns(connection, table) for table in tables
    }
    for _round in range(max(16, len(tables) * 4)):
        changed = False

        if selected.get("task_state"):
            task_values = _selected_column_values(
                connection, "task_state", selected["task_state"], ("task_id",))
            changed |= _add_matching_values(
                connection, selected, "graph_events", ("task_id",), task_values)

        for child, parent, child_columns, parent_columns in foreign_keys:
            if selected[parent]:
                parent_values = _selected_column_values(
                    connection, parent, selected[parent], parent_columns)
                if parent_values:
                    changed |= _add_matching_values(
                        connection, selected, child, child_columns, parent_values)

            if not selected[child]:
                continue
            owns_event = parent == "graph_events"
            owns_node = (
                parent == "nodes"
                and set(child_columns).issubset(primary_keys[child])
            )
            if owns_event or owns_node:
                child_values = _selected_column_values(
                    connection, child, selected[child], child_columns)
                if child_values:
                    changed |= _add_matching_values(
                        connection, selected, parent, parent_columns, child_values)

        if not changed:
            break
    else:
        raise RuntimeError("deletion dependency closure did not converge")
    return {table: rowids for table, rowids in selected.items() if rowids}


def _deletion_plan(
    connection: sqlite3.Connection,
    selector: dict[str, str],
) -> dict[str, Any]:
    selected = _deletion_selection(connection, selector)
    counts = {table: len(rowids) for table, rowids in sorted(selected.items())}
    if "nodes" in selected:
        node_ids = _selected_column_values(
            connection, "nodes", selected["nodes"], ("node_id",))
        search_rows = 0
        ordered_ids = sorted(str(item[0]) for item in node_ids)
        for offset in range(0, len(ordered_ids), 400):
            batch = ordered_ids[offset:offset + 400]
            placeholders = ",".join("?" for _ in batch)
            search_rows += int(connection.execute(
                f"SELECT COUNT(*) FROM memory_fts "
                f"WHERE claim_id IN ({placeholders})",
                batch,
            ).fetchone()[0])
        if search_rows:
            counts["memory_fts"] = search_rows
    return {
        "scope": selector["scope"],
        "selector_sha256": _selector_sha256(selector),
        "matched": bool(counts),
        "rows": counts,
        "total_rows": sum(counts.values()),
    }


def plan_private_deletion(
    database: str | Path,
    scope: str,
    *,
    value: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Return content-free row counts for one bounded deletion selector."""
    source = Path(database).expanduser()
    _private_regular_metadata(source, minimum=1, maximum=MAX_DATABASE_BYTES)
    selector = _normalize_deletion_selector(
        scope, value=value, start=start, end=end)
    with _open_read_only(source) as connection:
        return _deletion_plan(connection, selector)


def _drop_and_restore_triggers(
    connection: sqlite3.Connection,
) -> list[str]:
    triggers = [
        (str(row[0]), str(row[1])) for row in connection.execute(
            "SELECT name,sql FROM sqlite_master "
            "WHERE type='trigger' AND sql IS NOT NULL ORDER BY name"
        )
        if not str(row[0]).startswith("deletion_tombstones_")
    ]
    for name, _sql in triggers:
        connection.execute(f"DROP TRIGGER {_quote_identifier(name)}")
    return [sql for _name, sql in triggers]


def _delete_selected_rows(
    connection: sqlite3.Connection,
    selector: dict[str, str],
) -> dict[str, Any]:
    plan = _deletion_plan(connection, selector)
    if not plan["matched"]:
        return plan
    selected = _deletion_selection(connection, selector)
    node_ids = set()
    if "nodes" in selected:
        node_ids = {
            str(item[0]) for item in _selected_column_values(
                connection, "nodes", selected["nodes"], ("node_id",))
        }

    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("BEGIN IMMEDIATE")
    try:
        trigger_sql = _drop_and_restore_triggers(connection)
        if node_ids:
            for offset in range(0, len(node_ids), 400):
                batch = sorted(node_ids)[offset:offset + 400]
                placeholders = ",".join("?" for _ in batch)
                connection.execute(
                    f"DELETE FROM memory_fts WHERE claim_id IN ({placeholders})",
                    batch,
                )
        for table, rowids in sorted(selected.items()):
            ordered = sorted(rowids)
            for offset in range(0, len(ordered), 400):
                batch = ordered[offset:offset + 400]
                placeholders = ",".join("?" for _ in batch)
                connection.execute(
                    f"DELETE FROM {_quote_identifier(table)} "
                    f"WHERE rowid IN ({placeholders})",
                    batch,
                )
        deletion_id = f"deletion_{uuid.uuid4().hex}"
        connection.execute(
            """INSERT INTO deletion_tombstones
               (deletion_id,scope,selector_sha256,deleted_at,rows_json)
               VALUES (?,?,?,?,?)""",
            (
                deletion_id,
                selector["scope"],
                plan["selector_sha256"],
                _utc_now(),
                _canonical_json(plan["rows"]).decode("utf-8").strip(),
            ),
        )
        for sql in trigger_sql:
            connection.execute(sql)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError("selective deletion would leave dangling references")
    integrity = connection.execute("PRAGMA integrity_check").fetchall()
    if integrity != [("ok",)]:
        raise RuntimeError("selective deletion failed SQLite integrity check")
    return plan | {"deletion_id": deletion_id}


def _validate_sqlite_sidecars(database: Path) -> list[Path]:
    sidecars: list[Path] = []
    for suffix in ("-wal", "-shm"):
        candidate = Path(str(database) + suffix)
        if not os.path.lexists(candidate):
            continue
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise RuntimeError("database sidecar identity cannot be verified") from exc
        if (not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1):
            raise RuntimeError("database sidecar identity is invalid")
        sidecars.append(candidate)
    return sidecars


def delete_private_data(
    database: str | Path,
    scope: str,
    *,
    value: str | None = None,
    start: str | None = None,
    end: str | None = None,
    runtime_stopped: bool = False,
) -> dict[str, Any]:
    """Rebuild and atomically replace a stopped Friday database after deletion."""
    if not runtime_stopped:
        raise RuntimeError("selective deletion requires the Friday service to be stopped")
    source_request = Path(database).expanduser()
    source = source_request.parent.resolve() / source_request.name
    original = _private_regular_metadata(
        source, minimum=1, maximum=MAX_DATABASE_BYTES)
    selector = _normalize_deletion_selector(
        scope, value=value, start=start, end=end)
    selector_digest = _selector_sha256(selector)
    sidecars = _validate_sqlite_sidecars(source)

    lock_path = Path(str(source) + ".data-lifecycle.lock")
    lock_flags = (os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
                  | getattr(os, "O_NOFOLLOW", 0))
    lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    temporary: Path | None = None
    try:
        lock_metadata = os.fstat(lock_descriptor)
        if (not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_uid != os.getuid()
                or lock_metadata.st_nlink != 1):
            raise RuntimeError("data-lifecycle lock identity is invalid")
        os.fchmod(lock_descriptor, 0o600)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        before_digest, before_metadata = _sha256_private(
            source, maximum=MAX_DATABASE_BYTES)
        if (before_metadata.st_dev, before_metadata.st_ino) != (
                original.st_dev, original.st_ino):
            raise RuntimeError("Friday database changed before deletion began")

        temporary = Path(tempfile.mkdtemp(
            prefix=f".{source.name}.delete-", dir=source.parent))
        os.chmod(temporary, 0o700)
        replacement = temporary / source.name
        _backup_database(source, replacement)
        snapshot_digest, snapshot_metadata = _sha256_private(
            replacement, maximum=MAX_DATABASE_BYTES)
        with sqlite3.connect(replacement, timeout=15) as connection:
            connection.execute("PRAGMA busy_timeout = 15000")
            apply_schema_migrations(connection)
            existing = connection.execute(
                "SELECT deletion_id,deleted_at,rows_json FROM deletion_tombstones "
                "WHERE scope=? AND selector_sha256=?",
                (selector["scope"], selector_digest),
            ).fetchone()
            if existing is not None:
                return {
                    "status": "already_deleted",
                    "scope": selector["scope"],
                    "selector_sha256": selector_digest,
                    "deletion_id": str(existing[0]),
                    "deleted_at": str(existing[1]),
                    "rows": json.loads(str(existing[2])),
                }
            result = _delete_selected_rows(connection, selector)
            if not result["matched"]:
                raise RuntimeError("deletion selector matched no durable records")
            connection.execute("VACUUM")
            connection.execute("PRAGMA journal_mode = DELETE")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise RuntimeError("compacted database has dangling references")
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError("compacted database failed SQLite integrity check")
        os.chmod(replacement, 0o600)
        replacement_descriptor = os.open(
            replacement, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(replacement_descriptor)
        finally:
            os.close(replacement_descriptor)
        after_digest, after_metadata = _sha256_private(
            replacement, maximum=MAX_DATABASE_BYTES)

        current_digest, current_metadata = _sha256_private(
            source, maximum=MAX_DATABASE_BYTES)
        if (current_digest != before_digest
                or (current_metadata.st_dev, current_metadata.st_ino)
                != (original.st_dev, original.st_ino)):
            raise RuntimeError("Friday database changed while deletion was running")
        os.replace(replacement, source)
        for sidecar in sidecars:
            sidecar.unlink()
        _fsync_directory(source.parent)
        result.update({
            "status": "deleted",
            "database_sha256_before": snapshot_digest,
            "database_sha256_after": after_digest,
            "database_bytes_before": snapshot_metadata.st_size,
            "database_bytes_after": after_metadata.st_size,
            "storage_note": (
                "logical records and SQLite free pages were removed; physical "
                "media recovery remains outside Friday's guarantees"),
        })
        return result
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
