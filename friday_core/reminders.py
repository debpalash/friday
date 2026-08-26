"""Persistent user-created reminders and a restart-safe delivery worker."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from .graph import GraphStore, new_id, utc_now


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("due_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("due_at must include a timezone")
    return parsed.astimezone(UTC)


class ReminderService:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    def create(self, text: str, due_at: str, *, interval_seconds: int | None = None,
               source_task_id: str | None = None, actor: str = "friday") -> dict[str, Any]:
        message = text.strip()
        if not message:
            raise ValueError("reminder text cannot be empty")
        due = _parse_time(due_at)
        if interval_seconds is not None and interval_seconds < 60:
            raise ValueError("recurring reminders must be at least 60 seconds apart")
        normalized_due = due.isoformat().replace("+00:00", "Z")
        now = utc_now()
        body = {"text": message, "due_at": normalized_due,
                "interval_seconds": interval_seconds,
                "source_task_id": source_task_id, "status": "scheduled"}
        with self.graph.transaction() as conn:
            event_id, seq = self.graph.append_event(
                conn, "reminder.created", body, actor=actor, task_id=source_task_id)
            reminder_id = self.graph.append_node(
                conn, "reminder", body, event_id=event_id,
                node_id=new_id("reminder"))
            if source_task_id:
                self.graph.append_edge(conn, source_task_id, "scheduled", reminder_id,
                                       event_id=event_id)
            conn.execute(
                """INSERT INTO reminder_state
                   (reminder_id,text,due_at,interval_seconds,status,source_task_id,
                    created_at,updated_at,last_event_seq)
                   VALUES (?,?,?,?,'scheduled',?,?,?,?)""",
                (reminder_id, message, normalized_due, interval_seconds,
                 source_task_id, now, now, seq))
        return {"reminder_id": reminder_id, **body, "created_at": now}

    def cancel(self, reminder_id: str, *, actor: str = "user") -> dict[str, Any]:
        with self.graph.transaction() as conn:
            row = conn.execute("SELECT * FROM reminder_state WHERE reminder_id=?",
                               (reminder_id,)).fetchone()
            if row is None:
                raise ValueError("reminder does not exist")
            body = {"reminder_id": reminder_id, "status": "cancelled"}
            event_id, seq = self.graph.append_event(
                conn, "reminder.cancelled", body, actor=actor,
                task_id=row["source_task_id"])
            conn.execute(
                """UPDATE reminder_state SET status='cancelled',updated_at=?,
                   last_event_seq=? WHERE reminder_id=?""",
                (utc_now(), seq, reminder_id))
        return body

    def due(self, *, now: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        at = _parse_time(now).isoformat().replace("+00:00", "Z") if now else utc_now()
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM reminder_state WHERE status='scheduled' AND due_at<=?
                   ORDER BY due_at LIMIT ?""", (at, limit)).fetchall()
        return [dict(row) for row in rows]

    def mark_fired(self, reminder_id: str, *, expected_due_at: str | None = None,
                   actor: str = "scheduler") -> dict[str, Any]:
        with self.graph.transaction() as conn:
            row = conn.execute("SELECT * FROM reminder_state WHERE reminder_id=?",
                               (reminder_id,)).fetchone()
            if row is None or row["status"] != "scheduled":
                raise ValueError("reminder is not scheduled")
            if expected_due_at is not None and row["due_at"] != expected_due_at:
                raise ValueError("reminder due time changed before delivery")
            fired_at = utc_now()
            if row["interval_seconds"]:
                due = _parse_time(row["due_at"])
                now = datetime.now(UTC)
                interval = timedelta(seconds=int(row["interval_seconds"]))
                while due <= now:
                    due += interval
                next_due = due.isoformat().replace("+00:00", "Z")
                status = "scheduled"
            else:
                next_due = row["due_at"]
                status = "fired"
            body = {"reminder_id": reminder_id, "fired_at": fired_at,
                    "status": status, "next_due_at": next_due}
            event_id, seq = self.graph.append_event(
                conn, "reminder.fired", body, actor=actor,
                task_id=row["source_task_id"])
            observation_id = self.graph.append_node(
                conn, "observation", body, event_id=event_id)
            self.graph.append_edge(conn, reminder_id, "produced", observation_id,
                                   event_id=event_id)
            conn.execute(
                """UPDATE reminder_state SET status=?,due_at=?,last_fired_at=?,
                   updated_at=?,last_event_seq=? WHERE reminder_id=?""",
                (status, next_due, fired_at, fired_at, seq, reminder_id))
        return {"reminder_id": reminder_id, "text": row["text"], **body}

    def list(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM reminder_state"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY due_at LIMIT ?"
        params.append(limit)
        with self.graph._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]


ReminderDelivery = Callable[[dict[str, Any]], Awaitable[None]]
ReminderConfirmation = Callable[[dict[str, Any]], Awaitable[None]]


class ReminderWorker:
    def __init__(self, reminders: ReminderService, delivery: ReminderDelivery,
                 *, confirmation: ReminderConfirmation | None = None,
                 poll_seconds: float = 5.0):
        self.reminders = reminders
        self.delivery = delivery
        self.confirmation = confirmation
        self.poll_seconds = poll_seconds
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if not self.is_running:
            self._task = asyncio.create_task(self._loop(), name="friday-reminders")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _loop(self) -> None:
        while True:
            for reminder in self.reminders.due():
                try:
                    # Delivery is the side effect that matters.  Keep the
                    # reminder scheduled when it fails instead of recording a
                    # false success that can never be retried.  A crash after
                    # delivery and before the state transition can cause an
                    # at-least-once duplicate, which is safer than silently
                    # losing a reminder and is surfaced in the roadmap's
                    # durable-step reconciliation milestone.
                    await self.delivery({
                        "reminder_id": reminder["reminder_id"],
                        "text": reminder["text"],
                        "due_at": reminder["due_at"],
                        "status": "due",
                    })
                    receipt = self.reminders.mark_fired(
                        reminder["reminder_id"], expected_due_at=reminder["due_at"])
                    if self.confirmation is not None:
                        await self.confirmation(receipt)
                except Exception:
                    continue
            await asyncio.sleep(self.poll_seconds)
