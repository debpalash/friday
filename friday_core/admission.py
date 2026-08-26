"""Durable, transaction-safe admission control for executable work.

``ResourceAdmissionController`` keeps policy deliberately separate from workers:
call :meth:`acquire` for a self-contained transaction, or call
:meth:`acquire_in_transaction` while atomically claiming a step in an existing
``BEGIN IMMEDIATE`` transaction.  A lease is fenced by its exact lease, attempt,
worker, and runtime identities; an expired or released attempt cannot be reused.

Live snapshots describe currently *available* machine resources.  Admission uses
the lower of live availability and the static budget remaining after active lease
reservations.  Consequently, impossible claims are permanently rejected while
ordinary contention is retryable and deferred.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .cognition import ResourceClaim
from .graph import GraphStore, canonical_json, new_id, sha256_text


def _normalize_accelerator(value: str) -> str:
    value = value.strip().lower()
    if value == "cuda":
        return "cuda:0"
    if value.startswith("cuda:"):
        suffix = value.removeprefix("cuda:")
        if suffix.isdigit():
            return f"cuda:{int(suffix)}"
    return value


def _normalize_vram_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TypeError("accelerator VRAM capacity must be a mapping")
    normalized: dict[str, int] = {}
    for raw_key, raw_capacity in value.items():
        key = _normalize_accelerator(str(raw_key))
        if not key or key == "none":
            raise ValueError("accelerator capacity keys must identify a device")
        if key in normalized:
            raise ValueError(f"duplicate accelerator alias: {key}")
        capacity = int(raw_capacity)
        if capacity < 0:
            raise ValueError("accelerator VRAM capacity cannot be negative")
        normalized[key] = capacity
    return normalized


def _as_utc(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z")


class AdmissionBudget(BaseModel):
    """Static upper bounds that Friday is permitted to reserve."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    cpu_millis: int = Field(ge=0)
    ram_mib: int = Field(ge=0)
    concurrency_slots: int = Field(ge=0)
    network_slots: int = Field(default=0, ge=0)
    accelerator_vram_mib: dict[str, int] = Field(default_factory=dict)

    @field_validator("accelerator_vram_mib", mode="before")
    @classmethod
    def normalize_accelerators(cls, value: Any) -> dict[str, int]:
        return _normalize_vram_map(value)


class ResourceSnapshot(BaseModel):
    """A point-in-time view of resources available to new Friday work."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    available_cpu_millis: int = Field(ge=0)
    available_ram_mib: int = Field(ge=0)
    available_network_slots: int = Field(default=0, ge=0)
    available_accelerator_vram_mib: dict[str, int] = Field(default_factory=dict)
    captured_at: datetime | None = None

    @field_validator("available_accelerator_vram_mib", mode="before")
    @classmethod
    def normalize_accelerators(cls, value: Any) -> dict[str, int]:
        return _normalize_vram_map(value)


class AdmissionDecision(BaseModel):
    """Result of an admission attempt; only ``admitted`` has a lease."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["admitted", "deferred", "rejected"]
    reason: str
    retryable: bool
    lease_id: str | None = None
    expires_at: str | None = None
    deficits: dict[str, int] = Field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return self.status == "admitted"


SnapshotProvider = Callable[[], ResourceSnapshot | dict[str, Any]]


_CONTROL_LANE_OPERATIONS = frozenset({"inspect", "terminate"})
_CONTROL_LEASE_TTL_SECONDS = 15
_WORKLOAD_RESERVING_STATUSES = frozenset({"active", "reconciling"})
_WORKLOAD_ACTIVE_INSTANCE_STATES = frozenset({
    "prepared",
    "starting",
    "running",
    "stop_requested",
    "stopping",
    "reconcile_required",
})


class ResourceAdmissionController:
    """Reserve bounded machine resources using durable fenced leases.

    ``acquire_in_transaction`` and the other ``*_in_transaction`` methods never
    commit or roll back their supplied connection.  Their caller owns that
    transaction, allowing step claims and resource reservations to be indivisible.
    """

    def __init__(
        self,
        graph: GraphStore,
        budget: AdmissionBudget | dict[str, Any],
        snapshot_provider: SnapshotProvider | None = None,
        *,
        snapshot_ttl_seconds: float = 2.0,
        max_snapshot_age_seconds: float = 10.0,
        max_snapshot_future_skew_seconds: float = 1.0,
        lease_ttl_seconds: int = 60,
        runtime_id: str | None = None,
        profile_fingerprint: str | None = None,
    ) -> None:
        if snapshot_ttl_seconds < 0:
            raise ValueError("snapshot_ttl_seconds cannot be negative")
        if max_snapshot_age_seconds <= 0:
            raise ValueError("max_snapshot_age_seconds must be positive")
        if max_snapshot_future_skew_seconds < 0:
            raise ValueError(
                "max_snapshot_future_skew_seconds cannot be negative")
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        self.graph = graph
        self.budget = AdmissionBudget.model_validate(budget)
        self.snapshot_provider = snapshot_provider
        self.snapshot_ttl_seconds = float(snapshot_ttl_seconds)
        self.max_snapshot_age_seconds = float(max_snapshot_age_seconds)
        self.max_snapshot_future_skew_seconds = float(
            max_snapshot_future_skew_seconds)
        self.lease_ttl_seconds = int(lease_ttl_seconds)
        self.runtime_id = runtime_id or f"runtime_{uuid.uuid4().hex}"
        self._validate_identity("runtime_id", self.runtime_id)
        self.profile_fingerprint = profile_fingerprint or sha256_text(
            canonical_json(self.budget.model_dump(mode="json")))
        self._validate_identity(
            "profile_fingerprint", self.profile_fingerprint)
        self._snapshot_lock = threading.Lock()
        self._cached_snapshot: ResourceSnapshot | None = None
        self._snapshot_cached_at = 0.0

    @staticmethod
    def is_control_lane_operation(operation: str) -> bool:
        """Return whether an operation may use the cleanup-only lane."""
        return isinstance(operation, str) and operation in _CONTROL_LANE_OPERATIONS

    @classmethod
    def control_lane_allows(
        cls, operation: str, claim: ResourceClaim | dict[str, Any],
    ) -> bool:
        """Authorize only resource-free inspect/terminate control work.

        This helper is deliberately separate from ordinary admission.  A caller
        cannot turn the reserved cleanup lane into workload launch capacity by
        merely labelling a resource-consuming claim as ``control``.
        """
        if not cls.is_control_lane_operation(operation):
            return False
        try:
            validated = ResourceClaim.model_validate(claim)
        except (TypeError, ValueError):
            return False
        return cls._is_valid_control_claim(validated)

    def invalidate_snapshot(self) -> None:
        """Force the provider to be consulted on the next admission attempt."""
        with self._snapshot_lock:
            self._cached_snapshot = None
            self._snapshot_cached_at = 0.0

    def get_snapshot(self, *, force: bool = False) -> ResourceSnapshot:
        """Return a validated live snapshot, cached for the configured TTL."""
        if self.snapshot_provider is None:
            return ResourceSnapshot(
                available_cpu_millis=self.budget.cpu_millis,
                available_ram_mib=self.budget.ram_mib,
                available_network_slots=self.budget.network_slots,
                available_accelerator_vram_mib=(
                    self.budget.accelerator_vram_mib),
                captured_at=datetime.now(UTC),
            )
        with self._snapshot_lock:
            now = time.monotonic()
            if (not force and self._cached_snapshot is not None
                    and now - self._snapshot_cached_at
                    <= self.snapshot_ttl_seconds):
                try:
                    self._validate_snapshot_freshness(
                        self._cached_snapshot, datetime.now(UTC))
                    return self._cached_snapshot
                except RuntimeError:
                    self._cached_snapshot = None
                    self._snapshot_cached_at = 0.0
            snapshot = ResourceSnapshot.model_validate(self.snapshot_provider())
            self._validate_snapshot_freshness(snapshot, datetime.now(UTC))
            self._cached_snapshot = snapshot
            self._snapshot_cached_at = now
            return snapshot

    def acquire(
        self,
        claim: ResourceClaim,
        *,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        snapshot: ResourceSnapshot | dict[str, Any] | None = None,
        now: datetime | str | None = None,
        lease_ttl_seconds: int | None = None,
    ) -> AdmissionDecision:
        """Atomically decide and acquire using a new ``BEGIN IMMEDIATE`` txn."""
        validated_claim = ResourceClaim.model_validate(claim)
        resolved = (ResourceSnapshot.model_validate(snapshot)
                    if snapshot is not None else None)
        with self.graph.transaction() as conn:
            return self.acquire_in_transaction(
                conn, validated_claim, step_id=step_id, attempt_id=attempt_id,
                worker_id=worker_id, snapshot=resolved, now=now,
                lease_ttl_seconds=lease_ttl_seconds)

    def acquire_in_transaction(
        self,
        conn: sqlite3.Connection,
        claim: ResourceClaim,
        *,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        snapshot: ResourceSnapshot | dict[str, Any] | None = None,
        now: datetime | str | None = None,
        lease_ttl_seconds: int | None = None,
    ) -> AdmissionDecision:
        """Decide and insert a lease without owning the supplied transaction."""
        claim = ResourceClaim.model_validate(claim)
        self._validate_identity("step_id", step_id)
        self._validate_identity("attempt_id", attempt_id)
        self._validate_identity("worker_id", worker_id)
        is_control = claim.latency_class == "control"
        instant = _as_utc(now)
        ttl = self._ttl(lease_ttl_seconds)
        if is_control:
            ttl = min(ttl, _CONTROL_LEASE_TTL_SECONDS)
        now_text = _stamp(instant)
        self._expire_due_in_transaction(conn, instant)
        requested = self._claim_units(claim)

        if is_control and not self._is_valid_control_claim(claim):
            decision = AdmissionDecision(
                status="rejected", reason="invalid_control_lane_claim",
                retryable=False)
            self._record_decision(conn, decision, step_id, attempt_id)
            return decision

        existing = conn.execute(
            """SELECT lease_id, worker_id, runtime_id, status, expires_at,
                      cpu_millis, ram_mib, concurrency_slots, network_slots,
                      accelerator, vram_mib, profile_fingerprint,
                      latency_class
               FROM resource_leases
               WHERE step_id = ? AND attempt_id = ?""",
            (step_id, attempt_id),
        ).fetchone()
        if existing is not None:
            if (existing[3] == "active" and existing[1] == worker_id
                    and existing[2] == self.runtime_id
                    and str(existing[4]) > now_text
                    and self._existing_matches_claim(existing, claim,
                                                     requested)):
                return AdmissionDecision(
                    status="admitted", reason="already_admitted",
                    retryable=False, lease_id=str(existing[0]),
                    expires_at=str(existing[4]))
            decision = AdmissionDecision(
                status="rejected", reason="attempt_already_fenced",
                retryable=False)
            self._record_decision(conn, decision, step_id, attempt_id)
            return decision

        if is_control:
            control_in_use = conn.execute(
                """SELECT 1 FROM resource_leases
                   WHERE status='active' AND latency_class='control'
                   LIMIT 1""").fetchone()
            if control_in_use is not None:
                decision = AdmissionDecision(
                    status="deferred", reason="control_lane_busy",
                    retryable=True, deficits={"control_slots": 1})
                self._record_decision(conn, decision, step_id, attempt_id)
                return decision
        else:
            observed = (ResourceSnapshot.model_validate(snapshot)
                        if snapshot is not None else self.get_snapshot())
            self._validate_snapshot_freshness(observed, instant)
            static_deficits = self._deficits(
                requested, self._budget_units(claim))
            if static_deficits:
                decision = AdmissionDecision(
                    status="rejected", reason="static_budget_exceeded",
                    retryable=False, deficits=static_deficits)
                self._record_decision(conn, decision, step_id, attempt_id)
                return decision

            leased = self._leased_units(
                conn, claim.accelerator, include_control=False)
            remaining = self._remaining_budget(
                claim, leased, reserve_interactive=False)
            if claim.latency_class != "interactive":
                noninteractive_leased = self._leased_units(
                    conn, claim.accelerator, include_control=False,
                    noninteractive_only=True)
                quota_remaining = self._remaining_budget(
                    claim, noninteractive_leased,
                    reserve_interactive=True)
                remaining = {
                    key: min(remaining[key], quota_remaining[key])
                    for key in remaining
                }
            live = self._snapshot_units(observed, claim)
            live_remaining = {
                key: max(0, live[key] - leased.get(key, 0))
                for key in requested
            }
            effective = {
                key: min(remaining[key], live_remaining[key])
                for key in requested
            }
            deficits = self._deficits(requested, effective)
            if deficits:
                decision = AdmissionDecision(
                    status="deferred",
                    reason="capacity_temporarily_unavailable",
                    retryable=True, deficits=deficits)
                self._record_decision(conn, decision, step_id, attempt_id)
                return decision

        lease_id = new_id("resource_lease")
        expires_at = _stamp(instant + timedelta(seconds=ttl))
        accelerator = _normalize_accelerator(claim.accelerator)
        event_payload = {
            "lease_id": lease_id,
            "step_id": step_id,
            "attempt_id": attempt_id,
            "worker_id": worker_id,
            "runtime_id": self.runtime_id,
            "profile_fingerprint": self.profile_fingerprint,
            "latency_class": claim.latency_class,
            "cpu_millis": requested["cpu_millis"],
            "ram_mib": requested["ram_mib"],
            "concurrency_slots": requested["concurrency_slots"],
            "network_slots": requested["network_slots"],
            "accelerator": accelerator,
            "vram_mib": requested[self._vram_key(claim.accelerator)],
            "expires_at": expires_at,
        }
        event_id, seq = self.graph.append_event(
            conn, "resource.lease_acquired", event_payload,
            actor="resource_admission")
        self.graph.append_node(
            conn, "resource_lease", event_payload, event_id=event_id,
            node_id=lease_id)
        conn.execute(
            """INSERT INTO resource_leases
               (lease_id, step_id, attempt_id, worker_id, runtime_id,
                profile_fingerprint, latency_class,
                cpu_millis, ram_mib, concurrency_slots, network_slots,
                accelerator, vram_mib, status, acquired_at, heartbeat_at,
                expires_at, last_event_seq)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active',
                       ?, ?, ?, ?)""",
            (lease_id, step_id, attempt_id, worker_id, self.runtime_id,
             self.profile_fingerprint, claim.latency_class,
             requested["cpu_millis"], requested["ram_mib"],
             requested["concurrency_slots"], requested["network_slots"],
             accelerator, requested[self._vram_key(claim.accelerator)],
             now_text, now_text, expires_at, seq),
        )
        return AdmissionDecision(
            status="admitted", reason="capacity_reserved", retryable=False,
            lease_id=lease_id, expires_at=expires_at)

    def acquire_workload(
        self,
        claim: ResourceClaim,
        instance_id: str,
        enforcement_json: dict[str, Any] | str,
        now: datetime | str | None = None,
        *,
        snapshot: ResourceSnapshot | dict[str, Any] | None = None,
    ) -> AdmissionDecision:
        """Atomically reserve resources for a durable process instance."""
        validated_claim = ResourceClaim.model_validate(claim)
        resolved = (ResourceSnapshot.model_validate(snapshot)
                    if snapshot is not None else None)
        with self.graph.transaction() as conn:
            return self.acquire_workload_in_transaction(
                conn, validated_claim, instance_id, enforcement_json, now,
                snapshot=resolved)

    def acquire_workload_in_transaction(
        self,
        conn: sqlite3.Connection,
        claim: ResourceClaim,
        instance_id: str,
        enforcement_json: dict[str, Any] | str,
        now: datetime | str | None = None,
        *,
        snapshot: ResourceSnapshot | dict[str, Any] | None = None,
    ) -> AdmissionDecision:
        """Reserve for an instance without taking ownership of ``conn``.

        Workload leases deliberately have no automatic expiry path.  Their
        ``expires_at`` value is a monitor deadline only: missing that deadline
        must move the lease to reconciliation, where it continues reserving
        capacity until the process cgroup is authoritatively empty.
        """
        claim = ResourceClaim.model_validate(claim)
        self._validate_identity("instance_id", instance_id)
        enforcement_text = self._canonical_enforcement_json(enforcement_json)
        instant = _as_utc(now)
        requested = self._claim_units(claim)

        if claim.latency_class == "control":
            decision = AdmissionDecision(
                status="rejected", reason="control_lane_cannot_launch_workload",
                retryable=False)
            self._record_workload_decision(conn, decision, instance_id)
            return decision

        existing = conn.execute(
            """SELECT * FROM workload_resource_leases
               WHERE instance_id = ?""",
            (instance_id,),
        ).fetchone()
        if existing is not None:
            if (str(existing["status"]) in _WORKLOAD_RESERVING_STATUSES
                    and str(existing["runtime_id"]) == self.runtime_id
                    and self._existing_workload_matches(
                        existing, claim, requested, enforcement_text,
                        source_step_lease_id=None,
                        source_attempt_id=None,
                        source_worker_id=None)):
                return AdmissionDecision(
                    status="admitted", reason="already_admitted",
                    retryable=False, lease_id=str(existing["lease_id"]),
                    expires_at=str(existing["expires_at"]))
            decision = AdmissionDecision(
                status="rejected", reason="workload_instance_already_fenced",
                retryable=False)
            self._record_workload_decision(conn, decision, instance_id)
            return decision

        if not self._workload_instance_is_launchable(conn, instance_id):
            decision = AdmissionDecision(
                status="rejected", reason="workload_instance_not_launchable",
                retryable=False)
            self._record_workload_decision(conn, decision, instance_id)
            return decision

        self._expire_due_in_transaction(conn, instant)
        observed = (ResourceSnapshot.model_validate(snapshot)
                    if snapshot is not None else self.get_snapshot())
        self._validate_snapshot_freshness(observed, instant)

        static_deficits = self._deficits(requested, self._budget_units(claim))
        if static_deficits:
            decision = AdmissionDecision(
                status="rejected", reason="static_budget_exceeded",
                retryable=False, deficits=static_deficits)
            self._record_workload_decision(conn, decision, instance_id)
            return decision

        leased = self._leased_units(
            conn, claim.accelerator, include_control=False)
        remaining = self._remaining_budget(
            claim, leased, reserve_interactive=False)
        if claim.latency_class != "interactive":
            noninteractive_leased = self._leased_units(
                conn, claim.accelerator, include_control=False,
                noninteractive_only=True)
            quota_remaining = self._remaining_budget(
                claim, noninteractive_leased, reserve_interactive=True)
            remaining = {
                key: min(remaining[key], quota_remaining[key])
                for key in remaining
            }
        live = self._snapshot_units(observed, claim)
        live_remaining = {
            key: max(0, live[key] - leased.get(key, 0))
            for key in requested
        }
        effective = {
            key: min(remaining[key], live_remaining[key])
            for key in requested
        }
        deficits = self._deficits(requested, effective)
        if deficits:
            decision = AdmissionDecision(
                status="deferred", reason="capacity_temporarily_unavailable",
                retryable=True, deficits=deficits)
            self._record_workload_decision(conn, decision, instance_id)
            return decision

        return self._insert_workload_lease(
            conn, claim, requested, instance_id, enforcement_text, instant,
            source_step_lease_id=None,
            source_attempt_id=None,
            source_worker_id=None,
            reason="capacity_reserved")

    def transfer_step_to_workload(
        self,
        claim: ResourceClaim,
        instance_id: str,
        source_step_lease_id: str,
        source_attempt_id: str,
        source_worker_id: str,
        enforcement_json: dict[str, Any] | str,
        now: datetime | str | None = None,
    ) -> AdmissionDecision:
        """Atomically replace an exact active step reservation with a workload."""
        with self.graph.transaction() as conn:
            return self.transfer_step_to_workload_in_transaction(
                conn, claim, instance_id, source_step_lease_id,
                source_attempt_id, source_worker_id, enforcement_json, now)

    def transfer_step_to_workload_in_transaction(
        self,
        conn: sqlite3.Connection,
        claim: ResourceClaim,
        instance_id: str,
        source_step_lease_id: str,
        source_attempt_id: str,
        source_worker_id: str,
        enforcement_json: dict[str, Any] | str,
        now: datetime | str | None = None,
    ) -> AdmissionDecision:
        """Transfer an exact step lease with neither a gap nor double charge."""
        claim = ResourceClaim.model_validate(claim)
        self._validate_identity("instance_id", instance_id)
        self._validate_identity("source_step_lease_id", source_step_lease_id)
        self._validate_identity("source_attempt_id", source_attempt_id)
        self._validate_identity("source_worker_id", source_worker_id)
        enforcement_text = self._canonical_enforcement_json(enforcement_json)
        instant = _as_utc(now)
        requested = self._claim_units(claim)

        if claim.latency_class == "control":
            decision = AdmissionDecision(
                status="rejected", reason="control_lane_cannot_launch_workload",
                retryable=False)
            self._record_workload_decision(conn, decision, instance_id)
            return decision

        existing_rows = conn.execute(
            """SELECT * FROM workload_resource_leases
               WHERE instance_id = ? OR source_step_lease_id = ?""",
            (instance_id, source_step_lease_id),
        ).fetchall()
        if existing_rows:
            existing = existing_rows[0]
            if (len(existing_rows) == 1
                    and str(existing["status"])
                    in _WORKLOAD_RESERVING_STATUSES
                    and str(existing["runtime_id"]) == self.runtime_id
                    and self._existing_workload_matches(
                        existing, claim, requested, enforcement_text,
                        source_step_lease_id=source_step_lease_id,
                        source_attempt_id=source_attempt_id,
                        source_worker_id=source_worker_id)
                    and self.is_step_lease_safely_discharged_in_transaction(
                        conn, source_step_lease_id, source_attempt_id,
                        worker_id=source_worker_id)):
                return AdmissionDecision(
                    status="admitted", reason="already_transferred",
                    retryable=False, lease_id=str(existing["lease_id"]),
                    expires_at=str(existing["expires_at"]))
            decision = AdmissionDecision(
                status="rejected", reason="workload_transfer_already_fenced",
                retryable=False)
            self._record_workload_decision(conn, decision, instance_id)
            return decision

        if not self._workload_instance_is_launchable(conn, instance_id):
            decision = AdmissionDecision(
                status="rejected", reason="workload_instance_not_launchable",
                retryable=False)
            self._record_workload_decision(conn, decision, instance_id)
            return decision

        # Expiring the source before reading it prevents a dead step attempt
        # from manufacturing a durable reservation.
        self._expire_due_in_transaction(conn, instant)
        source = conn.execute(
            """SELECT lease_id,step_id,attempt_id,worker_id,runtime_id,status,
                      expires_at,cpu_millis,ram_mib,concurrency_slots,
                      network_slots,accelerator,vram_mib,
                      profile_fingerprint,latency_class
               FROM resource_leases WHERE lease_id = ?""",
            (source_step_lease_id,),
        ).fetchone()
        instance_binding = conn.execute(
            "SELECT step_id FROM process_instances WHERE instance_id=?",
            (instance_id,),
        ).fetchone()
        if (source is None
                or str(source["attempt_id"]) != source_attempt_id
                or str(source["worker_id"]) != source_worker_id
                or str(source["runtime_id"]) != self.runtime_id
                or str(source["profile_fingerprint"])
                != self.profile_fingerprint
                or str(source["status"]) != "active"
                or str(source["expires_at"]) <= _stamp(instant)
                or (instance_binding is None
                    or (instance_binding[0] is not None
                        and str(instance_binding[0])
                        != str(source["step_id"])))
                or not self._resource_row_matches_claim(
                    source, claim, requested)):
            decision = AdmissionDecision(
                status="rejected", reason="source_step_lease_fence_mismatch",
                retryable=False)
            self._record_workload_decision(conn, decision, instance_id)
            return decision

        decision = self._insert_workload_lease(
            conn, claim, requested, instance_id, enforcement_text, instant,
            source_step_lease_id=source_step_lease_id,
            source_attempt_id=source_attempt_id,
            source_worker_id=source_worker_id,
            reason="step_lease_transferred")
        if not self._finish_exact(
                conn, source_step_lease_id, source_attempt_id,
                worker_id=source_worker_id, status="released",
                reason="transferred_to_workload", now=instant):
            # The caller owns this transaction, so raising is the only way to
            # ensure an unexpected partial transfer cannot be committed.
            raise RuntimeError("step lease changed during workload transfer")
        return decision

    def is_step_lease_safely_discharged(
        self,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
    ) -> bool:
        with self.graph._connect() as conn:
            return self.is_step_lease_safely_discharged_in_transaction(
                conn, lease_id, attempt_id, worker_id=worker_id)

    def is_step_lease_safely_discharged_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
    ) -> bool:
        """Confirm an exact transfer, so step finish does not report staleness."""
        self._validate_identity("lease_id", lease_id)
        self._validate_identity("attempt_id", attempt_id)
        self._validate_identity("worker_id", worker_id)
        row = conn.execute(
            """SELECT s.cpu_millis,s.ram_mib,s.concurrency_slots,
                      s.network_slots,s.accelerator,s.vram_mib,
                      s.profile_fingerprint,s.latency_class,s.runtime_id,
                      w.cpu_millis,w.ram_mib,w.concurrency_slots,
                      w.network_slots,w.accelerator,w.vram_mib,
                      w.profile_fingerprint,w.latency_class,w.runtime_id,
                      w.status,p.state,p.finished_at
               FROM resource_leases AS s
               JOIN workload_resource_leases AS w
                 ON w.source_step_lease_id = s.lease_id
                AND w.source_attempt_id = s.attempt_id
                AND w.source_worker_id = s.worker_id
               JOIN process_instances AS p ON p.instance_id = w.instance_id
               WHERE s.lease_id = ? AND s.attempt_id = ? AND s.worker_id = ?
                 AND s.runtime_id = ? AND s.profile_fingerprint = ?
                 AND s.status = 'released'
                 AND s.release_reason = 'transferred_to_workload'""",
            (lease_id, attempt_id, worker_id, self.runtime_id,
             self.profile_fingerprint),
        ).fetchone()
        if row is None:
            return False
        step_binding = tuple(row[index] for index in range(9))
        workload_binding = tuple(row[index] for index in range(9, 18))
        if step_binding != workload_binding:
            return False
        workload_status = str(row[18])
        process_state = str(row[19])
        if (workload_status in _WORKLOAD_RESERVING_STATUSES
                and (process_state in _WORKLOAD_ACTIVE_INSTANCE_STATES
                     or process_state == "identity_mismatch")):
            return True
        return (
            workload_status in {"released", "fenced"}
            and process_state in {"launch_failed", "terminated", "exited"}
            and row[20] is not None
        )

    def heartbeat_workload(
        self,
        lease_id: str,
        instance_id: str,
        *,
        now: datetime | str | None = None,
    ) -> bool:
        with self.graph.transaction() as conn:
            return self.heartbeat_workload_in_transaction(
                conn, lease_id, instance_id, now=now)

    def heartbeat_workload_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        instance_id: str,
        *,
        now: datetime | str | None = None,
    ) -> bool:
        """Renew only the active workload owned by this runtime and profile."""
        self._validate_identity("lease_id", lease_id)
        self._validate_identity("instance_id", instance_id)
        instant = _as_utc(now)
        now_text = _stamp(instant)
        row = conn.execute(
            """SELECT w.heartbeat_at,w.expires_at,w.latency_class,p.state
               FROM workload_resource_leases AS w
               JOIN process_instances AS p ON p.instance_id=w.instance_id
               WHERE w.lease_id=? AND w.instance_id=?
                 AND w.runtime_id=? AND w.profile_fingerprint=?
                 AND w.status='active'""",
            (lease_id, instance_id, self.runtime_id,
             self.profile_fingerprint),
        ).fetchone()
        if (row is None
                or str(row["state"]) not in _WORKLOAD_ACTIVE_INSTANCE_STATES
                or now_text < str(row["heartbeat_at"])
                or now_text >= str(row["expires_at"])):
            return False
        expires_at = _stamp(
            instant + timedelta(seconds=self.lease_ttl_seconds))
        _, seq = self.graph.append_event(
            conn, "resource.workload_lease_heartbeat",
            {"lease_id": lease_id, "instance_id": instance_id,
             "runtime_id": self.runtime_id,
             "profile_fingerprint": self.profile_fingerprint,
             "latency_class": str(row["latency_class"]),
             "expires_at": expires_at},
            actor="resource_admission")
        changed = conn.execute(
            """UPDATE workload_resource_leases
               SET heartbeat_at=?, expires_at=?, last_event_seq=?
               WHERE lease_id=? AND instance_id=? AND runtime_id=?
                 AND profile_fingerprint=? AND status='active'
                 AND heartbeat_at <= ? AND expires_at >= ?""",
            (now_text, expires_at, seq, lease_id, instance_id,
             self.runtime_id, self.profile_fingerprint, now_text,
             now_text),
        ).rowcount
        return changed == 1

    def mark_workload_reconciling(
        self,
        lease_id: str,
        instance_id: str,
        *,
        reason: str = "monitor_lost",
        now: datetime | str | None = None,
    ) -> bool:
        with self.graph.transaction() as conn:
            return self.mark_workload_reconciling_in_transaction(
                conn, lease_id, instance_id, reason=reason, now=now)

    def mark_workload_reconciling_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        instance_id: str,
        *,
        reason: str = "monitor_lost",
        now: datetime | str | None = None,
    ) -> bool:
        """Record monitor loss while deliberately retaining the reservation."""
        self._validate_identity("lease_id", lease_id)
        self._validate_identity("instance_id", instance_id)
        reason = self._validate_reason(reason)
        instant = _as_utc(now)
        now_text = _stamp(instant)
        row = conn.execute(
            """SELECT heartbeat_at,status,latency_class
               FROM workload_resource_leases
               WHERE lease_id=? AND instance_id=? AND runtime_id=?
                 AND profile_fingerprint=?
                 AND status IN ('active','reconciling')""",
            (lease_id, instance_id, self.runtime_id,
             self.profile_fingerprint),
        ).fetchone()
        if row is None or now_text < str(row["heartbeat_at"]):
            return False
        if str(row["status"]) == "reconciling":
            return True
        _, seq = self.graph.append_event(
            conn, "resource.workload_lease_reconciling",
            {"lease_id": lease_id, "instance_id": instance_id,
             "runtime_id": self.runtime_id,
             "profile_fingerprint": self.profile_fingerprint,
             "latency_class": str(row["latency_class"]),
             "status": "reconciling", "reason": reason},
            actor="resource_admission")
        changed = conn.execute(
            """UPDATE workload_resource_leases
               SET status='reconciling', reconcile_started_at=?,
                   last_event_seq=?
               WHERE lease_id=? AND instance_id=? AND runtime_id=?
                 AND profile_fingerprint=? AND status='active'
                 AND heartbeat_at <= ?""",
            (now_text, seq, lease_id, instance_id, self.runtime_id,
             self.profile_fingerprint, now_text),
        ).rowcount
        return changed == 1

    # Explicit monitor-loss spelling for callers that model the transition by
    # cause rather than destination.
    mark_workload_monitor_lost = mark_workload_reconciling
    mark_workload_monitor_lost_in_transaction = (
        mark_workload_reconciling_in_transaction)

    def workloads_needing_reconciliation(
        self,
        *,
        now: datetime | str | None = None,
    ) -> list[dict[str, str]]:
        """List opaque, profile-bound workload fences needing inspection.

        This method is intentionally read-only.  In particular, an expired
        monitor deadline never changes the lease status or releases capacity.
        """
        now_text = _stamp(_as_utc(now))
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT lease_id,instance_id,runtime_id,
                          profile_fingerprint,status,expires_at
                   FROM workload_resource_leases
                   WHERE profile_fingerprint=?
                     AND (status='reconciling'
                          OR (status='active' AND expires_at <= ?))
                   ORDER BY acquired_at,lease_id""",
                (self.profile_fingerprint, now_text),
            ).fetchall()
        return [
            {"lease_id": str(row[0]), "instance_id": str(row[1]),
             "runtime_id": str(row[2]),
             "profile_fingerprint": str(row[3]), "status": str(row[4]),
             "expires_at": str(row[5])}
            for row in rows
        ]

    def mark_stale_workload_runtime_reconciling(
        self,
        lease_id: str,
        instance_id: str,
        previous_runtime_id: str,
        *,
        reason: str = "runtime_restarted",
        now: datetime | str | None = None,
    ) -> bool:
        with self.graph.transaction() as conn:
            return self.mark_stale_workload_runtime_reconciling_in_transaction(
                conn, lease_id, instance_id, previous_runtime_id,
                reason=reason, now=now)

    def mark_stale_workload_runtime_reconciling_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        instance_id: str,
        previous_runtime_id: str,
        *,
        reason: str = "runtime_restarted",
        now: datetime | str | None = None,
    ) -> bool:
        """Fence one exact old runtime into reconciliation without releasing."""
        self._validate_identity("lease_id", lease_id)
        self._validate_identity("instance_id", instance_id)
        self._validate_identity("previous_runtime_id", previous_runtime_id)
        reason = self._validate_reason(reason)
        if previous_runtime_id == self.runtime_id:
            return False
        instant = _as_utc(now)
        now_text = _stamp(instant)
        row = conn.execute(
            """SELECT heartbeat_at,expires_at,latency_class
               FROM workload_resource_leases
               WHERE lease_id=? AND instance_id=? AND runtime_id=?
                 AND profile_fingerprint=? AND status='active'""",
            (lease_id, instance_id, previous_runtime_id,
             self.profile_fingerprint),
        ).fetchone()
        if (row is None
                or now_text < str(row["heartbeat_at"])
                or now_text < str(row["expires_at"])):
            return False
        _, seq = self.graph.append_event(
            conn, "resource.workload_lease_reconciling",
            {"lease_id": lease_id, "instance_id": instance_id,
             "previous_runtime_id": previous_runtime_id,
             "runtime_id": previous_runtime_id,
             "recovery_runtime_id": self.runtime_id,
             "profile_fingerprint": self.profile_fingerprint,
             "latency_class": str(row["latency_class"]),
             "status": "reconciling", "reason": reason},
            actor="resource_admission")
        changed = conn.execute(
            """UPDATE workload_resource_leases
               SET status='reconciling',reconcile_started_at=?,
                   last_event_seq=?
               WHERE lease_id=? AND instance_id=? AND runtime_id=?
                 AND profile_fingerprint=? AND status='active'
                 AND heartbeat_at <= ? AND expires_at <= ?""",
            (now_text, seq, lease_id, instance_id, previous_runtime_id,
             self.profile_fingerprint, now_text, now_text),
        ).rowcount
        return changed == 1

    def adopt_workload(
        self,
        lease_id: str,
        instance_id: str,
        *,
        previous_runtime_id: str,
        now: datetime | str | None = None,
    ) -> bool:
        with self.graph.transaction() as conn:
            return self.adopt_workload_in_transaction(
                conn, lease_id, instance_id,
                previous_runtime_id=previous_runtime_id, now=now)

    def adopt_workload_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        instance_id: str,
        *,
        previous_runtime_id: str,
        now: datetime | str | None = None,
    ) -> bool:
        """Adopt an explicitly reconciling lease across a runtime restart."""
        self._validate_identity("lease_id", lease_id)
        self._validate_identity("instance_id", instance_id)
        self._validate_identity("previous_runtime_id", previous_runtime_id)
        instant = _as_utc(now)
        now_text = _stamp(instant)
        expires_at = _stamp(
            instant + timedelta(seconds=self.lease_ttl_seconds))
        row = conn.execute(
            """SELECT w.runtime_id,w.status,w.heartbeat_at,w.latency_class,
                      p.state
               FROM workload_resource_leases AS w
               JOIN process_instances AS p ON p.instance_id=w.instance_id
               WHERE w.lease_id=? AND w.instance_id=?
                 AND w.profile_fingerprint=?""",
            (lease_id, instance_id, self.profile_fingerprint),
        ).fetchone()
        if row is None:
            return False
        if (str(row["runtime_id"]) == self.runtime_id
                and str(row["status"]) == "active"):
            return conn.execute(
                """SELECT 1 FROM graph_events
                   WHERE event_type='resource.workload_lease_adopted'
                     AND json_extract(payload_json,'$.lease_id')=?
                     AND json_extract(payload_json,'$.instance_id')=?
                     AND json_extract(payload_json,'$.previous_runtime_id')=?
                     AND json_extract(payload_json,'$.runtime_id')=?
                     AND json_extract(payload_json,'$.profile_fingerprint')=?
                   ORDER BY seq DESC LIMIT 1""",
                (lease_id, instance_id, previous_runtime_id,
                 self.runtime_id, self.profile_fingerprint),
            ).fetchone() is not None
        if (str(row["runtime_id"]) != previous_runtime_id
                or str(row["status"]) != "reconciling"
                or str(row["state"]) not in _WORKLOAD_ACTIVE_INSTANCE_STATES
                or now_text < str(row["heartbeat_at"])):
            return False
        _, seq = self.graph.append_event(
            conn, "resource.workload_lease_adopted",
            {"lease_id": lease_id, "instance_id": instance_id,
             "previous_runtime_id": previous_runtime_id,
             "runtime_id": self.runtime_id,
             "profile_fingerprint": self.profile_fingerprint,
             "latency_class": str(row["latency_class"]),
             "expires_at": expires_at},
            actor="resource_admission")
        changed = conn.execute(
            """UPDATE workload_resource_leases
               SET runtime_id=?, status='active', heartbeat_at=?,
                   expires_at=?, reconcile_started_at=NULL,
                   last_event_seq=?
               WHERE lease_id=? AND instance_id=? AND runtime_id=?
                 AND profile_fingerprint=? AND status='reconciling'
                 AND heartbeat_at <= ?""",
            (self.runtime_id, now_text, expires_at, seq, lease_id,
             instance_id, previous_runtime_id, self.profile_fingerprint,
             now_text),
        ).rowcount
        return changed == 1

    def release_workload(
        self,
        lease_id: str,
        instance_id: str,
        *,
        cgroup_empty: bool,
        previous_runtime_id: str | None = None,
        reason: str = "completed",
        now: datetime | str | None = None,
    ) -> bool:
        with self.graph.transaction() as conn:
            return self.release_workload_in_transaction(
                conn, lease_id, instance_id, cgroup_empty=cgroup_empty,
                previous_runtime_id=previous_runtime_id, reason=reason,
                now=now)

    def release_workload_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        instance_id: str,
        *,
        cgroup_empty: bool,
        previous_runtime_id: str | None = None,
        reason: str = "completed",
        now: datetime | str | None = None,
    ) -> bool:
        """Release capacity only after positive proof the cgroup is empty."""
        if cgroup_empty is not True:
            return False
        return self._finish_workload_exact(
            conn, lease_id, instance_id, status="released", reason=reason,
            now=_as_utc(now), previous_runtime_id=previous_runtime_id)

    def fence_workload(
        self,
        lease_id: str,
        instance_id: str,
        *,
        cgroup_empty: bool,
        previous_runtime_id: str | None = None,
        reason: str = "identity_fenced",
        now: datetime | str | None = None,
    ) -> bool:
        """Fence an unsafe workload, still requiring empty-cgroup proof."""
        with self.graph.transaction() as conn:
            return self.fence_workload_in_transaction(
                conn, lease_id, instance_id, cgroup_empty=cgroup_empty,
                previous_runtime_id=previous_runtime_id, reason=reason,
                now=now)

    def fence_workload_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        instance_id: str,
        *,
        cgroup_empty: bool,
        previous_runtime_id: str | None = None,
        reason: str = "identity_fenced",
        now: datetime | str | None = None,
    ) -> bool:
        if cgroup_empty is not True:
            return False
        return self._finish_workload_exact(
            conn, lease_id, instance_id, status="fenced", reason=reason,
            now=_as_utc(now), previous_runtime_id=previous_runtime_id)

    def heartbeat(
        self,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
        now: datetime | str | None = None,
        lease_ttl_seconds: int | None = None,
    ) -> bool:
        with self.graph.transaction() as conn:
            return self.heartbeat_in_transaction(
                conn, lease_id, attempt_id, worker_id=worker_id, now=now,
                lease_ttl_seconds=lease_ttl_seconds)

    def heartbeat_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
        now: datetime | str | None = None,
        lease_ttl_seconds: int | None = None,
    ) -> bool:
        """Renew only the live lease owned by this exact execution attempt."""
        instant = _as_utc(now)
        self._expire_due_in_transaction(conn, instant)
        now_text = _stamp(instant)
        row = conn.execute(
            """SELECT lease_id,latency_class FROM resource_leases
               WHERE lease_id = ? AND attempt_id = ? AND worker_id = ?
                 AND runtime_id = ? AND profile_fingerprint = ?
                 AND status = 'active'
                 AND expires_at > ? AND heartbeat_at <= ?""",
            (lease_id, attempt_id, worker_id, self.runtime_id,
             self.profile_fingerprint, now_text, now_text),
        ).fetchone()
        if row is None:
            return False
        ttl = self._ttl(lease_ttl_seconds)
        if str(row[1]) == "control":
            ttl = min(ttl, _CONTROL_LEASE_TTL_SECONDS)
        expires_at = _stamp(instant + timedelta(seconds=ttl))
        _, seq = self.graph.append_event(
            conn, "resource.lease_heartbeat",
            {"lease_id": lease_id, "attempt_id": attempt_id,
             "worker_id": worker_id, "runtime_id": self.runtime_id,
             "profile_fingerprint": self.profile_fingerprint,
             "latency_class": str(row[1]),
             "expires_at": expires_at},
            actor="resource_admission")
        changed = conn.execute(
            """UPDATE resource_leases
               SET heartbeat_at = ?, expires_at = ?, last_event_seq = ?
               WHERE lease_id = ? AND attempt_id = ? AND worker_id = ?
                 AND runtime_id = ? AND profile_fingerprint = ?
                 AND status = 'active' AND heartbeat_at <= ?""",
            (now_text, expires_at, seq, lease_id, attempt_id, worker_id,
             self.runtime_id, self.profile_fingerprint, now_text),
        ).rowcount
        return changed == 1

    def release(
        self,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
        reason: str = "completed",
        now: datetime | str | None = None,
    ) -> bool:
        with self.graph.transaction() as conn:
            return self.release_in_transaction(
                conn, lease_id, attempt_id, worker_id=worker_id,
                reason=reason, now=now)

    def release_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
        reason: str = "completed",
        now: datetime | str | None = None,
    ) -> bool:
        """Release an active lease only when its entire fence matches."""
        return self._finish_exact(
            conn, lease_id, attempt_id, worker_id=worker_id,
            status="released", reason=reason, now=_as_utc(now))

    def fence(
        self,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
        reason: str = "attempt_fenced",
        now: datetime | str | None = None,
    ) -> bool:
        with self.graph.transaction() as conn:
            return self.fence_in_transaction(
                conn, lease_id, attempt_id, worker_id=worker_id,
                reason=reason, now=now)

    def fence_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
        reason: str = "attempt_fenced",
        now: datetime | str | None = None,
    ) -> bool:
        """Fence an exact live attempt so late heartbeats cannot revive it."""
        return self._finish_exact(
            conn, lease_id, attempt_id, worker_id=worker_id,
            status="fenced", reason=reason, now=_as_utc(now))

    def reap_stale(self, *, now: datetime | str | None = None) -> int:
        """Expire TTL-dead leases and return the number fenced from capacity."""
        with self.graph.transaction() as conn:
            return self._expire_due_in_transaction(conn, _as_utc(now))

    def reap_runtime(
        self, runtime_id: str, *, now: datetime | str | None = None,
    ) -> int:
        """Fence active leases for one runtime known by the caller to be dead."""
        self._validate_identity("runtime_id", runtime_id)
        with self.graph.transaction() as conn:
            rows = conn.execute(
                """SELECT lease_id, attempt_id, worker_id
                   FROM resource_leases
                   WHERE runtime_id = ? AND status = 'active'""",
                (runtime_id,),
            ).fetchall()
            instant = _as_utc(now)
            count = 0
            for row in rows:
                if self._finish_any_runtime(
                        conn, str(row[0]), str(row[1]), str(row[2]),
                        runtime_id, status="fenced", reason="stale_runtime",
                        now=instant):
                    count += 1
            return count

    def status(self) -> dict[str, Any]:
        """Return argument-free capacity and active reservation telemetry."""
        snapshot = self.get_snapshot()
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT 'step' AS lease_kind,status,cpu_millis,ram_mib,
                          concurrency_slots,network_slots,accelerator,vram_mib,
                          expires_at,latency_class
                   FROM resource_leases WHERE status='active'
                   UNION ALL
                   SELECT 'workload' AS lease_kind,status,cpu_millis,ram_mib,
                          concurrency_slots,network_slots,accelerator,vram_mib,
                          expires_at,latency_class
                   FROM workload_resource_leases
                   WHERE status IN ('active','reconciling')""").fetchall()
        by_accelerator: dict[str, int] = {}
        by_latency_class: dict[str, int] = {}
        for row in rows:
            accelerator = str(row[6])
            if accelerator != "none":
                by_accelerator[accelerator] = (
                    by_accelerator.get(accelerator, 0) + int(row[7]))
            latency_class = str(row[9])
            by_latency_class[latency_class] = (
                by_latency_class.get(latency_class, 0) + 1)
        return {
            "runtime_id": self.runtime_id,
            "profile_fingerprint": self.profile_fingerprint,
            "budget": self.budget.model_dump(mode="json"),
            "snapshot": snapshot.model_dump(mode="json"),
            "active": {
                "leases": len(rows),
                "cpu_millis": sum(int(row[2]) for row in rows),
                "ram_mib": sum(int(row[3]) for row in rows),
                "concurrency_slots": sum(int(row[4]) for row in rows),
                "network_slots": sum(int(row[5]) for row in rows),
                "accelerator_vram_mib": by_accelerator,
                "latency_classes": by_latency_class,
                "earliest_expiry": min(
                    (str(row[8]) for row in rows), default=None),
            },
        }

    def fence_interrupted_in_transaction(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
        reason: str = "worker_process_interrupted",
        now: datetime | str | None = None,
    ) -> bool:
        """Fence an attempt whose prior runtime is authoritatively dead.

        Recovery runs in a new runtime, so normal owner fencing must reject it.
        This narrowly scoped method first resolves the recorded runtime for the
        exact lease/attempt/worker tuple, then closes that tuple only.
        """
        self._validate_identity("lease_id", lease_id)
        self._validate_identity("attempt_id", attempt_id)
        self._validate_identity("worker_id", worker_id)
        reason = self._validate_reason(reason)
        row = conn.execute(
            """SELECT runtime_id FROM resource_leases
               WHERE lease_id=? AND attempt_id=? AND worker_id=?
                 AND status='active'""",
            (lease_id, attempt_id, worker_id),
        ).fetchone()
        if row is None:
            return False
        return self._finish_any_runtime(
            conn, lease_id, attempt_id, worker_id, str(row[0]),
            status="fenced", reason=reason, now=_as_utc(now))

    @staticmethod
    def _is_valid_control_claim(claim: ResourceClaim) -> bool:
        return (
            claim.latency_class == "control"
            and float(claim.cpu_cores) == 0.0
            and int(claim.ram_mib) == 0
            and int(claim.vram_mib) == 0
            and _normalize_accelerator(claim.accelerator) == "none"
            and claim.network is False
            and int(claim.concurrency_slots) == 1
        )

    @staticmethod
    def _canonical_enforcement_json(
        enforcement: dict[str, Any] | str,
    ) -> str:
        if isinstance(enforcement, str):
            try:
                value = json.loads(enforcement)
            except json.JSONDecodeError as exc:
                raise ValueError("enforcement_json must be valid JSON") from exc
        else:
            value = enforcement
        if not isinstance(value, dict):
            raise ValueError("enforcement_json must describe a JSON object")
        try:
            encoded = json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("enforcement_json must contain JSON values") from exc
        if len(encoded.encode("utf-8")) > 65_536:
            raise ValueError("enforcement_json exceeds 65536 encoded bytes")
        return encoded

    @staticmethod
    def _workload_instance_is_launchable(
        conn: sqlite3.Connection, instance_id: str,
    ) -> bool:
        row = conn.execute(
            "SELECT state FROM process_instances WHERE instance_id=?",
            (instance_id,),
        ).fetchone()
        return (row is not None
                and str(row[0]) in _WORKLOAD_ACTIVE_INSTANCE_STATES)

    def _resource_row_matches_claim(
        self,
        row: sqlite3.Row,
        claim: ResourceClaim,
        requested: dict[str, int],
    ) -> bool:
        return (
            int(row["cpu_millis"]) == requested["cpu_millis"]
            and int(row["ram_mib"]) == requested["ram_mib"]
            and int(row["concurrency_slots"])
            == requested["concurrency_slots"]
            and int(row["network_slots"]) == requested["network_slots"]
            and str(row["accelerator"])
            == _normalize_accelerator(claim.accelerator)
            and int(row["vram_mib"])
            == requested[self._vram_key(claim.accelerator)]
            and str(row["profile_fingerprint"])
            == self.profile_fingerprint
            and str(row["latency_class"]) == claim.latency_class
        )

    def _existing_workload_matches(
        self,
        row: sqlite3.Row,
        claim: ResourceClaim,
        requested: dict[str, int],
        enforcement_text: str,
        *,
        source_step_lease_id: str | None,
        source_attempt_id: str | None,
        source_worker_id: str | None,
    ) -> bool:
        return (
            self._resource_row_matches_claim(row, claim, requested)
            and str(row["enforcement_json"]) == enforcement_text
            and row["source_step_lease_id"] == source_step_lease_id
            and row["source_attempt_id"] == source_attempt_id
            and row["source_worker_id"] == source_worker_id
        )

    def _record_workload_decision(
        self,
        conn: sqlite3.Connection,
        decision: AdmissionDecision,
        instance_id: str,
    ) -> None:
        idempotency_key = (
            f"workload-admission:{instance_id}:{self.profile_fingerprint}:"
            f"{decision.status}:{decision.reason}")
        if conn.execute(
                "SELECT 1 FROM graph_events WHERE idempotency_key=?",
                (idempotency_key,)).fetchone() is not None:
            return
        self.graph.append_event(
            conn, f"resource.workload_admission_{decision.status}",
            {"instance_id": instance_id,
             "runtime_id": self.runtime_id,
             "profile_fingerprint": self.profile_fingerprint,
             "status": decision.status, "reason": decision.reason,
             "retryable": decision.retryable,
             "deficits": decision.deficits},
            actor="resource_admission", idempotency_key=idempotency_key)

    def _insert_workload_lease(
        self,
        conn: sqlite3.Connection,
        claim: ResourceClaim,
        requested: dict[str, int],
        instance_id: str,
        enforcement_text: str,
        instant: datetime,
        *,
        source_step_lease_id: str | None,
        source_attempt_id: str | None,
        source_worker_id: str | None,
        reason: str,
    ) -> AdmissionDecision:
        lease_id = new_id("workload_lease")
        now_text = _stamp(instant)
        expires_at = _stamp(
            instant + timedelta(seconds=self.lease_ttl_seconds))
        accelerator = _normalize_accelerator(claim.accelerator)
        event_payload = {
            "lease_id": lease_id,
            "instance_id": instance_id,
            "source_step_lease_id": source_step_lease_id,
            "source_attempt_id": source_attempt_id,
            "source_worker_id": source_worker_id,
            "runtime_id": self.runtime_id,
            "profile_fingerprint": self.profile_fingerprint,
            "latency_class": claim.latency_class,
            "cpu_millis": requested["cpu_millis"],
            "ram_mib": requested["ram_mib"],
            "concurrency_slots": requested["concurrency_slots"],
            "network_slots": requested["network_slots"],
            "accelerator": accelerator,
            "vram_mib": requested[self._vram_key(claim.accelerator)],
            "enforcement_sha256": sha256_text(enforcement_text),
            "expires_at": expires_at,
            "reason": reason,
        }
        event_id, seq = self.graph.append_event(
            conn, "resource.workload_lease_acquired", event_payload,
            actor="resource_admission")
        self.graph.append_node(
            conn, "workload_resource_lease", event_payload,
            event_id=event_id, node_id=lease_id)
        self.graph.append_edge(
            conn, lease_id, "reserves_for", instance_id, event_id=event_id)
        if source_step_lease_id is not None:
            self.graph.append_edge(
                conn, lease_id, "transferred_from", source_step_lease_id,
                event_id=event_id)
        conn.execute(
            """INSERT INTO workload_resource_leases
               (lease_id,instance_id,source_step_lease_id,source_attempt_id,
                source_worker_id,runtime_id,profile_fingerprint,latency_class,
                cpu_millis,ram_mib,concurrency_slots,network_slots,accelerator,
                vram_mib,enforcement_json,status,acquired_at,heartbeat_at,
                expires_at,last_event_seq)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'active',?,?,?,?)""",
            (lease_id, instance_id, source_step_lease_id, source_attempt_id,
             source_worker_id, self.runtime_id, self.profile_fingerprint,
             claim.latency_class, requested["cpu_millis"],
             requested["ram_mib"], requested["concurrency_slots"],
             requested["network_slots"], accelerator,
             requested[self._vram_key(claim.accelerator)], enforcement_text,
             now_text, now_text, expires_at, seq),
        )
        return AdmissionDecision(
            status="admitted", reason=reason, retryable=False,
            lease_id=lease_id, expires_at=expires_at)

    def _finish_workload_exact(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        instance_id: str,
        *,
        status: Literal["released", "fenced"],
        reason: str,
        now: datetime,
        previous_runtime_id: str | None = None,
    ) -> bool:
        self._validate_identity("lease_id", lease_id)
        self._validate_identity("instance_id", instance_id)
        reason = self._validate_reason(reason)
        expected_runtime_id = self.runtime_id
        cross_runtime = previous_runtime_id is not None
        if previous_runtime_id is not None:
            self._validate_identity(
                "previous_runtime_id", previous_runtime_id)
            if previous_runtime_id == self.runtime_id:
                return False
            expected_runtime_id = previous_runtime_id
        now_text = _stamp(now)
        row = conn.execute(
            """SELECT heartbeat_at,latency_class,status
               FROM workload_resource_leases
               WHERE lease_id=? AND instance_id=? AND runtime_id=?
                 AND profile_fingerprint=?
                 AND status IN ('active','reconciling')""",
            (lease_id, instance_id, expected_runtime_id,
             self.profile_fingerprint),
        ).fetchone()
        if (row is None or now_text < str(row["heartbeat_at"])
                or (cross_runtime and str(row["status"]) != "reconciling")):
            return False
        event_payload = {
            "lease_id": lease_id, "instance_id": instance_id,
            "runtime_id": expected_runtime_id,
            "profile_fingerprint": self.profile_fingerprint,
            "latency_class": str(row["latency_class"]),
            "previous_status": str(row["status"]),
            "status": status, "reason": reason,
            "cgroup_empty": True,
        }
        if cross_runtime:
            event_payload["recovery_runtime_id"] = self.runtime_id
        _, seq = self.graph.append_event(
            conn, f"resource.workload_lease_{status}", event_payload,
            actor="resource_admission")
        changed = conn.execute(
            """UPDATE workload_resource_leases
               SET status=?, released_at=?, release_reason=?,last_event_seq=?
               WHERE lease_id=? AND instance_id=? AND runtime_id=?
                 AND profile_fingerprint=?
                 AND status=?
                 AND heartbeat_at <= ?""",
            (status, now_text, reason, seq, lease_id, instance_id,
             expected_runtime_id, self.profile_fingerprint,
             str(row["status"]), now_text),
        ).rowcount
        return changed == 1

    @staticmethod
    def _validate_identity(name: str, value: str) -> None:
        if (not isinstance(value, str) or len(value) > 200
                or re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None):
            raise ValueError(
                f"{name} must be a 1-200 character opaque identifier")

    @staticmethod
    def _validate_reason(reason: str) -> str:
        if (not isinstance(reason, str) or len(reason) > 80
                or re.fullmatch(r"[A-Za-z0-9_.:-]+", reason) is None):
            raise ValueError("reason must be a short non-sensitive code")
        return reason

    def _validate_snapshot_freshness(
        self, snapshot: ResourceSnapshot, reference_time: datetime,
    ) -> None:
        if snapshot.captured_at is None:
            raise RuntimeError("resource telemetry has no capture timestamp")
        age_seconds = (
            _as_utc(reference_time) - _as_utc(snapshot.captured_at)
        ).total_seconds()
        if age_seconds > self.max_snapshot_age_seconds:
            raise RuntimeError("resource telemetry is stale")
        if age_seconds < -self.max_snapshot_future_skew_seconds:
            raise RuntimeError("resource telemetry timestamp is in the future")

    def _ttl(self, override: int | None) -> int:
        ttl = self.lease_ttl_seconds if override is None else int(override)
        if ttl <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        return ttl

    @staticmethod
    def _vram_key(accelerator: str) -> str:
        return f"vram_mib:{_normalize_accelerator(accelerator)}"

    def _claim_units(self, claim: ResourceClaim) -> dict[str, int]:
        return {
            "cpu_millis": int(math.ceil(claim.cpu_cores * 1000.0)),
            "ram_mib": claim.ram_mib,
            "concurrency_slots": claim.concurrency_slots,
            "network_slots": 1 if claim.network else 0,
            self._vram_key(claim.accelerator): claim.vram_mib,
        }

    def _existing_matches_claim(
        self,
        row: sqlite3.Row | tuple[Any, ...],
        claim: ResourceClaim,
        requested: dict[str, int],
    ) -> bool:
        return (
            int(row[5]) == requested["cpu_millis"]
            and int(row[6]) == requested["ram_mib"]
            and int(row[7]) == requested["concurrency_slots"]
            and int(row[8]) == requested["network_slots"]
            and str(row[9]) == _normalize_accelerator(claim.accelerator)
            and int(row[10]) == requested[self._vram_key(claim.accelerator)]
            and str(row[11]) == self.profile_fingerprint
            and str(row[12]) == claim.latency_class
        )

    def _budget_units(
        self,
        claim: ResourceClaim,
        *,
        reserve_interactive: bool = True,
    ) -> dict[str, int]:
        accelerator = _normalize_accelerator(claim.accelerator)
        interactive = claim.latency_class == "interactive"
        concurrency_capacity = self.budget.concurrency_slots
        network_capacity = self.budget.network_slots
        if reserve_interactive and not interactive:
            # Background/batch work cannot consume the final interactive lane.
            concurrency_capacity = max(0, concurrency_capacity - 1)
            network_capacity = max(0, network_capacity - 1)
        return {
            "cpu_millis": self.budget.cpu_millis,
            "ram_mib": self.budget.ram_mib,
            "concurrency_slots": concurrency_capacity,
            "network_slots": network_capacity,
            self._vram_key(accelerator): (
                0 if accelerator == "none" else
                self.budget.accelerator_vram_mib.get(accelerator, 0)),
        }

    def _snapshot_units(
        self, snapshot: ResourceSnapshot, claim: ResourceClaim,
    ) -> dict[str, int]:
        accelerator = _normalize_accelerator(claim.accelerator)
        return {
            "cpu_millis": snapshot.available_cpu_millis,
            "ram_mib": snapshot.available_ram_mib,
            # Concurrency is a Friday-owned reservation, not an OS snapshot.
            "concurrency_slots": self.budget.concurrency_slots,
            "network_slots": snapshot.available_network_slots,
            self._vram_key(accelerator): (
                0 if accelerator == "none" else
                snapshot.available_accelerator_vram_mib.get(accelerator, 0)),
        }

    def _leased_units(
        self,
        conn: sqlite3.Connection,
        accelerator: str,
        *,
        include_control: bool = True,
        noninteractive_only: bool = False,
    ) -> dict[str, int]:
        normalized = _normalize_accelerator(accelerator)
        row = conn.execute(
            """SELECT COALESCE(SUM(cpu_millis), 0),
                      COALESCE(SUM(ram_mib), 0),
                      COALESCE(SUM(concurrency_slots), 0),
                      COALESCE(SUM(network_slots), 0),
                      COALESCE(SUM(CASE WHEN accelerator = ?
                                        THEN vram_mib ELSE 0 END), 0)
               FROM (
                   SELECT cpu_millis,ram_mib,concurrency_slots,network_slots,
                          accelerator,vram_mib
                   FROM resource_leases
                   WHERE status='active'
                     AND (? = 1 OR latency_class <> 'control')
                     AND (? = 0 OR latency_class IN ('background','batch'))
                   UNION ALL
                   SELECT cpu_millis,ram_mib,concurrency_slots,network_slots,
                          accelerator,vram_mib
                   FROM workload_resource_leases
                   WHERE status IN ('active','reconciling')
                     AND (? = 0 OR latency_class IN ('background','batch'))
               ) AS reserving""",
            (normalized, int(include_control), int(noninteractive_only),
             int(noninteractive_only)),
        ).fetchone()
        return {
            "cpu_millis": int(row[0]),
            "ram_mib": int(row[1]),
            "concurrency_slots": int(row[2]),
            "network_slots": int(row[3]),
            self._vram_key(normalized): int(row[4]),
        }

    def _remaining_budget(
        self,
        claim: ResourceClaim,
        leased: dict[str, int],
        *,
        reserve_interactive: bool = True,
    ) -> dict[str, int]:
        capacity = self._budget_units(
            claim, reserve_interactive=reserve_interactive)
        return {key: max(0, amount - leased.get(key, 0))
                for key, amount in capacity.items()}

    @staticmethod
    def _deficits(
        requested: dict[str, int], available: dict[str, int],
    ) -> dict[str, int]:
        return {key: amount - available.get(key, 0)
                for key, amount in requested.items()
                if amount > available.get(key, 0)}

    def _record_decision(
        self,
        conn: sqlite3.Connection,
        decision: AdmissionDecision,
        step_id: str,
        attempt_id: str,
    ) -> None:
        # Deliberately contains resource accounting only, never model/tool args.
        idempotency_key = (
            f"admission:{step_id}:{attempt_id}:{decision.status}:"
            f"{decision.reason}")
        if conn.execute(
                "SELECT 1 FROM graph_events WHERE idempotency_key=?",
                (idempotency_key,)).fetchone() is not None:
            return
        self.graph.append_event(
            conn, f"resource.admission_{decision.status}",
            {"step_id": step_id, "attempt_id": attempt_id,
             "status": decision.status, "reason": decision.reason,
             "retryable": decision.retryable,
             "deficits": decision.deficits},
            actor="resource_admission", idempotency_key=idempotency_key)

    def _finish_exact(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        attempt_id: str,
        *,
        worker_id: str,
        status: Literal["released", "fenced"],
        reason: str,
        now: datetime,
    ) -> bool:
        self._validate_identity("lease_id", lease_id)
        self._validate_identity("attempt_id", attempt_id)
        self._validate_identity("worker_id", worker_id)
        reason = self._validate_reason(reason)
        self._expire_due_in_transaction(conn, now)
        return self._finish_any_runtime(
            conn, lease_id, attempt_id, worker_id, self.runtime_id,
            status=status, reason=reason, now=now,
            expected_profile_fingerprint=self.profile_fingerprint)

    def _finish_any_runtime(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        attempt_id: str,
        worker_id: str,
        runtime_id: str,
        *,
        status: Literal["released", "fenced", "expired"],
        reason: str,
        now: datetime,
        expected_profile_fingerprint: str | None = None,
    ) -> bool:
        row = conn.execute(
            """SELECT lease_id,profile_fingerprint,latency_class,heartbeat_at
               FROM resource_leases
               WHERE lease_id = ? AND attempt_id = ? AND worker_id = ?
                 AND runtime_id = ? AND status = 'active'
                 AND heartbeat_at <= ?""",
            (lease_id, attempt_id, worker_id, runtime_id, _stamp(now)),
        ).fetchone()
        if (row is None or (expected_profile_fingerprint is not None
                            and str(row[1])
                            != expected_profile_fingerprint)):
            return False
        recorded_profile = str(row[1])
        latency_class = str(row[2])
        finished_at = _stamp(now)
        _, seq = self.graph.append_event(
            conn, f"resource.lease_{status}",
            {"lease_id": lease_id, "attempt_id": attempt_id,
             "worker_id": worker_id, "runtime_id": runtime_id,
             "profile_fingerprint": recorded_profile,
             "latency_class": latency_class,
             "status": status, "reason": reason},
            actor="resource_admission")
        changed = conn.execute(
            """UPDATE resource_leases
               SET status = ?, released_at = ?, release_reason = ?,
                   last_event_seq = ?
               WHERE lease_id = ? AND attempt_id = ? AND worker_id = ?
                 AND runtime_id = ? AND profile_fingerprint = ?
                 AND status = 'active' AND heartbeat_at <= ?""",
            (status, finished_at, reason, seq, lease_id, attempt_id,
             worker_id, runtime_id, recorded_profile, finished_at),
        ).rowcount
        return changed == 1

    def _expire_due_in_transaction(
        self, conn: sqlite3.Connection, now: datetime,
    ) -> int:
        now_text = _stamp(now)
        rows = conn.execute(
            """SELECT lease_id, attempt_id, worker_id, runtime_id
               FROM resource_leases
               WHERE status = 'active' AND expires_at <= ?""",
            (now_text,),
        ).fetchall()
        count = 0
        for row in rows:
            if self._finish_any_runtime(
                    conn, str(row[0]), str(row[1]), str(row[2]), str(row[3]),
                    status="expired", reason="lease_ttl_expired", now=now):
                count += 1
        return count
