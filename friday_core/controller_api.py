"""Framework-neutral controller endpoint orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .controller_auth import (
    ControllerAuthError,
    ControllerAuthService,
    ControllerPrincipal,
)
from .transport import controller_origin


@dataclass(frozen=True)
class ControllerAPIError(RuntimeError):
    status_code: int
    detail: str


def _require_fields(body: Mapping[str, Any], expected: set[str], detail: str) -> None:
    if not isinstance(body, Mapping) or set(body) != expected:
        raise ControllerAPIError(400, detail)


class ControllerAPI:
    """Strict controller pairing, session, identity, and revocation boundary."""

    def __init__(
        self,
        auth: ControllerAuthService,
        transport_binding_sha256: str,
    ) -> None:
        self.auth = auth
        self.transport_binding_sha256 = transport_binding_sha256

    def create_pairing(self) -> dict[str, Any]:
        return self.auth.create_pairing(self.transport_binding_sha256)

    def prepare_pairing(
        self, headers: Mapping[str, str], body: Mapping[str, Any],
    ) -> dict[str, Any]:
        _require_fields(
            body, {"pairing_token", "label", "public_jwk"},
            "pairing request fields are invalid",
        )
        try:
            return self.auth.prepare_pairing(
                body["pairing_token"], body["label"], body["public_jwk"],
                origin=controller_origin(headers),
                transport_binding_sha256=self.transport_binding_sha256,
            )
        except (ControllerAuthError, TypeError, ValueError) as exc:
            raise ControllerAPIError(
                401, "controller pairing was rejected") from exc

    def complete_pairing(
        self, headers: Mapping[str, str], body: Mapping[str, Any],
    ) -> dict[str, Any]:
        _require_fields(
            body,
            {"pairing_token", "label", "public_jwk", "signature_b64url"},
            "pairing proof fields are invalid",
        )
        try:
            return self.auth.complete_pairing(
                body["pairing_token"], body["label"], body["public_jwk"],
                body["signature_b64url"],
                origin=controller_origin(headers),
                transport_binding_sha256=self.transport_binding_sha256,
            )
        except (ControllerAuthError, TypeError, ValueError) as exc:
            raise ControllerAPIError(
                401, "controller pairing was rejected") from exc

    def create_session_challenge(
        self, headers: Mapping[str, str], body: Mapping[str, Any],
    ) -> dict[str, Any]:
        _require_fields(
            body, {"controller_id"},
            "controller challenge fields are invalid",
        )
        try:
            return self.auth.create_session_challenge(
                body["controller_id"], origin=controller_origin(headers),
                transport_binding_sha256=self.transport_binding_sha256,
            )
        except (ControllerAuthError, TypeError, ValueError) as exc:
            raise ControllerAPIError(
                401, "controller challenge was rejected") from exc

    def complete_session(self, body: Mapping[str, Any]) -> dict[str, Any]:
        _require_fields(
            body,
            {"challenge_id", "challenge", "proof_payload", "signature_b64url"},
            "controller proof fields are invalid",
        )
        try:
            return self.auth.complete_session(
                body["challenge_id"], body["challenge"],
                body["proof_payload"], body["signature_b64url"],
            )
        except (ControllerAuthError, TypeError, ValueError) as exc:
            raise ControllerAPIError(
                401, "controller proof was rejected") from exc

    @staticmethod
    def identity(principal: ControllerPrincipal) -> dict[str, Any]:
        return {
            "controller_id": principal.controller_id,
            "session_id": principal.session_id,
            "public_key_sha256": principal.public_key_sha256,
            "controller_epoch": principal.controller_epoch,
            "idle_expires_at": principal.idle_expires_at,
            "absolute_expires_at": principal.absolute_expires_at,
        }

    def list_controllers(self) -> dict[str, Any]:
        return {"controllers": self.auth.list_controllers()}

    def revoke_controller(
        self, principal: ControllerPrincipal, controller_id: str,
    ) -> dict[str, str]:
        try:
            self.auth.revoke_controller(principal, controller_id)
        except ControllerAuthError as exc:
            raise ControllerAPIError(
                403, "controller revocation was rejected") from exc
        return {"status": "revoked", "controller_id": controller_id}

    def revoke_session(self, principal: ControllerPrincipal) -> dict[str, str]:
        self.auth.revoke_session(principal)
        return {"status": "revoked", "session_id": principal.session_id}
