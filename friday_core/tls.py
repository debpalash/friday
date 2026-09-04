"""Private local-CA bootstrap for Friday's HTTPS control plane."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from friday_host import fs


_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_GENERATION = re.compile(r"generation_[0-9a-f]{32}\Z")


class TLSBootstrapError(RuntimeError):
    """Friday cannot prove a usable, locally anchored TLS identity."""


@dataclass(frozen=True)
class TLSMaterial:
    certfile: Path
    keyfile: Path
    cafile: Path
    hosts: tuple[str, ...]
    generation: str
    certificate_sha256: str
    transport_binding_sha256: str


def normalize_tls_hosts(hosts: Iterable[str]) -> tuple[str, ...]:
    """Return an exact, ambiguity-free SAN host set."""
    normalized: set[str] = {"localhost", "127.0.0.1", "::1"}
    for raw in hosts:
        if not isinstance(raw, str):
            raise TLSBootstrapError("TLS host must be text")
        value = raw.strip().lower()
        if (not value or len(value) > 253 or value.endswith(".")
                or any(ord(char) > 127 for char in value)
                or any(char in value for char in "/\\%[]@")):
            raise TLSBootstrapError("TLS host is ambiguous")
        try:
            value = ipaddress.ip_address(value).compressed
        except ValueError:
            labels = value.split(".")
            if any(not _DNS_LABEL.fullmatch(label) for label in labels):
                raise TLSBootstrapError("TLS DNS host is invalid") from None
        normalized.add(value)
    return tuple(sorted(normalized))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _secure_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TLSBootstrapError("TLS state directory is unavailable") from exc
    if (not stat.S_ISDIR(metadata.st_mode) or not fs.owned_by_caller(metadata)
            or not fs.private_mode_ok(metadata, mask=0o022)):
        raise TLSBootstrapError("TLS state directory identity is unsafe")
    fs.chmod_private(path, 0o700)


def _secure_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TLSBootstrapError(f"TLS material is unavailable: {path.name}") from exc
    if (not stat.S_ISREG(metadata.st_mode) or not fs.owned_by_caller(metadata)
            or metadata.st_nlink != 1):
        raise TLSBootstrapError(f"TLS material identity is unsafe: {path.name}")
    fs.chmod_private(path, 0o600)


def _write_private(path: Path, value: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | fs.PRIVATE_OPEN_FLAGS,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        fs.chmod_private(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fs.fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


_CA_COMMON_NAME = "Friday Local Controller CA"
_SERVER_COMMON_NAME = "Friday Local Assistant"
_CA_DAYS = 3650


def _load_private_key(path: Path) -> ec.EllipticCurvePrivateKey:
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise TLSBootstrapError("TLS certificate operation failed") from exc
    if (not isinstance(key, ec.EllipticCurvePrivateKey)
            or not isinstance(key.curve, ec.SECP256R1)):
        raise TLSBootstrapError("TLS certificate operation failed")
    return key


def _load_certificate(path: Path) -> x509.Certificate:
    try:
        return x509.load_pem_x509_certificate(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise TLSBootstrapError("TLS certificate operation failed") from exc


def _spki_der(public_key) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo)


def _public_key_digest(*, key: Path | None = None,
                       certificate: Path | None = None) -> str:
    if (key is None) == (certificate is None):
        raise ValueError("provide exactly one TLS identity")
    if key is not None:
        der = _spki_der(_load_private_key(key).public_key())
    else:
        assert certificate is not None
        der = _spki_der(_load_certificate(certificate).public_key())
    return hashlib.sha256(der).hexdigest()


def _certificate_hosts(certificate: Path) -> tuple[str, ...]:
    loaded = _load_certificate(certificate)
    try:
        extension = loaded.extensions.get_extension_for_class(
            x509.SubjectAlternativeName)
    except x509.ExtensionNotFound as exc:
        raise TLSBootstrapError("TLS certificate has no subjectAltName") from exc
    values: set[str] = set()
    for name in extension.value.get_values_for_type(x509.DNSName):
        values.add(name.strip().lower())
    for address in extension.value.get_values_for_type(x509.IPAddress):
        values.add(address.compressed)
    if not values:
        raise TLSBootstrapError("TLS certificate has no subjectAltName")
    return tuple(sorted(values))


def _private_key_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption())


def _generate_ca(root: Path) -> tuple[Path, Path]:
    key_path = root / "ca-key.pem"
    certificate_path = root / "friday-local-ca.crt"
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _CA_COMMON_NAME)])
    now = datetime.now(UTC).replace(microsecond=0)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=_CA_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None),
                       critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=False, content_commitment=False,
            key_encipherment=False, data_encipherment=False,
            key_agreement=False, key_cert_sign=True, crl_sign=True,
            encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
                       critical=False)
        .sign(key, hashes.SHA256())
    )
    _write_private(key_path, _private_key_pem(key))
    _write_private(certificate_path,
                   certificate.public_bytes(serialization.Encoding.PEM))
    for path in (key_path, certificate_path):
        _secure_file(path)
    return key_path, certificate_path


def _issue_server(
    root: Path, ca_key: Path, ca_certificate: Path,
    hosts: tuple[str, ...], *, days: int,
) -> Path:
    generation = root / f"generation_{secrets.token_hex(16)}"
    generation.mkdir(mode=0o700)
    key_path = generation / "server-key.pem"
    certificate_path = generation / "server-cert.pem"
    san_items: list[x509.GeneralName] = []
    for host in hosts:
        try:
            san_items.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            san_items.append(x509.DNSName(host))
    try:
        issuer_key = _load_private_key(ca_key)
        issuer = _load_certificate(ca_certificate)
        issuer_ski = issuer.extensions.get_extension_for_class(
            x509.SubjectKeyIdentifier).value
        key = ec.generate_private_key(ec.SECP256R1())
        now = datetime.now(UTC).replace(microsecond=0)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(
                NameOID.COMMON_NAME, _SERVER_COMMON_NAME)]))
            .issuer_name(issuer.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None),
                           critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True, content_commitment=False,
                key_encipherment=True, data_encipherment=False,
                key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.SubjectAlternativeName(san_items), critical=False)
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
                critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier
                .from_issuer_subject_key_identifier(issuer_ski),
                critical=False)
            .sign(issuer_key, hashes.SHA256())
        )
        _write_private(key_path, _private_key_pem(key))
        _write_private(certificate_path,
                       certificate.public_bytes(serialization.Encoding.PEM))
    except Exception:
        shutil.rmtree(generation, ignore_errors=True)
        raise
    for path in (key_path, certificate_path):
        _secure_file(path)
    os.chmod(generation, 0o700)
    return generation


def _verify_issued_by(ca_certificate: Path, certificate: Path) -> None:
    """Equivalent of ``openssl verify -CAfile ca cert`` for Friday's chain."""
    issuer = _load_certificate(ca_certificate)
    leaf = _load_certificate(certificate)
    now = datetime.now(UTC)
    try:
        constraints = issuer.extensions.get_extension_for_class(
            x509.BasicConstraints).value
    except x509.ExtensionNotFound as exc:
        raise TLSBootstrapError("TLS certificate operation failed") from exc
    if not constraints.ca or leaf.issuer != issuer.subject:
        raise TLSBootstrapError("TLS certificate operation failed")
    for item in (issuer, leaf):
        if not item.not_valid_before_utc <= now <= item.not_valid_after_utc:
            raise TLSBootstrapError("TLS certificate operation failed")
    algorithm = leaf.signature_hash_algorithm
    if algorithm is None:
        raise TLSBootstrapError("TLS certificate operation failed")
    try:
        issuer.public_key().verify(
            leaf.signature, leaf.tbs_certificate_bytes, ec.ECDSA(algorithm))
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise TLSBootstrapError("TLS certificate operation failed") from exc


def _certificate_current(path: Path, *, seconds: int) -> bool:
    """Equivalent of ``openssl x509 -checkend``."""
    certificate = _load_certificate(path)
    return certificate.not_valid_after_utc > (
        datetime.now(UTC) + timedelta(seconds=seconds))


def _manifest(ca: Path, generation: Path,
              hosts: tuple[str, ...]) -> dict:
    return {
        "schema_version": 1,
        "generation": generation.name,
        "hosts": list(hosts),
        "ca_certificate_sha256": _sha256_file(ca),
        "server_certificate_sha256": _sha256_file(
            generation / "server-cert.pem"),
        "server_key_sha256": _sha256_file(generation / "server-key.pem"),
    }


def _publish_generation(root: Path, ca: Path, generation: Path,
                        hosts: tuple[str, ...]) -> None:
    value = _manifest(ca, generation, hosts)
    _atomic_json(generation / "manifest.json", value)
    _atomic_json(root / "active.json", value)


def _load_json(path: Path) -> dict:
    _secure_file(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TLSBootstrapError(f"TLS metadata is invalid: {path.name}") from exc
    if not isinstance(value, dict):
        raise TLSBootstrapError(f"TLS metadata is invalid: {path.name}")
    return value


def _validate_active(
    root: Path, *, renew_before_seconds: int,
) -> tuple[dict, Path, tuple[str, ...], bool]:
    ca_key = root / "ca-key.pem"
    ca_certificate = root / "friday-local-ca.crt"
    active = _load_json(root / "active.json")
    for path in (ca_key, ca_certificate):
        _secure_file(path)
    generation_name = active.get("generation")
    if not isinstance(generation_name, str) or not _GENERATION.fullmatch(
            generation_name):
        raise TLSBootstrapError("active TLS generation is invalid")
    generation = root / generation_name
    _secure_directory(generation)
    certificate = generation / "server-cert.pem"
    key = generation / "server-key.pem"
    manifest = _load_json(generation / "manifest.json")
    for path in (certificate, key):
        _secure_file(path)
    if active != manifest or manifest.get("schema_version") != 1:
        raise TLSBootstrapError("active TLS metadata does not match its generation")
    hosts = manifest.get("hosts")
    if (not isinstance(hosts, list)
            or any(not isinstance(item, str) for item in hosts)):
        raise TLSBootstrapError("TLS SAN metadata is invalid")
    recorded_hosts = tuple(hosts)
    if (manifest.get("ca_certificate_sha256") != _sha256_file(ca_certificate)
            or manifest.get("server_certificate_sha256") !=
                _sha256_file(certificate)
            or manifest.get("server_key_sha256") != _sha256_file(key)):
        raise TLSBootstrapError("TLS material digest changed")
    if (_public_key_digest(key=ca_key)
            != _public_key_digest(certificate=ca_certificate)):
        raise TLSBootstrapError("TLS CA key does not match its certificate")
    if (_public_key_digest(key=key)
            != _public_key_digest(certificate=certificate)):
        raise TLSBootstrapError("TLS server key does not match its certificate")
    _verify_issued_by(ca_certificate, certificate)
    actual_hosts = _certificate_hosts(certificate)
    if actual_hosts != recorded_hosts:
        raise TLSBootstrapError("TLS certificate SANs changed")
    if not _certificate_current(ca_certificate, seconds=renew_before_seconds):
        raise TLSBootstrapError(
            "Friday local CA is expiring; explicit trust-anchor rotation is required")
    server_current = _certificate_current(
        certificate, seconds=renew_before_seconds)
    return manifest, generation, actual_hosts, server_current


def _initial_material(
    state_root: Path, root: Path,
    hosts: tuple[str, ...], *, certificate_days: int,
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=".tls-stage-", dir=state_root))
    os.chmod(staging, 0o700)
    try:
        ca_key, ca_certificate = _generate_ca(staging)
        generation = _issue_server(
            staging, ca_key, ca_certificate, hosts,
            days=certificate_days)
        _publish_generation(staging, ca_certificate, generation, hosts)
        os.rename(staging, root)
        fs.fsync_directory(state_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def ensure_tls_material(
    state_root: str | Path, hosts: Iterable[str], *,
    openssl_path: str = "/usr/bin/openssl", certificate_days: int = 397,
    renew_before_seconds: int = 30 * 24 * 60 * 60,
) -> TLSMaterial:
    """Return verified TLS paths, bootstrapping or renewing only the leaf.

    Once created, the local CA is never silently replaced. Any corruption or
    expiry of that trust anchor fails startup so a compromised or accidental
    certificate change cannot train clients to trust a new identity.

    ``openssl_path`` is accepted for compatibility and ignored: certificate
    work uses the ``cryptography`` package on every platform.
    """
    del openssl_path
    if not 30 <= certificate_days <= 825:
        raise ValueError("TLS certificate lifetime is invalid")
    if not 86_400 <= renew_before_seconds <= certificate_days * 86_400 // 2:
        raise ValueError("TLS renewal window is invalid")
    desired_hosts = normalize_tls_hosts(hosts)
    state = Path(state_root)
    _secure_directory(state, create=True)
    root = state / "tls"
    lock_path = state / "tls-bootstrap.lock"
    flags = os.O_RDWR | os.O_CREAT | fs.PRIVATE_OPEN_FLAGS
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise TLSBootstrapError("TLS bootstrap lock is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode)
                or not fs.owned_by_caller(metadata)
                or metadata.st_nlink != 1):
            raise TLSBootstrapError("TLS bootstrap lock identity is unsafe")
        fs.chmod_private(descriptor, 0o600)
        fs.lock_exclusive(descriptor)
        if not root.exists():
            _initial_material(
                state, root, desired_hosts,
                certificate_days=certificate_days)
        _secure_directory(root)
        manifest, generation, actual_hosts, server_current = _validate_active(
            root, renew_before_seconds=renew_before_seconds)
        if actual_hosts != desired_hosts or not server_current:
            ca_key = root / "ca-key.pem"
            ca_certificate = root / "friday-local-ca.crt"
            generation = _issue_server(
                root, ca_key, ca_certificate, desired_hosts,
                days=certificate_days)
            _publish_generation(
                root, ca_certificate, generation, desired_hosts)
            manifest, generation, actual_hosts, server_current = (
                _validate_active(
                    root, renew_before_seconds=renew_before_seconds))
        if actual_hosts != desired_hosts or not server_current:
            raise TLSBootstrapError("TLS server identity is not current")
        return TLSMaterial(
            certfile=generation / "server-cert.pem",
            keyfile=generation / "server-key.pem",
            cafile=root / "friday-local-ca.crt",
            hosts=actual_hosts,
            generation=generation.name,
            certificate_sha256=str(manifest["server_certificate_sha256"]),
            # Pair controllers to the stable local trust anchor, not the
            # routinely renewed leaf certificate.
            transport_binding_sha256=str(
                manifest["ca_certificate_sha256"]),
        )
    finally:
        try:
            fs.unlock(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "TLSBootstrapError", "TLSMaterial", "ensure_tls_material",
    "normalize_tls_hosts",
]
