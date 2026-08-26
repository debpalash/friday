"""Private local-CA bootstrap for Friday's HTTPS control plane."""

from __future__ import annotations

import fcntl
import hashlib
import ipaddress
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022):
        raise TLSBootstrapError("TLS state directory identity is unsafe")
    os.chmod(path, 0o700)


def _secure_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TLSBootstrapError(f"TLS material is unavailable: {path.name}") from exc
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1):
        raise TLSBootstrapError(f"TLS material identity is unsafe: {path.name}")
    os.chmod(path, 0o600)


def _run(openssl_path: str, arguments: list[str], *,
         input_bytes: bytes | None = None) -> bytes:
    executable = Path(openssl_path)
    if not executable.is_absolute() or not executable.is_file():
        raise TLSBootstrapError("OpenSSL executable is unavailable")
    try:
        options = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "check": False,
            "timeout": 15,
        }
        if input_bytes is None:
            options["stdin"] = subprocess.DEVNULL
        else:
            options["input"] = input_bytes
        result = subprocess.run([str(executable), *arguments], **options)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TLSBootstrapError("TLS certificate operation failed") from exc
    if result.returncode != 0:
        raise TLSBootstrapError("TLS certificate operation failed")
    return result.stdout


def _write_private(path: Path, value: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
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
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _public_key_digest(openssl_path: str, *, key: Path | None = None,
                       certificate: Path | None = None) -> str:
    if (key is None) == (certificate is None):
        raise ValueError("provide exactly one TLS identity")
    if key is not None:
        der = _run(openssl_path, [
            "pkey", "-in", str(key), "-pubout", "-outform", "DER",
        ])
    else:
        assert certificate is not None
        public_pem = _run(openssl_path, [
            "x509", "-in", str(certificate), "-pubkey", "-noout",
        ])
        der = _run(
            openssl_path, ["pkey", "-pubin", "-outform", "DER"],
            input_bytes=public_pem)
    return hashlib.sha256(der).hexdigest()


def _certificate_hosts(openssl_path: str, certificate: Path) -> tuple[str, ...]:
    output = _run(openssl_path, [
        "x509", "-in", str(certificate), "-noout", "-ext",
        "subjectAltName",
    ]).decode("utf-8", errors="strict")
    values: set[str] = set()
    for item in output.replace("\n", " ").split(","):
        token = item.strip()
        if "DNS:" in token:
            values.add(token.rsplit("DNS:", 1)[1].strip().lower())
        elif "IP Address:" in token:
            raw = token.rsplit("IP Address:", 1)[1].strip()
            try:
                values.add(ipaddress.ip_address(raw).compressed)
            except ValueError as exc:
                raise TLSBootstrapError(
                    "TLS certificate contains an invalid IP SAN") from exc
    if not values:
        raise TLSBootstrapError("TLS certificate has no subjectAltName")
    return tuple(sorted(values))


def _generate_ca(openssl_path: str, root: Path) -> tuple[Path, Path]:
    key = root / "ca-key.pem"
    certificate = root / "friday-local-ca.crt"
    _run(openssl_path, [
        "genpkey", "-algorithm", "EC", "-pkeyopt",
        "ec_paramgen_curve:P-256", "-out", str(key),
    ])
    _run(openssl_path, [
        "req", "-x509", "-new", "-sha256", "-key", str(key),
        "-out", str(certificate), "-days", "3650", "-subj",
        "/CN=Friday Local Controller CA", "-addext",
        "basicConstraints=critical,CA:TRUE", "-addext",
        "keyUsage=critical,keyCertSign,cRLSign", "-addext",
        "subjectKeyIdentifier=hash",
    ])
    for path in (key, certificate):
        _secure_file(path)
    return key, certificate


def _issue_server(
    openssl_path: str, root: Path, ca_key: Path, ca_certificate: Path,
    hosts: tuple[str, ...], *, days: int,
) -> Path:
    generation = root / f"generation_{secrets.token_hex(16)}"
    generation.mkdir(mode=0o700)
    key = generation / "server-key.pem"
    certificate = generation / "server-cert.pem"
    request = generation / "server.csr"
    extensions = generation / "server-extensions.cnf"
    san_items = []
    for host in hosts:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            san_items.append(f"DNS:{host}")
        else:
            san_items.append(f"IP:{host}")
    _write_private(extensions, (
        "[server_ext]\n"
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        f"subjectAltName={','.join(san_items)}\n"
        "subjectKeyIdentifier=hash\n"
        "authorityKeyIdentifier=keyid,issuer\n"
    ).encode())
    try:
        _run(openssl_path, [
            "genpkey", "-algorithm", "EC", "-pkeyopt",
            "ec_paramgen_curve:P-256", "-out", str(key),
        ])
        _run(openssl_path, [
            "req", "-new", "-sha256", "-key", str(key), "-out",
            str(request), "-subj", "/CN=Friday Local Assistant",
        ])
        _run(openssl_path, [
            "x509", "-req", "-sha256", "-in", str(request), "-CA",
            str(ca_certificate), "-CAkey", str(ca_key), "-set_serial",
            "0x" + secrets.token_hex(16), "-out", str(certificate),
            "-days", str(days), "-extfile", str(extensions), "-extensions",
            "server_ext",
        ])
    except Exception:
        shutil.rmtree(generation, ignore_errors=True)
        raise
    finally:
        for transient in (request, extensions):
            try:
                transient.unlink()
            except FileNotFoundError:
                pass
    for path in (key, certificate):
        _secure_file(path)
    os.chmod(generation, 0o700)
    return generation


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
    root: Path, openssl_path: str, *, renew_before_seconds: int,
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
    if (_public_key_digest(openssl_path, key=ca_key)
            != _public_key_digest(
                openssl_path, certificate=ca_certificate)):
        raise TLSBootstrapError("TLS CA key does not match its certificate")
    if (_public_key_digest(openssl_path, key=key)
            != _public_key_digest(openssl_path, certificate=certificate)):
        raise TLSBootstrapError("TLS server key does not match its certificate")
    _run(openssl_path, [
        "verify", "-CAfile", str(ca_certificate), str(certificate),
    ])
    actual_hosts = _certificate_hosts(openssl_path, certificate)
    if actual_hosts != recorded_hosts:
        raise TLSBootstrapError("TLS certificate SANs changed")
    ca_current = subprocess.run(
        [openssl_path, "x509", "-in", str(ca_certificate), "-noout",
         "-checkend", str(renew_before_seconds)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=15, check=False,
    ).returncode == 0
    if not ca_current:
        raise TLSBootstrapError(
            "Friday local CA is expiring; explicit trust-anchor rotation is required")
    server_current = subprocess.run(
        [openssl_path, "x509", "-in", str(certificate), "-noout",
         "-checkend", str(renew_before_seconds)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=15, check=False,
    ).returncode == 0
    return manifest, generation, actual_hosts, server_current


def _initial_material(
    state_root: Path, root: Path, openssl_path: str,
    hosts: tuple[str, ...], *, certificate_days: int,
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=".tls-stage-", dir=state_root))
    os.chmod(staging, 0o700)
    try:
        ca_key, ca_certificate = _generate_ca(openssl_path, staging)
        generation = _issue_server(
            openssl_path, staging, ca_key, ca_certificate, hosts,
            days=certificate_days)
        _publish_generation(staging, ca_certificate, generation, hosts)
        os.rename(staging, root)
        directory = os.open(
            state_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
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
    """
    if not 30 <= certificate_days <= 825:
        raise ValueError("TLS certificate lifetime is invalid")
    if not 86_400 <= renew_before_seconds <= certificate_days * 86_400 // 2:
        raise ValueError("TLS renewal window is invalid")
    desired_hosts = normalize_tls_hosts(hosts)
    state = Path(state_root)
    _secure_directory(state, create=True)
    root = state / "tls"
    lock_path = state / "tls-bootstrap.lock"
    flags = (os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
             | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise TLSBootstrapError("TLS bootstrap lock is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1):
            raise TLSBootstrapError("TLS bootstrap lock identity is unsafe")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if not root.exists():
            _initial_material(
                state, root, openssl_path, desired_hosts,
                certificate_days=certificate_days)
        _secure_directory(root)
        manifest, generation, actual_hosts, server_current = _validate_active(
            root, openssl_path,
            renew_before_seconds=renew_before_seconds)
        if actual_hosts != desired_hosts or not server_current:
            ca_key = root / "ca-key.pem"
            ca_certificate = root / "friday-local-ca.crt"
            generation = _issue_server(
                openssl_path, root, ca_key, ca_certificate, desired_hosts,
                days=certificate_days)
            _publish_generation(
                root, ca_certificate, generation, desired_hosts)
            manifest, generation, actual_hosts, server_current = (
                _validate_active(
                    root, openssl_path,
                    renew_before_seconds=renew_before_seconds))
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
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "TLSBootstrapError", "TLSMaterial", "ensure_tls_material",
    "normalize_tls_hosts",
]
