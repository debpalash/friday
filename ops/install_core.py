#!/usr/bin/env python3
"""Portable Friday install transaction (macOS today, Windows next).

``install.sh`` (and later ``install.ps1``) only preflight the host, bootstrap
uv, obtain the source tree, and hand over to this module. It performs the
same transaction as the Linux bash body: a fresh release directory, a hashed
Python environment, pinned assets, the local model engine, private
configuration, the service registration, an atomic ``current`` switch, a
doctor pass, and full rollback on any failure. It depends only on the
standard library and ``friday_host``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from friday_host import fs, paths  # noqa: E402
from friday_host.host import HostPlatform, detect_host  # noqa: E402
from friday_host.service import ServiceSpec, backend_for  # noqa: E402

COPY_IGNORE = {"venv", "models", "state", ".git", "server.log", "session.json",
               "__pycache__", "node_modules"}
SHARED_LINKS = ("models", "skills", "capabilities", "backups")
SMOKE_IMPORT = ("import cryptography, fastapi, numpy, onnxruntime, openai, pydantic, "
                "sherpa_onnx, uvicorn, websockets\nfrom friday_core import GraphStore\n"
                "print('app runtime imports verified')\n")


class InstallError(RuntimeError):
    pass


def step(label: str, message: str) -> None:
    print(f"\n  {label:<14} {message}", flush=True)


def _validate_root(value: Path, label: str, home: Path) -> Path:
    resolved = Path(os.path.abspath(os.path.expanduser(str(value))))
    if (resolved == Path(resolved.anchor) or resolved == home
            or resolved == home.parent or not resolved.is_absolute()):
        raise InstallError(f"unsafe {label}: {resolved}")
    text = str(resolved)
    if "\n" in text or "'" in text:
        raise InstallError(f"{label} contains unsupported characters")
    if resolved.is_symlink():
        raise InstallError(f"{label} must not be a symlink: {resolved}")
    return resolved


class Tee:
    def __init__(self, path: Path) -> None:
        self._file = path.open("a", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text: str) -> int:
        self._file.write(text)
        self._file.flush()
        return self._stdout.write(text)

    def flush(self) -> None:
        self._file.flush()
        self._stdout.flush()

    def fileno(self) -> int:
        return self._stdout.fileno()


class Installer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        environment = dict(os.environ)
        self.host: HostPlatform = detect_host(
            environment=environment, platform_name=args.host_os or None,
            machine=args.host_arch or None)
        if self.host.is_linux:
            raise InstallError("Linux installs use the bash body of install.sh")
        actual = detect_host(environment=environment)
        if actual.os != self.host.os and environment.get("FRIDAY_INSTALL_REHEARSAL") != "1":
            raise InstallError(
                f"host override {self.host.os} does not match this {actual.os} machine; "
                "set FRIDAY_INSTALL_REHEARSAL=1 for a fake-host rehearsal")
        self.home = Path.home()
        self.install_root = _validate_root(
            args.root or paths.default_install_root(environment, self.host),
            "install root", self.home)
        self.state_root = _validate_root(
            args.state_root or paths.default_state_root(environment, self.host),
            "state root", self.home)
        self.config_root = _validate_root(
            args.config_root or paths.default_config_root(environment, self.host),
            "config root", self.home)
        self.cache_root = _validate_root(
            args.cache_root or paths.default_cache_root(environment, self.host),
            "cache root", self.home)
        self.log_root = _validate_root(
            args.log_root or paths.default_log_root(environment, self.host),
            "log root", self.home)
        self.bin_root = _validate_root(paths.default_bin_root(environment, self.host),
                                       "binary root", self.home)
        self.source_dir = _validate_root(args.source_dir, "source", self.home)
        self.uv = Path(args.uv).resolve()
        self.owner = args.owner or (os.environ.get("USER") or os.environ.get("USERNAME")
                                    or "owner")
        self.runtime_root = self.install_root / "runtime"
        self.shared = self.install_root / "shared"
        self.model_root = self.shared / "models"
        self.env_file = self.config_root / "friday.env"
        self.cli_file = self.bin_root / "friday"
        self.current = self.install_root / "current"
        self.release_dir: Path | None = None
        self.previous_target: Path | None = None
        self.switched = False
        self.rollback_dir: Path | None = None
        self.backend = backend_for(self.host, home=self.home)
        self.service_was_active = False
        self.service_was_enabled = False
        self.lock = None
        self.engine = "unsupported"
        self.model_asset = ""

    # ---- preflight
    def preflight(self) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            raise InstallError("run the installer as your desktop user, not root")
        if not self.host.is_macos:
            raise InstallError(f"this installer does not support {self.host.os} yet")
        if self.host.arch != "aarch64":
            raise InstallError("macOS installs need Apple Silicon; Intel Macs are not "
                               "supported by this release")
        for name in ("curl", "tar", "shasum", "launchctl"):
            if shutil.which(name) is None:
                raise InstallError(f"{name} is required")
        if not self.uv.is_file():
            raise InstallError(f"uv is missing: {self.uv}")
        if not all((self.source_dir / item).is_file()
                   for item in ("server.py", "supervisor.py", "frontend/index.html",
                                "ops/install_core.py")):
            raise InstallError(f"source is not a Friday checkout: {self.source_dir}")
        if not (1 <= len(self.owner) <= 64
                and all(c.isalnum() or c in " ._-" for c in self.owner)):
            raise InstallError("owner name must be 1-64 letters, numbers, spaces, '.', "
                               "'_' or '-'")
        if not self.args.skip_hardware_check:
            memory = self._physical_memory_gib()
            if memory < 16:
                raise InstallError(
                    f"Friday needs at least 16 GiB of unified memory; found {memory:.0f} GiB")
        self.install_root.mkdir(parents=True, exist_ok=True)
        needed = 8 if self.args.skip_assets else 30
        free_gib = shutil.disk_usage(self.install_root).free / 1024 ** 3
        if free_gib < needed:
            raise InstallError(f"insufficient free disk: need at least {needed} GiB")

    def _physical_memory_gib(self) -> float:
        from friday_host.procs import physical_memory_bytes  # noqa: PLC0415

        value = physical_memory_bytes()
        return (value or 0) / 1024 ** 3

    # ---- rollback support
    def _snapshot(self, label: str, target: Path) -> None:
        assert self.rollback_dir is not None
        if target.is_dir() and not target.is_symlink():
            raise InstallError(f"installer-managed file is unexpectedly a directory: {target}")
        if target.exists() or target.is_symlink():
            shutil.copy2(target, self.rollback_dir / label, follow_symlinks=False)
            (self.rollback_dir / f"{label}.present").touch()

    def _restore(self, label: str, target: Path) -> None:
        assert self.rollback_dir is not None
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        if (self.rollback_dir / f"{label}.present").is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.rollback_dir / label, target, follow_symlinks=False)

    def managed_files(self) -> dict[str, Path]:
        files = {"environment": self.env_file, "cli": self.cli_file}
        for index, unit in enumerate(self.backend.unit_paths()):
            files[f"service{index}"] = unit
        return files

    def rollback(self) -> None:
        print("\n  rollback       restoring the previous Friday release", file=sys.stderr)
        try:
            self.backend.stop()
        except Exception:
            pass
        if self.switched:
            try:
                self.current.unlink()
            except FileNotFoundError:
                pass
            if self.previous_target is not None and self.previous_target.is_dir():
                self._switch_current(self.previous_target)
        if self.release_dir is not None and self.release_dir.is_dir():
            shutil.rmtree(self.release_dir, ignore_errors=True)
        if self.rollback_dir is not None:
            for label, target in self.managed_files().items():
                try:
                    self._restore(label, target)
                except OSError:
                    pass
        for action in ((self.backend.enable if self.service_was_enabled else self.backend.disable),
                       (self.backend.start if self.service_was_active else self.backend.stop)):
            try:
                action()
            except Exception:
                pass

    # ---- transaction pieces
    def _switch_current(self, target: Path) -> None:
        temporary = self.install_root / f".current-{os.getpid()}"
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        os.symlink(target, temporary)
        os.replace(temporary, self.current)
        fs.fsync_directory(self.install_root)

    def _source_revision(self) -> str:
        if self.args.source_revision:
            return self.args.source_revision
        if shutil.which("git") and (self.source_dir / ".git").exists():
            result = subprocess.run(["git", "-C", str(self.source_dir), "rev-parse", "HEAD"],
                                    text=True, capture_output=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        return "local"

    def _populate_release(self, revision: str) -> None:
        assert self.release_dir is not None
        if (shutil.which("git") and (self.source_dir / ".git").exists()
                and revision not in {"local"}):
            with subprocess.Popen(["git", "-C", str(self.source_dir), "archive",
                                   "--format=tar", "HEAD"], stdout=subprocess.PIPE) as git:
                subprocess.run(["tar", "-xf", "-", "-C", str(self.release_dir)],
                               stdin=git.stdout, check=True, timeout=600)
            if git.returncode != 0:
                raise InstallError("git archive failed")
        else:
            shutil.copytree(self.source_dir, self.release_dir, symlinks=False,
                            ignore=lambda _d, names: [n for n in names if n in COPY_IGNORE],
                            dirs_exist_ok=True)
        for relative in ("install.sh", "ops/fridayctl", "ops/provision_qwen_runtime.sh",
                         "scripts/uninstall.sh"):
            path = self.release_dir / relative
            if path.is_file():
                path.chmod(0o755)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (self.release_dir / "FRIDAY_RELEASE").write_text(
            f"revision={revision}\ninstalled_at={stamp}\n", encoding="utf-8")

    @staticmethod
    def _seed(source: Path, target: Path) -> None:
        if not source.is_dir():
            return
        target.mkdir(parents=True, exist_ok=True)
        for item in source.rglob("*"):
            relative = item.relative_to(source)
            destination = target / relative
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)

    def _link_shared(self) -> None:
        assert self.release_dir is not None
        for name in (*SHARED_LINKS, "state", "persona/voices"):
            shutil.rmtree(self.release_dir / name, ignore_errors=True)
            try:
                (self.release_dir / name).unlink()
            except FileNotFoundError:
                pass
        (self.release_dir / "persona").mkdir(exist_ok=True)
        targets = {"models": self.model_root, "skills": self.shared / "skills",
                   "capabilities": self.shared / "capabilities",
                   "backups": self.shared / "backups", "state": self.state_root,
                   "persona/voices": self.shared / "persona" / "voices"}
        for name, target in targets.items():
            os.symlink(target, self.release_dir / name)
        for name in ("session.json", "server.log"):
            try:
                (self.release_dir / name).unlink()
            except FileNotFoundError:
                pass
        (self.state_root / "logs").mkdir(parents=True, exist_ok=True)
        for path in (self.state_root / "session.json", self.state_root / "logs" / "server.log"):
            path.touch()
            path.chmod(0o600)

    def _python(self) -> Path:
        assert self.release_dir is not None
        return paths.venv_python(self.release_dir, self.host)

    def _run(self, command: list[str], *, cwd: Path | None = None,
             env: dict[str, str] | None = None, timeout: int = 3600) -> None:
        result = subprocess.run(command, cwd=cwd, env=env, timeout=timeout)
        if result.returncode != 0:
            raise InstallError(f"{Path(command[0]).name} failed with status {result.returncode}")

    def _build_environment(self) -> None:
        assert self.release_dir is not None
        step("environment", "creating a pinned Python 3.12 runtime")
        venv = self.release_dir / "venv"
        self._run([str(self.uv), "venv", "--python", "3.12", str(venv)])
        lock = self.release_dir / "requirements" / f"runtime-{self.host.lock_id}.lock"
        if not lock.is_file():
            raise InstallError(f"{lock.name} is missing")
        self._run([str(self.uv), "pip", "sync", "--python", str(self._python()),
                   "--require-hashes", str(lock)])
        self._run([str(self._python()), "-c", SMOKE_IMPORT], cwd=self.release_dir)

    def _install_assets(self) -> None:
        assert self.release_dir is not None
        step("assets", "verifying pinned ASR, speech, and embedding models")
        python = str(self._python())
        ops = self.release_dir / "ops"
        self._run([python, str(ops / "install_asr_model.py"), "--model-root",
                   str(self.model_root), "--cache-root", str(self.cache_root / "downloads")])
        self._run([python, str(ops / "install_piper_voice.py")], cwd=self.release_dir)
        self._run([python, str(ops / "install_embedding_model.py")], cwd=self.release_dir)
        self._run([python, str(ops / "install_vad_model.py")], cwd=self.release_dir)

    def _select_engine(self) -> tuple[str, str]:
        assert self.release_dir is not None
        script = ("import json, os, sys\n"
                  "from friday_core.hardware import detect_hardware, select_runtime_profile\n"
                  "profile = select_runtime_profile(detect_hardware(), environment=os.environ)\n"
                  "print(json.dumps({'engine': profile.engine, 'asset': profile.model_asset,"
                  " 'backend': profile.engine_backend, 'tier': profile.tier}))\n")
        env = dict(os.environ)
        env["FRIDAY_LLM_ENGINE"] = self.args.engine
        result = subprocess.run([str(self._python()), "-c", script], cwd=self.release_dir,
                                env=env, text=True, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise InstallError(f"runtime profile selection failed: {result.stderr.strip()[-400:]}")
        import json  # noqa: PLC0415

        chosen = json.loads(result.stdout.strip().splitlines()[-1])
        if chosen["engine"] not in {"mlx_lm", "llama_server"}:
            raise InstallError(f"no supported local engine for this Mac: {chosen['tier']}")
        return chosen["engine"], chosen["asset"]

    def _install_engine(self) -> None:
        assert self.release_dir is not None
        self.engine, self.model_asset = self._select_engine()
        python = str(self._python())
        ops = self.release_dir / "ops"
        if self.engine == "mlx_lm":
            step("engine", "installing the pinned MLX runtime")
            self._run([python, str(ops / "install_mlx_runtime.py"), "--runtime-root",
                       str(self.runtime_root), "--uv", str(self.uv)], cwd=self.release_dir)
        else:
            step("engine", "installing the pinned llama-server build")
            self._run([python, str(ops / "install_llama_server.py"), "--runtime-root",
                       str(self.runtime_root), "--cache-root",
                       str(self.cache_root / "downloads")], cwd=self.release_dir)
        step("model", f"verifying the pinned Qwen3 checkpoint ({self.model_asset})")
        self._run([python, str(ops / "install_local_model.py"), "--asset", self.model_asset,
                   "--model-root", str(self.model_root)], cwd=self.release_dir)

    def _write_config(self) -> None:
        existing: dict[str, str] = {}
        if self.env_file.is_file():
            from friday_host.envfile import read_env_file  # noqa: PLC0415

            existing = read_env_file(self.env_file)
        engine = self.engine if self.engine != "unsupported" else (
            existing.get("FRIDAY_LLM_ENGINE") or self.args.engine)
        values = {
            "FRIDAY_INSTALL_ROOT": str(self.install_root),
            "FRIDAY_CONFIG_ROOT": str(self.config_root),
            "FRIDAY_STATE_DIR": str(self.state_root),
            "FRIDAY_LLM_REPO": str(self.runtime_root / "qwen"),
            "FRIDAY_RUNTIME_ROOT": str(self.runtime_root),
            "FRIDAY_LLM_ENGINE": engine,
            "FRIDAY_LOCAL_API_KEY_FILE": str(self.state_root / "local-api-key"),
            "FRIDAY_OWNER_NAME": self.owner,
            "FRIDAY_BIND_HOST": "127.0.0.1",
            "FRIDAY_PORT": existing.get("FRIDAY_PORT", "8500"),
            "FRIDAY_DESKTOP_MODE": "off",
        }
        from friday_host.envfile import render_env_text  # noqa: PLC0415

        self.config_root.mkdir(parents=True, exist_ok=True)
        def _opener(path: str, flags: int) -> int:
            return os.open(path, flags, 0o600)

        with open(self.env_file, "w", encoding="utf-8", opener=_opener) as handle:
            handle.write(render_env_text(values))
        self.env_file.chmod(0o600)

    def _write_cli(self) -> None:
        self.bin_root.mkdir(parents=True, exist_ok=True)
        body = ("#!/usr/bin/env bash\n"
                f"export FRIDAY_INSTALL_ROOT='{self.install_root}'\n"
                f"export FRIDAY_CONFIG_ROOT='{self.config_root}'\n"
                f"exec '{self.current}/venv/bin/python' '{self.current}/ops/fridayctl.py' \"$@\"\n")
        self.cli_file.write_text(body, encoding="utf-8")
        self.cli_file.chmod(0o755)

    def _register_service(self) -> None:
        assert self.release_dir is not None
        spec = ServiceSpec(
            current_dir=self.current, env_file=self.env_file,
            python=self.current / "venv" / "bin" / "python",
            supervisor=self.current / "supervisor.py",
            launcher=self.current / "ops" / "friday_launch.py", log_dir=self.log_root)
        self.backend.install(spec, self.release_dir / "ops" / "friday.launchd.plist.in")

    def _doctor(self, *, expect_running: bool) -> None:
        assert self.release_dir is not None
        from friday_host.envfile import read_env_file  # noqa: PLC0415

        env = dict(os.environ)
        env.update(read_env_file(self.env_file))
        command = [str(self._python()), str(self.release_dir / "ops" / "friday_doctor.py")]
        if expect_running:
            command.append("--expect-running")
        self._run(command, cwd=self.release_dir, env=env, timeout=900)

    # ---- driver
    def run(self) -> int:
        self.preflight()
        os.umask(0o077)
        for directory in (self.install_root, self.state_root, self.config_root,
                          self.cache_root, self.log_root, self.bin_root,
                          self.install_root / "releases"):
            directory.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.install_root / ".install.lock", "a+")  # noqa: SIM115
        try:
            fs.lock_exclusive(self.lock.fileno(), blocking=False)
        except OSError as exc:
            raise InstallError("another Friday install/update is already running") from exc
        sys.stdout = Tee(self.install_root / "install.log")  # type: ignore[assignment]

        if self.current.is_symlink():
            self.previous_target = Path(os.path.realpath(self.current))
        elif self.current.exists():
            raise InstallError(f"{self.current} exists but is not an installer-owned symlink")

        self.rollback_dir = self.install_root / f".rollback-{os.getpid()}"
        self.rollback_dir.mkdir()
        for label, target in self.managed_files().items():
            self._snapshot(label, target)

        print("\n  Friday Installer\n  " + "─" * 52)
        step("platform", f"{self.host.os} {self.host.arch} / {self.backend.kind}")
        step("privacy", "loopback-only UI; private login agent; no cloud dependency")
        step("install", str(self.install_root))
        self.service_was_active = self.backend.is_active()
        self.service_was_enabled = self.backend.is_enabled()

        try:
            self._transaction()
        except BaseException:
            self.rollback()
            raise
        finally:
            if self.rollback_dir is not None:
                shutil.rmtree(self.rollback_dir, ignore_errors=True)
        return 0

    def _transaction(self) -> None:
        self.backend.stop()
        if self.backend.is_active():
            raise InstallError("could not stop the existing Friday service safely")
        revision = self._source_revision()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.release_dir = self.install_root / "releases" / f"{revision[:12]}-{stamp}-{os.getpid()}"
        if self.release_dir.exists():
            raise InstallError(f"release already exists: {self.release_dir}")
        if (self.previous_target is not None and not self.args.repair
                and (self.previous_target / "FRIDAY_RELEASE").is_file()
                and revision != "local"
                and f"revision={revision}\n" in (self.previous_target / "FRIDAY_RELEASE").read_text()):
            step("source", f"revision {revision[:12]} is already installed; use --repair to reinstall")
            self.release_dir = None
            if self.args.start:
                self.backend.start()
            return
        self.release_dir.mkdir()
        step("source", f"local revision {revision[:12]}")
        self._populate_release(revision)

        for directory in (self.shared, self.model_root, self.shared / "skills",
                          self.shared / "capabilities", self.shared / "backups",
                          self.shared / "persona" / "voices", self.state_root / "logs"):
            directory.mkdir(parents=True, exist_ok=True)
        for name in ("skills", "persona/voices"):
            self._seed(self.source_dir / name, self.shared / name)
            self._seed(self.release_dir / name, self.shared / name)
        self._seed(self.source_dir / "models", self.model_root)
        self._seed(self.source_dir / "capabilities", self.shared / "capabilities")
        self._seed(self.source_dir / "backups", self.shared / "backups")
        if not (self.state_root / "friday.db").exists() and (self.source_dir / "state").is_dir():
            self._seed(self.source_dir / "state", self.state_root)
            for stale in list(self.state_root.glob("*.pid")) + list(self.state_root.glob("*.lock")):
                stale.unlink()
        self._link_shared()

        self._build_environment()
        if not self.args.skip_assets:
            self._install_assets()
            self._install_engine()
        else:
            step("assets", "skipped (--skip-assets); the doctor will report what is missing")

        step("config", "writing private per-user configuration")
        self._write_config()
        self._write_cli()
        self._register_service()
        self._switch_current(self.release_dir)
        self.switched = True
        self._doctor(expect_running=False)

        if self.args.start:
            step("launch", "starting Friday and loading the local model")
            self.backend.restart()
            self._run([str(self.cli_file), "start"], timeout=900)
            self._doctor(expect_running=True)
        else:
            step("launch", "installed; startup deferred (--no-start)")

        if self.previous_target is not None and self.previous_target != self.release_dir:
            (self.install_root / "previous-release").write_text(f"{self.previous_target}\n")
        print("\n  " + "─" * 52)
        print("  Friday installed\n  launch         friday open\n  status         friday status\n"
              "  diagnostics    friday doctor\n  stop model     friday stop\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--uv", required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--state-root", type=Path, default=None)
    parser.add_argument("--config-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--llm-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--owner", default="")
    parser.add_argument("--engine", default="auto", choices=("auto", "mlx_lm", "llama_server"))
    parser.add_argument("--build-venv", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--skip-hardware-check", action="store_true")
    parser.add_argument("--no-start", dest="start", action="store_false")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--host-os", default="", help=argparse.SUPPRESS)
    parser.add_argument("--host-arch", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return Installer(parse_args(argv)).run()
    except InstallError as exc:
        print(f"\nFriday install failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
