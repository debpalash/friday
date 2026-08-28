#!/usr/bin/env bash
# Friday installer for Linux/x86_64. The whole implementation is parsed before
# execution so a streamed invocation cannot strand a half-read script.

friday_install() {
  set -Eeuo pipefail

  local repository="${FRIDAY_REPOSITORY:-debpalash/friday}"
  local install_ref="${FRIDAY_INSTALL_REF:-main}"
  local source_sha256="${FRIDAY_SOURCE_SHA256:-}"
  local source_dir=""
  local source_archive=""
  install_root="${FRIDAY_INSTALL_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/friday}"
  local state_root="${FRIDAY_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/friday}"
  local config_root="${FRIDAY_CONFIG_ROOT:-${XDG_CONFIG_HOME:-$HOME/.config}/friday}"
  local cache_root="${FRIDAY_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/friday}"
  local bin_root="${XDG_BIN_HOME:-$HOME/.local/bin}"
  local data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
  local llm_root="${FRIDAY_LLM_REPO:-}"
  local owner_name="${FRIDAY_OWNER_NAME:-$(id -un)}"
  local start_after=1
  local seed_venv=1
  local skip_assets=0
  local skip_hardware=0
  local repair=0
  release_dir=""
  previous_target=""
  switched=0
  service_was_active=0
  service_was_enabled=0
  legacy_was_active=0
  legacy_was_enabled=0
  rollback_dir=""

  usage() {
    cat <<'EOF'
Install Friday as a private local desktop assistant.

Usage:
  Download and verify a versioned installer from GitHub Releases, then run it.
  Development checkout: ./install.sh --local . --build-venv
  ./install.sh --local PATH --llm-root PATH

Options:
  --local PATH          Install an exact local Git checkout
  --archive PATH        Install a release-style source tarball from disk
  --source-sha256 HASH  Required SHA-256 for --archive; optional for --ref
  --ref REF             Install a GitHub branch, tag, or commit (default: main)
  --root PATH           App/runtime root (default: ~/.local/share/friday)
  --state-root PATH     Personal state root (default: ~/.local/state/friday)
  --config-root PATH    Private configuration root (default: ~/.config/friday)
  --llm-root PATH       Reuse an existing verified Friday Qwen runtime
  --owner NAME          Owner identity used by Friday
  --build-venv          Build a fresh app environment instead of reflink-copying
                        a local checkout's existing venv
  --skip-assets         Do not install ASR, voice, or embedding assets
  --no-start            Install and enable Friday without starting it
  --repair              Reinstall even when the same source revision is present
  -h, --help            Show this help

Public install requirements: Linux x86_64, systemd user services, an NVIDIA
GPU with at least 22 GiB VRAM, and about 50 GiB free disk for the default model.
EOF
  }

  fail() { printf '\nFriday install failed: %s\n' "$*" >&2; exit 1; }
  step() { printf '\n  %-14s %s\n' "$1" "$2"; }

  while (($#)); do
    case "$1" in
      --local) [[ $# -ge 2 ]] || fail "--local needs a path"; source_dir="$2"; shift 2 ;;
      --archive) [[ $# -ge 2 ]] || fail "--archive needs a path"; source_archive="$2"; shift 2 ;;
      --source-sha256) [[ $# -ge 2 ]] || fail "--source-sha256 needs a value"; source_sha256="$2"; shift 2 ;;
      --ref) [[ $# -ge 2 ]] || fail "--ref needs a value"; install_ref="$2"; shift 2 ;;
      --root) [[ $# -ge 2 ]] || fail "--root needs a path"; install_root="$2"; shift 2 ;;
      --state-root) [[ $# -ge 2 ]] || fail "--state-root needs a path"; state_root="$2"; shift 2 ;;
      --config-root) [[ $# -ge 2 ]] || fail "--config-root needs a path"; config_root="$2"; shift 2 ;;
      --llm-root) [[ $# -ge 2 ]] || fail "--llm-root needs a path"; llm_root="$2"; shift 2 ;;
      --owner) [[ $# -ge 2 ]] || fail "--owner needs a value"; owner_name="$2"; shift 2 ;;
      --build-venv) seed_venv=0; shift ;;
      --skip-assets) skip_assets=1; shift ;;
      --skip-hardware-check) skip_hardware=1; shift ;;
      --no-start) start_after=0; shift ;;
      --repair) repair=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) fail "unknown argument: $1" ;;
    esac
  done

  [[ -z "$source_dir" || -z "$source_archive" ]] \
    || fail "--local and --archive are mutually exclusive"
  if [[ -n "$source_sha256" && ! "$source_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    fail "source SHA-256 must be 64 lowercase hexadecimal characters"
  fi
  [[ -z "$source_archive" || -n "$source_sha256" ]] \
    || fail "--archive requires --source-sha256"

  command -v realpath >/dev/null || fail "realpath is required"
  install_root="$(realpath -m "$install_root")"
  state_root="$(realpath -m "$state_root")"
  config_root="$(realpath -m "$config_root")"
  cache_root="$(realpath -m "$cache_root")"
  bin_root="$(realpath -m "$bin_root")"
  data_home="$(realpath -m "$data_home")"
  [[ -z "$source_dir" ]] || source_dir="$(realpath -m "$source_dir")"
  [[ -z "$source_archive" ]] || source_archive="$(realpath -m "$source_archive")"
  [[ -z "$llm_root" ]] || llm_root="$(realpath -m "$llm_root")"

  validate_root() {
    local value="$1" label="$2"
    [[ "$value" == /* && "$value" != / && "$value" != "$HOME" \
        && "$value" != "$(dirname "$HOME")" ]] || fail "unsafe $label: $value"
    [[ "$value" != *$'\n'* && "$value" != *"'"* ]] \
      || fail "$label contains unsupported characters"
    [[ ! -L "$value" ]] || fail "$label must not be a symlink: $value"
  }
  validate_root "$install_root" "install root"
  validate_root "$state_root" "state root"
  validate_root "$config_root" "config root"
  validate_root "$cache_root" "cache root"
  validate_root "$bin_root" "binary root"
  validate_root "$data_home" "data root"
  [[ -z "$llm_root" ]] || validate_root "$llm_root" "Qwen runtime root"
  if [[ -n "$source_archive" ]]; then
    [[ -f "$source_archive" && ! -L "$source_archive" ]] \
      || fail "source archive must be a regular non-symlink file"
    printf '%s  %s\n' "$source_sha256" "$source_archive" \
      | sha256sum -c - >/dev/null \
      || fail "source archive SHA-256 does not match"
  fi
  [[ "$owner_name" =~ ^[[:alnum:]_.[:space:]-]{1,64}$ ]] \
    || fail "owner name must be 1-64 letters, numbers, spaces, '.', '_' or '-'"

  [[ "$(id -u)" -ne 0 ]] || fail "run the installer as your desktop user, not root"
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] \
    || fail "this release supports Linux x86_64 only"
  for command_name in bash bwrap curl tar sha256sum systemctl openssl flock git patch python3; do
    command -v "$command_name" >/dev/null || fail "$command_name is required"
  done
  systemctl --user show-environment >/dev/null 2>&1 \
    || fail "a working systemd user session is required"
  if (( ! skip_hardware )); then
    command -v nvidia-smi >/dev/null || fail "NVIDIA driver tools are required"
    local maximum_vram
    maximum_vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
      | sort -nr | head -n1 | tr -d '[:space:]')"
    [[ "$maximum_vram" =~ ^[0-9]+$ && "$maximum_vram" -ge 22528 ]] \
      || fail "Friday's default 27B runtime needs an NVIDIA GPU with at least 22 GiB VRAM"
  fi
  if [[ -n "$source_dir" ]]; then
    [[ -f "$source_dir/server.py" && -f "$source_dir/supervisor.py" \
        && -f "$source_dir/frontend/index.html" ]] \
      || fail "local source is not a Friday checkout: $source_dir"
  fi

  local required_kib available_kib
  required_kib=$((8 * 1024 * 1024))
  if [[ -z "$llm_root" || ! -x "$llm_root/venv/bin/vllm" ]]; then
    required_kib=$((45 * 1024 * 1024))
  fi
  mkdir -p "$install_root"
  available_kib="$(df -Pk "$install_root" | awk 'NR==2 {print $4}')"
  [[ "$available_kib" =~ ^[0-9]+$ && "$available_kib" -ge "$required_kib" ]] \
    || fail "insufficient free disk: need at least $((required_kib / 1024 / 1024)) GiB"

  umask 077
  mkdir -p "$install_root" "$state_root" "$config_root" "$cache_root" \
    "$bin_root" "$data_home/applications" \
    "$data_home/icons/hicolor/scalable/apps" "$HOME/.config/systemd/user"
  exec 9>"$install_root/.install.lock"
  flock -n 9 || fail "another Friday install/update is already running"
  exec > >(tee -a "$install_root/install.log") 2>&1

  if [[ -L "$install_root/current" ]]; then
    previous_target="$(readlink -f "$install_root/current")"
  elif [[ -e "$install_root/current" ]]; then
    fail "$install_root/current exists but is not an installer-owned symlink"
  fi

  # These remain global because an EXIT trap can run after friday_install's
  # local scope has unwound under `set -e`.
  unit_file="$HOME/.config/systemd/user/friday.service"
  config_file="$config_root/friday.env"
  cli_file="$bin_root/friday"
  desktop_file="$data_home/applications/friday.desktop"
  icon_file="$data_home/icons/hicolor/scalable/apps/friday.svg"
  rollback_dir="$install_root/.rollback-$$"
  mkdir "$rollback_dir"

  snapshot_managed_file() {
    local label="$1" target="$2"
    [[ ! -d "$target" || -L "$target" ]] \
      || fail "installer-managed file is unexpectedly a directory: $target"
    if [[ -e "$target" || -L "$target" ]]; then
      cp -a -- "$target" "$rollback_dir/$label"
      touch "$rollback_dir/$label.present"
    fi
  }
  restore_managed_file() {
    local label="$1" target="$2"
    rm -f -- "$target"
    if [[ -f "$rollback_dir/$label.present" ]]; then
      mkdir -p "$(dirname "$target")"
      cp -a -- "$rollback_dir/$label" "$target"
    fi
  }
  snapshot_managed_file environment "$config_file"
  snapshot_managed_file service "$unit_file"
  snapshot_managed_file cli "$cli_file"
  snapshot_managed_file desktop "$desktop_file"
  snapshot_managed_file icon "$icon_file"

  printf '\n  Friday Installer\n  %s\n' '────────────────────────────────────────────────────'
  step platform "Linux x86_64 / NVIDIA (${maximum_vram:-unprobed} MiB)"
  step privacy "loopback-only UI; private user service; no cloud dependency"
  step install "$install_root"

  systemctl --user is-active --quiet friday.service && service_was_active=1 || true
  systemctl --user is-enabled --quiet friday.service && service_was_enabled=1 || true
  systemctl --user is-active --quiet friday-supervisor.service && legacy_was_active=1 || true
  systemctl --user is-enabled --quiet friday-supervisor.service && legacy_was_enabled=1 || true

  rollback() {
    local status=$?
    trap - EXIT HUP INT TERM
    if (( status != 0 )); then
      # Rollback must keep going even if one cleanup step is racing or damaged.
      set +e
      printf '\n  rollback       restoring the previous Friday release\n' >&2
      # The failed release may still be the service working directory and may
      # recreate state while it is being removed. Quiesce it first.
      systemctl --user stop friday.service >/dev/null 2>&1
      if (( switched )); then
        rm -f -- "$install_root/current"
        if [[ -n "$previous_target" && -d "$previous_target" ]]; then
          ln -s "$previous_target" "$install_root/.current-rollback-$$"
          mv -Tf "$install_root/.current-rollback-$$" "$install_root/current"
        fi
      fi
      [[ -z "$release_dir" || ! -d "$release_dir" ]] || rm -rf -- "$release_dir"
      restore_managed_file environment "$config_file"
      restore_managed_file service "$unit_file"
      restore_managed_file cli "$cli_file"
      restore_managed_file desktop "$desktop_file"
      restore_managed_file icon "$icon_file"
      systemctl --user daemon-reload >/dev/null 2>&1 || true
      if (( service_was_enabled )); then
        systemctl --user enable friday.service >/dev/null 2>&1 || true
      else
        systemctl --user disable friday.service >/dev/null 2>&1 || true
      fi
      if (( service_was_active )); then
        systemctl --user start friday.service >/dev/null 2>&1 || true
      else
        systemctl --user stop friday.service >/dev/null 2>&1 || true
      fi
      if (( legacy_was_enabled )); then
        systemctl --user enable friday-supervisor.service >/dev/null 2>&1 || true
      fi
      (( legacy_was_active )) && systemctl --user start friday-supervisor.service >/dev/null 2>&1 || true
    fi
    [[ -z "$rollback_dir" || ! -d "$rollback_dir" ]] || rm -rf -- "$rollback_dir"
    exit "$status"
  }
  trap rollback EXIT HUP INT TERM

  systemctl --user stop friday.service friday-supervisor.service 2>/dev/null || true
  if systemctl --user is-active --quiet friday.service \
      || systemctl --user is-active --quiet friday-supervisor.service; then
    fail "could not stop the existing Friday service safely"
  fi
  systemctl --user disable friday-supervisor.service 2>/dev/null || true

  local source_revision archive top release_id
  mkdir -p "$install_root/releases"
  if [[ -n "$source_dir" ]]; then
    if git -C "$source_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      source_revision="$(git -C "$source_dir" rev-parse HEAD)"
    else
      source_revision="local"
    fi
  elif [[ -n "$source_archive" ]]; then
    source_revision="archive-${source_sha256:0:12}"
  else
    source_revision="${install_ref//[^A-Za-z0-9._-]/_}"
  fi
  release_id="${source_revision:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  release_dir="$install_root/releases/$release_id"
  [[ ! -e "$release_dir" ]] || fail "release already exists: $release_dir"
  mkdir "$release_dir"

  if [[ -n "$source_dir" ]]; then
    step source "local revision ${source_revision:0:12}"
    if git -C "$source_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      git -C "$source_dir" archive --format=tar HEAD | tar -xf - -C "$release_dir"
    else
      tar --exclude='./venv' --exclude='./models' --exclude='./state' \
        --exclude='./server.log' --exclude='./session.json' \
        -C "$source_dir" -cf - . | tar -xf - -C "$release_dir"
    fi
  else
    if [[ -n "$source_archive" ]]; then
      step source "verified local release archive ${source_sha256:0:12}"
      archive="$cache_root/friday-source-${release_id}.tar.gz"
      cp --reflink=auto -- "$source_archive" "$archive.part"
      chmod 0400 "$archive.part"
      mv "$archive.part" "$archive"
    else
      step source "GitHub $repository@$install_ref"
      archive="$cache_root/friday-source-${release_id}.tar.gz"
      local auth_args=()
      [[ -z "${GH_TOKEN:-}" ]] || auth_args=(-H "Authorization: Bearer $GH_TOKEN")
      curl --fail --location --retry 5 --retry-all-errors \
        "${auth_args[@]}" --output "$archive.part" \
        "https://codeload.github.com/$repository/tar.gz/$install_ref"
      mv "$archive.part" "$archive"
    fi
    if [[ -n "$source_sha256" ]]; then
      printf '%s  %s\n' "$source_sha256" "$archive" | sha256sum -c -
    fi
    top="$(python3 - "$archive" <<'PY'
import pathlib
import sys
import tarfile

archive = pathlib.Path(sys.argv[1])
members = 0
total = 0
root = None
with tarfile.open(archive, "r:gz") as source:
    for item in source:
        members += 1
        if members > 20_000:
            raise SystemExit("source archive has too many members")
        path = pathlib.PurePosixPath(item.name)
        parts = path.parts
        if (not parts or item.name.startswith("/") or ".." in parts
                or any(part in {"", "."} for part in parts)):
            raise SystemExit("unsafe source archive member path")
        if root is None:
            root = parts[0]
        if parts[0] != root:
            raise SystemExit("source archive has multiple top-level roots")
        if not (item.isdir() or item.isreg()):
            raise SystemExit("source archive contains a link or special member")
        if item.size < 0 or item.size > 128 * 1024 * 1024:
            raise SystemExit("source archive member exceeds the size limit")
        total += item.size
        if total > 1024 * 1024 * 1024:
            raise SystemExit("source archive exceeds the expansion limit")
if not root or members < 2:
    raise SystemExit("source archive is empty")
print(root)
PY
)" || fail "source archive failed structural validation"
    [[ -n "$top" ]] || fail "source archive is empty"
    tar -xzf "$archive" --strip-components=1 -C "$release_dir"
  fi
  [[ -f "$release_dir/install.sh" && -f "$release_dir/server.py" \
      && -f "$release_dir/frontend/index.html" ]] \
    || fail "source archive is missing Friday install/runtime files"
  chmod 755 "$release_dir/install.sh" "$release_dir/ops/fridayctl" \
    "$release_dir/ops/provision_qwen_runtime.sh" \
    "$release_dir/scripts/uninstall.sh"
  printf 'revision=%s\ninstalled_at=%s\n' "$source_revision" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$release_dir/FRIDAY_RELEASE"

  local shared="$install_root/shared"
  local model_root="$shared/models"
  mkdir -p "$shared" "$model_root" "$shared/skills" "$shared/capabilities" \
    "$shared/backups" "$shared/persona/voices" "$state_root/logs"

  seed_directory() {
    local from="$1" to="$2"
    [[ -d "$from" ]] || return 0
    mkdir -p "$to"
    cp -a --reflink=auto --no-clobber "$from"/. "$to"/ 2>/dev/null \
      || cp -a --reflink=auto "$from"/. "$to"/
  }
  if [[ -n "$source_dir" ]]; then
    seed_directory "$source_dir/models" "$model_root"
    seed_directory "$source_dir/skills" "$shared/skills"
    seed_directory "$source_dir/capabilities" "$shared/capabilities"
    seed_directory "$source_dir/backups" "$shared/backups"
    seed_directory "$source_dir/persona/voices" "$shared/persona/voices"
    if [[ ! -f "$state_root/friday.db" && -d "$source_dir/state" ]]; then
      seed_directory "$source_dir/state" "$state_root"
      rm -f -- "$state_root"/*.pid "$state_root"/*.lock
    fi
    [[ -e "$state_root/session.json" || ! -f "$source_dir/session.json" ]] \
      || cp -a --reflink=auto "$source_dir/session.json" "$state_root/session.json"
  fi
  seed_directory "$release_dir/skills" "$shared/skills"
  seed_directory "$release_dir/persona/voices" "$shared/persona/voices"

  rm -rf -- "$release_dir/models" "$release_dir/skills" \
    "$release_dir/capabilities" "$release_dir/backups" "$release_dir/state"
  ln -s "$model_root" "$release_dir/models"
  ln -s "$shared/skills" "$release_dir/skills"
  ln -s "$shared/capabilities" "$release_dir/capabilities"
  ln -s "$shared/backups" "$release_dir/backups"
  ln -s "$state_root" "$release_dir/state"
  mkdir -p "$release_dir/persona"
  rm -rf -- "$release_dir/persona/voices"
  ln -s "$shared/persona/voices" "$release_dir/persona/voices"
  touch "$state_root/session.json" "$state_root/logs/server.log"
  chmod 600 "$state_root/session.json" "$state_root/logs/server.log"
  rm -f -- "$release_dir/session.json" "$release_dir/server.log"

  local uv_bin
  ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
      command -v uv
      return
    fi
    local version=0.12.1
    local tool_root="$install_root/tools/uv/$version"
    local bundle="$cache_root/uv-$version-x86_64-unknown-linux-gnu.tar.gz"
    local expected="90b2f223fb69d19db49e117da601f64978593417988530aa733d456141b4bcbb"
    if [[ ! -x "$tool_root/uv" ]]; then
      mkdir -p "$tool_root"
      curl --fail --location --retry 5 --retry-all-errors --output "$bundle.part" \
        "https://github.com/astral-sh/uv/releases/download/$version/uv-x86_64-unknown-linux-gnu.tar.gz"
      printf '%s  %s\n' "$expected" "$bundle.part" | sha256sum -c -
      tar -xzf "$bundle.part" --strip-components=1 -C "$tool_root"
      mv "$bundle.part" "$bundle"
    fi
    printf '%s\n' "$tool_root/uv"
  }
  uv_bin="$(ensure_uv)"

  if [[ -n "$source_dir" && -x "$source_dir/venv/bin/python" && $seed_venv -eq 1 ]]; then
    step environment "reflink-copying the verified local Python environment"
    cp -a --reflink=auto "$source_dir/venv" "$release_dir/venv"
    python3 - "$source_dir/venv" "$release_dir/venv" <<'PY'
from pathlib import Path
import sys
old, new = map(lambda value: str(Path(value).resolve()), sys.argv[1:])
for path in (Path(new) / "bin").iterdir():
    if not path.is_file() or path.is_symlink():
        continue
    try:
        body = path.read_bytes()
    except OSError:
        continue
    prefix = ("#!" + old).encode()
    if body.startswith(prefix):
        path.write_bytes(("#!" + new).encode() + body[len(prefix):])
PY
  else
    step environment "creating a pinned Python 3.12 runtime"
    [[ -f "$release_dir/requirements/runtime.lock" ]] \
      || fail "requirements/runtime.lock is missing"
    "$uv_bin" venv --python 3.12 "$release_dir/venv"
    "$uv_bin" pip sync --python "$release_dir/venv/bin/python" \
      --require-hashes "$release_dir/requirements/runtime.lock"
  fi
  (cd "$release_dir" && venv/bin/python - <<'PY'
import fastapi, numpy, openai, pydantic, sherpa_onnx, silero_vad, torch, uvicorn
from friday_core import GraphStore
print("app runtime imports verified")
PY
  )

  if (( ! skip_assets )); then
    step assets "verifying pinned ASR, speech, and embedding models"
    "$release_dir/venv/bin/python" "$release_dir/ops/install_asr_model.py" \
      --model-root "$model_root" --cache-root "$cache_root/downloads"
    "$release_dir/venv/bin/python" "$release_dir/ops/install_piper_voice.py"
    "$release_dir/venv/bin/python" "$release_dir/ops/install_omnivoice_model.py"
    "$release_dir/venv/bin/python" "$release_dir/ops/install_embedding_model.py"
  fi

  if [[ -z "$llm_root" ]]; then
    llm_root="$install_root/runtime/qwen"
  fi
  step model "verifying or provisioning the pinned Qwen/vLLM runtime"
  "$release_dir/ops/provision_qwen_runtime.sh" "$llm_root" \
    "$shared/qwen-models" "$uv_bin"
  [[ -s "$llm_root/api_key.txt" ]] || {
    openssl rand -hex 24 > "$llm_root/api_key.txt"
  }
  chmod 600 "$llm_root/api_key.txt"

  step config "writing private per-user configuration"
  cat > "$config_file" <<EOF
FRIDAY_INSTALL_ROOT='$install_root'
FRIDAY_CONFIG_ROOT='$config_root'
FRIDAY_STATE_DIR='$state_root'
FRIDAY_LLM_REPO='$llm_root'
FRIDAY_LOCAL_API_KEY_FILE='$llm_root/api_key.txt'
FRIDAY_OWNER_NAME='$owner_name'
FRIDAY_BIND_HOST='127.0.0.1'
FRIDAY_PORT='8500'
FRIDAY_ALLOWED_HOSTS='localhost,127.0.0.1,::1'
FRIDAY_ALLOWED_ORIGINS='https://localhost:8500,https://127.0.0.1:8500,https://[::1]:8500'
FRIDAY_DESKTOP_MODE='auto'
EOF
  chmod 600 "$config_file"

  python3 - "$release_dir/ops/friday.service.in" \
    "$unit_file" "$install_root/current" "$config_file" <<'PY'
from pathlib import Path
import sys
source, target, current, env_file = map(Path, sys.argv[1:])
body = source.read_text().replace("@CURRENT@", str(current)).replace("@ENV_FILE@", str(env_file))
target.write_text(body)
target.chmod(0o600)
PY
  cat > "$cli_file" <<EOF
#!/usr/bin/env bash
export FRIDAY_INSTALL_ROOT='$install_root'
export FRIDAY_CONFIG_ROOT='$config_root'
exec '$install_root/current/ops/fridayctl' "\$@"
EOF
  chmod 755 "$cli_file"
  python3 - "$release_dir/ops/friday.desktop.in" \
    "$desktop_file" "$cli_file" <<'PY'
from pathlib import Path
import sys
source, target, cli = map(Path, sys.argv[1:])
target.write_text(source.read_text().replace("@CLI@", str(cli)))
target.chmod(0o644)
PY
  install -m 0644 "$release_dir/assets/friday.svg" \
    "$icon_file"
  command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$data_home/applications" >/dev/null 2>&1 || true

  ln -s "$release_dir" "$install_root/.current-$$"
  mv -Tf "$install_root/.current-$$" "$install_root/current"
  switched=1
  systemctl --user daemon-reload
  systemctl --user enable friday.service >/dev/null

  set -a
  # shellcheck disable=SC1090
  source "$config_file"
  set +a
  "$release_dir/venv/bin/python" "$release_dir/ops/friday_doctor.py"

  if (( start_after )); then
    step launch "starting Friday and loading the local model"
    systemctl --user restart friday.service
    "$cli_file" start
    "$release_dir/venv/bin/python" "$release_dir/ops/friday_doctor.py" --expect-running
  else
    step launch "installed; startup deferred (--no-start)"
  fi

  if [[ -n "$previous_target" && "$previous_target" != "$release_dir" ]]; then
    printf '%s\n' "$previous_target" > "$install_root/previous-release"
  fi
  rm -rf -- "$rollback_dir"
  rollback_dir=""
  trap - EXIT HUP INT TERM
  printf '\n  %s\n' '────────────────────────────────────────────────────'
  printf '  Friday installed\n  launch         friday open\n  status         friday status\n  diagnostics    friday doctor\n  stop model     friday stop\n\n'
}

friday_install "$@"
