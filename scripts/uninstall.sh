#!/usr/bin/env bash
set -Eeuo pipefail

INSTALL_ROOT="${FRIDAY_INSTALL_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/friday}"
STATE_ROOT="${FRIDAY_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/friday}"
CONFIG_ROOT="${FRIDAY_CONFIG_ROOT:-${XDG_CONFIG_HOME:-$HOME/.config}/friday}"
BIN_ROOT="${XDG_BIN_HOME:-$HOME/.local/bin}"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
PURGE=0

case "${1:-}" in
  --purge) PURGE=1 ;;
  "") ;;
  -h|--help)
    cat <<'EOF'
Usage: friday uninstall [--purge]

Default: remove the app, runtime service, CLI, and launcher while preserving
personal state, downloaded models, and configuration.

--purge: also remove Friday's exact state, models, and configuration roots.
EOF
    exit 0
    ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

safe_root() {
  local value
  value="$(realpath -m "$1")"
  [[ "$value" == /* && "$value" != / && "$value" != "$HOME" \
      && "$value" != "$(dirname "$HOME")" ]] || {
    echo "Refusing unsafe uninstall root: $value" >&2
    exit 1
  }
  [[ ! -L "$value" ]] || {
    echo "Refusing symlink uninstall root: $value" >&2
    exit 1
  }
  printf '%s\n' "$value"
}

INSTALL_ROOT="$(safe_root "$INSTALL_ROOT")"
STATE_ROOT="$(safe_root "$STATE_ROOT")"
CONFIG_ROOT="$(safe_root "$CONFIG_ROOT")"

systemctl --user stop friday.service 2>/dev/null || true
systemctl --user disable friday.service 2>/dev/null || true
rm -f -- "$HOME/.config/systemd/user/friday.service"
systemctl --user daemon-reload

rm -f -- "$BIN_ROOT/friday"
rm -f -- "$DATA_HOME/applications/friday.desktop"
rm -f -- "$DATA_HOME/icons/hicolor/scalable/apps/friday.svg"
command -v update-desktop-database >/dev/null 2>&1 \
  && update-desktop-database "$DATA_HOME/applications" >/dev/null 2>&1 || true

if (( PURGE )); then
  rm -rf -- "$INSTALL_ROOT" "$STATE_ROOT" "$CONFIG_ROOT"
  echo "Friday and its personal data were permanently removed."
else
  rm -rf -- "$INSTALL_ROOT/releases" "$INSTALL_ROOT/current" \
    "$INSTALL_ROOT/tools" "$INSTALL_ROOT/runtime/qwen/venv"
  echo "Friday was uninstalled. Personal state, configuration, and models were preserved."
  echo "Preserved: $STATE_ROOT, $CONFIG_ROOT, and $INSTALL_ROOT/shared"
fi
