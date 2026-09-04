#!/usr/bin/env bash
# Regenerate every hash-pinned dependency lock from the .in inputs.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/requirements"
compile() {
  local output="$1" platform="$2"; shift 2
  # onnxruntime and cryptography publish macOS wheels for 14.0 and newer.
  MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile "$@" --output-file "$output" \
    --python-version 3.12 --python-platform "$platform" --generate-hashes --quiet
  echo "compiled $output for $platform"
}
# The CUDA superset lock resolves first; the portable locks are constrained to
# its versions so every shared package agrees across platforms.
compile cuda-linux-x86_64.lock x86_64-unknown-linux-gnu runtime.in cuda.in
compile runtime-linux-x86_64.lock x86_64-unknown-linux-gnu runtime.in -c cuda-linux-x86_64.lock
compile runtime-linux-arm64.lock aarch64-unknown-linux-gnu runtime.in -c cuda-linux-x86_64.lock
compile runtime-macos-arm64.lock aarch64-apple-darwin runtime.in -c cuda-linux-x86_64.lock
compile runtime-windows-x86_64.lock x86_64-pc-windows-msvc runtime.in -c cuda-linux-x86_64.lock
compile legacy-cli-linux-x86_64.lock x86_64-unknown-linux-gnu runtime.in legacy-cli.in -c cuda-linux-x86_64.lock
compile mlx-runtime.lock aarch64-apple-darwin mlx-runtime.in
cd "$ROOT" && "${PYTHON:-python3}" scripts/update_lock_ledger.py
