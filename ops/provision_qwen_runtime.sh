#!/usr/bin/env bash
set -Eeuo pipefail

QWEN_REPOSITORY="https://github.com/syv-ai/qwen38-27b-rtx3090.git"
QWEN_COMMIT="f238b9320a2ef1a48cfe47c4c2db3b0ef89d93b1"
MODEL_ID="ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound"
MODEL_REVISION="92600100b5c2b97bf1fd1745479c1e0f8007e008"
MODEL_DIRECTORY="Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound"

usage() {
  echo "Usage: provision_qwen_runtime.sh RUNTIME_ROOT MODEL_ROOT UV_BINARY" >&2
}

[[ $# -eq 3 ]] || { usage; exit 2; }
RUNTIME_ROOT="$(realpath -m "$1")"
MODEL_ROOT="$(realpath -m "$2")"
UV="$(realpath -m "$3")"

[[ -x "$UV" ]] || { echo "uv is not executable: $UV" >&2; exit 1; }
[[ "$RUNTIME_ROOT" != / && "$RUNTIME_ROOT" != "$HOME" ]] || {
  echo "refusing unsafe Qwen runtime root: $RUNTIME_ROOT" >&2
  exit 1
}

MODEL="$MODEL_ROOT/$MODEL_DIRECTORY"
if [[ -x "$RUNTIME_ROOT/venv/bin/vllm" \
      && -f "$RUNTIME_ROOT/single-user/start_qwen.sh" \
      && -f "$MODEL/config.json" \
      && -s "$RUNTIME_ROOT/api_key.txt" ]]; then
  MODEL="models/$MODEL_DIRECTORY" bash "$RUNTIME_ROOT/verify.sh" --no-server
  echo "Verified existing Friday model runtime: $RUNTIME_ROOT"
  exit 0
fi

PARENT="$(dirname "$RUNTIME_ROOT")"
mkdir -p "$PARENT" "$MODEL_ROOT"
STAGING="$PARENT/.qwen-install-$$"
ROLLBACK="$PARENT/.qwen-rollback-$$"
cleanup() {
  local status=$?
  [[ -d "$STAGING" ]] && rm -rf -- "$STAGING"
  if (( status != 0 )) && [[ -d "$ROLLBACK" && ! -e "$RUNTIME_ROOT" ]]; then
    mv -- "$ROLLBACK" "$RUNTIME_ROOT"
  fi
  exit "$status"
}
trap cleanup EXIT HUP INT TERM

echo "Cloning pinned Qwen runtime..."
git clone --filter=blob:none --no-checkout "$QWEN_REPOSITORY" "$STAGING"
git -C "$STAGING" checkout --detach "$QWEN_COMMIT"
[[ "$(git -C "$STAGING" rev-parse HEAD)" == "$QWEN_COMMIT" ]] || {
  echo "Qwen source revision mismatch" >&2
  exit 1
}

echo "Creating pinned vLLM environment..."
"$UV" venv --python 3.12 "$STAGING/venv"
QWEN_LOCK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/requirements/qwen-runtime.lock"
[[ -f "$QWEN_LOCK" ]] || {
  echo "Qwen runtime lock is missing: $QWEN_LOCK" >&2
  exit 1
}
"$UV" pip sync --python "$STAGING/venv/bin/python" \
  --require-hashes "$QWEN_LOCK"

SITE="$($STAGING/venv/bin/python -c 'import os,vllm; print(os.path.dirname(vllm.__file__))')"
for patch_file in "$STAGING"/patches/*.patch; do
  patch --batch --forward -p1 -d "$SITE" < "$patch_file"
done
PY="$STAGING/venv/bin/python" bash "$STAGING/kvarn/install.sh"

echo "Downloading the pinned Friday model (~16 GiB, resumable)..."
HF_HUB_ENABLE_HF_TRANSFER=1 "$STAGING/venv/bin/hf" download \
  "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL"
mkdir -p "$STAGING/models"
ln -s "$MODEL" "$STAGING/models/$MODEL_DIRECTORY"
openssl rand -hex 24 > "$STAGING/api_key.txt"
chmod 600 "$STAGING/api_key.txt"

MODEL="models/$MODEL_DIRECTORY" bash "$STAGING/verify.sh" --no-server
if [[ -e "$RUNTIME_ROOT" ]]; then
  mv -- "$RUNTIME_ROOT" "$ROLLBACK"
fi
mv -- "$STAGING" "$RUNTIME_ROOT"
rm -rf -- "$ROLLBACK"
trap - EXIT HUP INT TERM
echo "Installed Friday model runtime: $RUNTIME_ROOT"
