import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.platform_markers import require_platform

require_platform('linux')


ROOT = Path(__file__).resolve().parents[1]
PROVISIONER = ROOT / "ops" / "provision_qwen_runtime.sh"
QWEN_COMMIT = "f238b9320a2ef1a48cfe47c4c2db3b0ef89d93b1"


def _executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


class QwenProvisionerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.bin = self.root / "bin"
        self.runtime = self.root / "runtime"
        self.models = self.root / "models"
        self.home.mkdir(mode=0o700)
        self.bin.mkdir(mode=0o700)
        self.models.mkdir(mode=0o700)
        self._write_fixtures()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_fixtures(self):
        _executable(self.bin / "git", f"""#!/usr/bin/env bash
set -eu
if [[ "${{1:-}}" == clone ]]; then
  target="${{@: -1}}"
  mkdir -p "$target/patches" "$target/kvarn" "$target/single-user"
  printf 'fixture\n' > "$target/patches/fixture.patch"
  printf '#!/bin/sh\nexit 0\n' > "$target/kvarn/install.sh"
  printf '#!/bin/sh\nexit 0\n' > "$target/verify.sh"
  printf '#!/bin/sh\nexit 0\n' > "$target/single-user/start_qwen.sh"
  chmod 755 "$target/kvarn/install.sh" "$target/verify.sh" \
    "$target/single-user/start_qwen.sh"
  exit 0
fi
if [[ "${{1:-}}" == -C && "${{3:-}}" == rev-parse ]]; then
  printf '%s\n' '{QWEN_COMMIT}'
fi
exit 0
""")
        _executable(self.bin / "patch", "#!/bin/sh\nexit 0\n")
        _executable(self.bin / "uv", r'''#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == venv ]]; then
  target="${@: -1}"
  mkdir -p "$target/bin" "$target/site/vllm"
  cat > "$target/bin/python" <<PY
#!/usr/bin/env bash
if [[ "\${1:-}" == -c ]]; then
  printf '%s\n' "$target/site/vllm"
fi
exit 0
PY
  chmod 755 "$target/bin/python"
  cat > "$target/bin/hf" <<'SH'
#!/usr/bin/env bash
set -eu
target="${@: -1}"
mkdir -p "$target"
printf '{}\n' > "$target/config.json"
SH
  chmod 755 "$target/bin/hf"
  if [[ " $* " == *" --relocatable "* ]]; then
    cat > "$target/bin/vllm" <<'SH'
#!/usr/bin/env bash
set -eu
if [[ "${FAIL_RELOCATED_VLLM:-0}" == 1 \
      && "$0" != *"/.qwen-install-"* ]]; then
  exit 19
fi
printf 'fixture-vllm\n'
SH
  else
    cat > "$target/bin/vllm" <<SH
#!$target/bin/python
fixture
SH
  fi
  chmod 755 "$target/bin/vllm"
fi
exit 0
''')

    def _run(self, **extra_environment):
        environment = os.environ.copy()
        environment.update({
            "HOME": str(self.home),
            "PATH": str(self.bin) + os.pathsep + environment["PATH"],
            **extra_environment,
        })
        return subprocess.run(
            ["bash", str(PROVISIONER), str(self.runtime),
             str(self.models), str(self.bin / "uv")],
            cwd=ROOT, env=environment, text=True, capture_output=True,
            timeout=30, check=False,
        )

    def test_relocated_vllm_launcher_executes_from_final_runtime(self):
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        launched = subprocess.run(
            [str(self.runtime / "venv" / "bin" / "vllm"), "--version"],
            text=True, capture_output=True, timeout=10, check=False)
        self.assertEqual(launched.returncode, 0, launched.stderr)
        self.assertEqual(launched.stdout.strip(), "fixture-vllm")
        self.assertFalse(any(self.root.glob(".qwen-install-*")))

    def test_failed_post_move_launcher_restores_previous_runtime(self):
        self.runtime.mkdir(mode=0o700)
        (self.runtime / "previous-canary").write_text("preserve\n")
        completed = self._run(FAIL_RELOCATED_VLLM="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(
            (self.runtime / "previous-canary").read_text(), "preserve\n")
        self.assertFalse(any(self.root.glob(".qwen-rollback-*")))


if __name__ == "__main__":
    unittest.main()
