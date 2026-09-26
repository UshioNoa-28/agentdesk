"""``python -m agent`` 内嵌入口:真实启停一轮,无 traceback。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

REPO_ROOT = Path(__file__).resolve().parents[1]


class AgentModuleEntryTests(TestCase):
    def test_start_stop_cycle_without_errors(self) -> None:
        with TemporaryDirectory() as directory:
            env = dict(os.environ)
            env.update(
                {
                    "DATABASE_URL": f"sqlite+aiosqlite:///{Path(directory) / 'e2e.db'}",
                    "WORKSPACE_ROOT": directory,
                    "SESSION_LOCK_DIR": str(Path(directory) / "locks"),
                }
            )
            (Path(directory) / ".agent-desk").mkdir(parents=True)
            Path(directory, ".agent-desk", "mcp.json").write_text(
                '{"mcpServers": {}}', encoding="utf-8"
            )
            Path(directory, ".agent-desk", "settings.json").write_text(
                '{"model": {"test-model": {'
                '"name": "Test Model", "base_url": "https://test/v1", "api_key": "test-key", '
                '"context_window": 200000, "max_output_tokens": 4096'
                "}}}",
                encoding="utf-8",
            )
            env.pop("PYTHONWARNINGS", None)
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "agent"],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                deadline = time.monotonic() + 20
                ready = False
                while time.monotonic() < deadline:
                    line = proc.stdout.readline()  # type: ignore[union-attr]
                    if not line:
                        break
                    if "local runtime ready" in line:
                        ready = True
                        break
                self.assertTrue(ready, "startup did not reach ready state")

                proc.send_signal(signal.SIGINT)
                remaining = proc.communicate(timeout=20)[0]
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)

            output = remaining or ""
            self.assertNotIn("Traceback", output)
            self.assertEqual(0, proc.returncode)
