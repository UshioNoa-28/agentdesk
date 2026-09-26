"""会话驾驶锁:checkout/release 语义、跨进程互斥、进程死亡自动回收。"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agent.exceptions import SessionDriverBusyError
from agent.infrastructure.driver_locks import FileSessionDriverLocks

_CHILD_HOLD = """
import fcntl, sys, time
path = sys.argv[1]
handle = open(path, "a+b")
fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
print("locked", flush=True)
time.sleep(float(sys.argv[2]))
"""


class FileSessionDriverLockTests(TestCase):
    def test_same_session_conflicts_and_releases(self) -> None:
        with TemporaryDirectory() as directory:
            locks = FileSessionDriverLocks(Path(directory) / "locks")

            first = locks.checkout("sess-1")
            with self.assertRaises(SessionDriverBusyError):
                locks.checkout("sess-1")
            # 不同会话互不影响。
            second = locks.checkout("sess-2")
            first.release()
            third = locks.checkout("sess-1")  # 释放后可重取
            second.release()
            third.release()

    def test_release_is_idempotent(self) -> None:
        with TemporaryDirectory() as directory:
            locks = FileSessionDriverLocks(Path(directory) / "locks")
            handle = locks.checkout("sess-a")
            handle.release()
            handle.release()
            locks.checkout("sess-a").release()

    def test_lock_released_when_holding_process_dies(self) -> None:
        """内核负责回收:flock 随持锁进程退出自动消失,无需租约/心跳。"""

        with TemporaryDirectory() as directory:
            lock_dir = Path(directory) / "locks"
            locks = FileSessionDriverLocks(lock_dir)
            lock_path = lock_dir / "sess-1.lock"

            child = subprocess.Popen(
                [sys.executable, "-c", _CHILD_HOLD, str(lock_path), "3"],
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                assert child.stdout is not None
                self.assertEqual("locked", child.stdout.readline().strip())

                with self.assertRaises(SessionDriverBusyError):
                    locks.checkout("sess-1")

                child.kill()
                child.wait(timeout=5)

                started = time.monotonic()
                while True:  # SIGKILL 后内核释放通常瞬时,留 2s 容忍调度
                    try:
                        handle = locks.checkout("sess-1")
                        handle.release()
                        break
                    except SessionDriverBusyError:
                        if time.monotonic() - started > 2:
                            self.fail("lock not reclaimed after holder died")
                        time.sleep(0.05)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)

    def test_lock_files_created_lazily_in_configured_dir(self) -> None:
        with TemporaryDirectory() as directory:
            lock_dir = Path(directory) / "deep" / "locks"
            locks = FileSessionDriverLocks(lock_dir)
            handle = locks.checkout("sess-a")
            self.assertTrue((lock_dir / "sess-a.lock").exists())
            handle.release()
