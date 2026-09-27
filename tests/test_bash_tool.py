"""固定 ``bash`` Meta Tool：权限段适配与子进程执行语义。"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import tracemalloc
from unittest import IsolatedAsyncioTestCase, TestCase

from agent.domain.tools import ToolContext
from agent.infrastructure.metatools.bash import BashTool


class BashPermissionTargetsTests(TestCase):
    def test_clean_chain_splits_into_segments(self) -> None:
        tool = BashTool()

        self.assertEqual(
            ("git status", "rm -rf x"),
            tool.permission_targets({"command": "git status && rm -rf x"}),
        )

    def test_opaque_command_becomes_single_whole_segment(self) -> None:
        # 拆不干净时整条命令是唯一段：scoped allow 匹配不上，必然进询问。
        tool = BashTool()
        for command in (
            "if true; then ls; fi",
            "echo hi > out.txt",
            "echo $(date)",
            "kill ${PID?}",
        ):
            with self.subTest(command=command):
                self.assertEqual(
                    (command,),
                    tool.permission_targets({"command": command}),
                )


class BashExecutionTests(IsolatedAsyncioTestCase):
    async def _run(self, arguments: dict[str, object]) -> dict[str, object]:
        tool = BashTool()
        result = await tool.aexecute(
            arguments,
            context=ToolContext(caller_session_id="session-1"),
        )
        return json.loads(result.content)

    async def test_echo_returns_ok_payload(self) -> None:
        payload = await self._run({"command": "echo hi"})

        self.assertTrue(payload["ok"])
        self.assertEqual(0, payload["exit_code"])
        self.assertEqual("hi\n", payload["stdout"])
        self.assertEqual("", payload["stderr"])

    async def test_nonzero_exit_is_failure_not_tool_error(self) -> None:
        # 命令失败=工具完成了执行：ok=False 但载荷完整、无 error 信封。
        payload = await self._run({"command": "echo oops >&2; exit 3"})

        self.assertFalse(payload["ok"])
        self.assertEqual(3, payload["exit_code"])
        self.assertEqual("oops\n", payload["stderr"])
        self.assertNotIn("error", payload)

    async def test_timeout_kills_process_and_reports_error(self) -> None:
        started = time.monotonic()

        payload = await self._run({"command": "sleep 30", "timeout_seconds": 0.2})

        self.assertFalse(payload["ok"])
        self.assertEqual("bash_timeout", payload["error"]["code"])
        # 0.2s 就该返回；没杀干净会拖到 sleep 自然结束。
        self.assertLess(time.monotonic() - started, 5.0)

    async def test_output_capped_per_stream(self) -> None:
        payload = await self._run({"command": r"head -c 200000 /dev/zero | tr '\0' 'x'"})

        self.assertTrue(payload["ok"])
        self.assertEqual(65_536, len(payload["stdout"]))
        self.assertTrue(payload["stdout_truncated"])
        self.assertFalse(payload["stderr_truncated"])

    async def test_truncated_output_keeps_head_and_tail(self) -> None:
        """窗口是头 32Ki + 尾 32Ki：两端内容必须原样在场。"""

        payload = await self._run(
            {"command": r"printf HEAD; head -c 100000 /dev/zero | tr '\0' 'x'; printf TAIL"}
        )

        stdout = payload["stdout"]
        self.assertTrue(payload["stdout_truncated"])
        self.assertEqual(65_536, len(stdout))
        self.assertTrue(stdout.startswith("HEAD"))
        self.assertTrue(stdout.endswith("TAIL"))
        self.assertTrue(set(stdout[4 : len(stdout) - 4]) == {"x"})

    async def test_both_streams_flood_simultaneously_without_deadlock(self) -> None:
        """两路各自持续刷屏：收集必须并行排空，否则 pipe 写满即挂死。

        命令本身 ~200KB 双路输出，若有一路不被持续读取，子进程永远退不
        出来，最终只会看到 bash_timeout 而不是 ok。
        """

        started = time.monotonic()
        payload = await self._run(
            {
                "command": (
                    r"head -c 200000 /dev/zero | tr '\0' 'x' 1>&2 &"
                    r" head -c 200000 /dev/zero | tr '\0' 'y'; wait"
                )
            }
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(65_536, len(payload["stdout"]))
        self.assertEqual(65_536, len(payload["stderr"]))
        self.assertTrue(payload["stdout_truncated"])
        self.assertTrue(payload["stderr_truncated"])
        self.assertLess(time.monotonic() - started, 20.0)

    async def test_memory_stays_bounded_under_large_output(self) -> None:
        """8MB 输出下堆增长有界：返回值只有 64KiB，其余字节过路不入账。"""

        tracemalloc.start()
        try:
            baseline, _ = tracemalloc.get_traced_memory()
            payload = await self._run({"command": r"head -c 8000000 /dev/zero | tr '\0' 'x'"})
            peak, _ = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        self.assertTrue(payload["stdout_truncated"])
        self.assertLess(peak - baseline, 2_000_000)

    async def test_cancellation_kills_process_group(self) -> None:
        # bash 对单命令会 exec：ps 里只剩 "sleep 31.415"，探针用时长本身。
        task = asyncio.create_task(
            BashTool().aexecute(
                {"command": "sleep 31.415", "timeout_seconds": 30.0},
                context=ToolContext(caller_session_id="session-1"),
            )
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.2)
        listing = subprocess.run(
            ["ps", "-eo", "args="], capture_output=True, text=True, check=False
        ).stdout
        self.assertNotIn("31.415", listing)
