"""Unit tests for the wait_for_replies meta tool (inbound-watermark polling)."""

from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase

from agent.domain.entities import Session
from agent.domain.tools import ToolContext
from agent.metatools.wait_for_replies import WaitForRepliesTool


class _WatermarkSessions:
    """按脚本回放 inbound_last_seq;耗尽后重复最后一个值。"""

    def __init__(self, values: list[int], *, main_session_id: str | None = None) -> None:
        self.values = values
        self.calls = 0
        self._main_session_id = main_session_id

    async def inbound_last_seq(self, session_id: str) -> int:
        self.calls += 1
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]

    async def get(self, session_id: str):
        return Session.create(title="x", main_session_id=self._main_session_id)


class WaitForRepliesToolTests(IsolatedAsyncioTestCase):
    async def test_subagent_caller_raises(self) -> None:
        """subagent 的 main_session_id 非 None;拿到本工具即越权,必须抛。"""
        sessions = _WatermarkSessions([5, 6], main_session_id="owner-1")
        tool = WaitForRepliesTool(default_timeout=60, max_timeout=600, session_service=sessions)

        with self.assertRaises(RuntimeError):
            await tool.aexecute({}, context=ToolContext(caller_session_id="sub-session"))
        self.assertEqual(0, sessions.calls)  # 越权在查水位之前就拦下

    async def test_missing_session_service_raises(self) -> None:
        """工具仅主控可用,None 只可能是装配错误,必须炸而不是静默降级。"""
        tool = WaitForRepliesTool(default_timeout=1, max_timeout=2, session_service=None)

        with self.assertRaises(RuntimeError):
            await tool.aexecute({}, context=ToolContext(caller_session_id="s-1"))

    async def test_invalid_timeout_falls_back_to_default(self) -> None:
        sessions = _WatermarkSessions([4])
        tool = WaitForRepliesTool(default_timeout=1, max_timeout=600, session_service=sessions)

        res = await tool.aexecute(
            {"timeout_seconds": "nonsense"}, context=ToolContext(caller_session_id="s-1")
        )

        payload = json.loads(res.content)
        self.assertIn("Timed out", payload["note"])
        self.assertEqual(1, payload["waited_seconds"])


    async def test_wakes_early_when_inbound_message_arrives(self) -> None:
        sessions = _WatermarkSessions([5, 6])
        tool = WaitForRepliesTool(default_timeout=60, max_timeout=600, session_service=sessions)

        res = await tool.aexecute({}, context=ToolContext(caller_session_id="s-1"))

        payload = json.loads(res.content)
        self.assertIn("A new inbound message arrived", payload["note"])
        self.assertLess(payload["waited_seconds"], 60)

    async def test_silent_session_times_out(self) -> None:
        sessions = _WatermarkSessions([5])
        tool = WaitForRepliesTool(default_timeout=1, max_timeout=600, session_service=sessions)

        res = await tool.aexecute({}, context=ToolContext(caller_session_id="s-1"))

        payload = json.loads(res.content)
        self.assertIn("Timed out with no new message", payload["note"])
        self.assertEqual(1, payload["waited_seconds"])
        self.assertGreaterEqual(sessions.calls, 2)

    async def test_timeout_argument_still_clamped_when_polling(self) -> None:
        sessions = _WatermarkSessions([7])
        tool = WaitForRepliesTool(default_timeout=60, max_timeout=1, session_service=sessions)

        res = await tool.aexecute(
            {"timeout_seconds": 9999}, context=ToolContext(caller_session_id="s-1")
        )

        payload = json.loads(res.content)
        self.assertIn("Timed out", payload["note"])
        self.assertEqual(1, payload["waited_seconds"])


__all__ = ["WaitForRepliesToolTests"]
