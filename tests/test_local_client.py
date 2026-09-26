"""LocalApiClient 读路径:嵌入式运行时的形状与错误契约。

形状判据来自 HTTP 实现:同名字段、同类型(JSON-safe),UI 层因此
不需要区分传输。ask_stream 属于下一里程碑,这里只钉住其占位错误。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import TestCase, mock

from cli.local_client import LocalApiClient
from cli.ports import AgentApiError, AgentClientPort


class LocalApiClientTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        skill_dir = root / ".agent-desk" / "skills" / "structured-answer"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: structured-answer\n"
            "description: Organize answers into conclusion, evidence, limits.\n"
            "---\n\n"
            "Answer with conclusion first.\n",
            encoding="utf-8",
        )
        (root / ".agent-desk" / "mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
        (root / ".agent-desk" / "settings.json").write_text(
            '{"model": {"test-model": {'
            '"name": "Test Model", "base_url": "https://test/v1", "api_key": "test-key", '
            '"context_window": 200000, "max_output_tokens": 4096'
            "}}}",
            encoding="utf-8",
        )
        cls._env = mock.patch.dict(
            os.environ,
            {
                "DATABASE_URL": f"sqlite+aiosqlite:///{root / 'local.db'}",
                "SESSION_LOCK_DIR": str(root / "locks"),
                "WORKSPACE_ROOT": str(root),
            },
        )
        cls._env.start()
        cls.client = LocalApiClient()
        cls.addClassCleanup(cls.client.close)
        cls.addClassCleanup(cls._env.stop)
        cls.addClassCleanup(cls._directory.cleanup)

    # -- 基本形状 ------------------------------------------------------

    def test_health_matches_http_shape(self) -> None:
        self.assertEqual({"status": "ok"}, self.client.health())

    def test_satisfies_client_port(self) -> None:
        self.assertIsInstance(self.client, AgentClientPort)

    def test_session_crud_roundtrip(self) -> None:
        created = self.client.create_session(title="Local Roundtrip")
        self.assertEqual(
            {
                "id",
                "title",
                "user_id",
                "parent_session_id",
                "main_session_id",
                "allowed_tools",
                "parent_last_seq",
                "created_at",
                "updated_at",
            },
            set(created),
        )
        self.assertEqual("Local Roundtrip", created["title"])
        self.assertIsInstance(created["created_at"], str)
        self.assertIsNone(created["main_session_id"])

        listed = self.client.list_sessions()
        self.assertIn(created["id"], [item["id"] for item in listed])

        fetched = self.client.get_session(created["id"])
        self.assertEqual(created["id"], fetched["id"])

        renamed = self.client.rename_session(created["id"], "Renamed Local")
        self.assertEqual("Renamed Local", renamed["title"])

        # 默认会话不入库 SYSTEM 行：空会话分叉时 parent_last_seq 自然取 0
        forked = self.client.resume_session(created["id"])
        self.assertEqual(created["id"], forked["parent_session_id"])
        self.assertEqual(0, forked["parent_last_seq"])

        self.client.delete_session(forked["id"])
        self.assertNotIn(forked["id"], [item["id"] for item in self.client.list_sessions()])

    def test_list_messages_skips_system_prompt(self) -> None:
        session = self.client.create_session(title="Local History")
        self.assertEqual([], self.client.list_messages(session["id"]))

    def test_compact_returns_dto_envelope(self) -> None:
        session = self.client.create_session(title="Local Compact")
        payload = self.client.compact_session(session["id"])
        self.assertEqual(
            {
                "session_id",
                "status",
                "kind",
                "checkpoint_message_id",
                "estimated_tokens_before",
                "estimated_tokens_after",
                "message",
            },
            set(payload),
        )
        self.assertEqual(session["id"], payload["session_id"])

    def test_management_surfaces(self) -> None:
        mcps = self.client.list_mcps()
        self.assertIsInstance(mcps, list)
        skills = self.client.list_skills()
        names = {item["name"] for item in skills}
        self.assertIn("structured-answer", names)
        memories = self.client.list_memories()
        self.assertIsInstance(memories, list)

    # -- 驾驶权签出 ----------------------------------------------------

    def test_checkout_blocks_other_clients_until_released(self) -> None:
        session_id = self.client.create_session(title="Checkout Hold")["id"]
        self.client.checkout(session_id)
        self.client.checkout(session_id)  # 同会话重复签出幂等

        second = LocalApiClient()
        try:
            with self.assertRaises(AgentApiError) as caught:
                second.checkout(session_id)
            self.assertEqual(409, caught.exception.status_code)

            self.client.release(session_id)
            second.checkout(session_id)
            second.release(session_id)
        finally:
            second.close()

    # -- 错误契约 ------------------------------------------------------

    def test_missing_session_maps_to_404(self) -> None:
        with self.assertRaises(AgentApiError) as caught:
            self.client.get_session("00000000-0000-0000-0000-000000000000")
        self.assertEqual(404, caught.exception.status_code)

    def test_permission_validation_before_dispatch(self) -> None:
        with self.assertRaises(AgentApiError):
            self.client.reply_permission("perm-1", "always")

    def test_unknown_permission_maps_like_http(self) -> None:
        for decision in ("once", "allow_project", "reject", "allow"):
            with self.assertRaises(AgentApiError) as caught:
                self.client.reply_permission("00000000-0000-0000-0000-000000000000", decision)
            self.assertEqual(404, caught.exception.status_code)

    def test_permission_mode_roundtrip(self) -> None:
        initial = self.client.get_permission_mode()
        self.assertIn(initial, {"default", "dont_ask", "bypass"})

        updated = self.client.set_permission_mode("dont_ask")
        self.assertEqual("dont_ask", updated)
        self.assertEqual("dont_ask", self.client.get_permission_mode())

        self.client.set_permission_mode(initial)

    def test_reply_ask_user_validation(self) -> None:
        with self.assertRaises(AgentApiError) as caught:
            self.client.reply_ask_user("   ", "answer")
        self.assertEqual(400, caught.exception.status_code)

    def test_unknown_ask_user_interruption_returns_404(self) -> None:
        with self.assertRaises(AgentApiError) as caught:
            self.client.reply_ask_user("non-existent-id", "answer")
        self.assertEqual(404, caught.exception.status_code)

    def test_cancel_validation(self) -> None:
        with self.assertRaises(AgentApiError) as caught:
            self.client.cancel("   ")
        self.assertEqual(400, caught.exception.status_code)

    def test_cancel_dispatches_safely(self) -> None:
        self.client.cancel("sess-idle-test")



_GOOD_MESSAGE = {
    "id": "m-1",
    "session_id": "s-1",
    "seq": 2,
    "role": "assistant",
    "content": "hi there",
    "tool_calls": [],
    "metadata": {},
}


class TurnBridgeRuleTests(TestCase):
    """SSE→管道替换后,终止规则必须与 HTTP 实现逐条一致。"""

    def _events(self, frames, session_id="s-1"):
        from cli.local_client import _TurnBridge

        bridge = _TurnBridge()
        for frame in frames:
            bridge.push(frame)
        bridge.push(None)
        return list(bridge.events(session_id))

    def test_forwards_frames_as_event_dicts(self) -> None:
        events = self._events(
            [
                ("text_delta", {"content": "a"}),
                ("tool_call", {"id": "c1", "name": "read_file", "arguments": {}}),
                ("message_end", {"message": dict(_GOOD_MESSAGE)}),
            ]
        )
        self.assertEqual(
            ["text_delta", "tool_call", "message_end"],
            [event["event"] for event in events],
        )

    def test_rejects_multiple_terminal_events(self) -> None:
        from cli.ports import AgentApiError

        with self.assertRaises(AgentApiError):
            self._events(
                [
                    ("message_end", {"message": dict(_GOOD_MESSAGE)}),
                    ("queued", {"session_id": "s-1"}),
                ]
            )

    def test_rejects_event_after_terminal(self) -> None:
        from cli.ports import AgentApiError

        with self.assertRaises(AgentApiError):
            self._events(
                [
                    ("message_end", {"message": dict(_GOOD_MESSAGE)}),
                    ("text_delta", {"content": "leak"}),
                ]
            )

    def test_queued_frame_must_match_session(self) -> None:
        from cli.ports import AgentApiError

        with self.assertRaises(AgentApiError):
            self._events([("queued", {"session_id": "other"})])
        self._events([("queued", {"session_id": "s-1"})])  # 匹配则通过

    def test_message_end_payload_is_validated(self) -> None:
        from cli.ports import AgentApiError

        for broken in (
            {"message": "not-a-dict"},
            {"message": {"id": "m", "role": "assistant"}},  # 缺 content
            {"message": {**_GOOD_MESSAGE, "session_id": "other"}},
            {"message": {**_GOOD_MESSAGE, "seq": 0}},
        ):
            with self.subTest(broken=broken), self.assertRaises(AgentApiError):
                self._events([("message_end", broken)])
