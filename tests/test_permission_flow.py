from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

from agent.application.graph.agent_graph import AgentGraph
from agent.application.graph.hook_registry import AgentHookRegistry
from agent.application.services.permission_service import PermissionService
from agent.domain.context_settings import ContextSettings
from agent.domain.entities import Session
from agent.domain.exceptions import DomainValidationError
from agent.domain.interruption import (
    AskUserRequest,
    InterruptionReplyStatus,
    InterruptionResult,
    InterruptionStatus,
    PermissionRequest,
)
from agent.domain.model_messages import ToolCall
from agent.domain.permissions import (
    PermissionAction,
    PermissionMode,
    ToolRule,
)
from agent.exceptions import PermissionAlreadyResolvedError, PermissionNotFoundError
from agent.infrastructure.interruption import InterruptionBroker
from agent.infrastructure.metatools.ask_user import AskUserTool
from agent.infrastructure.metatools.execute_mcp import ExecuteMcpTool
from agent.infrastructure.metatools.workspace.edit_file import EditFileTool
from agent.infrastructure.metatools.workspace.read_file import ReadFileTool
from agent.infrastructure.permission import PermissionManager
from agent.infrastructure.runtime.stream_hub import StreamHub


def _permission_request(**overrides: object) -> PermissionRequest:
    """带合理默认的 PermissionRequest，测试只覆盖自己关心的字段。"""

    values: dict[str, object] = {
        "interruption_id": "permission-1",
        "session_id": "session-1",
        "tool_call_id": "call-1",
        "name": "edit_file",
        "arguments": {},
        "targets": (),
    }
    values.update(overrides)
    return PermissionRequest(  # type: ignore[arg-type]
        interruption_id=str(values["interruption_id"]),
        session_id=str(values["session_id"]),
        tool_call_id=str(values["tool_call_id"]),
        name=str(values["name"]),
        arguments=dict(values["arguments"]),
        targets=tuple(values["targets"]),  # type: ignore[arg-type]
    )


def _resolved(decision: str) -> InterruptionResult:
    return InterruptionResult(
        status=InterruptionStatus.RESOLVED, payload={"decision": decision}
    )


class InterruptionBrokerTests(IsolatedAsyncioTestCase):
    async def test_request_publishes_event_and_resumes_once(self) -> None:
        hub = StreamHub()
        queue = hub.register("session-1")
        assert queue is not None
        broker = InterruptionBroker(hub)
        interruption_id = str(uuid4())

        waiting = asyncio.create_task(
            broker.request(
                _permission_request(interruption_id=interruption_id)
            )
        )
        event, payload = await asyncio.wait_for(queue.get(), timeout=1)

        self.assertEqual("permission_request", event)
        self.assertEqual("session-1", payload["session_id"])
        self.assertEqual("call-1", payload["tool_call_id"])
        self.assertEqual("edit_file", payload["name"])
        self.assertEqual(interruption_id, payload["permission_id"])
        self.assertEqual(
            InterruptionReplyStatus.ACCEPTED,
            broker.reply(
                interruption_id=interruption_id, payload={"decision": "allow"}
            ),
        )
        result = await waiting
        self.assertEqual(InterruptionStatus.RESOLVED, result.status)
        self.assertEqual("allow", result.payload["decision"])
        self.assertEqual(0, broker.pending_count)

    async def test_ask_user_request_publishes_ask_user_frame(self) -> None:
        hub = StreamHub()
        queue = hub.register("session-1")
        assert queue is not None
        broker = InterruptionBroker(hub)

        waiting = asyncio.create_task(
            broker.request(
                AskUserRequest(
                    interruption_id="ask-1",
                    session_id="session-1",
                    question="用哪个数据库？",
                    options=("PostgreSQL", "SQLite"),
                    recommended="PostgreSQL",
                )
            )
        )
        event, payload = await asyncio.wait_for(queue.get(), timeout=1)

        self.assertEqual("ask_user_request", event)
        self.assertEqual("用哪个数据库？", payload["question"])
        self.assertEqual(["PostgreSQL", "SQLite"], payload["options"])
        self.assertEqual("PostgreSQL", payload["recommended"])
        self.assertEqual(
            InterruptionReplyStatus.ACCEPTED,
            broker.reply(interruption_id="ask-1", payload={"answer": "SQLite"}),
        )
        result = await waiting
        self.assertEqual("SQLite", result.payload["answer"])

    async def test_request_waits_until_explicit_reply(self) -> None:
        hub = StreamHub()
        queue = hub.register("session-1")
        assert queue is not None
        broker = InterruptionBroker(hub)

        waiting = asyncio.create_task(
            broker.request(_permission_request(interruption_id="int-1"))
        )
        _, payload = await queue.get()

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(waiting), timeout=0.02)
        self.assertFalse(waiting.done())
        self.assertEqual(
            InterruptionReplyStatus.ACCEPTED,
            broker.reply(
                interruption_id=payload["permission_id"],
                payload={"decision": "reject"},
            ),
        )
        result = await waiting
        self.assertEqual("reject", result.payload["decision"])

    async def test_concurrent_requests_are_resolved_by_their_own_ids(self) -> None:
        hub = StreamHub()
        queue = hub.register("session-1")
        assert queue is not None
        broker = InterruptionBroker(hub)
        first = asyncio.create_task(
            broker.request(
                _permission_request(interruption_id="int-1", tool_call_id="call-1")
            )
        )
        second = asyncio.create_task(
            broker.request(
                _permission_request(interruption_id="int-2", tool_call_id="call-2")
            )
        )
        _, first_payload = await queue.get()
        _, second_payload = await queue.get()

        self.assertNotEqual(
            first_payload["permission_id"],
            second_payload["permission_id"],
        )
        self.assertEqual(
            InterruptionReplyStatus.ACCEPTED,
            broker.reply(
                interruption_id=second_payload["permission_id"],
                payload={"decision": "allow"},
            ),
        )
        self.assertEqual(
            InterruptionReplyStatus.ACCEPTED,
            broker.reply(
                interruption_id=first_payload["permission_id"],
                payload={"decision": "reject"},
            ),
        )
        self.assertEqual("reject", (await first).payload["decision"])
        self.assertEqual("allow", (await second).payload["decision"])

    async def test_cancelled_request_is_removed_from_pending_map(self) -> None:
        hub = StreamHub()
        queue = hub.register("session-1")
        assert queue is not None
        broker = InterruptionBroker(hub)
        waiting = asyncio.create_task(
            broker.request(_permission_request(interruption_id="int-1"))
        )
        _, payload = await queue.get()

        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting

        self.assertEqual(0, broker.pending_count)
        self.assertEqual(
            InterruptionReplyStatus.NOT_FOUND,
            broker.reply(
                interruption_id=payload["permission_id"],
                payload={"decision": "allow"},
            ),
        )

    async def test_unknown_or_empty_reply_is_not_accepted(self) -> None:
        broker = InterruptionBroker(StreamHub())

        self.assertEqual(
            InterruptionReplyStatus.NOT_FOUND,
            broker.reply(interruption_id="missing", payload={"decision": "allow"}),
        )
        self.assertEqual(
            InterruptionReplyStatus.NOT_FOUND,
            broker.reply(interruption_id="   ", payload={"decision": "allow"}),
        )

    async def test_duplicate_reply_is_rejected_before_waiter_resumes(self) -> None:
        hub = StreamHub()
        queue = hub.register("session-1")
        assert queue is not None
        broker = InterruptionBroker(hub)
        waiting = asyncio.create_task(
            broker.request(_permission_request(interruption_id="int-1"))
        )
        _, payload = await queue.get()

        self.assertEqual(
            InterruptionReplyStatus.ACCEPTED,
            broker.reply(
                interruption_id=payload["permission_id"],
                payload={"decision": "allow"},
            ),
        )
        self.assertEqual(
            InterruptionReplyStatus.ALREADY_RESOLVED,
            broker.reply(
                interruption_id=payload["permission_id"],
                payload={"decision": "reject"},
            ),
        )
        self.assertEqual("allow", (await waiting).payload["decision"])

    async def test_no_active_stream_fails_closed(self) -> None:
        broker = InterruptionBroker(StreamHub())

        result = await broker.request(
            _permission_request(interruption_id="int-1")
        )

        self.assertEqual(InterruptionStatus.UNAVAILABLE, result.status)
        self.assertEqual(0, broker.pending_count)

    async def test_pending_exposes_original_request_until_resolved(self) -> None:
        hub = StreamHub()
        queue = hub.register("session-1")
        assert queue is not None
        broker = InterruptionBroker(hub)

        waiting = asyncio.create_task(
            broker.request(
                _permission_request(interruption_id="int-1", name="read_file")
            )
        )
        _, payload = await queue.get()
        interruption_id = payload["permission_id"]

        pending = broker.pending(interruption_id)
        self.assertIsInstance(pending, PermissionRequest)
        assert isinstance(pending, PermissionRequest)
        self.assertEqual("read_file", pending.name)
        self.assertEqual(
            InterruptionReplyStatus.ACCEPTED,
            broker.reply(interruption_id=interruption_id, payload={"decision": "allow"}),
        )
        await waiting
        self.assertIsNone(broker.pending(interruption_id))

    async def test_unknown_id_has_no_pending_request(self) -> None:
        broker = InterruptionBroker(StreamHub())

        self.assertIsNone(broker.pending("missing"))

    async def test_close_wakes_pending_requests_and_rejects_new_ones(self) -> None:
        hub = StreamHub()
        queue = hub.register("session-1")
        assert queue is not None
        broker = InterruptionBroker(hub)
        waiting = asyncio.create_task(
            broker.request(_permission_request(interruption_id="int-1"))
        )
        await queue.get()

        broker.close()

        result = await waiting
        self.assertEqual(InterruptionStatus.CANCELLED, result.status)
        self.assertEqual(0, broker.pending_count)
        self.assertEqual(
            InterruptionStatus.UNAVAILABLE,
            (
                await broker.request(
                    _permission_request(interruption_id="int-2", tool_call_id="call-2")
                )
            ).status,
        )



class PermissionServiceTests(IsolatedAsyncioTestCase):
    @staticmethod
    def _service(
        broker: object,
        *,
        manager: object | None = None,
        session_service: object | None = None,
    ) -> PermissionService:
        return PermissionService(
            broker=broker,
            manager=manager if manager is not None else PermissionManager(start=Path()),
            session_service=session_service or AsyncMock(),
        )

    async def test_accepted_reply_returns_decision(self) -> None:
        broker = MagicMock()
        broker.reply.return_value = InterruptionReplyStatus.ACCEPTED
        service = self._service(broker)

        result = await service.reply(permission_id=" permission-1 ", decision="allow")

        self.assertEqual("allow", result)
        broker.reply.assert_called_once_with(
            interruption_id="permission-1",
            payload={"decision": "allow"},
        )

    async def test_runtime_statuses_become_application_errors(self) -> None:
        cases = (
            (InterruptionReplyStatus.NOT_FOUND, PermissionNotFoundError),
            (InterruptionReplyStatus.ALREADY_RESOLVED, PermissionAlreadyResolvedError),
        )
        for status, error_type in cases:
            broker = MagicMock()
            broker.reply.return_value = status
            service = self._service(broker)
            with self.assertRaises(error_type):
                await service.reply(permission_id="permission-1", decision="reject")

    async def test_invalid_decision_is_rejected_before_broker(self) -> None:
        broker = MagicMock()
        service = self._service(broker)

        with self.assertRaises(DomainValidationError):
            await service.reply(permission_id="permission-1", decision="always")
        broker.reply.assert_not_called()

    async def test_allow_policy_skips_session_lookup_and_broker(self) -> None:
        broker = AsyncMock()
        manager = PermissionManager(
            start=Path(),
            permission_rule={"allow": [ToolRule("read_file", "")]},
        )
        sessions = AsyncMock()
        service = self._service(broker, manager=manager, session_service=sessions)

        result = await service.authorize(
            caller_session_id="session-1",
            tool_call_id="call-1",
            name="read_file",
            arguments={"path": "README.md"},
            targets=("README.md",),
        )

        self.assertEqual("allow", result)
        sessions.get.assert_not_awaited()
        broker.request.assert_not_awaited()

    async def test_explicit_empty_scope_persists_whole_tool(self) -> None:
        broker = MagicMock()
        broker.reply.return_value = InterruptionReplyStatus.ACCEPTED
        broker.pending.return_value = _permission_request(name="edit_file")
        manager = MagicMock()
        service = self._service(broker, manager=manager)

        result = await service.reply(permission_id="permission-1", decision="allow", scope=[""])

        self.assertEqual("allow", result)
        manager.persist_rule.assert_called_once_with("edit_file", PermissionAction.ALLOW, "")

    async def test_allow_reply_forwards_scope_to_persist(self) -> None:
        broker = MagicMock()
        broker.reply.return_value = InterruptionReplyStatus.ACCEPTED
        broker.pending.return_value = _permission_request(name="read_file")
        manager = MagicMock()
        service = self._service(broker, manager=manager)

        await service.reply(permission_id="permission-1", decision="allow", scope=["src/**"])

        manager.persist_rule.assert_called_once_with("read_file", PermissionAction.ALLOW, "src/**")

    async def test_multi_segment_scope_persists_one_rule_per_segment(self) -> None:
        # CLI "always allow all"：段原文逐条落盘，顺序即事件里的顺序。
        broker = MagicMock()
        broker.reply.return_value = InterruptionReplyStatus.ACCEPTED
        broker.pending.return_value = _permission_request(name="bash")
        manager = MagicMock()
        service = self._service(broker, manager=manager)

        await service.reply(
            permission_id="permission-1",
            decision="allow",
            scope=["cd agent", "uv run agentdesk"],
        )

        self.assertEqual(
            [
                call("bash", PermissionAction.ALLOW, "cd agent"),
                call("bash", PermissionAction.ALLOW, "uv run agentdesk"),
            ],
            manager.persist_rule.call_args_list,
        )

    async def test_bare_string_scope_is_rejected_before_broker(self) -> None:
        # 裸 str 迭代是单字符：这不是兼容问题，是必须拒收的错误形状。
        broker = MagicMock()
        manager = MagicMock()
        service = self._service(broker, manager=manager)

        with self.assertRaises(DomainValidationError):
            await service.reply(permission_id="permission-1", decision="allow", scope="src/**")
        broker.reply.assert_not_called()
        manager.persist_rule.assert_not_called()

    async def test_empty_sequence_scope_is_rejected_before_broker(self) -> None:
        broker = MagicMock()
        service = self._service(broker)

        for bad in ([], ()):
            with self.subTest(scope=bad):
                with self.assertRaises(DomainValidationError):
                    await service.reply(permission_id="permission-1", decision="allow", scope=bad)
        broker.reply.assert_not_called()

    async def test_allow_without_scope_never_persists(self) -> None:
        # 缺省 scope=None 即"只放行本次"：忘传参数不可能意外落盘。
        broker = MagicMock()
        broker.reply.return_value = InterruptionReplyStatus.ACCEPTED
        broker.pending.return_value = _permission_request(name="edit_file")
        manager = MagicMock()
        service = self._service(broker, manager=manager)

        await service.reply(permission_id="permission-1", decision="allow")

        manager.persist_rule.assert_not_called()

    async def test_reject_with_scope_is_invalid_combination(self) -> None:
        broker = MagicMock()
        manager = MagicMock()
        service = self._service(broker, manager=manager)

        with self.assertRaises(DomainValidationError):
            await service.reply(permission_id="permission-1", decision="reject", scope=["src/**"])
        broker.reply.assert_not_called()

    async def test_failed_or_stale_reply_never_grants(self) -> None:
        broker = MagicMock()
        broker.reply.return_value = InterruptionReplyStatus.NOT_FOUND
        broker.pending.return_value = None
        manager = MagicMock()
        service = self._service(broker, manager=manager)

        with self.assertRaises(PermissionNotFoundError):
            await service.reply(permission_id="gone", decision="allow")
        manager.persist_rule.assert_not_called()

    async def test_allow_outcome_authorizes_tool_call(self) -> None:
        session = Session.create(title="main")
        broker = AsyncMock()
        broker.request.return_value = _resolved("allow")
        manager = PermissionManager(start=Path())
        sessions = AsyncMock()
        sessions.get.return_value = session
        service = self._service(broker, manager=manager, session_service=sessions)

        result = await service.authorize(
            caller_session_id=session.id,
            tool_call_id="call-1",
            name="edit_file",
            arguments={},
            targets=("notes.txt",),
        )

        self.assertEqual("allow", result)

    async def test_approval_for_subagent_is_published_to_main_session(self) -> None:
        main_session_id = "main-session"
        sub_session = Session.create(title="worker", main_session_id=main_session_id)
        broker = AsyncMock()
        broker.request.return_value = _resolved("allow")
        manager = PermissionManager(start=Path())
        sessions = AsyncMock()
        sessions.get.return_value = sub_session
        service = self._service(broker, manager=manager, session_service=sessions)

        result = await service.authorize(
            caller_session_id=sub_session.id,
            tool_call_id="call-1",
            name="edit_file",
            arguments={"path": "notes.txt", "old_text": "a", "new_text": "b"},
            targets=("notes.txt",),
        )

        self.assertEqual("allow", result)
        broker.request.assert_awaited_once()
        request = broker.request.await_args.args[0]
        self.assertIsInstance(request, PermissionRequest)
        self.assertEqual(main_session_id, request.session_id)
        self.assertEqual("call-1", request.tool_call_id)
        self.assertEqual("edit_file", request.name)
        self.assertEqual(
            {"path": "notes.txt", "old_text": "a", "new_text": "b"}, request.arguments
        )
        self.assertEqual(("notes.txt",), request.targets)

    async def test_broker_denial_outcomes_remain_distinct(self) -> None:
        session = Session.create(title="main")
        manager = PermissionManager(start=Path())
        sessions = AsyncMock()
        sessions.get.return_value = session

        outcomes = (
            ("reject", _resolved("reject")),
            ("unavailable", InterruptionResult(status=InterruptionStatus.UNAVAILABLE)),
            # CANCELLED(本轮被打断)未获批准，等同拒绝，但来源与人工拒绝不同。
            ("reject", InterruptionResult(status=InterruptionStatus.CANCELLED)),
        )
        for expected, result_value in outcomes:
            broker = AsyncMock()
            broker.request.return_value = result_value
            service = self._service(broker, manager=manager, session_service=sessions)

            result = await service.authorize(
                caller_session_id=session.id,
                tool_call_id="call-1",
                name="edit_file",
                arguments={},
                targets=("notes.txt",),
            )

            self.assertEqual(expected, result)


class GraphPermissionTests(IsolatedAsyncioTestCase):
    @staticmethod
    def _service(
        *,
        broker: object,
        session_service: object,
        modes: dict[str, str],
    ) -> PermissionService:
        rules: dict[str, list[ToolRule]] = {}
        for name, action in modes.items():
            rules.setdefault(action, []).append(ToolRule(tool=name, scope=""))
        return PermissionService(
            broker=broker,
            manager=PermissionManager(start=Path(), permission_rule=rules),
            session_service=session_service,
        )

    @staticmethod
    def _graph(service: PermissionService) -> AgentGraph:
        return AgentGraph(
            max_turns=4,
            settings=ContextSettings(),
            hook_registry=AgentHookRegistry(),
            permission_service=service,
        )

    async def test_edit_file_waits_for_each_decision_before_mutating(self) -> None:
        session = Session.create(title="main")
        sessions = AsyncMock()
        sessions.get.return_value = session
        hub = StreamHub()
        queue = hub.register(session.id)
        assert queue is not None
        broker = InterruptionBroker(hub)
        service = self._service(
            broker=broker,
            session_service=sessions,
            modes={"edit_file": "ask"},
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("first", encoding="utf-8")
            tool = EditFileTool()
            state = {"tools": [tool], "caller_session_id": session.id}
            first_call = ToolCall(
                id="call-1",
                name="edit_file",
                arguments={
                    "path": str(root / "notes.txt"),
                    "old_text": "first",
                    "new_text": "second",
                },
            )
            rejected = asyncio.create_task(self._graph(service)._execute_tool(state, first_call))
            _, first_request = await queue.get()
            self.assertEqual("first", (root / "notes.txt").read_text(encoding="utf-8"))
            broker.reply(
                interruption_id=first_request["permission_id"],
                payload={"decision": "reject"},
            )
            rejected_payload = json.loads((await rejected).content)
            self.assertEqual("permission_reject", rejected_payload["error"]["code"])
            self.assertEqual("first", (root / "notes.txt").read_text(encoding="utf-8"))

            second_call = ToolCall(
                id="call-2",
                name="edit_file",
                arguments={
                    "path": str(root / "notes.txt"),
                    "old_text": "first",
                    "new_text": "second",
                },
            )
            approved = asyncio.create_task(self._graph(service)._execute_tool(state, second_call))
            _, second_request = await queue.get()
            self.assertEqual("first", (root / "notes.txt").read_text(encoding="utf-8"))
            broker.reply(
                interruption_id=second_request["permission_id"],
                payload={"decision": "allow"},
            )
            approved_payload = json.loads((await approved).content)

            self.assertTrue(approved_payload["ok"])
            self.assertEqual("second", (root / "notes.txt").read_text(encoding="utf-8"))

    async def test_invalid_arguments_do_not_open_permission_request(self) -> None:
        broker = AsyncMock()
        sessions = AsyncMock()
        service = self._service(
            broker=broker,
            session_service=sessions,
            modes={"edit_file": "ask"},
        )
        graph = self._graph(service)

        result = json.loads(
            (
                await graph._execute_tool(
                    {
                        "tools": [EditFileTool()],
                        "caller_session_id": "session-1",
                    },
                    ToolCall(
                        id="call-1",
                        name="edit_file",
                        # 入参已收敛为 path/old_text/new_text，多余键被 additionalProperties 拒掉。
                        arguments={
                            "path": "notes.txt",
                            "operation": "create",
                            "content": "x",
                        },
                    ),
                )
            ).content
        )

        self.assertEqual("invalid_tool_arguments", result["error"]["code"])
        broker.request.assert_not_awaited()

    async def test_unavailable_permission_does_not_execute_tool(self) -> None:
        permissions = AsyncMock()
        permissions.authorize.return_value = "unavailable"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tool = EditFileTool()
            graph = AgentGraph(
                max_turns=4,
                settings=ContextSettings(),
                hook_registry=AgentHookRegistry(),
                permission_service=permissions,
            )

            result = json.loads(
                (
                    await graph._execute_tool(
                        {"tools": [tool], "caller_session_id": "session-1"},
                        ToolCall(
                            id="call-1",
                            name="edit_file",
                            arguments={
                                "path": str(root / "notes.txt"),
                                "old_text": "a",
                                "new_text": "not written",
                            },
                        ),
                    )
                ).content
            )

            self.assertEqual("permission_unavailable", result["error"]["code"])
            self.assertFalse((root / "notes.txt").exists())

    async def test_permission_failure_becomes_tool_result(self) -> None:
        permissions = AsyncMock()
        permissions.authorize.side_effect = RuntimeError("permission backend failed")
        tool = EditFileTool()
        graph = AgentGraph(
            max_turns=4,
            settings=ContextSettings(),
            hook_registry=AgentHookRegistry(),
            permission_service=permissions,
        )

        result = json.loads(
            (
                await graph._execute_tool(
                    {"tools": [tool], "caller_session_id": "session-1"},
                    ToolCall(
                        id="call-1",
                        name="edit_file",
                        arguments={"path": "/tmp/notes.txt", "old_text": "a", "new_text": "y"},
                    ),
                )
            ).content
        )

        self.assertEqual("permission_error", result["error"]["code"])
        self.assertEqual("permission backend failed", result["error"]["message"])


class AuthorizeDecisionTests(IsolatedAsyncioTestCase):
    @staticmethod
    def _service(
        *,
        rules: dict[str, list[ToolRule]] | None = None,
        mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> tuple[PermissionService, AsyncMock, PermissionManager]:
        broker = AsyncMock()
        broker.request.return_value = _resolved("allow")
        sessions = AsyncMock()
        sessions.get.return_value = Session.create(title="main")
        manager = PermissionManager(start=Path(), permission_rule=rules or {}, mode=mode)
        service = PermissionService(
            broker=broker,
            manager=manager,
            session_service=sessions,
        )
        return service, broker, manager

    @staticmethod
    async def _authorize(
        service: PermissionService,
        name: str,
        arguments: dict[str, object],
        *,
        targets: tuple[str, ...] | None = None,
    ) -> str:
        return await service.authorize(
            caller_session_id="session-1",
            tool_call_id="call-1",
            name=name,
            arguments=arguments,
            targets=targets,
        )

    async def test_deny_beats_allow_without_broker(self) -> None:
        service, broker, _ = self._service(
            rules={
                "allow": [ToolRule("edit_file", "")],
                "deny": [ToolRule("edit_file", "")],
            }
        )

        result = await self._authorize(service, "edit_file", {})

        self.assertEqual("reject", result)
        broker.request.assert_not_awaited()

    async def test_ask_rule_beats_allow_rule(self) -> None:
        service, broker, _ = self._service(
            rules={
                "allow": [ToolRule("send_message", "")],
                "ask": [ToolRule("send_message", "")],
            }
        )

        result = await self._authorize(service, "send_message", {})

        self.assertEqual("allow", result)
        broker.request.assert_awaited_once()

    async def test_dont_ask_allows_unmatched_and_still_denies(self) -> None:
        service, broker, _ = self._service(
            rules={"deny": [ToolRule("edit_file", "")]},
            mode=PermissionMode.DONT_ASK,
        )

        self.assertEqual("allow", await self._authorize(service, "send_message", {}))
        self.assertEqual("reject", await self._authorize(service, "edit_file", {}))
        broker.request.assert_not_awaited()

    async def test_bypass_allows_even_denied_tools(self) -> None:
        service, broker, _ = self._service(
            rules={"deny": [ToolRule("edit_file", "")]},
            mode=PermissionMode.BYPASS,
        )

        result = await self._authorize(service, "edit_file", {})

        self.assertEqual("allow", result)
        broker.request.assert_not_awaited()

    async def test_unmatched_tool_falls_back_to_ask(self) -> None:
        service, broker, _ = self._service(rules={"allow": [ToolRule("read_file", "")]})

        result = await self._authorize(service, "edit_file", {}, targets=("notes.txt",))

        self.assertEqual("allow", result)
        broker.request.assert_awaited_once()

    async def test_path_scoped_allow_matches_within_and_asks_outside(
        self,
    ) -> None:
        service, broker, _ = self._service(rules={"allow": [ToolRule("read_file", "src/**")]})

        inside = await self._authorize(
            service,
            "read_file",
            {"path": "src/nested/a.ts"},
            targets=("src/nested/a.ts",),
        )
        outside = await self._authorize(
            service,
            "read_file",
            {"path": "docs/b.md"},
            targets=("docs/b.md",),
        )

        self.assertEqual("allow", inside)
        self.assertEqual("allow", outside)
        broker.request.assert_awaited_once()
        request = broker.request.await_args.args[0]
        self.assertIsInstance(request, PermissionRequest)
        self.assertEqual("read_file", request.name)
        self.assertEqual({"path": "docs/b.md"}, request.arguments)
        self.assertEqual(("docs/b.md",), request.targets)

    async def test_no_surface_tool_passes_without_any_rule(self) -> None:
        # permission_targets=None 的无权限面工具：不落进"未匹配即询问"的兜底。
        service, broker, _ = self._service()

        result = await self._authorize(service, "send_message", {})

        self.assertEqual("allow", result)
        broker.request.assert_not_awaited()

    async def test_scoped_rule_does_not_reach_no_surface_tool(self) -> None:
        # send_message 没有 target 概念：scoped 规则碰不到它，依旧默认放行。
        service, broker, _ = self._service(rules={"allow": [ToolRule("send_message", "boss")]})

        result = await self._authorize(service, "send_message", {})

        self.assertEqual("allow", result)
        broker.request.assert_not_awaited()

    async def test_multi_segment_allow_requires_every_segment(self) -> None:
        # bash 拆出的段必须全部命中 allow 才免问；一段缺规则整条进询问。
        service, broker, _ = self._service(rules={"allow": [ToolRule("bash", "git *")]})

        both_git = await self._authorize(service, "bash", {}, targets=("git pull", "git status"))
        self.assertEqual("allow", both_git)
        broker.request.assert_not_awaited()

        mixed = await self._authorize(service, "bash", {}, targets=("git status", "rm -rf x"))
        self.assertEqual("allow", mixed)  # broker mock 模拟人点了 allow
        broker.request.assert_awaited_once()
        self.assertEqual(
            ("git status", "rm -rf x"),
            broker.request.await_args.args[0].targets,
        )

    async def test_deny_on_any_segment_beats_otherwise_allowed_segments(self) -> None:
        service, broker, _ = self._service(
            rules={
                "allow": [ToolRule("bash", "git *")],
                "deny": [ToolRule("bash", "rm *")],
            }
        )

        result = await self._authorize(service, "bash", {}, targets=("git status", "rm -rf x"))

        self.assertEqual("reject", result)
        broker.request.assert_not_awaited()

    async def test_ask_on_any_segment_overrides_allowed_segments(self) -> None:
        service, broker, _ = self._service(
            rules={
                "allow": [ToolRule("bash", "git *")],
                "ask": [ToolRule("bash", "docker *")],
            }
        )

        result = await self._authorize(service, "bash", {}, targets=("git status", "docker ps"))

        self.assertEqual("allow", result)  # 人工过问后放行
        broker.request.assert_awaited_once()

    async def test_allow_reply_grants_tool_for_next_call(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PermissionManager(start=root)
            broker = MagicMock()
            broker.request = AsyncMock(return_value=_resolved("allow"))
            sessions = AsyncMock()
            sessions.get.return_value = Session.create(title="main")
            service = PermissionService(
                broker=broker,
                manager=manager,
                session_service=sessions,
            )

            first = await self._authorize(service, "edit_file", {}, targets=("notes.txt",))
            self.assertEqual("allow", first)

            broker.reply.return_value = InterruptionReplyStatus.ACCEPTED
            broker.pending.return_value = _permission_request(name="edit_file")
            await service.reply(permission_id="permission-1", decision="allow", scope=[""])

            broker.request.reset_mock()
            second = await self._authorize(service, "edit_file", {}, targets=("notes.txt",))

            self.assertEqual("allow", second)
            broker.request.assert_not_awaited()
            written = json.loads(
                (root / ".agent-desk" / "settings.json").read_text(encoding="utf-8")
            )
            self.assertEqual(["edit_file"], written["permissions"]["allow"])

    async def test_read_file_hook_extracts_absolute_target(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "a.py").write_text("x", encoding="utf-8")
            tool = ReadFileTool()

            self.assertEqual(
                ((root / "src" / "a.py").as_posix(),),
                tool.permission_targets({"path": str(root / "src" / "a.py")}),
            )
            # 非法 path 不再退回空段：直接抛参数错误，graph 在授权前拦下。
            for bad in ({"path": "src/a.py"}, {"path": "../outside.txt"}, {}):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        tool.permission_targets(bad)
            # 未声明钩子的工具落在协议默认 None 上(无权限面)。
            untouched = AskUserTool(AsyncMock())
            self.assertIsNone(untouched.permission_targets({"question": "continue?"}))

    async def test_metatool_target_is_mcp_server_granularity(self) -> None:
        """execute_mcp 的 target 是 server 标识，粒度到 server 不细化到 tool。

        缺失/空白的 mcp 由 schema 先验证挡掉(见 graph 层用例)，钩子不再抛错。
        """

        tool = ExecuteMcpTool(MagicMock())
        self.assertEqual(
            ("github",),
            tool.permission_targets({"mcp": " github ", "tool": "create_issue"}),
        )

    async def test_workspace_tools_share_path_target(self) -> None:
        for tool in (EditFileTool(), ReadFileTool()):
            with self.subTest(tool=tool.definition.name):
                self.assertEqual(
                    ("/etc/passwd",),
                    tool.permission_targets({"path": "/etc/passwd"}),
                )
                self.assertEqual(
                    ("/data/x.md",),
                    tool.permission_targets({"path": "/data/./x/../x.md"}),
                )
                with self.assertRaises(ValueError):
                    tool.permission_targets({"path": "docs/a.md"})

    async def test_graph_passes_tool_targets_into_authorize(self) -> None:
        permissions = AsyncMock()
        permissions.authorize.return_value = "allow"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("hi", encoding="utf-8")
            graph = AgentGraph(
                max_turns=4,
                settings=ContextSettings(),
                hook_registry=AgentHookRegistry(),
                permission_service=permissions,
            )

            await graph._execute_tool(
                {
                    "tools": [ReadFileTool()],
                    "caller_session_id": "session-1",
                },
                ToolCall(
                    id="call-1",
                    name="read_file",
                    arguments={"path": str(root / "notes.txt")},
                ),
            )

            permissions.authorize.assert_awaited_once_with(
                caller_session_id="session-1",
                tool_call_id="call-1",
                name="read_file",
                arguments={"path": str(root / "notes.txt")},
                targets=((root / "notes.txt").as_posix(),),
            )

    async def test_graph_intercepts_invalid_path_before_authorize(self) -> None:
        # 钩子抛 ValueError = 参数错误：不询问、不执行，直接回错误结果。
        permissions = AsyncMock()
        graph = AgentGraph(
            max_turns=4,
            settings=ContextSettings(),
            hook_registry=AgentHookRegistry(),
            permission_service=permissions,
        )

        result = json.loads(
            (
                await graph._execute_tool(
                    {"tools": [ReadFileTool()], "caller_session_id": "session-1"},
                    ToolCall(id="call-1", name="read_file", arguments={"path": "../secret"}),
                )
            ).content
        )

        self.assertEqual("invalid_tool_arguments", result["error"]["code"])
        permissions.authorize.assert_not_awaited()

    async def test_blank_mcp_rejected_by_schema_before_authorize(self) -> None:
        # 能进 schema 的约束(pattern \S)不进钩子：先验证再匹配。
        permissions = AsyncMock()
        graph = AgentGraph(
            max_turns=4,
            settings=ContextSettings(),
            hook_registry=AgentHookRegistry(),
            permission_service=permissions,
        )

        result = json.loads(
            (
                await graph._execute_tool(
                    {"tools": [ExecuteMcpTool(MagicMock())], "caller_session_id": "session-1"},
                    ToolCall(
                        id="call-1",
                        name="execute_mcp",
                        arguments={"mcp": "   ", "tool": "x", "arguments": {}},
                    ),
                )
            ).content
        )

        self.assertEqual("invalid_tool_arguments", result["error"]["code"])
        permissions.authorize.assert_not_awaited()

    async def test_graph_passes_none_targets_for_surfaceless_tool(self) -> None:
        permissions = AsyncMock()
        permissions.authorize.return_value = "reject"  # 拦在真正执行前
        graph = AgentGraph(
            max_turns=4,
            settings=ContextSettings(),
            hook_registry=AgentHookRegistry(),
            permission_service=permissions,
        )

        await graph._execute_tool(
            {"tools": [AskUserTool(AsyncMock())], "caller_session_id": "session-1"},
            ToolCall(
                id="call-1",
                name="ask_user",
                arguments={"question": "?", "options": ["yes", "no"]},
            ),
        )

        self.assertIsNone(permissions.authorize.await_args.kwargs["targets"])


class PermissionModeUseCaseTests(IsolatedAsyncioTestCase):
    """mode 用例：CLI 入口只给原始字符串，校验归 service，落盘归 manager。"""

    @staticmethod
    def _service(
        root: Path,
        *,
        mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> tuple[PermissionService, PermissionManager]:
        manager = PermissionManager(start=root, permission_rule={}, mode=mode)
        service = PermissionService(
            broker=AsyncMock(),
            manager=manager,
            session_service=AsyncMock(),
        )
        return service, manager

    async def test_change_mode_takes_effect_and_persists(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Path(directory) / ".agent-desk" / "settings.json"
            service, manager = self._service(Path(directory))

            self.assertEqual(PermissionMode.DONT_ASK, service.change_mode(" dont_ask "))
            self.assertIs(PermissionMode.DONT_ASK, manager.mode)
            self.assertIs(PermissionMode.DONT_ASK, service.current_mode())
            written = json.loads(settings.read_text(encoding="utf-8"))
            self.assertEqual("dont_ask", written["permissions"]["mode"])

    async def test_invalid_mode_rejected_without_touching_state(self) -> None:
        with TemporaryDirectory() as directory:
            service, manager = self._service(Path(directory), mode=PermissionMode.BYPASS)

            for bad in ("yolo", "", "DEFAULT"):
                with self.subTest(mode=bad):
                    with self.assertRaises(DomainValidationError):
                        service.change_mode(bad)
            self.assertIs(PermissionMode.BYPASS, manager.mode)
            self.assertFalse((Path(directory) / ".agent-desk" / "settings.json").exists())
