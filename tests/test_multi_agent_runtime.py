"""Unit and integration tests for the Multi-Agent subsystem."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import ANY, AsyncMock, MagicMock

from autogen_core import SingleThreadedAgentRuntime

from agent.domain.entities import Session
from agent.domain.messages import Message, MessageKind
from agent.domain.multi_agent import MAIN_AGENT_NAME, AgentMessage
from agent.domain.tools import ToolContext
from agent.metatools.define_subagent import DefineSubagentTool
from agent.metatools.execute_mcp import ExecuteMcpTool
from agent.metatools.execute_python import ExecutePythonTool
from agent.metatools.list_subagents import ListSubagentsTool
from agent.metatools.load_skill import LoadSkillTool
from agent.metatools.registry import MetaToolRegistry
from agent.metatools.search_mcp import SearchMcpTool
from agent.metatools.send_message import SendMessageTool
from agent.metatools.wait_for_replies import WaitForRepliesTool
from agent.ports.memory import MemoryServicePort
from agent.ports.model import AgentModelPort, AgentWorkflow
from agent.ports.runtime import AgentStreamHubPort
from agent.ports.services import MessageServicePort, SessionServicePort
from agent.ports.tools import MetaToolRegistryPort
from agent.runtime import AgentFactory, AgentRuntimeManager
from agent.runtime.event_publisher import RuntimeEventPublisher
from agent.runtime.stream_hub import StreamHub
from tests.runtime_scopes import FakeRuntimeScopes


def _make_msg(session_id: str, content: str, seq: int = 1) -> Message:
    return Message(
        id="msg-123",
        session_id=session_id,
        seq=seq,
        role="assistant",
        content=content,
        metadata={"kind": MessageKind.ASSISTANT_ANSWER},
    )


def _returning(session: Session) -> AsyncMock:
    """构造 get() 恒定返回指定会话的 SessionService 测试替身。"""
    service = AsyncMock()
    service.get.return_value = session
    return service


class _StubMetaTools:
    def get_tools(self, allowed_tools: tuple[str, ...] = ()) -> tuple[Any, ...]:
        return ()

    @property
    def tools(self) -> tuple[Any, ...]:
        return ()


def _build_subagent_scopes(
    *,
    coder_session: Session,
    assistant: Message,
) -> tuple[AsyncMock, FakeRuntimeScopes]:
    message_service = AsyncMock()
    session_service = AsyncMock()
    session_service.history.return_value = [assistant]
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "turns": 1,
        "persisted_assistant": assistant,
    }
    model = AsyncMock()
    memory = AsyncMock()
    scopes = FakeRuntimeScopes({
        MessageServicePort: message_service,
        SessionServicePort: session_service,
        MetaToolRegistryPort: _StubMetaTools(),
        AgentWorkflow: graph,
        AgentModelPort: model,
        MemoryServicePort: memory,
        AgentStreamHubPort: StreamHub(),
    })
    return session_service, scopes


class MultiAgentManagerTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.autogen_runtime = SingleThreadedAgentRuntime()

    def _manager(self, scopes: FakeRuntimeScopes) -> AgentRuntimeManager:
        return AgentRuntimeManager(
            runtime=self.autogen_runtime,
            factory=AgentFactory(scopes=scopes),
            scopes=scopes,
            publisher=scopes._services.get(
                RuntimeEventPublisher, RuntimeEventPublisher(self.autogen_runtime)
            ),
        )

    async def asyncTearDown(self) -> None:
        if self.autogen_runtime._run_context is not None:
            await self.autogen_runtime.stop_when_idle()

    async def test_send_message_executes_in_subagent_scope(self) -> None:
        main_sess = Session.create(title="main")
        coder_sess = Session.create(
            title=f"{main_sess.id}_subagent_coder", main_session_id=main_sess.id
        )
        assistant = _make_msg(coder_sess.id, "def binary_search(): pass")

        session_service, scopes = _build_subagent_scopes(
            coder_session=coder_sess, assistant=assistant
        )
        session_service.get.side_effect = lambda session_id: (
            coder_sess if session_id == coder_sess.id else main_sess
        )
        session_service.list_by_main_session.return_value = [coder_sess]

        manager = self._manager(scopes)

        with self.assertRaises(ValueError) as raised:
            await manager.send_message(
                AgentMessage(
                    sender=MAIN_AGENT_NAME,
                    recipient="coder",
                    content="Please write binary search",
                    session_id=main_sess.id,
                )
            )
        self.assertIn("dispatch", str(raised.exception))

    async def test_dispatch_to_undefined_subagent_raises_error_with_available(self) -> None:
        main_sess = Session.create(title="main")
        coder_sess = Session.create(
            title=f"{main_sess.id}_subagent_coder", main_session_id=main_sess.id
        )
        assistant = _make_msg(coder_sess.id, "unused")

        session_service, scopes = _build_subagent_scopes(
            coder_session=coder_sess, assistant=assistant
        )
        session_service.get.return_value = main_sess
        session_service.list_by_main_session.return_value = [coder_sess]

        manager = self._manager(scopes)

        with self.assertRaises(ValueError) as raised:
            await manager.dispatch(
                AgentMessage(
                    sender=MAIN_AGENT_NAME,
                    recipient="non_existent",
                    content="hello",
                    session_id=main_sess.id,
                )
            )
        self.assertIn("coder", str(raised.exception))

    async def test_send_without_session_envelope_is_rejected(self) -> None:
        manager = self._manager(FakeRuntimeScopes())
        with self.assertRaises(ValueError):
            await manager.send_message(
                AgentMessage(sender="user", recipient=MAIN_AGENT_NAME, content="hi", session_id="")
            )


class DefineSubagentToolTests(IsolatedAsyncioTestCase):
    async def test_define_subagent_creates_titled_session(self) -> None:
        root_sess = Session.create(title="root")
        mock_session_service = AsyncMock()
        mock_session_service.get.return_value = root_sess
        created = Session.create(title=f"{root_sess.id}_subagent_coder")
        mock_session_service.create.return_value = created

        tool = DefineSubagentTool(session_service=mock_session_service)

        args = {
            "name": "coder",
            "description": "Software engineer subagent",
            "system_prompt": "You are a senior python engineer.",
            "allowed_tools": ["execute_python", "execute_mcp"],
        }
        res = await tool.aexecute(
            args, context=ToolContext(caller_session_id=root_sess.id)
        )

        self.assertIn('"ok": true', res.content)
        self.assertIn("coder", res.content)
        mock_session_service.create.assert_awaited_once_with(
            title=f"{root_sess.id}_subagent_coder",
            main_session_id=root_sess.id,
            allowed_tools=("execute_python", "execute_mcp"),
            custom_system_prompt=ANY,
        )

    async def test_define_subagent_requires_name(self) -> None:
        tool = DefineSubagentTool(session_service=AsyncMock())
        res = await tool.aexecute(
            {"name": "", "description": "d", "system_prompt": "p"},
            context=ToolContext(caller_session_id="s-1"),
        )
        self.assertIn('"ok": false', res.content)


class SendMessageToolTests(IsolatedAsyncioTestCase):
    async def test_dispatch_is_fire_and_forget_with_agent_identity(self) -> None:
        mock_runtime = AsyncMock()
        main_session = Session.create(title="Main Session")
        send_tool = SendMessageTool(
            runtime=mock_runtime, session_service=_returning(main_session)
        )

        res = await send_tool.aexecute(
            {
                "recipient": "coder",
                "message": "Please implement add function",
            },
            context=ToolContext(caller_session_id=main_session.id),
        )

        self.assertIn('"ok": true', res.content)
        self.assertIn('"status": "dispatched"', res.content)
        called_msg: AgentMessage = mock_runtime.dispatch.call_args[0][0]
        self.assertEqual(MAIN_AGENT_NAME, called_msg.sender)
        self.assertEqual("coder", called_msg.recipient)
        self.assertEqual(main_session.id, called_msg.session_id)

    async def test_subagent_report_carries_subagent_identity(self) -> None:
        mock_runtime = AsyncMock()
        sub_session = Session.create(
            title="owner-1_subagent_coder", main_session_id="owner-1"
        )
        send_tool = SendMessageTool(
            runtime=mock_runtime, session_service=_returning(sub_session)
        )

        res = await send_tool.aexecute(
            {
                "recipient": MAIN_AGENT_NAME,
                "message": "I finished the sorting algorithm.",
            },
            context=ToolContext(caller_session_id=sub_session.id),
        )

        self.assertIn('"ok": true', res.content)
        called_msg: AgentMessage = mock_runtime.dispatch.call_args[0][0]
        self.assertEqual("coder", called_msg.sender)
        self.assertEqual(MAIN_AGENT_NAME, called_msg.recipient)
        # Bug A 回归：子代理回报必须路由到归属主会话，而不是它自己的会话。
        self.assertEqual("owner-1", called_msg.session_id)

    async def test_subagent_cannot_message_another_subagent(self) -> None:
        mock_runtime = AsyncMock()
        sub_session = Session.create(
            title="owner-1_subagent_coder", main_session_id="owner-1"
        )
        send_tool = SendMessageTool(
            runtime=mock_runtime, session_service=_returning(sub_session)
        )

        res = await send_tool.aexecute(
            {"recipient": "researcher", "message": "peer ping"},
            context=ToolContext(caller_session_id=sub_session.id),
        )

        self.assertIn('"ok": false', res.content)
        self.assertIn("star topology", res.content)
        mock_runtime.dispatch.assert_not_awaited()

    async def test_execution_failure_is_reported_as_tool_error(self) -> None:
        mock_runtime = AsyncMock()
        mock_runtime.dispatch.side_effect = RuntimeError("boom")
        main_session = Session.create(title="Main Session")
        tool = SendMessageTool(
            runtime=mock_runtime, session_service=_returning(main_session)
        )

        res = await tool.aexecute(
            {"recipient": "coder", "message": "hi"},
            context=ToolContext(caller_session_id=main_session.id),
        )
        self.assertIn('"ok": false', res.content)
        self.assertIn("boom", res.content)


class EndToEndSubagentToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.autogen_runtime = SingleThreadedAgentRuntime()

    def _manager(self, scopes: FakeRuntimeScopes) -> AgentRuntimeManager:
        return AgentRuntimeManager(
            runtime=self.autogen_runtime,
            factory=AgentFactory(scopes=scopes),
            scopes=scopes,
            publisher=scopes._services.get(
                RuntimeEventPublisher, RuntimeEventPublisher(self.autogen_runtime)
            ),
        )

    async def asyncTearDown(self) -> None:
        if self.autogen_runtime._run_context is not None:
            await self.autogen_runtime.stop_when_idle()

    async def test_send_message_tool_drives_subagent_session(self) -> None:
        main_sess = Session.create(title="main")
        coder_sess = Session.create(
            title=f"{main_sess.id}_subagent_coder", main_session_id=main_sess.id
        )
        assistant = _make_msg(coder_sess.id, "class QuickSort: ...")

        session_service, scopes = _build_subagent_scopes(
            coder_session=coder_sess, assistant=assistant
        )
        session_service.get.side_effect = lambda session_id: (
            coder_sess if session_id == coder_sess.id else main_sess
        )
        session_service.list_by_main_session.return_value = [coder_sess]
        scopes.register(RuntimeEventPublisher, RuntimeEventPublisher(self.autogen_runtime))

        manager = self._manager(scopes)
        message_service = scopes._services[MessageServicePort]

        send_tool = SendMessageTool(
            runtime=manager, session_service=session_service
        )
        res = await send_tool.aexecute(
            {
                "recipient": "coder",
                "message": "Write quicksort in Python",
            },
            context=ToolContext(caller_session_id=main_sess.id),
        )

        # 工具立即返回投递确认；stop() 会排空在途事件处理。
        self.assertIn('"ok": true', res.content)
        self.assertIn('"status": "dispatched"', res.content)
        await manager.stop()
        self.assertGreaterEqual(message_service.add.await_count, 1)


class FullAsyncRoundTripTests(IsolatedAsyncioTestCase):
    """端到端：ask → 主控派发并等待 → 子代理事件回报 → 主控收尾返回。"""

    def setUp(self) -> None:
        self.autogen_runtime = SingleThreadedAgentRuntime()

    def _manager(self, scopes: FakeRuntimeScopes) -> AgentRuntimeManager:
        return AgentRuntimeManager(
            runtime=self.autogen_runtime,
            factory=AgentFactory(scopes=scopes),
            scopes=scopes,
            publisher=scopes._services[RuntimeEventPublisher],
        )

    async def asyncTearDown(self) -> None:
        if self.autogen_runtime._run_context is not None:
            await self.autogen_runtime.stop_when_idle()

    async def test_ask_returns_final_answer_after_subagent_report(self) -> None:
        from agent.application.services.agent_service import AgentService

        main_sess = Session.create(title="main")
        coder_sess = Session.create(
            title=f"{main_sess.id}_subagent_coder", main_session_id=main_sess.id
        )

        interim = _make_msg(main_sess.id, "Dispatch planned.", seq=2)
        report = _make_msg(coder_sess.id, "Report: quicksort done.", seq=3)
        final = _make_msg(main_sess.id, "Final answer from coder's report.", seq=4)

        report_arrived = asyncio.Event()

        message_service = AsyncMock()
        message_service.add.return_value = interim
        session_service = AsyncMock()
        session_service.get.side_effect = lambda sid: (
            coder_sess if sid == coder_sess.id else main_sess
        )
        session_service.list_by_main_session.return_value = [coder_sess]
        session_service.history.return_value = [interim, report, final]

        turn_counts: dict[str, int] = {main_sess.id: 0, coder_sess.id: 0}

        async def _graph(**kwargs):
            sid = kwargs["caller_session_id"]
            turn_counts[sid] += 1
            if sid == main_sess.id and turn_counts[sid] == 1:
                # 首轮：非阻塞派发任务，随后模拟 wait_for_replies 阻塞到回报到来。
                await manager.dispatch(
                    AgentMessage(
                        sender=MAIN_AGENT_NAME,
                        recipient="coder",
                        content="write quicksort",
                        session_id=main_sess.id,
                    )
                )
                await asyncio.wait_for(report_arrived.wait(), timeout=5.0)
                # 回报已持久化；下一轮 before_model 会读到它。这里直接给出最终回答。
                return {"turns": 2, "persisted_assistant": final}
            if sid == coder_sess.id:
                await manager.dispatch(
                    AgentMessage(
                        sender="coder",
                        recipient=MAIN_AGENT_NAME,
                        content=report.content,
                        session_id=main_sess.id,
                    )
                )
                report_arrived.set()
                return {"turns": 1, "persisted_assistant": report}
            return {"turns": 1, "persisted_assistant": interim}

        graph = AsyncMock()
        graph.ainvoke.side_effect = _graph

        scopes = FakeRuntimeScopes({
            MessageServicePort: message_service,
            SessionServicePort: session_service,
            MetaToolRegistryPort: _StubMetaTools(),
            AgentWorkflow: graph,
            AgentModelPort: AsyncMock(),
            MemoryServicePort: AsyncMock(),
            AgentStreamHubPort: StreamHub(),
        })
        scopes.register(RuntimeEventPublisher, RuntimeEventPublisher(self.autogen_runtime))
        manager = self._manager(scopes)
        service = AgentService(runtime=manager)

        answer = await asyncio.wait_for(
            service.ask(session_id=main_sess.id, question="build quicksort", sender="user"),
            timeout=10.0,
        )

        self.assertEqual("Final answer from coder's report.", answer.message.content)
        # 核心语义：一个 request = 一次主控 graph.invoke；回报事件不会触发第二次。
        self.assertEqual(1, turn_counts[main_sess.id])
        self.assertEqual(1, turn_counts[coder_sess.id])


class ListSubagentsToolTests(IsolatedAsyncioTestCase):
    async def test_lists_bare_names_from_db_sessions(self) -> None:
        main_sess = Session.create(title="main")
        coder_sess = Session.create(
            title=f"{main_sess.id}_subagent_coder", main_session_id=main_sess.id
        )
        reviewer_sess = Session.create(
            title=f"{main_sess.id}_subagent_reviewer", main_session_id=main_sess.id
        )
        session_service = AsyncMock()
        session_service.get.return_value = main_sess
        session_service.list_by_main_session.return_value = [coder_sess, reviewer_sess]

        list_tool = ListSubagentsTool(session_service=session_service)
        list_res = await list_tool.aexecute(
            {}, context=ToolContext(caller_session_id=main_sess.id)
        )

        self.assertIn('"ok": true', list_res.content)
        self.assertIn('"total": 2', list_res.content)
        self.assertIn("coder", list_res.content)
        self.assertIn("reviewer", list_res.content)

    async def test_lists_bare_names_by_caller_session(self) -> None:
        """caller_session_id 决定查询范围（主控专用工具，caller 即主会话）。"""
        main_sess = Session.create(title="main")
        coder_sess = Session.create(
            title=f"{main_sess.id}_subagent_coder", main_session_id=main_sess.id
        )
        session_service = AsyncMock()
        session_service.get.return_value = main_sess
        session_service.list_by_main_session.return_value = [coder_sess]

        list_tool = ListSubagentsTool(session_service=session_service)
        await list_tool.aexecute({}, context=ToolContext(caller_session_id=main_sess.id))
        session_service.list_by_main_session.assert_awaited_with(main_sess.id)


def _build_registry() -> MetaToolRegistry:
    session_service = AsyncMock()
    return MetaToolRegistry(
        tools=[
            SearchMcpTool(MagicMock()),
            LoadSkillTool(MagicMock()),
            ExecuteMcpTool(MagicMock()),
            ExecutePythonTool(),
            DefineSubagentTool(session_service=session_service),
            SendMessageTool(runtime=AsyncMock(), session_service=session_service),
            ListSubagentsTool(session_service=session_service),
            WaitForRepliesTool(
                session_service=session_service,
                default_timeout=60,
                max_timeout=600,
            ),
        ]
    )


class MetaToolRegistryTests(IsolatedAsyncioTestCase):
    def test_registry_has_all_8_meta_tools(self) -> None:
        tool_names = [t.definition.name for t in _build_registry().tools]
        for expected in (
            "search_mcp",
            "execute_mcp",
            "load_skill",
            "execute_python",
            "define_subagent",
            "send_message",
            "list_subagents",
            "wait_for_replies",
        ):
            self.assertIn(expected, tool_names)

    def test_get_tools_filters_by_allowed_tools_whitelist(self) -> None:
        granted = _build_registry().get_tools(("execute_python", "send_message"))
        granted_names = [t.definition.name for t in granted]
        self.assertEqual(["execute_python", "send_message"], granted_names)


class StarTopologyRulesTests(IsolatedAsyncioTestCase):
    def test_subagent_cannot_define_another_subagent(self) -> None:
        registry = _build_registry()

        subagent_allowed = (
            "execute_python",
            "search_mcp",
            "execute_mcp",
            "load_skill",
            "send_message",
        )
        granted_names = [t.definition.name for t in registry.get_tools(subagent_allowed)]

        self.assertNotIn("define_subagent", granted_names)
        self.assertNotIn("list_subagents", granted_names)


__all__ = [
    "DefineSubagentToolTests",
    "EndToEndSubagentToolTests",
    "ListSubagentsToolTests",
    "MetaToolRegistryTests",
    "MultiAgentManagerTests",
    "SendMessageToolTests",
    "StarTopologyRulesTests",
]
