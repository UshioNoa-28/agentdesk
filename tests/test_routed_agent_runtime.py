"""Unit tests for the RoutedAgent actor, AgentFactory, and AgentRuntimeManager."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from autogen_core import AgentId, MessageContext, SingleThreadedAgentRuntime

from agent.application.services.agent_service import AgentService
from agent.domain.entities import Session
from agent.domain.exceptions import DomainValidationError
from agent.domain.messages import Message, MessageKind
from agent.domain.multi_agent import (
    MAIN_AGENT_NAME,
    AgentMessage,
    AgentStatus,
    Answered,
    Queued,
)
from agent.exceptions import AgentExecutionError, AgentInternalError
from agent.graph.agent_graph import AgentGraphExecutionError
from agent.infrastructure.model import ModelProviderError
from agent.ports.memory import MemoryServicePort
from agent.ports.model import AgentModelPort, AgentWorkflow
from agent.ports.runtime import AgentStreamHubPort
from agent.ports.services import MessageServicePort, SessionServicePort
from agent.ports.tools import AgentRuntimePort, MetaToolRegistryPort
from agent.runtime import (
    AgentFactory,
    AgentRuntimeManager,
    RoutedAgent,
)
from agent.runtime.event_publisher import RuntimeEventPublisher
from agent.runtime.stream_hub import StreamHub
from tests.runtime_scopes import FakeRuntimeScopes


def _assistant_message(session_id: str, content: str, *, seq: int = 2) -> Message:
    return Message(
        id="msg-final",
        session_id=session_id,
        seq=seq,
        role="assistant",
        content=content,
        metadata={"kind": MessageKind.ASSISTANT_ANSWER},
    )


def _rpc_ctx(message: AgentMessage) -> MessageContext:
    """复刻 ``runtime.send_message`` 交给 handler 的那份上下文（RPC 语义）。"""

    return MessageContext(
        sender=AgentId(type=message.sender, key=message.session_id),
        topic_id=None,
        is_rpc=True,
        cancellation_token=None,
        message_id=message.id,
    )


class _StubMetaTools:
    def get_tools(self, allowed_tools: tuple[str, ...] = ()) -> tuple[Any, ...]:
        return ()

    @property
    def tools(self) -> tuple[Any, ...]:
        return ()


def _build_agent_services(
    *,
    session: Session,
    assistant: Message,
    graph_delay: float = 0.0,
    fail_graph: bool = False,
    sent_to_main: bool = False,
) -> tuple[AsyncMock, AsyncMock, AsyncMock, FakeRuntimeScopes]:
    message_service = AsyncMock()
    session_service = AsyncMock()
    session_service.get.return_value = session
    session_service.history.return_value = [assistant]
    meta_tools = _StubMetaTools()

    graph = AsyncMock()

    async def _ainvoke(**_kwargs: Any) -> dict[str, Any]:
        if graph_delay:
            await asyncio.sleep(graph_delay)
        if fail_graph:
            raise RuntimeError("graph exploded")
        return {
            "turns": 1,
            "persisted_assistant": assistant,
            "sent_to_main": sent_to_main,
        }

    graph.ainvoke.side_effect = _ainvoke

    model = AsyncMock()
    memory = AsyncMock()
    scopes = FakeRuntimeScopes({
        MessageServicePort: message_service,
        SessionServicePort: session_service,
        MetaToolRegistryPort: meta_tools,
        AgentWorkflow: graph,
        AgentModelPort: model,
        MemoryServicePort: memory,
        RuntimeEventPublisher: AsyncMock(spec=RuntimeEventPublisher),
        AgentStreamHubPort: StreamHub(),
    })
    return message_service, session_service, graph, scopes


class AgentServiceRoutingTests(IsolatedAsyncioTestCase):
    async def test_ask_wraps_question_into_agent_message_for_main_agent(self) -> None:
        expected = _assistant_message("sess-1", "Processed answer")
        runtime = AsyncMock(spec=AgentRuntimePort)
        runtime.send_message.return_value = Answered(message=expected)
        service = AgentService(runtime=runtime)
        res = await service.ask(session_id="sess-1", question="What is 2+2?", sender="alice")

        self.assertEqual("Processed answer", res.message.content)
        called_msg: AgentMessage = runtime.send_message.call_args[0][0]
        self.assertEqual("What is 2+2?", called_msg.content)
        self.assertEqual("alice", called_msg.sender)
        self.assertEqual(MAIN_AGENT_NAME, called_msg.recipient)
        self.assertEqual("sess-1", called_msg.session_id)

    async def test_ask_rejects_empty_question(self) -> None:
        service = AgentService(runtime=AsyncMock())
        with self.assertRaises(DomainValidationError):
            await service.ask(session_id="sess-1", question="   ", sender="user")


class RoutedAgentActorTests(IsolatedAsyncioTestCase):
    async def test_rpc_message_persists_and_returns_final_answer(self) -> None:
        session = Session.create(title="Main Test Session")
        assistant = _assistant_message(session.id, "Main agent response")
        message_service, _, graph, scopes = _build_agent_services(
            session=session, assistant=assistant
        )

        agent = RoutedAgent(name="main_agent", session_id=session.id, scopes=scopes)

        self.assertEqual(AgentStatus.IDLE, agent.status)
        request = AgentMessage(
            content="Hello",
            sender="user",
            recipient=MAIN_AGENT_NAME,
            session_id=session.id,
        )
        outcome = await agent.on_message(request, _rpc_ctx(request))

        assert isinstance(outcome, Answered)
        self.assertEqual("Main agent response", outcome.message.content)
        self.assertEqual(AgentStatus.IDLE, agent.status)
        message_service.add.assert_awaited_once()
        graph.ainvoke.assert_awaited_once()

    async def test_graph_receives_caller_session_id(self) -> None:
        """工具上下文的会话来源收敛为 caller_session_id（执行者自己的会话）。"""
        session = Session.create(title="owner-1_subagent_coder", main_session_id="owner-1")
        assistant = _assistant_message(session.id, "done")
        _, _, graph, scopes = _build_agent_services(session=session, assistant=assistant)

        agent = RoutedAgent(name="subagent", session_id=session.id, scopes=scopes)
        request = AgentMessage(
            content="task",
            sender=MAIN_AGENT_NAME,
            recipient="coder",
            session_id=session.id,
        )
        await agent.on_message(request, _rpc_ctx(request))

        self.assertEqual(
            session.id,
            graph.ainvoke.call_args.kwargs["caller_session_id"],
        )
        self.assertNotIn("runtime", graph.ainvoke.call_args.kwargs)

    async def test_subagent_auto_delivers_result_when_not_reported(self) -> None:
        """子代理跑完图却没 send_message 给主控时，最终文本被自动兜底投递。"""

        session = Session.create(title="owner-1_subagent_coder", main_session_id="owner-1")
        assistant = _assistant_message(session.id, "the answer")
        _, _, _, scopes = _build_agent_services(
            session=session, assistant=assistant, sent_to_main=False
        )

        agent = RoutedAgent(name="subagent", session_id=session.id, scopes=scopes)
        request = AgentMessage(
            content="task",
            sender=MAIN_AGENT_NAME,
            recipient="coder",
            session_id=session.id,
        )
        outcome = await agent.on_message(request, _rpc_ctx(request))

        assert isinstance(outcome, Answered)
        publisher = scopes._services[RuntimeEventPublisher]
        publisher.publish.assert_awaited_once()
        published = publisher.publish.call_args.args[0]
        self.assertEqual("the answer", published.content)
        self.assertEqual("coder", published.sender)
        self.assertEqual(MAIN_AGENT_NAME, published.recipient)
        self.assertEqual("owner-1", published.session_id)

    async def test_subagent_does_not_redeliver_when_already_reported(self) -> None:
        """子代理已 send_message 给主控（sent_to_main=True）时不重复投递。"""

        session = Session.create(title="owner-1_subagent_coder", main_session_id="owner-1")
        assistant = _assistant_message(session.id, "the answer")
        _, _, _, scopes = _build_agent_services(
            session=session, assistant=assistant, sent_to_main=True
        )

        agent = RoutedAgent(name="subagent", session_id=session.id, scopes=scopes)
        request = AgentMessage(
            content="task",
            sender=MAIN_AGENT_NAME,
            recipient="coder",
            session_id=session.id,
        )
        outcome = await agent.on_message(request, _rpc_ctx(request))

        assert isinstance(outcome, Answered)
        publisher = scopes._services[RuntimeEventPublisher]
        publisher.publish.assert_not_awaited()

    async def test_main_agent_never_auto_delivers(self) -> None:
        """主控（main_session_id 为 None）不走兜底投递路径。"""

        session = Session.create(title="Main Test Session")
        assistant = _assistant_message(session.id, "answer")
        _, _, _, scopes = _build_agent_services(
            session=session, assistant=assistant, sent_to_main=False
        )

        agent = RoutedAgent(name="main_agent", session_id=session.id, scopes=scopes)
        request = AgentMessage(
            content="hello",
            sender="user",
            recipient=MAIN_AGENT_NAME,
            session_id=session.id,
        )
        outcome = await agent.on_message(request, _rpc_ctx(request))

        assert isinstance(outcome, Answered)
        publisher = scopes._services[RuntimeEventPublisher]
        publisher.publish.assert_not_awaited()

    async def test_running_agent_accepts_second_rpc_as_queued(self) -> None:
        session = Session.create(title="Busy Session")
        assistant = _assistant_message(session.id, "slow answer")
        message_service, _, graph, scopes = _build_agent_services(
            session=session, assistant=assistant, graph_delay=0.05
        )

        agent = RoutedAgent(name="main_agent", session_id=session.id, scopes=scopes)

        first_request = AgentMessage(
            content="first",
            sender="user",
            recipient=MAIN_AGENT_NAME,
            session_id=session.id,
        )
        second_request = AgentMessage(
            content="second",
            sender="user",
            recipient=MAIN_AGENT_NAME,
            session_id=session.id,
        )
        first = asyncio.create_task(
            agent.on_message(first_request, _rpc_ctx(first_request))
        )
        await asyncio.sleep(0.01)
        self.assertEqual(AgentStatus.RUNNING, agent.status)

        # RPC 调用方在等回复：本轮不会有回复，但消息必须已经入库。
        outcome = await agent.on_message(second_request, _rpc_ctx(second_request))

        self.assertEqual(Queued(session_id=session.id), outcome)
        self.assertEqual(2, message_service.add.await_count)
        self.assertEqual(1, graph.ainvoke.call_count)
        first_outcome = await first
        assert isinstance(first_outcome, Answered)
        self.assertEqual("slow answer", first_outcome.message.content)

    async def test_running_agent_still_accepts_events_as_none(self) -> None:
        session = Session.create(
            title="main-session_subagent_coder",
            main_session_id="main-session",
        )
        assistant = _assistant_message(session.id, "slow answer")
        message_service, _, _, scopes = _build_agent_services(
            session=session, assistant=assistant, graph_delay=0.05
        )

        agent = RoutedAgent(name="coder", session_id=session.id, scopes=scopes)
        ctx = MessageContext(
            sender=AgentId(type=MAIN_AGENT_NAME, key="main-session"),
            topic_id=None,
            is_rpc=False,
            cancellation_token=None,
            message_id="evt-1",
        )

        first = asyncio.create_task(
            agent.on_message(
                AgentMessage(
                    content="do the work",
                    sender=MAIN_AGENT_NAME,
                    recipient="coder",
                    session_id=session.id,
                ),
                ctx,
            )
        )
        await asyncio.sleep(0.01)
        self.assertEqual(AgentStatus.RUNNING, agent.status)

        # 事件投递（非 RPC）以 None 表示"已接收"；publish 方忽略返回值，绝不能抛。
        reply = await agent.on_message(
            AgentMessage(
                content="clarifying follow-up",
                sender=MAIN_AGENT_NAME,
                recipient="coder",
                session_id=session.id,
            ),
            ctx,
        )

        self.assertIsNone(reply)
        self.assertEqual(2, message_service.add.await_count)
        await first

    async def test_failure_marks_agent_error_and_wraps_exception(self) -> None:
        session = Session.create(title="Broken Session")
        assistant = _assistant_message(session.id, "unused")
        _, _, _, scopes = _build_agent_services(
            session=session, assistant=assistant, fail_graph=True
        )

        agent = RoutedAgent(name="main_agent", session_id=session.id, scopes=scopes)

        request = AgentMessage(
            content="boom",
            sender="user",
            recipient=MAIN_AGENT_NAME,
            session_id=session.id,
        )
        with self.assertRaises(AgentInternalError):
            await agent.on_message(request, _rpc_ctx(request))
        self.assertEqual(AgentStatus.ERROR, agent.status)

    async def test_upstream_failure_maps_to_agent_execution_error(self) -> None:
        """异常链上出现 ModelProviderError → AgentExecutionError（502 语义）。"""
        session = Session.create(title="Upstream Session")
        assistant = _assistant_message(session.id, "unused")
        _, _, graph, scopes = _build_agent_services(session=session, assistant=assistant)

        async def _upstream_failure(**_kwargs: Any) -> dict[str, Any]:
            raise AgentGraphExecutionError("graph failed") from ModelProviderError(
                "provider unavailable"
            )

        graph.ainvoke.side_effect = _upstream_failure
        agent = RoutedAgent(name="main_agent", session_id=session.id, scopes=scopes)

        request = AgentMessage(
            content="boom",
            sender="user",
            recipient=MAIN_AGENT_NAME,
            session_id=session.id,
        )
        with self.assertRaises(AgentExecutionError):
            await agent.on_message(request, _rpc_ctx(request))


class RuntimeManagerEndToEndTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.autogen_runtime = SingleThreadedAgentRuntime()

    async def asyncTearDown(self) -> None:
        if self.autogen_runtime._run_context is not None:
            await self.autogen_runtime.stop_when_idle()

    def _manager(self, scopes: FakeRuntimeScopes) -> AgentRuntimeManager:
        return AgentRuntimeManager(
            runtime=self.autogen_runtime,
            factory=AgentFactory(scopes=scopes),
            scopes=scopes,
            publisher=scopes._services.get(
                RuntimeEventPublisher, RuntimeEventPublisher(self.autogen_runtime)
            ),
        )

    async def test_main_agent_round_trip_through_autogen_runtime(self) -> None:
        main_session = Session.create(title="main")
        assistant = _assistant_message(main_session.id, "final answer")
        _, _, _, scopes = _build_agent_services(session=main_session, assistant=assistant)
        manager = self._manager(scopes)
        reply = await manager.send_message(
            AgentMessage(
                content="hello",
                sender="user",
                recipient=MAIN_AGENT_NAME,
                session_id=main_session.id,
            )
        )

        assert isinstance(reply, Answered)
        self.assertEqual("final answer", reply.message.content)

    async def test_rpc_into_running_session_returns_queued(self) -> None:
        """第二发 ask 穿过真实 runtime 必须原样拿到 Queued，不被边界包成别的东西。"""

        main_session = Session.create(title="main")
        assistant = _assistant_message(main_session.id, "final answer")
        message_service, _, _, scopes = _build_agent_services(
            session=main_session, assistant=assistant, graph_delay=0.2
        )
        manager = self._manager(scopes)

        first = asyncio.create_task(
            manager.send_message(
                AgentMessage(
                    content="first",
                    sender="user",
                    recipient=MAIN_AGENT_NAME,
                    session_id=main_session.id,
                )
            )
        )
        await asyncio.sleep(0.05)
        self.assertEqual(1, message_service.add.await_count)

        outcome = await manager.send_message(
            AgentMessage(
                content="second",
                sender="user",
                recipient=MAIN_AGENT_NAME,
                session_id=main_session.id,
            )
        )

        # 被排队的那一发仍然入库，并且第一轮照常收尾。
        self.assertEqual(Queued(session_id=main_session.id), outcome)
        self.assertEqual(2, message_service.add.await_count)
        first_outcome = await first
        assert isinstance(first_outcome, Answered)
        self.assertEqual("final answer", first_outcome.message.content)

    async def test_dispatch_publishes_event_to_subagent_topic(self) -> None:
        main_session = Session.create(title="main")
        coder_session = Session.create(
            title=f"{main_session.id}_subagent_coder", main_session_id=main_session.id
        )
        assistant = _assistant_message(coder_session.id, "unused")
        _, session_service, _, scopes = _build_agent_services(
            session=coder_session, assistant=assistant
        )
        session_service.get.side_effect = lambda session_id: (
            coder_session if session_id == coder_session.id else main_session
        )
        session_service.list_by_main_session.return_value = [coder_session]
        publisher = scopes._services[RuntimeEventPublisher]
        manager = self._manager(scopes)
        # 手动启动（不经过 stop 的用例自行停机）
        await manager._ensure_started()
        try:
            await manager.dispatch(
                AgentMessage(
                    content="write quicksort",
                    sender=MAIN_AGENT_NAME,
                    recipient="coder",
                    session_id=main_session.id,
                )
            )
            publisher.publish.assert_awaited_once()
            self.assertIn(coder_session.id, str(publisher.publish.call_args))
        finally:
            await manager.stop()

    async def test_unknown_recipient_raises_with_available_names(self) -> None:
        main_session = Session.create(title="main")
        coder_session = Session.create(
            title=f"{main_session.id}_subagent_coder", main_session_id=main_session.id
        )
        assistant = _assistant_message(coder_session.id, "unused")
        _, session_service, _, scopes = _build_agent_services(
            session=coder_session, assistant=assistant
        )
        session_service.get.return_value = main_session
        session_service.list_by_main_session.return_value = [coder_session]

        manager = self._manager(scopes)

        with self.assertRaises(ValueError) as raised:
            await manager.dispatch(
                AgentMessage(
                    content="hi",
                    sender=MAIN_AGENT_NAME,
                    recipient="researcher",
                    session_id=main_session.id,
                )
            )
        self.assertIn("coder", str(raised.exception))

    async def test_send_message_requires_session_id_envelope(self) -> None:
        manager = self._manager(FakeRuntimeScopes())
        with self.assertRaises(ValueError):
            await manager.send_message(
                AgentMessage(
                    content="hi",
                    sender="user",
                    recipient=MAIN_AGENT_NAME,
                    session_id="",
                )
            )
