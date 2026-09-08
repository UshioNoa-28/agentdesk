from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from autogen_core import SingleThreadedAgentRuntime

from agent.application.context.compactor import ContextCompactor
from agent.application.context.manager import ContextManager
from agent.application.context.projector import (
    MessageContextProjector,
    SummaryContextProjector,
)
from agent.application.persistence.manager import PersistenceManager
from agent.application.services.agent_service import AgentService
from agent.application.services.message_service import MessageService
from agent.application.services.session_service import SessionService
from agent.application.services.token_counter_service import TokenCounterService
from agent.domain.entities import Session
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage, ModelTurn, ToolCall
from agent.exceptions import AgentExecutionError
from agent.graph.agent_graph import AgentGraph, AgentGraphExecutionError
from agent.graph.hook_registry import AgentHookRegistry
from agent.infrastructure.model import ModelProviderError
from agent.infrastructure.settings import ContextSettings
from agent.ports.memory import MemoryServicePort
from agent.ports.model import AgentModelPort, AgentWorkflow
from agent.ports.runtime import AgentStreamHubPort
from agent.ports.services import MessageServicePort, SessionServicePort
from agent.ports.tools import MetaToolRegistryPort
from agent.runtime import AgentFactory, AgentRuntimeManager
from agent.runtime.event_publisher import RuntimeEventPublisher
from agent.runtime.stream_hub import StreamHub
from tests.runtime_scopes import FakeRuntimeScopes
from tests.tokenizer_support import budget_token_counter


class _FakeUnitOfWork:
    async def __aenter__(self) -> "_FakeUnitOfWork":
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def commit(self) -> None:
        return None


class _Sessions:
    def __init__(self, session: Session) -> None:
        self.items = {session.id: session}

    async def add(self, session: Session) -> None:
        self.items[session.id] = session

    async def get(self, session_id: str):
        return self.items.get(session_id)

    async def get_for_update(self, session_id: str):
        return self.items.get(session_id)

    async def list_by_main_session(self, main_session_id: str) -> list[Session]:
        return [s for s in self.items.values() if s.main_session_id == main_session_id]

    async def save(self, session: Session):
        self.items[session.id] = session
        return session


class _Messages:
    def __init__(self) -> None:
        self.items: dict[str, list[Message]] = {}

    async def add(self, message: Message) -> Message:
        self.items.setdefault(message.session_id, []).append(message)
        return message

    async def seed(self, message: Message) -> None:
        self.items.setdefault(message.session_id, []).append(message)

    async def list_for_session(self, session_id: str, *, max_seq: int | None = None):
        messages = sorted(self.items.get(session_id, ()), key=lambda message: message.seq)
        return [message for message in messages if max_seq is None or message.seq <= max_seq]

    async def last_seq(self, session_id: str) -> int:
        return max((message.seq for message in self.items.get(session_id, ())), default=0)

    async def delete_for_session(self, session_id: str) -> int:
        return len(self.items.pop(session_id, []))


class _FakeMcpRegistry:
    def server_descriptions(self) -> tuple[tuple[str, str], ...]:
        return ()


class _FakeSkillCatalog:
    def metadata(self) -> tuple:
        return ()


class _MetaTools:
    tools = ()

    def get_tools(self, allowed_tools=()):
        allowed_set = set(allowed_tools or ())
        return tuple(t for t in self.tools if t.definition.name in allowed_set)


class _AnswerGraph:
    def __init__(self) -> None:
        self.tools = ()
        self.hook_registry: AgentHookRegistry | None = None
        self.persistence_manager: PersistenceManager | None = None

    async def ainvoke(
        self,
        *,
        messages,
        model,
        tools,
        caller_session_id=None,
        events=None,
    ):
        self.tools = tuple(tools)
        assert self.hook_registry is not None
        assert self.persistence_manager is not None
        state = {
            "messages": list(messages),
            "model": model,
            "tools": tools,
            "caller_session_id": caller_session_id,
        }
        for hook in self.hook_registry.before_model_hooks():
            update = await hook(state)
            if update:
                state.update(update)
        answer = ModelMessage.assistant(content="done")
        state["messages"].append(answer)
        state["persisted_assistant"] = await self.persistence_manager.persist_assistant(
            session_id=caller_session_id,
            turn=ModelTurn(message=answer, tool_calls=()),
        )
        for hook in self.hook_registry.after_model_hooks():
            update = await hook(state)
            if update:
                state.update(update)
        return {
            "messages": state["messages"],
            "turns": 1,
            "persisted_assistant": state.get("persisted_assistant"),
        }


class _FailingGraph:
    async def ainvoke(self, **_kwargs):
        # 上游 provider 失败：异常链携带 ModelProviderError，入口层应落 502。
        raise AgentGraphExecutionError("graph failed") from ModelProviderError(
            "provider unavailable"
        )


class _ServiceFixture:
    def __init__(
        self,
        graph,
        *,
        model=None,
        hook_registry=None,
    ) -> None:
        self.session = Session.create(title="test")
        self.sessions = _Sessions(self.session)
        self.messages = _Messages()
        self.mcp_registry = _FakeMcpRegistry()
        self.skill_catalog = _FakeSkillCatalog()
        self.meta_tools = _MetaTools()
        self.hook_registry = hook_registry or AgentHookRegistry()
        if isinstance(graph, _AnswerGraph):
            graph.hook_registry = self.hook_registry
        self.session_service = SessionService(
            repository=self.sessions,
            messages=self.messages,
            unit_of_work=_FakeUnitOfWork(),
            mcp_registry=self.mcp_registry,
            skill_catalog=self.skill_catalog,
        )
        # Seed the root system message for the session
        self.messages.items[self.session.id] = [
            Message.create(
                session_id=self.session.id,
                seq=1,
                message=ModelMessage.system("You are an AI agent."),
                metadata={"kind": MessageKind.SYSTEM},
            )
        ]
        message_service = MessageService(
            repository=self.messages,
            session_repository=self.sessions,
            unit_of_work=_FakeUnitOfWork(),
            session_service=self.session_service,
        )
        counter = TokenCounterService(budget_token_counter())
        projector = MessageContextProjector(
            token_counter=counter,
            max_tool_result_tokens=8_000,
            clear_tool_result_threshold_tokens=100,
        )
        settings = ContextSettings(
            context_window_tokens=200_000,
            context_reserve_ratio=0.1,
        )
        compactor = ContextCompactor(
            projector=projector,
            summary_projector=SummaryContextProjector(
                token_counter=counter,
                max_tool_result_tokens=8_000,
                clear_tool_result_threshold_tokens=100,
            ),
            token_counter=counter,
            model=object(),
            settings=settings,
        )
        context_manager = ContextManager(
            message_service=message_service,
            projector=projector,
            compactor=compactor,
            token_counter=counter,
            settings=settings,
            session_service=self.session_service,
            meta_tools=self.meta_tools,
        )
        persistence_manager = PersistenceManager(
            message_service=message_service,
            projector=projector,
        )
        if isinstance(graph, _AnswerGraph):
            graph.persistence_manager = persistence_manager
        if isinstance(graph, AgentGraph):
            graph._context_manager = context_manager
            graph._persistence_manager = persistence_manager
        self.autogen_runtime = SingleThreadedAgentRuntime()
        self.scopes = FakeRuntimeScopes({
            MessageServicePort: message_service,
            SessionServicePort: self.session_service,
            MetaToolRegistryPort: self.meta_tools,
            AgentWorkflow: graph,
            AgentModelPort: model or object(),
            MemoryServicePort: _MemoryStub(),
            RuntimeEventPublisher: RuntimeEventPublisher(self.autogen_runtime),
        AgentStreamHubPort: StreamHub(),
        })
        self.runtime = AgentRuntimeManager(
            runtime=self.autogen_runtime,
            factory=AgentFactory(scopes=self.scopes),
            scopes=self.scopes,
            publisher=self.scopes._services[RuntimeEventPublisher],
        )
        self.service = AgentService(runtime=self.runtime)

    async def stop(self) -> None:
        if self.autogen_runtime._run_context is not None:
            await self.autogen_runtime.stop_when_idle()


class _MemoryStub:
    async def add(self, *, messages, user_id) -> None:
        return None


class AgentServiceMessageTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._fixtures = []

    def _fixture(self, *args, **kwargs) -> _ServiceFixture:
        fixture = _ServiceFixture(*args, **kwargs)
        self._fixtures.append(fixture)
        return fixture

    async def asyncTearDown(self) -> None:
        for fixture in self._fixtures:
            await fixture.stop()

    async def test_success_persists_user_and_final_assistant_messages(self) -> None:
        fixture = self._fixture(_AnswerGraph())

        answer = await fixture.service.ask(
            session_id=fixture.session.id, question="hello", sender="user"
        )
        messages = fixture.messages.items[fixture.session.id]

        self.assertEqual("assistant", answer.message.role)
        self.assertEqual(["system", "human", "assistant"], [message.role for message in messages])
        self.assertEqual([1, 2, 3], [message.seq for message in messages])
        self.assertEqual(0, len(fixture.hook_registry.before_model_hooks()))
        self.assertEqual(0, len(fixture.hook_registry.after_model_hooks()))
        self.assertEqual(0, len(fixture.hook_registry.after_tool_hooks()))

    async def test_provider_failure_keeps_user_message_without_failure_message(self) -> None:
        fixture = self._fixture(_FailingGraph())

        with self.assertRaises(AgentExecutionError) as raised:
            await fixture.service.ask(
                session_id=fixture.session.id, question="hello", sender="user"
            )

        messages = fixture.messages.items[fixture.session.id]
        self.assertEqual(["system", "human"], [message.role for message in messages])
        self.assertEqual(fixture.session.id, raised.exception.details["session_id"])

    async def test_missing_tool_result_is_left_for_projection_not_persistence(self) -> None:
        call = ToolCall(id="call-1", name="execute_mcp", arguments={"mcp": "rag"})
        fixture = self._fixture(_AnswerGraph())
        old = Message.create(
            session_id=fixture.session.id,
            seq=2,
            message=ModelMessage.assistant(content="", tool_calls=(call,)),
            metadata={"kind": MessageKind.ASSISTANT_TOOL_CALL},
        )
        await fixture.messages.seed(old)

        await fixture.service.ask(session_id=fixture.session.id, question="continue", sender="user")
        messages = fixture.messages.items[fixture.session.id]

        # 持久化层不再自动补齐 synthetic tool result；协议缺口由投影层修复。
        self.assertEqual(
            ["system", "assistant", "human", "assistant"],
            [message.role for message in messages],
        )
        self.assertNotIn("synthetic", messages[2].metadata)

    async def test_graph_receives_only_the_fixed_meta_tools(self) -> None:
        graph = _AnswerGraph()
        fixture = self._fixture(graph)

        await fixture.service.ask(session_id=fixture.session.id, question="hello", sender="user")

        self.assertEqual((), graph.tools)

    async def test_real_graph_runs_explicit_context_and_persistence_steps(self) -> None:
        registry = AgentHookRegistry()
        graph = AgentGraph(max_turns=4, hook_registry=registry)
        fixture = self._fixture(
            graph,
            model=_OneTurnAnswerModel(),
            hook_registry=registry,
        )

        answer = await fixture.service.ask(
            session_id=fixture.session.id,
            question="hello through the real graph",
            sender="user",
        )

        self.assertEqual("assistant", answer.message.role)
        self.assertEqual("real graph answer", answer.message.content)
        messages = fixture.messages.items[fixture.session.id]
        self.assertEqual(["system", "human", "assistant"], [item.role for item in messages])
        self.assertEqual([1, 2, 3], [item.seq for item in messages])

    async def test_agent_service_ask_does_not_rebuild_prompt(self) -> None:
        fixture = self._fixture(_AnswerGraph())
        initial_system = fixture.messages.items[fixture.session.id][0]

        for round_idx in range(5):
            await fixture.service.ask(
                session_id=fixture.session.id,
                question=f"question {round_idx}",
                sender="user",
            )

        current_messages = fixture.messages.items[fixture.session.id]
        self.assertEqual(initial_system.content, current_messages[0].content)
        self.assertEqual("system", current_messages[0].role)


class _OneTurnAnswerModel:
    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        assert messages[0].role == "system"
        assert messages[-1].role == "human"
        return ModelTurn(
            message=ModelMessage.assistant(content="real graph answer"),
            tool_calls=(),
        )
