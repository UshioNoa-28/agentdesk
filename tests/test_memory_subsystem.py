"""Unit and integration tests for Mem0 Long-term Memory subsystem & SearchMemoryTool."""

from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from autogen_core import SingleThreadedAgentRuntime

from agent.application.services.agent_service import AgentService
from agent.domain.entities import Session
from agent.domain.messages import Message, MessageKind
from agent.domain.tools import ALL_META_TOOL_NAMES, ToolContext
from agent.infrastructure.memory import Mem0MemoryService
from agent.infrastructure.settings import MemorySettings
from agent.metatools.registry import MetaToolRegistry
from agent.metatools.search_memory import SearchMemoryTool
from agent.ports.memory import MemoryServicePort
from agent.ports.model import AgentModelPort, AgentWorkflow
from agent.ports.runtime import AgentStreamHubPort
from agent.ports.services import MessageServicePort, SessionServicePort
from agent.ports.tools import MetaToolRegistryPort
from agent.runtime import AgentFactory, AgentRuntimeManager
from agent.runtime.event_publisher import RuntimeEventPublisher
from agent.runtime.stream_hub import StreamHub
from tests.runtime_scopes import FakeRuntimeScopes


class _StubMetaTools:
    def get_tools(self, allowed_tools: tuple[str, ...] = ()) -> tuple:
        return ()

    @property
    def tools(self) -> tuple:
        return ()


class MemorySubsystemTests(IsolatedAsyncioTestCase):
    async def test_session_user_id_default_and_custom(self) -> None:
        session_default = Session.create(title="Session 1")
        self.assertEqual("default_user", session_default.user_id)

        session_custom = Session.create(title="Session 2", user_id="alice")
        self.assertEqual("alice", session_custom.user_id)

    async def test_mem0_service_disabled_by_default(self) -> None:
        settings = MemorySettings(memory_enabled=False)
        service = Mem0MemoryService(settings=settings)

        self.assertEqual([], await service.search("python", user_id="default_user"))
        self.assertEqual([], await service.get_all(user_id="default_user"))
        # add and delete should complete silently without error
        await service.add("some messages", user_id="default_user")
        await service.delete("mem-123")

    async def test_mem0_service_enabled_with_mock_client(self) -> None:
        settings = MemorySettings(
            memory_enabled=True,
            memory_llm_provider="ollama",
            memory_embedder_provider="ollama",
            memory_embedder_dims=1024,
        )
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [
                {"id": "1", "memory": "User prefers Python 3.13"},
                {"id": "2", "memory": "User is based in Hangzhou"},
            ]
        }
        mock_client.get_all.return_value = [{"id": "1", "memory": "User prefers Python 3.13"}]
        mock_client.add.return_value = {"results": [{"event": "ADD"}]}
        mock_client.delete.return_value = None

        service = Mem0MemoryService(settings=settings, client=mock_client)

        memories = await service.search("code", user_id="alice", limit=2)
        self.assertEqual(["User prefers Python 3.13", "User is based in Hangzhou"], memories)
        mock_client.search.assert_awaited_once_with(
            query="code", filters={"user_id": "alice"}, top_k=2
        )

        all_mems = await service.get_all(user_id="alice")
        self.assertEqual(1, len(all_mems))
        mock_client.get_all.assert_awaited_once_with(filters={"user_id": "alice"})

        await service.add([{"role": "user", "content": "I like FastAPI"}], user_id="alice")
        mock_client.add.assert_awaited_once()
        self.assertEqual("alice", mock_client.add.call_args.kwargs.get("user_id"))

        await service.delete("1")
        mock_client.delete.assert_awaited_once_with("1")

    async def test_search_memory_tool_execution(self) -> None:
        mock_memory = AsyncMock()
        mock_memory.search.return_value = [
            "User prefers Clean Architecture",
            "User uses PostgreSQL",
        ]
        mock_session_svc = AsyncMock()
        session = Session.create(title="Session 1", user_id="alice")
        mock_session_svc.get.return_value = session

        tool = SearchMemoryTool(
            memory_service=mock_memory,
            settings=MemorySettings(),
            session_service=mock_session_svc,
        )

        self.assertEqual("search_memory", tool.definition.name)

        # Execute search with context
        result = await tool.aexecute(
            {"query": "architecture preferences", "limit": 2},
            context=ToolContext(caller_session_id=session.id),
        )

        mock_session_svc.get.assert_awaited_once_with(session.id)
        mock_memory.search.assert_awaited_once_with(
            query="architecture preferences",
            user_id="alice",
            limit=2,
        )
        self.assertIn("Found 2 long-term memory facts", result.content)
        self.assertIn("User prefers Clean Architecture", result.content)

    async def test_search_memory_tool_no_results(self) -> None:
        mock_memory = AsyncMock()
        mock_memory.search.return_value = []
        tool = SearchMemoryTool(memory_service=mock_memory, settings=MemorySettings())

        result = await tool.aexecute(
            {"query": "unknown"},
            context=ToolContext(caller_session_id="s-1"),
        )
        self.assertIn("No matching long-term memories found", result.content)

    async def test_meta_tool_registry_includes_search_memory(self) -> None:
        mock_session_svc = AsyncMock()
        mock_memory_svc = AsyncMock()

        registry = MetaToolRegistry(
            tools=[
                SearchMemoryTool(
                    memory_service=mock_memory_svc,
                    settings=MemorySettings(),
                    session_service=mock_session_svc,
                )
            ]
        )

        tool_names = [t.definition.name for t in registry.tools]
        self.assertIn("search_memory", tool_names)
        self.assertIn("search_memory", ALL_META_TOOL_NAMES)

    async def test_agent_service_spawns_memory_add_only_for_main_agent(self) -> None:
        mock_message_svc = AsyncMock()
        mock_graph = AsyncMock()
        mock_model = AsyncMock()
        mock_memory_svc = AsyncMock()

        main_session = Session.create(
            title="Main Session",
            main_session_id=None,
            user_id="bob",
        )
        session_service = AsyncMock()
        session_service.get.return_value = main_session
        session_service.history.return_value = []
        session_service.list_by_main_session.return_value = []

        assistant_msg = Message(
            id="msg-1",
            session_id=main_session.id,
            seq=2,
            role="assistant",
            content="Here is your answer",
            metadata={"kind": MessageKind.ASSISTANT_ANSWER},
        )
        mock_graph.ainvoke.return_value = {
            "turns": 1,
            "persisted_assistant": assistant_msg,
        }
        session_service.history.return_value = [assistant_msg]

        scopes = FakeRuntimeScopes({
            MessageServicePort: mock_message_svc,
            SessionServicePort: session_service,
            MetaToolRegistryPort: _StubMetaTools(),
            AgentWorkflow: mock_graph,
            AgentModelPort: mock_model,
            MemoryServicePort: mock_memory_svc,
        AgentStreamHubPort: StreamHub(),
        })
        autogen_runtime = SingleThreadedAgentRuntime()
        scopes.register(RuntimeEventPublisher, RuntimeEventPublisher(autogen_runtime))
        try:
            runtime = AgentRuntimeManager(
                runtime=autogen_runtime,
                factory=AgentFactory(scopes=scopes),
                scopes=scopes,
                publisher=scopes._services[RuntimeEventPublisher],
            )
            agent_service = AgentService(runtime=runtime)

            res = await agent_service.ask(
                session_id=main_session.id,
                question="Tell me a joke",
                sender="user",
            )
            self.assertEqual("Here is your answer", res.message.content)

            # Allow background asyncio.create_task to run
            await asyncio.sleep(0.01)
            mock_memory_svc.add.assert_called_once_with(
                messages=[
                    {"role": "user", "content": "Tell me a joke"},
                    {"role": "assistant", "content": "Here is your answer"},
                ],
                user_id="bob",
            )
        finally:
            if autogen_runtime._run_context is not None:
                await autogen_runtime.stop_when_idle()

    async def test_mem0_service_logs_to_file(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_memory.log"
            settings = MemorySettings(
                memory_enabled=True,
                memory_llm_provider="ollama",
                memory_embedder_provider="ollama",
                memory_embedder_dims=1024,
                memory_log_path=str(log_file),
            )

            mock_client = AsyncMock()
            mock_client.search.return_value = {"results": [{"memory": "Likes RPG games"}]}
            mock_client.add.return_value = {
                "results": [{"memory": "Likes RPG games", "event": "ADD"}]
            }

            service = Mem0MemoryService(settings=settings, client=mock_client)
            await service.add("I love RPG games", user_id="user_1")
            await service.search("game", user_id="user_1")

            self.assertTrue(log_file.exists())
            content = log_file.read_text(encoding="utf-8")
            self.assertIn("MEMORY_ADD", content)
            self.assertIn("MEMORY_SEARCH", content)
            self.assertIn("user_1", content)

