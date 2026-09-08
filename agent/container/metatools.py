"""Meta Tool 组合 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, collect, provide

from agent.domain.tools import AgentTool
from agent.infrastructure.settings import AgentRuntimeSettings, MemorySettings
from agent.metatools import MetaToolRegistry
from agent.metatools.ask_user import AskUserTool
from agent.metatools.define_subagent import DefineSubagentTool
from agent.metatools.execute_mcp import ExecuteMcpTool
from agent.metatools.execute_python import ExecutePythonTool
from agent.metatools.list_subagents import ListSubagentsTool
from agent.metatools.load_skill import LoadSkillTool
from agent.metatools.search_mcp import SearchMcpTool
from agent.metatools.search_memory import SearchMemoryTool
from agent.metatools.send_message import SendMessageTool
from agent.metatools.wait_for_replies import WaitForRepliesTool
from agent.ports.memory import MemoryServicePort
from agent.ports.services import SessionServicePort
from agent.ports.tools import (
    AgentRuntimePort,
    McpRegistryPort,
    MetaToolRegistryPort,
    SkillCatalogPort,
)


class MetaToolProvider(Provider):
    """把每个 AgentTool 作为独立依赖提供，并收集成 list[AgentTool]。"""

    tools = collect(AgentTool, scope=Scope.REQUEST)

    @provide(scope=Scope.APP)
    def provide_search_mcp(self, registry: McpRegistryPort) -> AgentTool:
        return SearchMcpTool(registry)

    @provide(scope=Scope.APP)
    def provide_load_skill(self, catalog: SkillCatalogPort) -> AgentTool:
        return LoadSkillTool(catalog)

    @provide(scope=Scope.APP)
    def provide_execute_mcp(self, registry: McpRegistryPort) -> AgentTool:
        return ExecuteMcpTool(registry)

    @provide(scope=Scope.APP)
    def provide_execute_python(self) -> AgentTool:
        return ExecutePythonTool()

    @provide(scope=Scope.REQUEST)
    def provide_define_subagent(self, session_service: SessionServicePort) -> AgentTool:
        return DefineSubagentTool(session_service=session_service)

    @provide(scope=Scope.REQUEST)
    def provide_send_message(
        self,
        runtime: AgentRuntimePort,
        session_service: SessionServicePort,
    ) -> AgentTool:
        return SendMessageTool(runtime=runtime, session_service=session_service)

    @provide(scope=Scope.REQUEST)
    def provide_list_subagents(self, session_service: SessionServicePort) -> AgentTool:
        return ListSubagentsTool(session_service=session_service)

    # wait 轮询要查库,与所注入的 REQUEST 作用域 SessionService 同域共生;
    # 绝不可留在 APP(那才是真并发风险面)。
    @provide(scope=Scope.REQUEST)
    def provide_wait_for_replies(
        self,
        settings: AgentRuntimeSettings,
        session_service: SessionServicePort,
    ) -> AgentTool:
        return WaitForRepliesTool(
            session_service=session_service,
            default_timeout=settings.wait_default_timeout_seconds,
            max_timeout=settings.wait_max_timeout_seconds,
        )

    @provide(scope=Scope.APP)
    def provide_ask_user(self) -> AgentTool:
        return AskUserTool()

    @provide(scope=Scope.REQUEST)
    def provide_search_memory(
        self,
        memory_service: MemoryServicePort,
        session_service: SessionServicePort,
        memory_settings: MemorySettings,
    ) -> AgentTool:
        return SearchMemoryTool(
            memory_service=memory_service,
            settings=memory_settings,
            session_service=session_service,
        )

    @provide(scope=Scope.REQUEST)
    def provide_meta_tool_registry(
        self,
        tools: list[AgentTool],
    ) -> MetaToolRegistryPort:
        return MetaToolRegistry(tools)


__all__ = ["MetaToolProvider"]
