"""Meta Tool 组合 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, collect, provide

from agent.domain.tools import AgentTool
from agent.infrastructure.metatools import MetaToolRegistry
from agent.infrastructure.metatools.ask_user import AskUserTool
from agent.infrastructure.metatools.bash import BashTool
from agent.infrastructure.metatools.define_subagent import DefineSubagentTool
from agent.infrastructure.metatools.execute_mcp import ExecuteMcpTool
from agent.infrastructure.metatools.forget import ForgetTool
from agent.infrastructure.metatools.list_subagents import ListSubagentsTool
from agent.infrastructure.metatools.load_skill import LoadSkillTool
from agent.infrastructure.metatools.memory import (
    DeleteNoteTool,
    ReadIndexTool,
    ReadNoteTool,
    UpdateNoteTool,
    WriteNoteTool,
)
from agent.infrastructure.metatools.remember import RememberTool
from agent.infrastructure.metatools.search_mcp import SearchMcpTool
from agent.infrastructure.metatools.send_message import SendMessageTool
from agent.infrastructure.metatools.wait_for_replies import WaitForRepliesTool
from agent.infrastructure.metatools.workspace import (
    EditFileTool,
    ReadFileTool,
    SearchTextTool,
)
from agent.infrastructure.settings import AgentRuntimeSettings
from agent.ports.runtime.interruption import InterruptionBrokerPort
from agent.ports.runtime.status import AgentStatusRegistryPort
from agent.ports.services import MemoryNoteBookPort, MemoryStorePort, SessionServicePort
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
    def provide_read_file(self) -> AgentTool:
        return ReadFileTool()

    @provide(scope=Scope.APP)
    def provide_search_text(self) -> AgentTool:
        return SearchTextTool()

    @provide(scope=Scope.APP)
    def provide_edit_file(self) -> AgentTool:
        return EditFileTool()

    @provide(scope=Scope.APP)
    def provide_bash(self) -> AgentTool:
        return BashTool()

    # 记忆写入只追加自己的日志，没有权限面；store 在建会话前就按
    # memory.enabled 定好，关闭时工具仍在，只在写入时报错。
    @provide(scope=Scope.APP)
    def provide_remember(self, store: MemoryStorePort) -> AgentTool:
        return RememberTool(store)

    @provide(scope=Scope.APP)
    def provide_forget(self, store: MemoryStorePort) -> AgentTool:
        return ForgetTool(store)

    # 整理者的笔记工具与普通 Meta Tool 同表注册：会话工具表是
    # get_tools(session.allowed_tools) 按名字筛出来的，而 MEMORY_TOOL_NAMES 刻意
    # 不在 ALL_META_TOOL_NAMES 里，所以主 Agent 与子代理都筛不到这几个。
    @provide(scope=Scope.APP)
    def provide_read_index(self, notebook: MemoryNoteBookPort) -> AgentTool:
        return ReadIndexTool(notebook)

    @provide(scope=Scope.APP)
    def provide_read_note(self, notebook: MemoryNoteBookPort) -> AgentTool:
        return ReadNoteTool(notebook)

    @provide(scope=Scope.APP)
    def provide_write_note(
        self,
        store: MemoryStorePort,
        notebook: MemoryNoteBookPort,
    ) -> AgentTool:
        return WriteNoteTool(store, notebook)

    @provide(scope=Scope.APP)
    def provide_update_note(
        self,
        store: MemoryStorePort,
        notebook: MemoryNoteBookPort,
    ) -> AgentTool:
        return UpdateNoteTool(store, notebook)

    @provide(scope=Scope.APP)
    def provide_delete_note(
        self,
        store: MemoryStorePort,
        notebook: MemoryNoteBookPort,
    ) -> AgentTool:
        return DeleteNoteTool(store, notebook)

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
    def provide_list_subagents(
        self,
        session_service: SessionServicePort,
        status_registry: AgentStatusRegistryPort,
    ) -> AgentTool:
        return ListSubagentsTool(
            session_service=session_service,
            status_registry=status_registry,
        )

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
    def provide_ask_user(self, broker: InterruptionBrokerPort) -> AgentTool:
        return AskUserTool(broker)

    @provide(scope=Scope.REQUEST)
    def provide_meta_tool_registry(
        self,
        tools: list[AgentTool],
    ) -> MetaToolRegistryPort:
        return MetaToolRegistry(tools)


__all__ = ["MetaToolProvider"]
