"""上下文投影、压缩与工作流 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.application.context.compactor import ContextCompactor
from agent.application.context.manager import ContextManager
from agent.application.context.projector import (
    MessageContextProjector,
    SummaryContextProjector,
)
from agent.application.graph.agent_graph import AgentGraph
from agent.application.graph.hook_registry import TurnHooks
from agent.application.graph.hooks import HookRegistryPort
from agent.application.persistence.manager import PersistenceManager
from agent.domain.context_settings import ContextSettings
from agent.infrastructure.model import ModelCatalog
from agent.infrastructure.settings import (
    AgentRuntimeSettings,
)
from agent.ports.context import (
    ContextCompactorPort,
    ContextManagerPort,
    ContextProjectorPort,
    PersistenceManagerPort,
    SummaryProjectorPort,
)
from agent.ports.model import AgentModelPort, AgentWorkflow
from agent.ports.services import (
    MessageServicePort,
    PermissionServicePort,
    SessionServicePort,
    TokenCounterServicePort,
)
from agent.ports.tools import (
    McpRegistryPort,
    MemoryCatalogPort,
    MetaToolRegistryPort,
    SkillCatalogPort,
)


class ContextProvider(Provider):
    """提供上下文投影、压缩、持久化和 LangGraph 工作流。"""

    @provide(scope=Scope.APP)
    def provide_turn_hooks(self) -> HookRegistryPort:
        """提供冻结的空 Hook 集合作为生产默认（无 Hook 运行）。

        ``TurnHooks`` 构造即不可变，无法逐条登记；要挂载 Hook 需在本 provider 上
        覆盖成填充好的 ``TurnHooks``，测试则直接向 ``AgentGraph`` 传入可变的
        ``AgentHookRegistry``。
        """

        return TurnHooks()

    @provide(scope=Scope.APP)
    def provide_context_settings(self) -> ContextSettings:
        """提供上下文层调度参数：代码内定，不进 settings.json。"""

        return ContextSettings()

    @provide(scope=Scope.REQUEST)
    def provide_graph(
        self,
        runtime_settings: AgentRuntimeSettings,
        context_settings: ContextSettings,
        hook_registry: HookRegistryPort,
        context_manager: ContextManagerPort,
        persistence_manager: PersistenceManagerPort,
        permission_service: PermissionServicePort,
        session_service: SessionServicePort,
    ) -> AgentWorkflow:
        """提供 LangGraph 工作流。"""

        return AgentGraph(
            max_turns=runtime_settings.max_turns,
            hook_registry=hook_registry,
            context_manager=context_manager,
            persistence_manager=persistence_manager,
            permission_service=permission_service,
            session_service=session_service,
            settings=context_settings,
        )

    @provide(scope=Scope.APP)
    def provide_context_projector(
        self,
        token_counter_service: TokenCounterServicePort,
        context_settings: ContextSettings,
    ) -> ContextProjectorPort:
        """提供带普通工具结果 token 上限的 Message 投影端口；上限是全局调度参数。"""

        return MessageContextProjector(
            token_counter=token_counter_service,
            settings=context_settings,
        )

    @provide(scope=Scope.APP)
    def provide_summary_projector(
        self,
        token_counter_service: TokenCounterServicePort,
        context_settings: ContextSettings,
    ) -> SummaryProjectorPort:
        """提供摘要视角投影端口：cold clearing 永不生效，其余行为一致。"""

        return SummaryContextProjector(
            token_counter=token_counter_service,
            settings=context_settings,
        )

    @provide(scope=Scope.REQUEST)
    def provide_context_compactor(
        self,
        projector: ContextProjectorPort,
        summary_projector: SummaryProjectorPort,
        token_counter_service: TokenCounterServicePort,
        model: AgentModelPort,
        catalog: ModelCatalog,
        context_settings: ContextSettings,
    ) -> ContextCompactorPort:
        """提供只负责 Compact 的组件；摘要与聊天共用同一模型实例，跟随当前 profile。"""

        return ContextCompactor(
            projector=projector,
            summary_projector=summary_projector,
            token_counter=token_counter_service,
            model=model,
            catalog=catalog,
            settings=context_settings,
        )

    @provide(scope=Scope.REQUEST)
    def provide_persistence_manager(
        self,
        message_service: MessageServicePort,
        projector: ContextProjectorPort,
    ) -> PersistenceManagerPort:
        """提供 Assistant 和 Tool 消息持久化管理器。"""

        return PersistenceManager(
            message_service=message_service,
            projector=projector,
        )

    @provide(scope=Scope.REQUEST)
    def provide_context_manager(
        self,
        message_service: MessageServicePort,
        projector: ContextProjectorPort,
        compactor: ContextCompactorPort,
        token_counter_service: TokenCounterServicePort,
        catalog: ModelCatalog,
        context_settings: ContextSettings,
        session_service: SessionServicePort,
        meta_tools: MetaToolRegistryPort,
        mcp_registry: McpRegistryPort,
        skill_catalog: SkillCatalogPort,
        memory_catalog: MemoryCatalogPort,
    ) -> ContextManagerPort:
        """组装上下文管理与压缩用例；预算随当前模型每次现取，默认 system
        prompt 每轮用活的 MCP/Skill/工具授权/记忆目录现拼，不落库。"""

        return ContextManager(
            message_service=message_service,
            projector=projector,
            compactor=compactor,
            token_counter=token_counter_service,
            catalog=catalog,
            settings=context_settings,
            session_service=session_service,
            meta_tools=meta_tools,
            mcp_registry=mcp_registry,
            skill_catalog=skill_catalog,
            memory_catalog=memory_catalog,
        )


__all__ = ["ContextProvider"]
