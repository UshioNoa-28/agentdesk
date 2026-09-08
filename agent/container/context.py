"""上下文投影、压缩与工作流 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.application.context.compactor import ContextCompactor
from agent.application.context.manager import ContextManager
from agent.application.context.projector import (
    MessageContextProjector,
    SummaryContextProjector,
)
from agent.application.persistence.manager import PersistenceManager
from agent.graph.agent_graph import AgentGraph
from agent.graph.hook_registry import TurnHooks
from agent.graph.hooks import HookRegistryPort
from agent.infrastructure.settings import (
    AgentRuntimeSettings,
    ContextSettings,
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
    SessionServicePort,
    TokenCounterServicePort,
)
from agent.ports.tools import MetaToolRegistryPort


class ContextProvider(Provider):
    """提供上下文投影、压缩、持久化和 LangGraph 工作流。"""

    @provide(scope=Scope.APP)
    def provide_turn_hooks(self) -> HookRegistryPort:
        """提供空钩子注册表；中间件或测试可通过它注入自定义钩子。"""

        return TurnHooks()

    @provide(scope=Scope.REQUEST)
    def provide_graph(
        self,
        runtime_settings: AgentRuntimeSettings,
        hook_registry: HookRegistryPort,
        context_manager: ContextManagerPort,
        persistence_manager: PersistenceManagerPort,
        session_service: SessionServicePort,
    ) -> AgentWorkflow:
        """提供 LangGraph 工作流。"""

        return AgentGraph(
            max_turns=runtime_settings.max_turns,
            hook_registry=hook_registry,
            context_manager=context_manager,
            persistence_manager=persistence_manager,
            session_service=session_service,
        )

    @provide(scope=Scope.APP)
    def provide_context_projector(
        self,
        token_counter_service: TokenCounterServicePort,
        context_settings: ContextSettings,
    ) -> ContextProjectorPort:
        """提供带普通工具结果 token 上限的 Message 投影端口。"""

        return MessageContextProjector(
            token_counter=token_counter_service,
            max_tool_result_tokens=context_settings.context_max_tool_result_tokens,
            clear_tool_result_threshold_tokens=(
                context_settings.context_clear_tool_result_threshold_tokens
            ),
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
            max_tool_result_tokens=context_settings.context_max_tool_result_tokens,
            clear_tool_result_threshold_tokens=(
                context_settings.context_clear_tool_result_threshold_tokens
            ),
        )

    @provide(scope=Scope.REQUEST)
    def provide_context_compactor(
        self,
        projector: ContextProjectorPort,
        summary_projector: SummaryProjectorPort,
        token_counter_service: TokenCounterServicePort,
        model: AgentModelPort,
        context_settings: ContextSettings,
    ) -> ContextCompactorPort:
        """提供只负责 Compact 的组件。"""

        return ContextCompactor(
            projector=projector,
            summary_projector=summary_projector,
            token_counter=token_counter_service,
            model=model,
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
        context_settings: ContextSettings,
        session_service: SessionServicePort,
        meta_tools: MetaToolRegistryPort,
    ) -> ContextManagerPort:
        """组装上下文管理与压缩用例。"""

        return ContextManager(
            message_service=message_service,
            projector=projector,
            compactor=compactor,
            token_counter=token_counter_service,
            settings=context_settings,
            session_service=session_service,
            meta_tools=meta_tools,
        )


__all__ = ["ContextProvider"]
