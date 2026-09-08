"""进程内模型适配器 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.infrastructure.model import LangChainAgentModel, LangChainToolAdapter
from agent.infrastructure.settings import (
    ChatModelSettings,
    ContextSettings,
    ObservabilitySettings,
)
from agent.ports.model import AgentModelPort


class AgentModelProvider(Provider):
    """提供进程内 LangChain 模型端口。"""

    @provide(scope=Scope.APP)
    def provide_tool_adapter(self) -> LangChainToolAdapter:
        """提供将工具 schema 绑定到 LangChain 的适配器。

        Returns:
            LangChainToolAdapter: 只负责 schema 包装、不执行 Agent 工具的适配器。
        """

        return LangChainToolAdapter()

    @provide(scope=Scope.APP)
    def provide_agent_model(
        self,
        settings: ChatModelSettings,
        context_settings: ContextSettings,
        tool_adapter: LangChainToolAdapter,
        observability: ObservabilitySettings,
    ) -> AgentModelPort:
        """直接装配进程内聊天模型，不经过任何内部 HTTP 服务。"""

        return LangChainAgentModel(
            settings=settings,
            context_settings=context_settings,
            tool_adapter=tool_adapter,
            observability=observability,
        )


__all__ = ["AgentModelProvider"]
