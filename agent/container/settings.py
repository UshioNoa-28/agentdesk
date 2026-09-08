"""Agent 私有的窄配置 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.infrastructure.settings import (
    AgentRuntimeSettings,
    ChatModelSettings,
    ContextSettings,
    DatabaseSettings,
    McpSettings,
    MemorySettings,
    ObservabilitySettings,
    SkillSettings,
)


class AgentSettingsProvider(Provider):
    """提供 Agent 进程需要的各项运行时配置。"""

    @provide(scope=Scope.APP)
    def provide_database_settings(self) -> DatabaseSettings:
        """提供 PostgreSQL 连接池所需的最小配置。"""

        return DatabaseSettings()

    @provide(scope=Scope.APP)
    def provide_chat_model_settings(self) -> ChatModelSettings:
        """提供聊天模型 provider 配置。"""

        return ChatModelSettings()

    @provide(scope=Scope.APP)
    def provide_observability_settings(self) -> ObservabilitySettings:
        """提供 wire 日志配置。"""

        return ObservabilitySettings()

    @provide(scope=Scope.APP)
    def provide_context_settings(self) -> ContextSettings:
        """提供上下文窗口和压缩策略配置。"""

        return ContextSettings()

    @provide(scope=Scope.APP)
    def provide_agent_runtime_settings(self) -> AgentRuntimeSettings:
        """提供模型-工具循环的最大 turn 配置。"""

        return AgentRuntimeSettings()

    @provide(scope=Scope.APP)
    def provide_mcp_settings(self) -> McpSettings:
        """提供 Agent MCP 配置文件路径。"""

        return McpSettings()

    @provide(scope=Scope.APP)
    def provide_skill_settings(self) -> SkillSettings:
        """提供 Agent Skill 配置文件路径。"""

        return SkillSettings()

    @provide(scope=Scope.APP)
    def provide_memory_settings(self) -> MemorySettings:
        """提供 Agent Mem0 长期记忆配置。"""

        return MemorySettings()


__all__ = ["AgentSettingsProvider"]
