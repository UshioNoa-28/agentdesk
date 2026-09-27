"""Agent 私有的窄配置 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.infrastructure.settings import (
    AgentRuntimeSettings,
    DatabaseSettings,
    ObservabilitySettings,
)


class AgentSettingsProvider(Provider):
    """提供 Agent 进程需要的各项运行时配置。"""

    @provide(scope=Scope.APP)
    def provide_database_settings(self) -> DatabaseSettings:
        """提供数据库连接配置（URL 决定方言）。"""

        return DatabaseSettings()

    @provide(scope=Scope.APP)
    def provide_observability_settings(self) -> ObservabilitySettings:
        """提供 wire 日志配置。"""

        return ObservabilitySettings()

    @provide(scope=Scope.APP)
    def provide_agent_runtime_settings(self) -> AgentRuntimeSettings:
        """提供 Agent 运行时配置：turn/等待超时上限、工作目录根与会话锁目录。"""

        return AgentRuntimeSettings()


__all__ = ["AgentSettingsProvider"]
