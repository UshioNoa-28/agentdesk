"""MCP 连接注册 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.infrastructure.mcp import McpConfiguration, McpRegistry, load_mcp_configuration
from agent.infrastructure.settings import AgentRuntimeSettings
from agent.ports.tools import McpRegistryPort


class McpProvider(Provider):
    """提供 MCP 配置读取和长生命周期 Server 注册表。"""

    @provide(scope=Scope.APP)
    def provide_mcp_configuration(
        self,
        settings: AgentRuntimeSettings,
    ) -> McpConfiguration:
        """从 workspace 根（缺省为进程 cwd）逐级向上发现 .agent-desk/mcp.json。"""

        start = settings.workspace_start
        return load_mcp_configuration(start)

    @provide(scope=Scope.APP)
    def provide_mcp_registry(
        self,
        configuration: McpConfiguration,
    ) -> McpRegistryPort:
        """根据已经加载的配置建立长生命周期 Server 注册表。"""

        return McpRegistry(configuration.servers)


__all__ = ["McpProvider"]
