"""MCP 连接注册 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.infrastructure.mcp import McpConfiguration, McpRegistry, load_mcp_configuration
from agent.infrastructure.settings import McpSettings
from agent.ports.tools import McpRegistryPort


class McpProvider(Provider):
    """提供 MCP 配置读取和长生命周期 Server 注册表。"""

    @provide(scope=Scope.APP)
    def provide_mcp_configuration(
        self,
        settings: McpSettings,
    ) -> McpConfiguration:
        """在组合根读取一次 MCP Server 配置，供注册表使用。"""

        return load_mcp_configuration(settings.mcp_config_path)

    @provide(scope=Scope.APP)
    def provide_mcp_registry(
        self,
        configuration: McpConfiguration,
    ) -> McpRegistryPort:
        """根据已经加载的配置建立长生命周期 Server 注册表。"""

        return McpRegistry(configuration.servers)


__all__ = ["McpProvider"]
