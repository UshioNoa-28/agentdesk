"""MCP 注册表与生命周期端口协议。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from agent.domain.tools import ToolDefinition, ToolResult


class McpRegistryPort(Protocol):
    """MCP Server 的生命周期、发现和调用边界。"""

    async def start_all_async(self) -> None: ...

    async def close_async(self) -> None: ...

    def server_descriptions(self) -> tuple[tuple[str, str], ...]: ...

    def list_statuses(self) -> tuple[object, ...]: ...

    async def retry_async(self, server_id: str) -> object: ...

    def search_tools(self, *, mcp: str, query: str, limit: int) -> tuple[ToolDefinition, ...]: ...

    async def execute_tool(
        self,
        *,
        mcp: str,
        tool: str,
        arguments: Mapping[str, object],
    ) -> ToolResult: ...


__all__ = ["McpRegistryPort"]
