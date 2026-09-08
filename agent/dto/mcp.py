from __future__ import annotations

from pydantic import BaseModel


class McpStatusResponse(BaseModel):
    """返回给管理 API 的 MCP Server 状态快照。"""

    id: str
    description: str
    state: str
    error: str | None
    tool_count: int


__all__ = ["McpStatusResponse"]
