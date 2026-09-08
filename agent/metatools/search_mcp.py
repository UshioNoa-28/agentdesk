"""固定 ``search_mcp`` Meta Tool。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from agent.domain.messages import MessageKind
from agent.domain.tools import AgentTool, ToolContext, ToolDefinition, ToolResult
from agent.ports.tools import McpRegistryPort

SEARCH_MCP_DEFINITION = ToolDefinition(
    name="search_mcp",
    description=(
        "Search one configured MCP server for tools relevant to a capability. "
        "The result is a complete tool definition in the current conversation, "
        "including the remote tool name, description, JSON input schema, and use "
        "cases. The result does not bind a new model tool. After inspecting it, "
        "call execute_mcp with the exact mcp and tool names."
    ),
    parameters={
        "type": "object",
        "properties": {
            "mcp": {
                "type": "string",
                "minLength": 1,
                "description": "Configured MCP server id from the routing guide.",
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Describe the capability or task that requires a tool.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
                "description": "Maximum number of complete definitions to return.",
            },
        },
        "required": ["mcp", "query"],
        "additionalProperties": False,
    },
)


class SearchMcpTool(AgentTool):
    """搜索 MCP 工具定义，但不把远程工具绑定进模型 ``tools`` 字段。"""

    def __init__(
        self,
        registry: McpRegistryPort,
        definition: ToolDefinition = SEARCH_MCP_DEFINITION,
    ) -> None:
        self._registry = registry
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        """校验查询并返回完整的 MCP 工具定义 JSON。"""

        tool_name = self.definition.name
        mcp = arguments.get("mcp")
        query = arguments.get("query")
        if not isinstance(mcp, str) or not mcp.strip():
            return _error("invalid_arguments", f"{tool_name} requires a non-empty mcp")
        if not isinstance(query, str) or not query.strip():
            return _error("invalid_arguments", f"{tool_name} requires a non-empty query")
        limit = arguments.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
            return _error("invalid_arguments", "limit must be an integer between 1 and 10")

        try:
            definitions = self._registry.search_tools(
                mcp=mcp.strip(),
                query=query.strip(),
                limit=limit,
            )
        except ValueError as exc:
            return _error("search_mcp_failed", str(exc))

        return ToolResult(
            content=json.dumps(
                {
                    "ok": True,
                    "mcp": mcp.strip(),
                    "query": query.strip(),
                    "tools": [_definition_payload(definition) for definition in definitions],
                },
                ensure_ascii=False,
            ),
            kind=MessageKind.MCP_TOOL_DEFINITION,
        )


def _definition_payload(definition: ToolDefinition) -> dict[str, object]:
    """把供应商无关工具定义转换成模型可读的完整 JSON。"""

    return {
        "name": definition.name,
        "description": definition.description,
        "parameters": dict(definition.parameters),
        "keywords": list(definition.keywords),
        "use_cases": list(definition.use_cases),
    }


def _error(code: str, message: str) -> ToolResult:
    """构造不会抛出到 Graph 的结构化搜索错误。"""

    return ToolResult(
        content=json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        )
    )


__all__ = ["SEARCH_MCP_DEFINITION", "SearchMcpTool"]
