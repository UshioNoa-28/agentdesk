"""固定 ``search_mcp`` Meta Tool。"""

from __future__ import annotations

from collections.abc import Mapping

from agent.domain.messages import MessageKind
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
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
                "pattern": "\\S",
                "description": "Configured MCP server id from the routing guide.",
            },
            "query": {
                "type": "string",
                "pattern": "\\S",
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
        """搜索 MCP 工具定义，返回完整的远程工具 JSON。"""

        mcp = str(arguments["mcp"]).strip()
        query = str(arguments["query"]).strip()
        limit = arguments.get("limit", 5)

        try:
            definitions = self._registry.search_tools(
                mcp=mcp,
                query=query,
                limit=limit,
            )
        except ValueError as exc:
            return error_result("search_mcp_failed", str(exc))

        return ok_result(
            {
                "mcp": mcp,
                "query": query,
                "tools": [_definition_payload(definition) for definition in definitions],
            },
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


__all__ = ["SEARCH_MCP_DEFINITION", "SearchMcpTool"]
