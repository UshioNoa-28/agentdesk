"""固定 ``execute_mcp`` Meta Tool。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from agent.domain.messages import MessageKind
from agent.domain.tools import AgentTool, ToolContext, ToolDefinition, ToolResult
from agent.ports.tools import McpRegistryPort

EXECUTE_MCP_DEFINITION = ToolDefinition(
    name="execute_mcp",
    description=(
        "Execute one remote tool on a configured MCP server. Use the exact mcp and "
        "tool names returned by search_mcp; do not invent or flatten names. The "
        "arguments object must follow the returned input schema. The registry "
        "validates the server connection and remote tool before execution."
    ),
    parameters={
        "type": "object",
        "properties": {
            "mcp": {
                "type": "string",
                "minLength": 1,
                "description": "Configured MCP server id.",
            },
            "tool": {
                "type": "string",
                "minLength": 1,
                "description": "Exact remote MCP tool name returned by search_mcp.",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments matching the remote tool input schema.",
                "additionalProperties": True,
            },
        },
        "required": ["mcp", "tool", "arguments"],
        "additionalProperties": False,
    },
)


class ExecuteMcpTool(AgentTool):
    """使用 MCP ID、远程工具名和参数执行一个 MCP 工具。"""

    def __init__(
        self,
        registry: McpRegistryPort,
        definition: ToolDefinition = EXECUTE_MCP_DEFINITION,
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
        """校验固定入口参数，再由 Registry 校验并调用远程工具。"""

        mcp = arguments.get("mcp")
        tool = arguments.get("tool")
        remote_arguments = arguments.get("arguments")
        if not isinstance(mcp, str) or not mcp.strip():
            return _error("invalid_arguments", "execute_mcp requires a non-empty mcp")
        if not isinstance(tool, str) or not tool.strip():
            return _error("invalid_arguments", "execute_mcp requires a non-empty tool")
        if not isinstance(remote_arguments, Mapping):
            return _error("invalid_arguments", "execute_mcp requires an object arguments")

        try:
            result = await self._registry.execute_tool(
                mcp=mcp.strip(),
                tool=tool.strip(),
                arguments=dict(remote_arguments),
            )
            return ToolResult(
                content=result.content,
                kind=MessageKind.TOOL_RESULT,
            )
        except Exception as exc:
            return _error("execute_mcp_failed", str(exc))


def _error(code: str, message: str) -> ToolResult:
    """构造固定执行入口的结构化参数错误。"""

    return ToolResult(
        content=json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        ),
        kind=MessageKind.TOOL_RESULT,
    )


__all__ = ["EXECUTE_MCP_DEFINITION", "ExecuteMcpTool"]
