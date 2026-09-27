"""固定 ``execute_mcp`` Meta Tool。"""

from __future__ import annotations

from collections.abc import Mapping

from agent.domain.messages import MessageKind
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
)
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
                "pattern": "\\S",
                "description": "Configured MCP server id.",
            },
            "tool": {
                "type": "string",
                "pattern": "\\S",
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

    def permission_targets(self, arguments: Mapping[str, object]) -> tuple[str, ...]:
        """供权限层匹配的 MCP server 标识；粒度到 server，不细化到 tool。

        先验证再匹配：required + pattern ``\\S`` 已在 schema 层挡掉缺失/
        空白，graph 走到这里时值必然可用，钩子不再抛错。
        """

        mcp = arguments["mcp"]
        return (mcp.strip(),)

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        """把固定入口参数交给 Registry 校验并调用远程工具。"""

        mcp = str(arguments["mcp"]).strip()
        tool = str(arguments["tool"]).strip()
        remote_arguments = dict(arguments["arguments"])  # type: ignore[arg-type]

        try:
            result = await self._registry.execute_tool(
                mcp=mcp,
                tool=tool,
                arguments=remote_arguments,
            )
            return ToolResult(
                content=result.content,
                kind=MessageKind.TOOL_RESULT,
            )
        except Exception as exc:
            return error_result("execute_mcp_failed", str(exc))


__all__ = ["EXECUTE_MCP_DEFINITION", "ExecuteMcpTool"]
