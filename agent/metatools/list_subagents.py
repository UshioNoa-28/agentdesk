"""固定 ``list_subagents`` Meta Tool。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from agent.domain.tools import AgentTool, ToolContext, ToolDefinition, ToolResult
from agent.ports.services import SessionServicePort

LIST_SUBAGENTS_DEFINITION = ToolDefinition(
    name="list_subagents",
    description="List all subagents currently defined for this session.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
)


class ListSubagentsTool(AgentTool):
    """列出当前主会话下全部子智能体的 Meta Tool。

    子智能体以子会话形式持久化（标题格式 ``{主会话id}_subagent_{name}``），
    因此这里直接查询数据库，服务重启后依然准确。
    """

    def __init__(
        self,
        *,
        session_service: SessionServicePort,
        definition: ToolDefinition = LIST_SUBAGENTS_DEFINITION,
    ) -> None:
        self._session_service = session_service
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
        prefix = f"{context.caller_session_id}_subagent_"
        subagents_data = [
            {
                "name": sub.title.removeprefix(prefix),
                "description": "",
                "session_id": sub.id,
                "allowed_tools": list(sub.allowed_tools),
            }
            for sub in await self._session_service.list_by_main_session(
                context.caller_session_id
            )
        ]
        return ToolResult(
            content=json.dumps(
                {
                    "ok": True,
                    "total": len(subagents_data),
                    "subagents": subagents_data,
                },
                ensure_ascii=False,
            )
        )


__all__ = ["LIST_SUBAGENTS_DEFINITION", "ListSubagentsTool"]
