"""固定 ``list_subagents`` Meta Tool。"""

from __future__ import annotations

from collections.abc import Mapping

from agent.domain.multi_agent import AgentStatus
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    ok_result,
)
from agent.ports.runtime.status import AgentStatusRegistryPort
from agent.ports.services import SessionServicePort

LIST_SUBAGENTS_DEFINITION = ToolDefinition(
    name="list_subagents",
    description=(
        "List all subagents defined for this session, each with its name, "
        "session id, allowed tools, and live runtime status "
        "(running / idle / error; 'idle' also means not currently executing)."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
)


class ListSubagentsTool(AgentTool):
    """列出当前主会话下全部子智能体（含运行态状态）的 Meta Tool。

    子智能体以子会话形式持久化（标题格式 ``{主会话id}_subagent_{name}``），
    因此清单本身直接查询数据库，服务重启后依然准确。运行态状态是 actor 实例上
    的易失内存态，只存在于进程内注册表：注册表里没有记录（子代理从未被实例化，
    或进程刚重启）时回退为 ``idle``。
    """

    def __init__(
        self,
        *,
        session_service: SessionServicePort,
        status_registry: AgentStatusRegistryPort,
        definition: ToolDefinition = LIST_SUBAGENTS_DEFINITION,
    ) -> None:
        self._session_service = session_service
        self._status_registry = status_registry
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
                "status": self._status_of(sub.id),
            }
            for sub in await self._session_service.list_by_main_session(
                context.caller_session_id
            )
        ]
        return ok_result(
            {
                "total": len(subagents_data),
                "subagents": subagents_data,
            }
        )

    def _status_of(self, session_id: str) -> str:
        """读取子代理实时状态；无在途记录（从未实例化/进程重启）时回退 idle。"""

        status = self._status_registry.get_status(session_id)
        return status.value if status is not None else AgentStatus.IDLE.value


__all__ = ["LIST_SUBAGENTS_DEFINITION", "ListSubagentsTool"]
