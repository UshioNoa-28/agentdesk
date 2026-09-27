"""固定 ``remember`` Meta Tool：向待整理日志追加一条事实。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from agent.domain.memory import MemoryLayer, MemoryOperation, MemoryWriteError
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
from agent.infrastructure.memory import MEMORY_DISABLED_MESSAGE
from agent.ports.services import MemoryStorePort

REMEMBER_DEFINITION = ToolDefinition(
    name="remember",
    description=(
        "Append one durable fact to the pending memory log. This tool does not touch "
        "the memory notes or MEMORY.md; a background consolidation pass rewrites "
        "those later, so the record is only visible in this conversation until then. "
        "One fact per call, stated as a self-contained sentence that a reader with no "
        "other context can use. layer='project' is for facts about this repository "
        "(commands, conventions, architecture decisions); layer='user' is for personal "
        "facts (preferences, role, long-term goals) that must not end up in a committed "
        "repository. You MUST NOT store secrets, credentials or tokens, and MUST NOT "
        "repeat a fact already recorded earlier in this conversation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2_000,
                "description": "The fact to keep, as one self-contained sentence.",
            },
            "layer": {
                "type": "string",
                "enum": ["project", "user"],
                "description": "Which memory layer owns the fact.",
            },
        },
        "required": ["content", "layer"],
        "additionalProperties": False,
    },
    keywords=("memory", "remember", "note"),
    use_cases=("Keep a durable fact beyond this conversation",),
)


class RememberTool(AgentTool):
    """把一条事实追加进当前 session 的待整理记忆日志。"""

    def __init__(self, store: MemoryStorePort) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return REMEMBER_DEFINITION

    def permission_targets(self, arguments: Mapping[str, object]) -> None:
        """无权限面：只追加自己的日志，不触碰用户文件，因此永不审批。"""

        return None

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        if not self._store.enabled:
            return error_result("memory_disabled", MEMORY_DISABLED_MESSAGE)
        try:
            layer = MemoryLayer(str(arguments["layer"]).strip())
        except ValueError:
            options = ", ".join(repr(item.value) for item in MemoryLayer)
            return error_result("invalid_layer", f"layer must be one of: {options}.")
        try:
            pending = await asyncio.to_thread(
                self._store.record,
                MemoryOperation.REMEMBER,
                str(arguments["content"]),
                session_id=context.caller_session_id,
                layer=layer,
            )
        except (MemoryWriteError, OSError) as exc:
            return error_result("remember_failed", str(exc))
        return ok_result(
            {
                "recorded": {
                    "op": pending.operation.value,
                    "layer": layer.value,
                    "at": pending.created_at.isoformat(),
                    "content": pending.content,
                },
            }
        )


__all__ = ["REMEMBER_DEFINITION", "RememberTool"]
