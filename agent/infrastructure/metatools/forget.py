"""固定 ``forget`` Meta Tool：向待整理日志追加一条失效声明。"""

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

FORGET_DEFINITION = ToolDefinition(
    name="forget",
    description=(
        "Record that a previously remembered fact is wrong, outdated or no longer "
        "useful, so the consolidation pass drops it from the memory notes. Like "
        "remember, this only appends to the pending log and never edits the notes or "
        "MEMORY.md directly. State content the same way the fact was written when it "
        "was remembered (or close enough to identify it), as one self-contained "
        "sentence. Pass the layer the fact lives in. Forgetting is not a secret "
        "delete: never use it to hide credentials or sensitive data that were "
        "recorded by mistake."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2_000,
                "description": "The fact that should no longer be remembered.",
            },
            "layer": {
                "type": "string",
                "enum": ["project", "user"],
                "description": "Which memory layer holds the fact.",
            },
        },
        "required": ["content", "layer"],
        "additionalProperties": False,
    },
    keywords=("memory", "forget", "outdated"),
    use_cases=("Mark a remembered fact as wrong or obsolete",),
)


class ForgetTool(AgentTool):
    """把「这条不要再记着」追加进当前 session 的待整理记忆日志。"""

    def __init__(self, store: MemoryStorePort) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return FORGET_DEFINITION

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
                MemoryOperation.FORGET,
                str(arguments["content"]),
                session_id=context.caller_session_id,
                layer=layer,
            )
        except (MemoryWriteError, OSError) as exc:
            return error_result("forget_failed", str(exc))
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


__all__ = ["FORGET_DEFINITION", "ForgetTool"]
