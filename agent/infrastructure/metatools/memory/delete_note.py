"""固定 ``delete_note`` 工具：删除笔记与其索引行。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from agent.domain.memory import MemoryNotFound, MemoryWriteError
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
from agent.infrastructure.memory import MEMORY_DISABLED_MESSAGE
from agent.infrastructure.metatools.memory._base import LAYER_SCHEMA, TITLE_SCHEMA, layer_of
from agent.ports.services import MemoryNoteBookPort, MemoryStorePort

DELETE_NOTE_DEFINITION = ToolDefinition(
    name="delete_note",
    description=(
        "Drop a note and its index line. Use when the fact is wrong, outdated or "
        "explicitly retracted. If it still holds partly, update it instead of deleting."
    ),
    parameters={
        "type": "object",
        "properties": {"layer": LAYER_SCHEMA, "title": TITLE_SCHEMA},
        "required": ["layer", "title"],
        "additionalProperties": False,
    },
    keywords=("memory", "note", "remove"),
    use_cases=("Retire a note the pending records invalidated",),
)


class DeleteNoteTool(AgentTool):
    """删除范围由 layer + title 唯一确定。"""

    def __init__(self, store: MemoryStorePort, notebook: MemoryNoteBookPort) -> None:
        self._store = store
        self._notebook = notebook

    @property
    def definition(self) -> ToolDefinition:
        return DELETE_NOTE_DEFINITION

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        if not self._store.enabled:
            return error_result("memory_disabled", MEMORY_DISABLED_MESSAGE)
        layer = layer_of(arguments["layer"])
        if layer is None:
            return error_result("invalid_layer", "layer must be 'project' or 'user'.")
        try:
            await asyncio.to_thread(
                self._notebook.delete_note, layer, str(arguments["title"])
            )
        except MemoryNotFound as exc:
            return error_result("note_not_found", str(exc))
        except (MemoryWriteError, OSError) as exc:
            return error_result("delete_note_failed", str(exc))
        return ok_result(
            {"deleted": {"layer": layer.value, "title": str(arguments["title"])}}
        )


__all__ = ["DELETE_NOTE_DEFINITION", "DeleteNoteTool"]
