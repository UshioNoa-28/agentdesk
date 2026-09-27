"""固定 ``read_index`` 工具：列出一层记忆的索引条目。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from agent.domain.memory import MemoryWriteError
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
from agent.infrastructure.metatools.memory._base import LAYER_SCHEMA, layer_of
from agent.ports.services import MemoryNoteBookPort

READ_INDEX_DEFINITION = ToolDefinition(
    name="read_index",
    description=(
        "List every note in one memory layer as title, one-line description and "
        "timestamp, in index order. The title is the note's identity and its file "
        "name. Read this before deciding whether a pending record is new, an update "
        "of an existing note, or a duplicate."
    ),
    parameters={
        "type": "object",
        "properties": {"layer": LAYER_SCHEMA},
        "required": ["layer"],
        "additionalProperties": False,
    },
    keywords=("memory", "index", "list"),
    use_cases=("See what a memory layer already holds",),
)


class ReadIndexTool(AgentTool):
    """读一层索引；参数里没有路径，也没有可核验对象。"""

    def __init__(self, notebook: MemoryNoteBookPort) -> None:
        self._notebook = notebook

    @property
    def definition(self) -> ToolDefinition:
        return READ_INDEX_DEFINITION

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        layer = layer_of(arguments["layer"])
        if layer is None:
            return error_result("invalid_layer", "layer must be 'project' or 'user'.")
        try:
            entries = await asyncio.to_thread(self._notebook.list_entries, layer)
        except (MemoryWriteError, OSError) as exc:
            return error_result("read_index_failed", str(exc))
        return ok_result(
            {
                "layer": layer.value,
                "count": len(entries),
                "entries": [
                    {
                        "title": item.title,
                        "description": item.description,
                        "timestamp": item.timestamp.isoformat(),
                    }
                    for item in entries
                ],
            }
        )


__all__ = ["READ_INDEX_DEFINITION", "ReadIndexTool"]
