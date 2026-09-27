"""固定 ``read_note`` 工具：读出一条笔记的正文与索引条目。"""

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
from agent.infrastructure.metatools.memory._base import (
    LAYER_SCHEMA,
    TITLE_SCHEMA,
    layer_of,
    note_payload,
)
from agent.ports.services import MemoryNoteBookPort

READ_NOTE_DEFINITION = ToolDefinition(
    name="read_note",
    description=(
        "Read one note in full (body plus its index entry) so you can merge into it "
        "instead of overwriting facts you never saw."
    ),
    parameters={
        "type": "object",
        "properties": {"layer": LAYER_SCHEMA, "title": TITLE_SCHEMA},
        "required": ["layer", "title"],
        "additionalProperties": False,
    },
    keywords=("memory", "note", "read"),
    use_cases=("Inspect a note before revising it",),
)


class ReadNoteTool(AgentTool):
    """标题对照索引；落点只能是已登记的笔记。"""

    def __init__(self, notebook: MemoryNoteBookPort) -> None:
        self._notebook = notebook

    @property
    def definition(self) -> ToolDefinition:
        return READ_NOTE_DEFINITION

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
            note = await asyncio.to_thread(
                self._notebook.read_note, layer, str(arguments["title"])
            )
        except MemoryNotFound as exc:
            return error_result("note_not_found", str(exc))
        except (MemoryWriteError, OSError) as exc:
            return error_result("read_note_failed", str(exc))
        return ok_result({"note": note_payload(note)})


__all__ = ["READ_NOTE_DEFINITION", "ReadNoteTool"]
