"""固定 ``write_note`` 工具：新建笔记并追加索引行。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from functools import partial

from agent.domain.memory import MemoryWriteError
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
from agent.infrastructure.memory import MEMORY_DISABLED_MESSAGE
from agent.infrastructure.memory.notes import (
    MAX_BODY_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_TITLE_CHARS,
)
from agent.infrastructure.metatools.memory._base import LAYER_SCHEMA, layer_of, note_payload
from agent.ports.services import MemoryNoteBookPort, MemoryStorePort

WRITE_NOTE_DEFINITION = ToolDefinition(
    name="write_note",
    description=(
        "Create one note and add its index line. Use for a fact that nothing existing "
        "covers. The body is the whole durable statement, written for a reader with no "
        "other context: commands, paths and decisions spelled out, no secrets. One "
        "topic per note. Never hand-write MEMORY.md; this tool owns the index."
    ),
    parameters={
        "type": "object",
        "properties": {
            "layer": LAYER_SCHEMA,
            "title": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TITLE_CHARS,
                "description": (
                    "Short kebab-case slug (e.g. game-preference). It is the note's "
                    "identity and names its file, so it must be unique in the layer."
                ),
            },
            "description": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_DESCRIPTION_CHARS,
                "description": "One sentence saying when this note is worth opening.",
            },
            "body": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_BODY_CHARS,
                "description": "The note content in markdown.",
            },
        },
        "required": ["layer", "title", "description", "body"],
        "additionalProperties": False,
    },
    keywords=("memory", "note", "create"),
    use_cases=("Record a durable fact no existing note covers",),
)


class WriteNoteTool(AgentTool):
    """写入位置由 layer + title 唯一确定；标题规范化与索引行都由 ``NoteBook`` 负责。"""

    def __init__(self, store: MemoryStorePort, notebook: MemoryNoteBookPort) -> None:
        self._store = store
        self._notebook = notebook

    @property
    def definition(self) -> ToolDefinition:
        return WRITE_NOTE_DEFINITION

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
            note = await asyncio.to_thread(
                partial(
                    self._notebook.write_note,
                    layer,
                    title=str(arguments["title"]),
                    description=str(arguments["description"]),
                    body=str(arguments["body"]),
                )
            )
        except (MemoryWriteError, OSError) as exc:
            return error_result("write_note_failed", str(exc))
        return ok_result({"created": note_payload(note)})


__all__ = ["WRITE_NOTE_DEFINITION", "WriteNoteTool"]
