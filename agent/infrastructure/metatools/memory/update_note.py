"""固定 ``update_note`` 工具：改一条已有笔记。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from functools import partial

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
from agent.infrastructure.memory.notes import (
    MAX_BODY_CHARS,
    MAX_DESCRIPTION_CHARS,
)
from agent.infrastructure.metatools.memory._base import (
    LAYER_SCHEMA,
    TITLE_SCHEMA,
    layer_of,
    note_payload,
)
from agent.ports.services import MemoryNoteBookPort, MemoryStorePort

UPDATE_NOTE_DEFINITION = ToolDefinition(
    name="update_note",
    description=(
        "Revise an existing note, located by its title. Omit a field to keep it. Read "
        "the note first and carry over everything still true, so the rewrite does not "
        "silently delete an earlier fact. Prefer updating over creating a "
        "near-duplicate; the title itself never changes (write a new note and delete "
        "this one to rename)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "layer": LAYER_SCHEMA,
            "title": TITLE_SCHEMA,
            "description": {"type": "string", "minLength": 1, "maxLength": MAX_DESCRIPTION_CHARS},
            "body": {"type": "string", "minLength": 1, "maxLength": MAX_BODY_CHARS},
        },
        "required": ["layer", "title"],
        "additionalProperties": False,
    },
    keywords=("memory", "note", "revise"),
    use_cases=("Fold a new record into the note that already covers it",),
)


class UpdateNoteTool(AgentTool):
    """改索引里已有的条目；标题不存在就什么也不动。"""

    def __init__(self, store: MemoryStorePort, notebook: MemoryNoteBookPort) -> None:
        self._store = store
        self._notebook = notebook

    @property
    def definition(self) -> ToolDefinition:
        return UPDATE_NOTE_DEFINITION

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
        fields = {
            key: str(arguments[key])
            for key in ("description", "body")
            if arguments.get(key) is not None
        }
        if not fields:
            return error_result(
                "nothing_to_update",
                "Provide at least one of description or body.",
            )
        try:
            note = await asyncio.to_thread(
                partial(
                    self._notebook.update_note,
                    layer,
                    str(arguments["title"]),
                    **fields,
                )
            )
        except MemoryNotFound as exc:
            return error_result("note_not_found", str(exc))
        except (MemoryWriteError, OSError) as exc:
            return error_result("update_note_failed", str(exc))
        return ok_result({"updated": note_payload(note)})


__all__ = ["UPDATE_NOTE_DEFINITION", "UpdateNoteTool"]
