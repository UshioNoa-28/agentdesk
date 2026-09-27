from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from dishka import make_async_container

from agent.container import providers
from agent.domain.memory import MemoryLayer
from agent.domain.tools import (
    ALL_META_TOOL_NAMES,
    MEMORY_TOOL_NAMES,
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from agent.infrastructure.memory import MemoryStore
from agent.infrastructure.memory.notes import NoteBook
from agent.infrastructure.metatools.memory import (
    DeleteNoteTool,
    ReadIndexTool,
    ReadNoteTool,
    UpdateNoteTool,
    WriteNoteTool,
)
from agent.ports.tools import MetaToolRegistryPort

_CONTEXT = ToolContext(caller_session_id="consolidator-1")


def _store(root: Path, *, enabled: bool = True) -> MemoryStore:
    return MemoryStore(start=root / "workspace", home=root / "home", enabled=enabled)


def _tools(root: Path, *, enabled: bool = True) -> dict[str, AgentTool]:
    """整理者的五个工具：组合根之外按同样的方式手工装配一份。"""

    store = _store(root, enabled=enabled)
    notebook = NoteBook(store)
    tools: tuple[AgentTool, ...] = (
        ReadIndexTool(notebook),
        ReadNoteTool(notebook),
        WriteNoteTool(store, notebook),
        UpdateNoteTool(store, notebook),
        DeleteNoteTool(store, notebook),
    )
    return {tool.definition.name: tool for tool in tools}


def _definitions() -> tuple[ToolDefinition, ...]:
    with TemporaryDirectory() as directory:
        return tuple(tool.definition for tool in _tools(Path(directory)).values())


async def _run(
    root: Path,
    name: str,
    arguments: Mapping[str, object],
    *,
    enabled: bool = True,
) -> ToolResult:
    return await _tools(root, enabled=enabled)[name].aexecute(arguments, context=_CONTEXT)


def _payload(result: ToolResult) -> dict[str, object]:
    return json.loads(result.content)


def _error_code(result: ToolResult) -> object:
    error = _payload(result)["error"]  # type: ignore[index]
    assert isinstance(error, dict)
    return error["code"]


def _field(result: ToolResult, field: str) -> dict[str, object]:
    value = _payload(result)[field]
    assert isinstance(value, dict)
    return value


async def _read_index(root: Path) -> list[dict[str, object]]:
    result = await _run(root, "read_index", {"layer": "project"})
    entries = _payload(result)["entries"]  # type: ignore[index]
    assert isinstance(entries, list)
    return [entry for entry in entries if isinstance(entry, dict)]


_WRITE_ARGS = {
    "layer": "project",
    "title": "game preference",
    "description": "what type of game user like",
    "body": "User likes Slay the Spire.",
}


class MemoryNoteToolTests(IsolatedAsyncioTestCase):
    async def test_write_note_is_visible_through_read_index_and_read_note(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            created = await _run(root, "write_note", _WRITE_ARGS)

            self.assertEqual(_field(created, "created")["title"], "game-preference")
            self.assertEqual(
                await _read_index(root),
                [
                    {
                        "title": "game-preference",
                        "description": "what type of game user like",
                        "timestamp": _field(created, "created")["timestamp"],
                    }
                ],
            )

            note = await _run(root, "read_note", {"layer": "project", "title": "game-preference"})
            self.assertEqual(_field(note, "note")["body"], "User likes Slay the Spire.\n")

    async def test_update_note_keeps_omitted_fields(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            await _run(root, "write_note", _WRITE_ARGS)

            result = await _run(
                root,
                "update_note",
                {"layer": "project", "title": "game-preference", "body": "New body."},
            )

            updated = _field(result, "updated")
            self.assertEqual(updated["title"], "game-preference")
            self.assertEqual(updated["description"], "what type of game user like")
            self.assertEqual(updated["body"], "New body.\n")

    async def test_update_note_without_any_field_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            await _run(root, "write_note", _WRITE_ARGS)

            result = await _run(
                root, "update_note", {"layer": "project", "title": "game-preference"}
            )

            self.assertEqual(_error_code(result), "nothing_to_update")

    async def test_delete_note_removes_the_index_entry(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            await _run(root, "write_note", {**_WRITE_ARGS, "title": "outdated"})
            await _run(root, "write_note", _WRITE_ARGS)

            result = await _run(root, "delete_note", {"layer": "project", "title": "outdated"})

            self.assertIs(_payload(result)["ok"], True)
            self.assertEqual(
                [item["title"] for item in await _read_index(root)], ["game-preference"]
            )
            self.assertFalse(
                NoteBook(_store(root)).note_path(MemoryLayer.PROJECT, "outdated").exists()
            )

    async def test_unknown_title_is_reported_as_a_tool_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            for name, arguments in (
                ("read_note", {"layer": "user", "title": "missing"}),
                ("update_note", {"layer": "user", "title": "missing", "body": "b"}),
                ("delete_note", {"layer": "user", "title": "missing"}),
            ):
                result = await _run(root, name, arguments)
                self.assertEqual(_error_code(result), "note_not_found", name)

    async def test_writes_are_refused_while_memory_is_disabled(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            await _run(root, "write_note", _WRITE_ARGS)

            entries = await _run(root, "read_index", {"layer": "project"}, enabled=False)
            write = await _run(root, "write_note", _WRITE_ARGS, enabled=False)
            delete = await _run(
                root,
                "delete_note",
                {"layer": "project", "title": "game-preference"},
                enabled=False,
            )

            listed = _payload(entries)["entries"]  # type: ignore[index]
            assert isinstance(listed, list)
            self.assertEqual([item["title"] for item in listed], ["game-preference"])
            self.assertEqual(_error_code(write), "memory_disabled")
            self.assertEqual(_error_code(delete), "memory_disabled")

    async def test_invalid_layer_is_rejected_before_any_filesystem_call(self) -> None:
        with TemporaryDirectory() as directory:
            result = await _run(Path(directory), "read_index", {"layer": "global"})

            self.assertEqual(_error_code(result), "invalid_layer")

    async def test_hostile_title_never_reaches_the_filesystem(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            await _run(root, "write_note", _WRITE_ARGS)
            before = NoteBook(_store(root)).read_index(MemoryLayer.PROJECT)

            for name in ("read_note", "delete_note", "update_note"):
                arguments: dict[str, object] = {
                    "layer": "project",
                    "title": "../../etc/passwd",
                    "body": "b",
                }
                result = await _run(root, name, arguments)
                self.assertFalse(_payload(result)["ok"], name)

            self.assertEqual(NoteBook(_store(root)).read_index(MemoryLayer.PROJECT), before)


class MemoryToolIsolationTests(TestCase):
    def test_names_are_absent_from_the_main_agent_surface(self) -> None:
        self.assertFalse(set(MEMORY_TOOL_NAMES).intersection(ALL_META_TOOL_NAMES))

    def test_definitions_match_the_documented_names(self) -> None:
        self.assertEqual([d.name for d in _definitions()], list(MEMORY_TOOL_NAMES))

    def test_arguments_never_name_a_path(self) -> None:
        allowed = {"layer", "title", "description", "body"}
        for definition in _definitions():
            with self.subTest(tool=definition.name):
                properties = definition.parameters["properties"]
                assert isinstance(properties, Mapping)
                self.assertLessEqual(set(properties), allowed)

    def test_tools_have_no_permission_surface(self) -> None:
        with TemporaryDirectory() as directory:
            for tool in _tools(Path(directory)).values():
                with self.subTest(tool=tool.definition.name):
                    self.assertIsNone(tool.permission_targets({"layer": "user"}))


class MemoryToolContainerTests(IsolatedAsyncioTestCase):
    """工具进了总表，但会话按 ``ALL_META_TOOL_NAMES`` 筛，主 Agent 收不到它们。"""

    async def test_session_tool_surface_excludes_memory_tools(self) -> None:
        with TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"WORKSPACE_ROOT": directory}):
                container = make_async_container(*providers())
                try:
                    async with container() as request:
                        registry = await request.get(MetaToolRegistryPort)
                finally:
                    await container.close()

        all_names = {tool.definition.name for tool in registry.get_tools(ALL_META_TOOL_NAMES)}
        self.assertIn("remember", all_names)
        self.assertIn("forget", all_names)
        self.assertFalse(all_names.intersection(MEMORY_TOOL_NAMES))

        own = {tool.definition.name for tool in registry.get_tools(MEMORY_TOOL_NAMES)}
        self.assertEqual(own, set(MEMORY_TOOL_NAMES))
