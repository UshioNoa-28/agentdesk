from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from agent.application.memory import MemoryConsolidator
from agent.domain.memory import MemoryLayer, MemoryOperation
from agent.domain.model_messages import ModelMessage, ModelTurn, ToolCall
from agent.domain.tools import AgentTool
from agent.infrastructure.memory import MemoryStore
from agent.infrastructure.memory.notes import NoteBook
from agent.infrastructure.metatools import MetaToolRegistry
from agent.infrastructure.metatools.memory import (
    DeleteNoteTool,
    ReadIndexTool,
    ReadNoteTool,
    UpdateNoteTool,
    WriteNoteTool,
)
from agent.infrastructure.prompt import PromptBuilder
from agent.ports.model import ModelCallPurpose

_SESSION = "sess-1"


def _builder() -> PromptBuilder:
    return PromptBuilder(
        mcp_registry=SimpleNamespace(server_descriptions=lambda: ()),
        skill_catalog=SimpleNamespace(metadata=lambda: ()),
    )


class _ScriptedModel:
    """按脚本回放 turn；脚本用尽后可重复最后一 turn，用来演练超预算。"""

    def __init__(self, turns: list[ModelTurn], *, repeat_last: bool = False) -> None:
        self._turns = turns
        self._repeat_last = repeat_last
        self.messages: list[list[ModelMessage]] = []
        self.purposes: list[object] = []

    async def ainvoke(
        self,
        *,
        messages,
        tools=(),
        tool_choice=None,
        max_output_tokens=None,
        purpose=None,
    ) -> ModelTurn:
        self.messages.append(list(messages))
        self.purposes.append(purpose)
        if not self._turns:
            raise AssertionError("the consolidator asked for more turns than scripted")
        if self._repeat_last and len(self._turns) == 1:
            return self._turns[0]
        return self._turns.pop(0)


def _tool_turn(name: str, arguments: dict[str, object]) -> ModelTurn:
    call = ToolCall(id=f"call-{name}", name=name, arguments=arguments)
    return ModelTurn(
        message=ModelMessage.assistant(content="", tool_calls=(call,)),
        tool_calls=(call,),
    )


def _final_turn(content: str) -> ModelTurn:
    return ModelTurn(message=ModelMessage.assistant(content=content), tool_calls=())


def _store(root: Path, *, enabled: bool = True) -> MemoryStore:
    return MemoryStore(start=root / "workspace", home=root / "home", enabled=enabled)


def _remember(
    store: MemoryStore,
    content: str,
    *,
    layer: MemoryLayer = MemoryLayer.PROJECT,
) -> None:
    store.record(MemoryOperation.REMEMBER, content, session_id=_SESSION, layer=layer)


def _fixture(
    root: Path,
    model: object,
    *,
    enabled: bool = True,
    max_turns: int = 6,
) -> MemoryConsolidator:
    """组合根之外，按同样的方式手工装配一份 store/notebook/五个笔记工具。"""

    store = _store(root, enabled=enabled)
    notebook = NoteBook(store)
    tools: tuple[AgentTool, ...] = (
        ReadIndexTool(notebook),
        ReadNoteTool(notebook),
        WriteNoteTool(store, notebook),
        UpdateNoteTool(store, notebook),
        DeleteNoteTool(store, notebook),
    )
    return MemoryConsolidator(
        store=store,
        notebook=notebook,
        meta_tools=MetaToolRegistry(tools),
        model=model,  # type: ignore[arg-type]
        prompt_builder=_builder(),
        max_turns=max_turns,
    )


def _inbox_lines(root: Path, layer: MemoryLayer = MemoryLayer.PROJECT) -> list[str]:
    path = _store(root).inbox_path(layer, _SESSION)
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


class ConsolidationTests(IsolatedAsyncioTestCase):
    async def test_disabled_memory_never_reaches_the_model(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _remember(_store(root), "fact")
            model = _ScriptedModel([_final_turn("done")])

            await _fixture(root, model, enabled=False).consolidate(session_id=_SESSION)

            self.assertEqual(model.purposes, [])

    async def test_an_empty_inbox_never_reaches_the_model(self) -> None:
        with TemporaryDirectory() as directory:
            model = _ScriptedModel([_final_turn("done")])

            await _fixture(Path(directory), model).consolidate(session_id=_SESSION)

            self.assertEqual(model.purposes, [])

    async def test_prompt_carries_the_index_and_the_pending_records(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = NoteBook(_store(root))
            notebook.write_note(
                MemoryLayer.PROJECT,
                title="build-command",
                description="how to build",
                body="uv build",
            )
            _remember(_store(root), "user likes slay the spire")
            model = _ScriptedModel([_final_turn("done")])

            await _fixture(root, model).consolidate(session_id=_SESSION)

            system = model.messages[0][0]
            human = model.messages[0][1]
            self.assertEqual("system", system.role)
            self.assertIn("- build-command\t", system.content)
            self.assertIn("remember user likes slay the spire", system.content)
            self.assertIn('<memory_layer name="user">', system.content)
            self.assertEqual("human", human.role)

    async def test_a_record_becomes_a_note_and_leaves_the_inbox(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _remember(_store(root), "user likes slay the spire")
            model = _ScriptedModel(
                [
                    _tool_turn(
                        "write_note",
                        {
                            "layer": "project",
                            "title": "game preference",
                            "description": "what type of game user like",
                            "body": "User likes Slay the Spire.",
                        },
                    ),
                    _final_turn("created one note"),
                ]
            )

            await _fixture(root, model).consolidate(session_id=_SESSION)

            base = (root / "workspace").resolve() / ".agent-desk" / ".memory"
            self.assertEqual(
                (base / "game-preference.md").read_text(encoding="utf-8"),
                "User likes Slay the Spire.\n",
            )
            index = (base / "MEMORY.md").read_text(encoding="utf-8")
            self.assertIn("game-preference\t", index)
            self.assertIn("what type of game user like", index)
            self.assertEqual(_inbox_lines(root), [])
            self.assertEqual(model.purposes, [ModelCallPurpose.NON_CHAT] * 2)

    async def test_a_forget_record_takes_the_note_out_of_the_index(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root)
            NoteBook(store).write_note(
                MemoryLayer.USER,
                title="old-role",
                description="d",
                body="was a backend engineer",
            )
            store.record(
                MemoryOperation.FORGET,
                "the old role note is stale",
                session_id=_SESSION,
                layer=MemoryLayer.USER,
            )
            model = _ScriptedModel(
                [
                    _tool_turn("delete_note", {"layer": "user", "title": "old-role"}),
                    _final_turn("dropped it"),
                ]
            )

            await _fixture(root, model).consolidate(session_id=_SESSION)

            base = (root / "home").resolve() / ".agent-desk" / ".memory"
            self.assertFalse((base / "MEMORY.md").exists())
            self.assertFalse((base / "old-role.md").exists())
            self.assertEqual(_inbox_lines(root, MemoryLayer.USER), [])

    async def test_invalid_arguments_are_answered_and_the_run_still_finishes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _remember(_store(root), "fact")
            model = _ScriptedModel(
                [
                    _tool_turn("write_note", {"layer": "project", "title": "t"}),
                    _final_turn("gave up"),
                ]
            )

            await _fixture(root, model).consolidate(session_id=_SESSION)

            answer = model.messages[1][-1]
            self.assertEqual("tool", answer.role)
            self.assertEqual(json.loads(answer.content)["error"]["code"], "invalid_tool_arguments")

    async def test_model_failure_is_logged_and_the_records_survive(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _remember(_store(root), "fact")

            class _Boom:
                async def ainvoke(self, **_kwargs: object) -> ModelTurn:
                    raise RuntimeError("provider exploded")

            with self.assertLogs("agent.application.memory.consolidator", level="ERROR"):
                await _fixture(root, _Boom()).consolidate(session_id=_SESSION)

            self.assertEqual(len(_inbox_lines(root)), 1)

    async def test_exhausting_the_turn_budget_keeps_the_records(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _remember(_store(root), "fact")
            model = _ScriptedModel(
                [_tool_turn("read_index", {"layer": "project"})],
                repeat_last=True,
            )

            with self.assertLogs("agent.application.memory.consolidator", level="ERROR"):
                await _fixture(root, model, max_turns=2).consolidate(session_id=_SESSION)

            self.assertEqual(len(model.purposes), 2)
            self.assertEqual(len(_inbox_lines(root)), 1)


class ConsolidationPortTests(IsolatedAsyncioTestCase):
    async def test_note_update_goes_through_the_notebook(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root)
            NoteBook(store).write_note(
                MemoryLayer.PROJECT,
                title="game-preference",
                description="what type of game user like",
                body="User likes Slay the Spire.",
            )
            _remember(store, "user also likes into the breach")
            model = _ScriptedModel(
                [
                    _tool_turn(
                        "update_note",
                        {
                            "layer": "project",
                            "title": "game-preference",
                            "body": "User likes Slay the Spire and Into the Breach.",
                        },
                    ),
                    _final_turn("merged"),
                ]
            )

            await _fixture(root, model).consolidate(session_id=_SESSION)

            note = NoteBook(store).read_note(MemoryLayer.PROJECT, "game-preference")
            self.assertEqual(
                note.body,
                "User likes Slay the Spire and Into the Breach.\n",
            )
            self.assertEqual(_inbox_lines(root), [])
