from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase

from agent.domain.memory import MemoryLayer, PendingRecord
from agent.domain.tools import ToolContext
from agent.infrastructure.memory import MEMORY_DISABLED_MESSAGE, MemoryStore
from agent.infrastructure.metatools.forget import FORGET_DEFINITION, ForgetTool
from agent.infrastructure.metatools.remember import REMEMBER_DEFINITION, RememberTool

_CONTEXT = ToolContext(caller_session_id="session-1")


def _payload(result) -> dict[str, object]:
    return json.loads(result.content)


def _store(root: Path, *, enabled: bool) -> MemoryStore:
    return MemoryStore(start=root / "workspace", home=root / "home", enabled=enabled)


def _pending(store: MemoryStore, layer: MemoryLayer) -> list[PendingRecord]:
    """按真实读接口取待整理记录，而不是自己猜日志的行格式。"""

    return list(store.pending_records(layer))


class RememberToolTests(IsolatedAsyncioTestCase):
    async def test_records_one_pending_line(self) -> None:
        with TemporaryDirectory() as directory:
            store = _store(Path(directory), enabled=True)

            result = await RememberTool(store).aexecute(
                {"content": "  this repo installs deps with uv  ", "layer": "project"},
                context=_CONTEXT,
            )

            recorded = _payload(result)["recorded"]  # type: ignore[index]
            self.assertEqual(recorded["op"], "remember")  # type: ignore[index]
            self.assertEqual(recorded["content"], "this repo installs deps with uv")  # type: ignore[index]
            self.assertEqual(recorded["layer"], "project")  # type: ignore[index]
            pending = _pending(store, MemoryLayer.PROJECT)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].operation.value, "remember")
            self.assertEqual(pending[0].content, "this repo installs deps with uv")
            self.assertEqual(pending[0].session_id, _CONTEXT.caller_session_id)
            self.assertEqual(recorded["at"], pending[0].created_at.isoformat())  # type: ignore[index]

    async def test_disabled_settings_reject_without_writing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root, enabled=False)

            result = await RememberTool(store).aexecute(
                {"content": "anything", "layer": "user"}, context=_CONTEXT
            )

            payload = _payload(result)
            self.assertEqual(payload["error"]["code"], "memory_disabled")  # type: ignore[index]
            self.assertEqual(payload["error"]["message"], MEMORY_DISABLED_MESSAGE)  # type: ignore[index]
            self.assertFalse((root / "home" / ".agent-desk").exists())
            self.assertFalse((root / "workspace" / ".agent-desk").exists())

    async def test_unknown_layer_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            store = _store(Path(directory), enabled=True)

            result = await RememberTool(store).aexecute(
                {"content": "fact", "layer": "global"}, context=_CONTEXT
            )

            self.assertEqual(_payload(result)["error"]["code"], "invalid_layer")  # type: ignore[index]
            self.assertEqual(_pending(store, MemoryLayer.PROJECT), [])

    def test_remember_needs_no_permission_approval(self) -> None:
        with TemporaryDirectory() as directory:
            tool = RememberTool(_store(Path(directory), enabled=True))

            self.assertIsNone(tool.permission_targets({"content": "fact", "layer": "user"}))


class ForgetToolTests(IsolatedAsyncioTestCase):
    async def test_forget_appends_to_the_same_pending_log(self) -> None:
        with TemporaryDirectory() as directory:
            store = _store(Path(directory), enabled=True)

            await RememberTool(store).aexecute(
                {"content": "prefers terse answers", "layer": "user"}, context=_CONTEXT
            )
            result = await ForgetTool(store).aexecute(
                {"content": "prefers terse answers", "layer": "user"}, context=_CONTEXT
            )

            self.assertEqual(_payload(result)["recorded"]["op"], "forget")  # type: ignore[index]
            self.assertEqual(
                [item.operation.value for item in _pending(store, MemoryLayer.USER)],
                ["remember", "forget"],
            )

    def test_forget_needs_no_permission_approval(self) -> None:
        with TemporaryDirectory() as directory:
            tool = ForgetTool(_store(Path(directory), enabled=True))

            self.assertIsNone(tool.permission_targets({"content": "fact", "layer": "user"}))

    async def test_disabled_settings_reject_without_writing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root, enabled=False)

            result = await ForgetTool(store).aexecute(
                {"content": "anything", "layer": "project"}, context=_CONTEXT
            )

            self.assertEqual(_payload(result)["error"]["code"], "memory_disabled")  # type: ignore[index]
            self.assertFalse((root / "workspace" / ".agent-desk").exists())


class MemoryToolDefinitionTests(TestCase):
    def test_definitions_match_their_tools(self) -> None:
        with TemporaryDirectory() as directory:
            store = _store(Path(directory), enabled=True)

            self.assertEqual(REMEMBER_DEFINITION.name, RememberTool(store).definition.name)
            self.assertEqual(FORGET_DEFINITION.name, ForgetTool(store).definition.name)

    def test_layer_is_required_in_both_schemas(self) -> None:
        for definition in (REMEMBER_DEFINITION, FORGET_DEFINITION):
            self.assertEqual(
                sorted(definition.parameters["required"]),  # type: ignore[arg-type]
                ["content", "layer"],
            )
