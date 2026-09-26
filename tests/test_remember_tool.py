from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from agent.domain.messages import MessageKind
from agent.domain.tools import ToolContext
from agent.infrastructure.memory import MEMORY_INDEX_FILE_NAME
from agent.infrastructure.metatools.remember import RememberTool

_WRITTEN_RE = re.compile(
    r"^written: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$",
    re.MULTILINE,
)


def _context() -> ToolContext:
    return ToolContext(caller_session_id="s-1")


class RememberToolTests(IsolatedAsyncioTestCase):
    """remember 在两个注入的目录层里选一个写入，并同步索引。"""

    def setUp(self) -> None:
        self._dir = TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        base = Path(self._dir.name)
        self.user_dir = base / "user" / "memory"
        self.project_dir = base / "project" / "memory"
        self.tool = RememberTool(user_dir=self.user_dir, project_dir=self.project_dir)

    async def _remember(self, layer: str, **overrides: object):
        arguments: dict[str, object] = {
            "layer": layer,
            "title": "缩进偏好",
            "description": "用户坚持 tab 缩进，spaces 会被退回",
            "body": "多次纠正过：spaces 会被退回。",
        }
        arguments.update(overrides)
        return await self.tool.aexecute(arguments, context=_context())

    async def test_user_layer_writes_note_and_index(self) -> None:
        result = await self._remember("user")
        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        self.assertIs(MessageKind.TOOL_RESULT, result.kind)

        text = (self.user_dir / "缩进偏好.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\nwritten: "))
        self.assertRegex(text, _WRITTEN_RE)
        self.assertIn("多次纠正过：spaces 会被退回。", text)
        self.assertNotIn("title:", text)

        index = (self.user_dir / MEMORY_INDEX_FILE_NAME).read_text(encoding="utf-8")
        self.assertEqual(
            ["- [缩进偏好](缩进偏好.md) — 用户坚持 tab 缩进，spaces 会被退回"],
            index.splitlines(),
        )

    async def test_project_layer_is_independent(self) -> None:
        await self._remember("project")
        self.assertTrue((self.project_dir / "缩进偏好.md").exists())
        self.assertFalse(self.user_dir.exists())

    async def test_overwrite_same_title_refreshes_written_and_index(self) -> None:
        await self._remember("user")
        second = json.loads(
            (await self._remember("user", description="新描述：一律 tab")).content
        )
        index = (self.user_dir / MEMORY_INDEX_FILE_NAME).read_text(encoding="utf-8")
        self.assertEqual(
            ["- [缩进偏好](缩进偏好.md) — 新描述：一律 tab"],
            index.splitlines(),
        )
        text = (self.user_dir / "缩进偏好.md").read_text(encoding="utf-8")
        self.assertEqual(1, len(_WRITTEN_RE.findall(text)))
        self.assertIn(second["remembered"]["written"], text)

    async def test_different_titles_append_index_lines(self) -> None:
        await self._remember("user")
        await self._remember("user", title="游戏偏好", description="用户喜欢 A")
        index = (self.user_dir / MEMORY_INDEX_FILE_NAME).read_text(encoding="utf-8")
        self.assertEqual(
            [
                "- [缩进偏好](缩进偏好.md) — 用户坚持 tab 缩进，spaces 会被退回",
                "- [游戏偏好](游戏偏好.md) — 用户喜欢 A",
            ],
            index.splitlines(),
        )

    async def test_rejects_bad_title(self) -> None:
        cases = (
            "",
            "  ",
            "../escape",
            "a/b",
            "a\\b",
            "[ bracket",
            "带(括号)",
            ".hidden",
            "a" * 65,
            "两\n行",
        )
        for bad in cases:
            with self.subTest(bad=bad):
                result = await self._remember("user", title=bad)
                payload = json.loads(result.content)
                self.assertFalse(payload["ok"])
                self.assertEqual("invalid_memory", payload["error"]["code"])

    async def test_rejects_bad_layer(self) -> None:
        result = await self._remember("workspace")
        self.assertEqual("invalid_memory", json.loads(result.content)["error"]["code"])

    async def test_rejects_bad_description(self) -> None:
        for bad in ("", "   ", "两\n行", "带[方括号]的"):
            with self.subTest(bad=bad):
                result = await self._remember("user", description=bad)
                self.assertFalse(json.loads(result.content)["ok"])

    async def test_rejects_empty_body(self) -> None:
        result = await self._remember("user", body="   ")
        self.assertFalse(json.loads(result.content)["ok"])

    async def test_permission_targets_is_note_absolute_path(self) -> None:
        targets = self.tool.permission_targets({"layer": "user", "title": "缩进偏好"})
        self.assertEqual(
            ((self.user_dir / "缩进偏好.md").resolve(strict=False).as_posix(),),
            targets,
        )

    async def test_permission_targets_rejects_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            self.tool.permission_targets({"layer": "workspace", "title": "ok"})
        with self.assertRaises(ValueError):
            self.tool.permission_targets({"layer": "user", "title": "a/b"})
