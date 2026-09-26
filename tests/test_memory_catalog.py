from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from agent.domain.memory import MemoryLayer
from agent.domain.tools import ToolContext
from agent.infrastructure.memory import MEMORY_INDEX_FILE_NAME, MemoryCatalog
from agent.infrastructure.metatools.remember import RememberTool


class MemoryCatalogTests(IsolatedAsyncioTestCase):
    """MemoryCatalog 把两层 MEMORY.md 索引现读成 (layer, title, description) 列表。"""

    def setUp(self) -> None:
        self._dir = TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        base = Path(self._dir.name)
        self.user_dir = base / "user"
        self.project_dir = base / "project"
        self.catalog = MemoryCatalog(user_dir=self.user_dir, project_dir=self.project_dir)

    def _write_index(self, directory: Path, text: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MEMORY_INDEX_FILE_NAME).write_text(text, encoding="utf-8")

    def test_missing_layers_list_empty(self) -> None:
        self.assertEqual((), self.catalog.list_entries())

    def test_lists_both_layers_user_first(self) -> None:
        self._write_index(self.user_dir, "- [缩进偏好](缩进偏好.md) — 一律 tab\n")
        self._write_index(
            self.project_dir,
            "- [部署入口](部署入口.md) — make deploy\n- [假想敌](假想敌.md) — x\n",
        )
        entries = self.catalog.list_entries()
        self.assertEqual(
            [
                (MemoryLayer.USER, "缩进偏好", "一律 tab"),
                (MemoryLayer.PROJECT, "部署入口", "make deploy"),
                (MemoryLayer.PROJECT, "假想敌", "x"),
            ],
            [(e.layer, e.title, e.description) for e in entries],
        )

    def test_skips_lines_off_the_protocol(self) -> None:
        self._write_index(
            self.user_dir,
            "# 记忆索引\n- [老格式](old.md)\n- [合规](合规.md) — 能列出\n随便一句话\n",
        )
        entries = self.catalog.list_entries()
        self.assertEqual([("合规", "能列出")], [(e.title, e.description) for e in entries])

    def test_load_prompt_returns_paths_only(self) -> None:
        self._write_index(self.user_dir, "- [缩进偏好](缩进偏好.md) — 一律 tab\n")
        prompt = self.catalog.load_prompt()
        self.assertIn(self.user_dir.as_posix(), prompt)
        self.assertIn(self.project_dir.as_posix(), prompt)
        self.assertNotIn("一律 tab", prompt)

    async def test_lists_what_remember_wrote(self) -> None:
        tool = RememberTool(user_dir=self.user_dir, project_dir=self.project_dir)
        result = await tool.aexecute(
            {
                "layer": "project",
                "title": "游戏偏好",
                "description": "用户喜欢 A",
                "body": "user likes A",
            },
            context=ToolContext(caller_session_id="s-1"),
        )
        self.assertTrue(json.loads(result.content)["ok"])
        entries = self.catalog.list_entries()
        self.assertEqual(
            [(MemoryLayer.PROJECT, "游戏偏好", "用户喜欢 A")],
            [(e.layer, e.title, e.description) for e in entries],
        )

    def test_read_note(self) -> None:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        expected = "# 缩进偏好\n一律 tab"
        (self.user_dir / "缩进偏好.md").write_text(expected, encoding="utf-8")
        self.assertEqual(expected, self.catalog.read_note(MemoryLayer.USER, "缩进偏好"))
        self.assertIsNone(self.catalog.read_note(MemoryLayer.USER, "不存在"))
        self.assertIsNone(self.catalog.read_note(MemoryLayer.PROJECT, "缩进偏好"))
