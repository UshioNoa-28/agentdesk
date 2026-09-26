from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from jsonschema import ValidationError, validate

from agent.domain.tools import ToolContext
from agent.infrastructure.metatools.workspace import (
    EDIT_FILE_DEFINITION,
    READ_FILE_DEFINITION,
    SEARCH_TEXT_DEFINITION,
    EditFileTool,
    ReadFileTool,
    SearchTextTool,
)

_CONTEXT = ToolContext(caller_session_id="workspace-tests")


def _payload(result) -> dict[str, object]:
    payload = json.loads(result.content)
    if not isinstance(payload, dict):
        raise AssertionError(f"tool result is not an object: {payload!r}")
    return payload


def _error_code(payload: dict[str, object]) -> object:
    """取统一信封里的失败码；没有 error 对象就是信封用错了。"""

    error = payload.get("error")
    if not isinstance(error, dict):
        raise AssertionError(f"envelope has no error object: {payload!r}")
    return error.get("code")


def _assert_rejected_by_schema(
    test: TestCase,
    definition,
    arguments: dict[str, object],
) -> None:
    """参数在进入工具前就该被 definition.parameters 拒掉。"""

    with test.assertRaises(ValidationError):
        validate(instance=arguments, schema=dict(definition.parameters))


def _symlink_or_skip(test: TestCase, link: Path, target: Path | str) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        test.skipTest(f"symbolic links are unavailable: {exc}")


class WorkspaceDefinitionTests(TestCase):
    def test_workspace_definitions_have_strict_path_contracts(self) -> None:
        definitions = (
            READ_FILE_DEFINITION,
            SEARCH_TEXT_DEFINITION,
            EDIT_FILE_DEFINITION,
        )

        self.assertEqual(
            (
                "read_file",
                "search_text",
                "edit_file",
            ),
            tuple(definition.name for definition in definitions),
        )
        for definition in definitions:
            with self.subTest(tool=definition.name):
                self.assertEqual("object", definition.parameters["type"])
                self.assertFalse(definition.parameters.get("additionalProperties"))
                self.assertIn("path", definition.parameters["properties"])
                self.assertIn("path", definition.parameters["required"])

    def test_edit_definition_is_pure_replace(self) -> None:
        properties = EDIT_FILE_DEFINITION.parameters["properties"]
        self.assertEqual(
            ["path", "old_text", "new_text"],
            EDIT_FILE_DEFINITION.parameters["required"],
        )
        self.assertNotIn("content", properties)
        self.assertNotIn("operation", properties)

    def test_edit_definition_rejects_removed_and_missing_arguments(self) -> None:
        _assert_rejected_by_schema(
            self, EDIT_FILE_DEFINITION, {"path": "/tmp/a.txt", "operation": "replace"}
        )
        _assert_rejected_by_schema(
            self,
            EDIT_FILE_DEFINITION,
            {"path": "/tmp/a.txt", "content": "x", "new_text": "y"},
        )
        _assert_rejected_by_schema(
            self, EDIT_FILE_DEFINITION, {"path": "/tmp/a.txt", "new_text": "y"}
        )
        _assert_rejected_by_schema(
            self,
            EDIT_FILE_DEFINITION,
            {"path": "/tmp/a.txt", "old_text": "", "new_text": "y"},
        )


class WorkspacePathContractTests(IsolatedAsyncioTestCase):
    """绝对路径是接口约定：相对路径一律拒之门外，越界交给权限规则管。"""

    async def test_all_path_tools_reject_relative_empty_and_nul_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("relative sentinel", encoding="utf-8")
            cases = (
                ("read_file", ReadFileTool(), {"path": ""}),
                (
                    "search_text",
                    SearchTextTool(),
                    {"query": "sentinel"},
                ),
                ("edit_file", EditFileTool(), {"path": "", "old_text": "a", "new_text": "b"}),
            )
            paths = (
                "notes.txt",
                "./notes.txt",
                "../notes.txt",
                "nested/../notes.txt",
                r"..\notes.txt",
                r"C:\workspace\notes.txt",
                "bad\x00path",
            )
            for raw_path in paths:
                for name, tool, arguments in cases:
                    call_arguments = dict(arguments, path=raw_path)
                    with self.subTest(tool=name, path=raw_path):
                        payload = _payload(await tool.aexecute(call_arguments, context=_CONTEXT))
                        self.assertFalse(payload["ok"])
                        self.assertTrue(_error_code(payload))

    async def test_search_text_requires_a_path(self) -> None:
        payload = _payload(await SearchTextTool().aexecute({"query": "x"}, context=_CONTEXT))
        self.assertFalse(payload["ok"])
        self.assertTrue(_error_code(payload))

    async def test_absolute_paths_work_anywhere_on_disk(self) -> None:
        with TemporaryDirectory() as elsewhere:
            outside_file = Path(elsewhere) / "notes.txt"
            outside_file.write_text("outside content", encoding="utf-8")

            payload = _payload(
                await ReadFileTool().aexecute({"path": str(outside_file)}, context=_CONTEXT)
            )
            self.assertTrue(payload["ok"])
            self.assertEqual("     1\toutside content", payload["content"])
            self.assertEqual(outside_file.as_posix(), payload["path"])


class ReadFileToolTests(IsolatedAsyncioTestCase):
    async def test_reads_utf8_text_and_inclusive_line_window(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "notes.txt"
            content = "first\nsecond\nthird\n"
            path.write_text(content, encoding="utf-8")

            payload = _payload(
                await ReadFileTool().aexecute(
                    {"path": str(path), "start_line": 2, "end_line": 3},
                    context=_CONTEXT,
                )
            )

            self.assertTrue(payload["ok"])
            self.assertEqual("     2\tsecond\n     3\tthird\n", payload["content"])
            self.assertEqual(len(content.encode("utf-8")), payload["size_bytes"])
            self.assertFalse(payload["truncated"])

            whole = _payload(await ReadFileTool().aexecute({"path": str(path)}, context=_CONTEXT))
            self.assertEqual("     1\tfirst\n     2\tsecond\n     3\tthird\n", whole["content"])

    async def test_max_bytes_marks_file_truncation_without_invalid_json(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "large.txt"
            path.write_text("abcdefghij", encoding="utf-8")
            payload = _payload(
                await ReadFileTool().aexecute({"path": str(path), "max_bytes": 4}, context=_CONTEXT)
            )

            self.assertTrue(payload["ok"])
            self.assertEqual("     1\tabcd", payload["content"])
            self.assertTrue(payload["truncated"])

    async def test_rejects_directory_and_invalid_line_range(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "folder"
            folder.mkdir()
            tool = ReadFileTool()
            directory_payload = _payload(
                await tool.aexecute({"path": str(folder)}, context=_CONTEXT)
            )
            range_payload = _payload(
                await tool.aexecute(
                    {"path": str(folder), "start_line": 3, "end_line": 1},
                    context=_CONTEXT,
                )
            )

            self.assertFalse(directory_payload["ok"])
            self.assertFalse(range_payload["ok"])

    async def test_internal_file_symlink_can_be_read(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.txt").write_text("through link", encoding="utf-8")
            alias = root / "alias.txt"
            _symlink_or_skip(self, alias, root / "target.txt")

            payload = _payload(
                await ReadFileTool().aexecute({"path": str(alias)}, context=_CONTEXT)
            )

            self.assertTrue(payload["ok"])
            self.assertEqual("     1\tthrough link", payload["content"])
            self.assertEqual(alias.as_posix(), payload["path"])


class SearchTextToolTests(IsolatedAsyncioTestCase):
    async def test_literal_search_reports_paths_lines_and_case_mode(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            a_txt = root / "a.txt"
            a_txt.write_text("Needle here\nno match\nneedle again\n", encoding="utf-8")
            (root / "nested").mkdir()
            b_txt = root / "nested" / "b.txt"
            b_txt.write_text("NEEDLE nested\n", encoding="utf-8")
            (root / "binary.bin").write_bytes(b"\x00needle\n")
            _symlink_or_skip(self, root / "a-link.txt", a_txt)

            tool = SearchTextTool()
            sensitive = _payload(
                await tool.aexecute({"query": "Needle", "path": str(root)}, context=_CONTEXT)
            )
            insensitive = _payload(
                await tool.aexecute(
                    {"query": "needle", "path": str(root), "case_sensitive": False},
                    context=_CONTEXT,
                )
            )

            self.assertEqual(1, sensitive["match_count"])
            self.assertEqual(a_txt.as_posix(), sensitive["matches"][0]["path"])
            self.assertEqual(1, sensitive["matches"][0]["line"])
            self.assertEqual(3, insensitive["match_count"])
            self.assertEqual(
                {a_txt.as_posix(), b_txt.as_posix()},
                {match["path"] for match in insensitive["matches"]},
            )
            self.assertNotIn(
                (root / "binary.bin").as_posix(),
                {match["path"] for match in insensitive["matches"]},
            )
            # binary.bin 也被扫描（只是不产生匹配），目录里共三个普通文件。
            self.assertEqual(3, insensitive["files_scanned"])

    async def test_result_limit_and_file_byte_limit_are_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            matches_txt = root / "matches.txt"
            matches_txt.write_text("needle\nneedle\nneedle\n", encoding="utf-8")
            limited = _payload(
                await SearchTextTool().aexecute(
                    {"query": "needle", "path": str(matches_txt), "max_results": 2},
                    context=_CONTEXT,
                )
            )
            late = root / "late.txt"
            late.write_text("prefix\nneedle\n", encoding="utf-8")
            byte_limited = _payload(
                await SearchTextTool(max_file_bytes=8).aexecute(
                    {"query": "needle", "path": str(late)}, context=_CONTEXT
                )
            )

            self.assertEqual(2, limited["match_count"])
            self.assertTrue(limited["truncated"])
            self.assertEqual(0, byte_limited["match_count"])
            self.assertTrue(byte_limited["truncated"])

    def test_arguments_rejected_before_reaching_the_tool(self) -> None:
        # 类型与下界由 definition.parameters 声明，graph 在校验失败时直接回信封，
        # 工具内不再重复 isinstance/区间判断——这里守住那个上移后的边界。
        _assert_rejected_by_schema(self, SEARCH_TEXT_DEFINITION, {"query": "", "path": "/tmp"})
        _assert_rejected_by_schema(
            self,
            SEARCH_TEXT_DEFINITION,
            {"query": "x", "path": "/tmp", "case_sensitive": "false"},
        )
        _assert_rejected_by_schema(
            self, SEARCH_TEXT_DEFINITION, {"path": "/tmp", "case_sensitive": True}
        )
        _assert_rejected_by_schema(
            self, SEARCH_TEXT_DEFINITION, {"query": "x", "path": "/tmp", "max_results": 0}
        )


class EditFileToolTests(IsolatedAsyncioTestCase):
    async def test_replace_is_atomic_and_preserves_file_mode(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "target.txt"
            path.write_text("old", encoding="utf-8")
            os.chmod(path, 0o640)
            old_inode = path.stat().st_ino

            payload = _payload(
                await EditFileTool().aexecute(
                    {"path": str(path), "old_text": "old", "new_text": "new"},
                    context=_CONTEXT,
                )
            )

            self.assertTrue(payload["ok"])
            self.assertEqual("new", path.read_text(encoding="utf-8"))
            self.assertEqual(3, payload["bytes_written"])
            self.assertEqual(0o640, stat.S_IMODE(path.stat().st_mode))
            self.assertNotEqual(old_inode, path.stat().st_ino)
            self.assertEqual([], list(root.glob(".*.agentdesk-tmp")))

    async def test_replace_requires_exactly_one_match(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.txt"
            path.write_text("port=8000\nmode=dev\n", encoding="utf-8")
            tool = EditFileTool()

            missing = _payload(
                await tool.aexecute(
                    {"path": str(path), "old_text": "absent", "new_text": "x"},
                    context=_CONTEXT,
                )
            )
            self.assertFalse(missing["ok"])
            self.assertEqual("port=8000\nmode=dev\n", path.read_text(encoding="utf-8"))

            path.write_text("dup\ndup\n", encoding="utf-8")
            duplicated = _payload(
                await tool.aexecute(
                    {"path": str(path), "old_text": "dup", "new_text": "x"},
                    context=_CONTEXT,
                )
            )
            self.assertFalse(duplicated["ok"])
            self.assertEqual("dup\ndup\n", path.read_text(encoding="utf-8"))

    async def test_empty_new_text_deletes_the_snippet(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.txt"
            path.write_text("a=1\nb=2\n", encoding="utf-8")

            payload = _payload(
                await EditFileTool().aexecute(
                    {"path": str(path), "old_text": "b=2\n", "new_text": ""},
                    context=_CONTEXT,
                )
            )

            self.assertTrue(payload["ok"])
            self.assertEqual("a=1\n", path.read_text(encoding="utf-8"))

    async def test_failed_replace_keeps_file_untouched_and_cleans_temp(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.txt"
            path.write_text("port=8000\n", encoding="utf-8")
            tool = EditFileTool()

            with patch(
                "agent.infrastructure.metatools.workspace.os.replace",
                side_effect=OSError("injected replace failure"),
            ):
                failed = _payload(
                    await tool.aexecute(
                        {
                            "path": str(path),
                            "old_text": "port=8000",
                            "new_text": "x",
                        },
                        context=_CONTEXT,
                    )
                )

            self.assertFalse(failed["ok"])
            self.assertEqual("port=8000\n", path.read_text(encoding="utf-8"))
            self.assertEqual([], list(root.glob(".config.txt.*.agentdesk-tmp")))

    async def test_missing_file_and_directory_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "folder").mkdir()
            tool = EditFileTool()

            missing = _payload(
                await tool.aexecute(
                    {"path": str(root / "missing.txt"), "old_text": "a", "new_text": "b"},
                    context=_CONTEXT,
                )
            )
            directory_payload = _payload(
                await tool.aexecute(
                    {"path": str(root / "folder"), "old_text": "a", "new_text": "b"},
                    context=_CONTEXT,
                )
            )

            self.assertFalse(missing["ok"])
            self.assertFalse(directory_payload["ok"])
            self.assertTrue((root / "folder").is_dir())

    async def test_replace_via_internal_symlink_replaces_link_not_target(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.txt"
            link = root / "link.txt"
            target.write_text("keep target", encoding="utf-8")
            _symlink_or_skip(self, link, target)

            payload = _payload(
                await EditFileTool().aexecute(
                    {"path": str(link), "old_text": "keep", "new_text": "new"},
                    context=_CONTEXT,
                )
            )

            self.assertTrue(payload["ok"])
            self.assertFalse(link.is_symlink())
            self.assertEqual("new target", link.read_text(encoding="utf-8"))
            self.assertEqual("keep target", target.read_text(encoding="utf-8"))


__all__ = [
    "EditFileToolTests",
    "ReadFileToolTests",
    "SearchTextToolTests",
    "WorkspaceDefinitionTests",
    "WorkspacePathContractTests",
]
