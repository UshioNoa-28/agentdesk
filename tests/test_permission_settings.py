from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agent.domain.permissions import (
    PermissionAction,
    PermissionMode,
    ToolRule,
)
from agent.domain.tools import ALL_META_TOOL_NAMES
from agent.infrastructure.permission import (
    PermissionManager,
    load_permission_manager,
)
from agent.infrastructure.permission.settings import (
    append_permission_rule,
    load_permission_mode,
    load_permission_rule,
)
from agent.infrastructure.project_settings import project_settings_path


def _write_settings(base: Path, payload: object) -> Path:
    path = project_settings_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def _read_settings(base: Path) -> dict:
    return json.loads(project_settings_path(base).read_text(encoding="utf-8"))


class PermissionRuleParsingTests(TestCase):
    def test_bare_tool_name_parses_to_empty_scope(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"allow": ["read_file"]}})

            rule = load_permission_rule(root, home=root)

            self.assertEqual([ToolRule("read_file", "")], rule["allow"])

    def test_scoped_rule_parses_tool_and_scope(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"deny": ["edit_file(src/**)"]}})

            rule = load_permission_rule(root, home=root)

            self.assertEqual([ToolRule("edit_file", "src/**")], rule["deny"])
            self.assertEqual([], rule["allow"])

    def test_parenthesized_rule_without_close_paren_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"allow": ["edit_file(src"]}})

            with self.assertRaisesRegex(ValueError, "Invalid permission rule"):
                load_permission_rule(root, home=root)

    def test_empty_scope_parens_are_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"allow": ["edit_file()"]}})

            with self.assertRaisesRegex(ValueError, "Invalid permission rule"):
                load_permission_rule(root, home=root)

    def test_unknown_meta_tool_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"ask": ["typo_tool"]}})

            with self.assertRaisesRegex(ValueError, "Unknown Meta Tool"):
                load_permission_rule(root, home=root)

    def test_non_string_entry_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"deny": [42]}})

            with self.assertRaisesRegex(
                ValueError, r"settings 'permissions\.deny' requires non-empty strings"
            ):
                load_permission_rule(root, home=root)


class PermissionRuleMergingTests(TestCase):
    def test_layers_union_deduplicated(self) -> None:
        with TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            project = home / "work" / "repo"
            project.mkdir(parents=True)
            _write_settings(
                project,
                {"permissions": {"allow": ["read_file", "edit_file(src/**)"]}},
            )
            _write_settings(
                home,
                {"permissions": {"allow": ["read_file", "search_mcp"]}},
            )

            rule = load_permission_rule(project, home=home)

            # 从 home 向项目扫描、去重保留先见者:home 的规则排在前。
            self.assertEqual(
                [
                    ToolRule("read_file", ""),
                    ToolRule("search_mcp", ""),
                    ToolRule("edit_file", "src/**"),
                ],
                rule["allow"],
            )

    def test_duplicated_scoped_rule_keeps_first_layer(self) -> None:
        with TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            project = home / "repo"
            project.mkdir(parents=True)
            _write_settings(project, {"permissions": {"deny": ["execute_mcp"]}})
            _write_settings(
                home, {"permissions": {"deny": ["execute_mcp", "edit_file"]}}
            )

            rule = load_permission_rule(project, home=home)

            self.assertEqual(
                [ToolRule("execute_mcp", ""), ToolRule("edit_file", "")],
                rule["deny"],
            )

    def test_same_tool_in_different_actions_stays_separate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(
                root,
                {
                    "permissions": {
                        "allow": ["read_file(a)"],
                        "ask": ["read_file(b)"],
                    }
                },
            )

            rule = load_permission_rule(root, home=root)

            self.assertEqual([ToolRule("read_file", "a")], rule["allow"])
            self.assertEqual([ToolRule("read_file", "b")], rule["ask"])


class PermissionModeLoadingTests(TestCase):
    def test_nearest_layer_defining_mode_wins(self) -> None:
        with TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            project = home / "repo"
            project.mkdir(parents=True)
            _write_settings(project, {"permissions": {}})
            _write_settings(home, {"permissions": {"mode": "bypass"}})

            self.assertEqual("bypass", load_permission_mode(project, home=home))
            _write_settings(project, {"permissions": {"mode": "default"}})
            self.assertEqual("default", load_permission_mode(project, home=home))

    def test_broken_mode_in_overridden_layer_is_still_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            project = home / "repo"
            project.mkdir(parents=True)
            _write_settings(project, {"permissions": {"mode": "bypass"}})
            _write_settings(home, {"permissions": {"mode": ""}})

            with self.assertRaisesRegex(ValueError, "non-empty string"):
                load_permission_mode(project, home=home)

    def test_absent_files_default(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            self.assertEqual("default", load_permission_mode(root, home=root))

    def test_empty_mode_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"mode": ""}})

            with self.assertRaisesRegex(ValueError, "non-empty string"):
                load_permission_mode(root, home=root)

    def test_unknown_mode_fails_manager_construction(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"mode": "brpass"}})

            with self.assertRaisesRegex(ValueError, "not a valid PermissionMode"):
                load_permission_manager(root, home=root)


class AppendPermissionRuleTests(TestCase):
    def test_creates_file_and_preserves_unknown_keys(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"future": {"model": "gpt"}})

            append_permission_rule(root, PermissionAction.ALLOW, "edit_file", "")

            written = _read_settings(root)
            self.assertEqual(["edit_file"], written["permissions"]["allow"])
            self.assertEqual({"model": "gpt"}, written["future"])

    def test_scoped_rule_round_trips_through_loader(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            append_permission_rule(root, "ask", "edit_file", "src/**")
            append_permission_rule(root, "ask", "read_file", "")

            written = _read_settings(root)
            self.assertEqual(
                ["edit_file(src/**)", "read_file"],
                written["permissions"]["ask"],
            )
            rule = load_permission_rule(root, home=root)
            self.assertEqual(
                [ToolRule("edit_file", "src/**"), ToolRule("read_file", "")],
                rule["ask"],
            )

    def test_appends_to_existing_entries(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"allow": ["read_file"]}})

            append_permission_rule(root, PermissionAction.DENY, "edit_file", "x")

            written = _read_settings(root)
            self.assertEqual(["read_file"], written["permissions"]["allow"])
            self.assertEqual(["edit_file(x)"], written["permissions"]["deny"])

    def test_invalid_action_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaisesRegex(ValueError, "'bogus' is not a valid"):
                append_permission_rule(root, "bogus", "read_file", "")

    def test_non_string_list_target_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"allow": "read_file"}})

            with self.assertRaisesRegex(ValueError, "must be a string list"):
                append_permission_rule(root, "allow", "edit_file", "")


class PermissionManagerTests(TestCase):
    def test_persist_rule_updates_memory_and_disk_bare(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PermissionManager(start=root)

            manager.persist_rule("read_file", PermissionAction.ALLOW)

            self.assertTrue(manager.in_allow("read_file"))
            self.assertEqual(
                ["read_file"], _read_settings(root)["permissions"]["allow"]
            )
            reloaded = load_permission_manager(root, home=root)
            self.assertTrue(reloaded.in_allow("read_file", "any/path.ts"))

    def test_instances_do_not_share_default_rules(self) -> None:
        with TemporaryDirectory() as directory:
            first = PermissionManager(start=Path(directory) / "a")
            second = PermissionManager(start=Path(directory) / "b")

            first.persist_rule("edit_file", PermissionAction.ALLOW)

            self.assertTrue(first.in_allow("edit_file"))
            self.assertFalse(second.in_allow("edit_file"))

    def test_persist_rule_updates_memory_and_disk(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PermissionManager(start=root)

            manager.persist_rule("edit_file", PermissionAction.DENY, "src/**")

            # 带 scope 的规则:给了 target 才参与匹配,空 target 不误伤。
            self.assertFalse(manager.in_deny("edit_file"))
            self.assertTrue(manager.in_deny("edit_file", "src/a/b.ts"))
            self.assertFalse(manager.in_deny("edit_file", "docs/c.ts"))
            self.assertEqual(
                ["edit_file(src/**)"], _read_settings(root)["permissions"]["deny"]
            )
            reloaded = load_permission_manager(root, home=root)
            self.assertTrue(reloaded.in_deny("edit_file", "src/x.ts"))
            self.assertFalse(reloaded.in_allow("edit_file", "src/x.ts"))

    def test_plain_scope_is_exact_glob(self) -> None:
        manager = PermissionManager(
            start=Path(),
            permission_rule={"deny": [ToolRule("send_message", "boss")]},
        )

        self.assertTrue(manager.in_deny("send_message", "boss"))
        self.assertFalse(manager.in_deny("send_message", "boss,cc"))

    def test_persist_rule_rejects_unknown_tool_without_side_effects(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PermissionManager(start=root)

            with self.assertRaises(ValueError):
                manager.persist_rule("no_such_tool", PermissionAction.ALLOW)

            self.assertFalse(manager.in_allow("no_such_tool"))
            self.assertFalse((root / ".agent-desk" / "settings.json").exists())

    def test_persist_rule_rejects_malformed_scope(self) -> None:
        manager = PermissionManager(start=Path())

        for scope in ("rm\nrf /", "a\x00b", "x" * 257):
            with self.subTest(scope=scope):
                with self.assertRaises(ValueError):
                    manager.persist_rule(
                        "edit_file", PermissionAction.DENY, scope
                    )
                self.assertFalse(manager.denied)

    def test_change_mode_updates_memory_and_disk(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PermissionManager(start=root)

            manager.change_mode(PermissionMode.DONT_ASK)

            self.assertIs(PermissionMode.DONT_ASK, manager.mode)
            self.assertEqual(
                "dont_ask", _read_settings(root)["permissions"]["mode"]
            )
            reloaded = load_permission_manager(root, home=root)
            self.assertIs(PermissionMode.DONT_ASK, reloaded.mode)

    def test_persist_survives_write_failure_in_process(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": "broken"})
            manager = PermissionManager(start=root)

            manager.persist_rule("edit_file", PermissionAction.ALLOW)

            self.assertTrue(manager.in_allow("edit_file"))

    def test_load_injects_no_builtin_allows(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(
                root,
                {"permissions": {"ask": ["send_message"], "allow": ["load_skill"]}},
            )
            manager = load_permission_manager(root, home=root)

            # 默认放行名单已删除：免审语义由工具 permission_targets 返回 None
            # 表达，manager 只看见 settings 里真实配置过的规则。
            self.assertFalse(manager.in_allow("search_mcp"))
            self.assertTrue(manager.in_allow("load_skill"))
            # 用户仍可用整工具 ask 规则把无权限面工具拉回审核。
            self.assertTrue(manager.in_ask("send_message"))
            # 与磁盘上已有条目去重。
            self.assertEqual(
                1, sum(1 for rule in manager.allowed if rule.tool == "load_skill")
            )

    def test_defaults_and_mode_from_settings(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"mode": "dont_ask"}})

            self.assertEqual(
                PermissionMode.DEFAULT, PermissionManager(start=root).mode
            )
            self.assertEqual(
                PermissionMode.DONT_ASK, load_permission_manager(root, home=root).mode
            )

    def test_in_ask_reflects_ask_list(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"ask": ["search_mcp"]}})
            manager = load_permission_manager(root, home=root)

            self.assertTrue(manager.in_ask("search_mcp"))
            self.assertFalse(manager.in_ask("read_file"))


class ScopeMatcherTests(TestCase):
    """统一 fnmatch glob,一切从简;要命中目录内容就把模式写出来。"""

    def _manager(self, rules: dict[str, list[ToolRule]]) -> PermissionManager:
        return PermissionManager(start=Path(), permission_rule=rules)

    def test_path_rules_match_by_glob(self) -> None:
        manager = self._manager(
            {
                "allow": [
                    ToolRule("read_file", "src/*.ts"),
                    ToolRule("read_file", "docs/**"),
                ],
                "deny": [ToolRule("read_file", "vendor/**")],
            }
        )

        self.assertTrue(manager.in_allow("read_file", "src/a.ts"))
        # fnmatch 的 * 跨目录:src/*.ts 连子目录里的 .ts 也算命中
        self.assertTrue(manager.in_allow("read_file", "src/n/a.ts"))
        # ./ 前缀归一后按同一模式匹配
        self.assertTrue(manager.in_allow("read_file", "./docs/a/b.md"))
        # 含 slash 的模式锚定在 workspace 根
        self.assertFalse(manager.in_allow("read_file", "pkg/docs/a.md"))
        # /** 命中目录下内容;目录名本身不算文件,不匹配
        self.assertFalse(manager.in_allow("read_file", "docs"))
        self.assertTrue(manager.in_deny("read_file", "vendor/a/s.rs"))
        self.assertFalse(manager.in_deny("read_file", "vendors/x.ts"))

    def test_command_rules_match_by_glob(self) -> None:
        manager = self._manager(
            {
                "deny": [ToolRule("bash", "rm *")],
                "allow": [ToolRule("bash", "git *")],
            }
        )

        self.assertTrue(manager.in_deny("bash", "rm -rf build"))
        # 尾部 " *" 是可选尾巴:裸命令也算命中
        self.assertTrue(manager.in_deny("bash", "rm"))
        self.assertTrue(manager.in_allow("bash", "git"))
        # 模式是全串匹配,不是子串
        self.assertFalse(manager.in_deny("bash", "echo rm x"))
        self.assertTrue(manager.in_allow("bash", "git commit -m x"))
        self.assertFalse(manager.in_allow("bash", "github push"))


class RepositoryExampleTests(TestCase):
    def test_example_loads_and_demonstrates_rule_shapes(self) -> None:
        example = (
            Path(__file__).parents[1] / ".agent-desk" / "settings.json.example"
        ).read_text(encoding="utf-8")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, example)
            manager = load_permission_manager(root, home=root)

            # scope 级 allow 只在命中 target 时生效；文件工具的 target 是绝对路径。
            self.assertTrue(manager.in_allow("read_file", "/home/you/projects/src/a.ts"))
            self.assertFalse(manager.in_allow("read_file", "/home/you/projects/secrets/key.pem"))
            self.assertTrue(manager.in_allow("execute_mcp", "github"))
            self.assertFalse(manager.in_allow("execute_mcp", "other"))
            # 未写规则的危险工具仍走询问。
            self.assertFalse(manager.in_allow("edit_file"))
            # 无权限面工具不再靠种子 allow 表达：默认放行由 permission_targets
            # 返回 None 承担，settings 里只剩真实策略(显式 ask 仍可拉回审核)。
            self.assertFalse(manager.in_allow("ask_user"))
            self.assertTrue(
                all(
                    rule.tool in ALL_META_TOOL_NAMES
                    for rule in (*manager.allowed, *manager.denied, *manager.asked)
                )
            )


__all__ = [
    "PermissionRuleParsingTests",
    "PermissionRuleMergingTests",
    "PermissionModeLoadingTests",
    "AppendPermissionRuleTests",
    "PermissionManagerTests",
    "ScopeMatcherTests",
    "RepositoryExampleTests",
]
