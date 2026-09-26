"""``bash.command_targets`` 的解析契约测试。"""

from __future__ import annotations

from unittest import TestCase

from agent.infrastructure.metatools.bash import command_targets


class CommandTargetsSplitTests(TestCase):
    """能被逐段核验的命令行，返回引号剥离后的命令段。"""

    def test_single_simple_command(self) -> None:
        self.assertEqual(("git status",), command_targets("git status"))

    def test_whitespace_is_collapsed(self) -> None:
        self.assertEqual(("git status",), command_targets("  git\t  status  "))

    def test_quoted_argument_stays_in_one_segment(self) -> None:
        self.assertEqual(
            ("git commit -m fix: a && b; c",),
            command_targets('git commit -m "fix: a && b; c"'),
        )

    def test_chained_commands_become_one_segment_each(self) -> None:
        self.assertEqual(
            ("git status", "rm -rf x", "grep y"),
            command_targets("git status && rm -rf x | grep y"),
        )

    def test_newline_separated_commands(self) -> None:
        self.assertEqual(
            ("git pull", "git status"),
            command_targets("git pull\ngit status"),
        )

    def test_background_operator_still_yields_segment(self) -> None:
        self.assertEqual(("sleep 1",), command_targets("sleep 1 &"))

    def test_tilde_is_allowed_verbatim(self) -> None:
        self.assertEqual(("cat ~/notes.txt",), command_targets("cat ~/notes.txt"))

    def test_path_like_word_is_a_valid_single_segment(self) -> None:
        # bash 语法里路径就是合法单词命令，解析不会报错——这正说明"拆不拆"
        # 只能由产出 target 的工具决定，service 无法从字符串反推。
        self.assertEqual(("/etc/passwd",), command_targets("/etc/passwd"))


class CommandTargetsOpaqueTests(TestCase):
    """一切无法逐段核验的构造都退回空元组（必进询问）。"""

    OPAQUE = [
        # 重定向：写目标不体现在段里
        "echo hi > out.txt",
        "ls 2>/dev/null",
        "cat < in.txt",
        # 命令/参数替换与进程替换
        "echo $(date)",
        "echo `date`",
        'echo "$HOME"',
        "diff <(ls) <(find)",
        # 赋值前缀改变运行环境
        "FOO=1 git status",
        # glob 与花括号展开依赖文件系统
        "cat file*.txt",
        "rm -rf /tmp/{a,b}",
        "rm ?",
        "ls [a-z]*",
        # compound / 控制流
        "if true; then ls; fi",
        "while read x; do echo $x; done",
        "{ ls; }",
        "for f in a b; do rm $f; done",
        # 语法本身残缺（执行期由 bash 报错，这里只保证给不出段）
        'git commit -m "unclosed',
        "git status &&",
        "echo a ; ;",
        # 非字符串/空白
        "",
        "   ",
        None,
        42,
    ]

    def test_opaque_commands_return_no_segments(self) -> None:
        for command in self.OPAQUE:
            with self.subTest(command=command):
                self.assertEqual((), command_targets(command))

    def test_opaque_anywhere_in_compound_poisons_whole_line(self) -> None:
        # 一段干净 + 一段不透明：整条给不出段，而不是只丢不透明那段。
        self.assertEqual((), command_targets("git status && echo hi > out.txt"))

    def test_escaped_glob_characters(self) -> None:
        self.assertEqual((), command_targets(r"grep \* file.txt"))


__all__ = ["CommandTargetsSplitTests", "CommandTargetsOpaqueTests"]
