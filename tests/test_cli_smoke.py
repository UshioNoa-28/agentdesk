"""Unit tests for Python CLI client and commands."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock, patch

from prompt_toolkit.document import Document

from cli.commands import handle_new, handle_quit, handle_rename
from cli.completer import SlashCompleter
from cli.ui import (
    format_timestamp,
    print_compact_result,
    print_mcps_table,
    print_sessions_table,
    print_skills_table,
)


class CliSmokeTests(TestCase):
    def test_completer_suggests_slash_commands(self) -> None:
        completer = SlashCompleter()
        doc = Document(text="/co")
        completions = list(completer.get_completions(doc, None))
        values = [c.text for c in completions]

        self.assertIn("/compact", values)
        self.assertNotIn("/history", values)

        from cli.commands import COMMAND_MAP
        self.assertNotIn("history", COMMAND_MAP)
        self.assertNotIn("review", COMMAND_MAP)
        self.assertNotIn("health", COMMAND_MAP)
        self.assertIn("memory", COMMAND_MAP)

        doc_mem = Document(text="/me")
        mem_completions = [c.text for c in completer.get_completions(doc_mem, None)]
        self.assertIn("/memory", mem_completions)

        doc_mem_sub = Document(text="/memory ")
        mem_sub_completions = [c.text for c in completer.get_completions(doc_mem_sub, None)]
        self.assertIn("list", mem_sub_completions)
        self.assertIn("user", mem_sub_completions)
        self.assertIn("project", mem_sub_completions)

    def test_completer_suggests_resume_sessions(self) -> None:
        mock_sessions = [
            {"id": "session-12345", "title": "Architecture Work"},
            {"id": "session-67890", "title": "Testing"},
        ]
        completer = SlashCompleter(get_sessions_fn=lambda: mock_sessions)
        doc = Document(text="/resume ")
        completions = list(completer.get_completions(doc, None))
        displays = [c.display for c in completions]

        self.assertTrue(any("Architecture Work" in str(d) for d in displays))

    def test_prompt_enables_nonblocking_automatic_completion(self) -> None:
        """Typing /resume should open dynamic suggestions without blocking input."""
        from cli.app import AnnaCliApp

        with patch("cli.app.PromptSession") as prompt_session:
            app = AnnaCliApp()
        try:
            options = prompt_session.call_args.kwargs
            self.assertIs(True, options["complete_while_typing"])
            self.assertIs(True, options["complete_in_thread"])
            self.assertEqual(app.get_activity_text, options["rprompt"])
            self.assertEqual(8, options["reserve_space_for_menu"])
            self.assertEqual(0.1, options["refresh_interval"])
            self.assertIs(True, options["erase_when_done"])
        finally:
            app.client.close()

    def test_format_timestamp(self) -> None:
        self.assertEqual("unknown", format_timestamp(None))
        self.assertNotEqual("unknown", format_timestamp("2026-08-21T12:00:00Z"))

    def test_submitted_message_uses_you_label_and_compaction_hides_summary_details(self) -> None:
        """Committed chat is distinct from the editable prompt and summaries stay internal."""
        from cli.ui import print_user_prompt

        with patch("cli.ui.console.print") as print_line:
            print_user_prompt("hello")
            rendered_user = str(print_line.call_args.args[0])
        self.assertIn("You", rendered_user)
        self.assertNotIn("]>[/", rendered_user)

        with patch("cli.ui.console.print") as print_line:
            print_compact_result({
                "status": "compacted",
                "estimated_tokens_before": 2000,
                "estimated_tokens_after": 1200,
                "kind": "summary",
            })
            rendered_compaction = str(print_line.call_args.args[0])
        self.assertIn("Context compacted.", rendered_compaction)
        self.assertNotIn("summary", rendered_compaction.lower())
        self.assertNotIn("token", rendered_compaction.lower())

    def test_commands_dispatch(self) -> None:
        mock_app = MagicMock()
        mock_app.client.create_session.return_value = {"id": "s1", "title": "New"}
        mock_app.client.rename_session.return_value = {"id": "s1", "title": "Renamed"}
        mock_app.current_session = {"id": "s1", "title": "Old"}

        handle_new(mock_app, "New")
        mock_app.reset_to_new_session.assert_called_once_with(
            pending_title="New", pending_tools=None
        )

        handle_rename(mock_app, "Renamed")
        self.assertEqual("s1", mock_app.current_session["id"])

        handle_quit(mock_app, "")
        self.assertEqual(False, mock_app.running)

    def test_ui_render_smoke(self) -> None:
        # Verify formatting functions execute without raising
        from datetime import datetime, timezone

        from cli.commands import handle_memory
        from cli.picker import SessionPicker
        from cli.ui import (
            format_relative_time,
            print_banner,
            print_error_message,
            print_message,
            print_user_prompt,
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        self.assertEqual("unknown", format_relative_time(None))
        self.assertEqual("just now", format_relative_time(now_iso))

        # Test banner & messages
        print_banner(
            base_url="http://127.0.0.1:8000/api",
            session_title="Test Session",
            session_id="s1234567",
        )
        print_user_prompt("Test user query")
        print_error_message("exceeded retry limit: 429 Too Many Requests")
        print_message(
            {
                "role": "assistant",
                "content": "Hello",
                "metadata": {
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
                },
            }
        )
        print_message({"role": "tool", "tool_name": "search_mcp", "content": "{\"status\":\"ok\"}"})

        from cli.ui import format_tool_result_preview

        preview = format_tool_result_preview('{"ok": true, "query": "test query"}')
        self.assertEqual("test query", preview)
        preview = format_tool_result_preview('{"stdout": "Hello\\n", "status": "ok"}')
        self.assertEqual("Hello", preview)
        preview = format_tool_result_preview('{"ok": false, "error": "fail"}')
        self.assertEqual("Error: fail", preview)

        import io

        from rich.console import Console

        from cli.streamer import MarkdownStreamer
        buf = io.StringIO()
        mock_c = Console(file=buf, width=60)
        streamer = MarkdownStreamer(mock_c, live_window=3, min_delay=0.0)
        streamer.update("# Title\n\nParagraph one.\n\nParagraph two.\n\nParagraph three.")
        streamer.finish("# Title\n\nParagraph one.\n\nParagraph two.\n\nParagraph three.")
        self.assertTrue(len(buf.getvalue()) > 0)

        # Test session picker format (main session & subagent)
        mock_sessions = [
            {"id": "s1", "title": "Session 1", "updated_at": "2026-08-27T20:00:00Z"},
            {"id": "s2", "title": "Session 2", "main_session_id": "s1"},
        ]
        picker = SessionPicker(mock_sessions, current_session_id="s1")
        self.assertEqual(len(picker.sessions), 1)  # Strictly filters out subagent s2
        formatted = picker._get_formatted_text()
        self.assertTrue(len(formatted) > 0)

        from cli.app import generate_fork_title
        from cli.picker import HistoryMessageResumer, SubagentPicker

        sub_picker = SubagentPicker(
            main_session=mock_sessions[0],
            subagents=[mock_sessions[1]],
            current_session_id="s1",
        )
        self.assertEqual(len(sub_picker.items), 2)
        sub_formatted = sub_picker._get_formatted_text()
        self.assertTrue(len(sub_formatted) > 0)

        # Test HistoryMessageResumer
        history_msgs = [
            {"role": "human", "content": "First ask", "seq": 2},
            {"role": "assistant", "content": "First reply", "seq": 3},
            {"role": "human", "content": "Second ask", "seq": 4},
        ]
        resumer = HistoryMessageResumer(history_msgs)
        self.assertEqual(len(resumer.turns), 2)
        self.assertEqual(1, resumer.selected_turn_idx)  # Defaults to latest ask
        res_formatted = resumer._get_formatted_text()
        self.assertTrue(len(res_formatted) > 0)

        # Test fork title generation
        title1 = generate_fork_title("优化方案", set())
        self.assertEqual("优化方案 (fork 1)", title1)
        title2 = generate_fork_title("优化方案", {"优化方案 (fork 1)"})
        self.assertEqual("优化方案 (fork 2)", title2)

        # Test summary message filtering
        print_message(
            {"role": "assistant", "content": "Summary text", "metadata": {"kind": "summary"}}
        )
        print_message({"role": "system", "content": "system prompt"})

        # Test memory command
        from cli.ui import print_memories_table, print_memory_note
        mock_mem_app = MagicMock()
        mock_mem_app.client.list_memories.return_value = [
            {"layer": "user", "title": "缩进", "description": "一律 tab"},
            {"layer": "project", "title": "部署", "description": "make deploy"},
        ]
        handle_memory(mock_mem_app, "")
        mock_mem_app.client.list_memories.assert_called_once()
        mock_mem_app.client.get_memory.return_value = {
            "layer": "user",
            "title": "缩进",
            "content": "# 缩进\n一律 tab",
        }
        handle_memory(mock_mem_app, "缩进")
        mock_mem_app.client.get_memory.assert_called_with("user", "缩进")

        print_memories_table([])
        print_memories_table([{"layer": "user", "title": "t", "description": "d"}])
        print_memory_note("user", "t", "note content")

        print_sessions_table(
            [{"id": "s1", "title": "S1", "updated_at": "2026-08-21T12:00:00Z"}], "s1"
        )
        print_mcps_table(
            [{"id": "exa", "state": "connected", "tool_count": 2, "description": "Search"}]
        )
        print_skills_table([{"name": "test", "state": "loaded", "description": "d", "path": "p"}])
        print_compact_result({"status": "noop", "message": "Nothing to compact"})
        print_compact_result({
            "status": "compacted",
            "estimated_tokens_before": 2000,
            "estimated_tokens_after": 1200,
            "kind": "summary",
        })
        print_compact_result(None)

        from cli.commands import handle_compact
        mock_compact_app = MagicMock()
        mock_compact_app.current_session = {"id": "sess-1"}
        mock_compact_app.client.compact_session.return_value = {
            "status": "noop",
            "message": "No eligible history",
        }
        handle_compact(mock_compact_app, "")
        mock_compact_app.client.compact_session.assert_called_once_with("sess-1")

        # Test handle_subagents strictly filters by main_session_id and excludes forked sessions
        from cli.commands import handle_subagents

        mock_sub_app = MagicMock()
        mock_sub_app.current_session = {"id": "main1", "title": "Main Session"}
        mock_sub_app.client.list_sessions.return_value = [
            {"id": "main1", "title": "Main Session"},
            {
                "id": "fork1",
                "title": "Forked Main",
                "parent_session_id": "main1",
                "main_session_id": None,
            },
            {
                "id": "sub1",
                "title": "Subagent Coder",
                "main_session_id": "main1",
                "parent_session_id": None,
            },
        ]
        handle_subagents(mock_sub_app, "sub1")
        mock_sub_app.open_session.assert_called_once_with(
            {
                "id": "sub1",
                "title": "Subagent Coder",
                "main_session_id": "main1",
                "parent_session_id": None,
            }
        )

        # Switching sessions renders the complete history automatically.
        from cli.app import AnnaCliApp
        with patch.object(AnnaCliApp, "_safe_health", return_value={"status": "ok"}):
            real_app = AnnaCliApp()
            real_app.client = MagicMock()
            messages = [
                {"role": "human", "content": "Earlier question", "seq": 1},
                {"role": "assistant", "content": "Earlier answer", "seq": 2},
            ]
            real_app.client.list_messages.return_value = messages
            with patch("cli.ui.print_history_list") as print_history:
                real_app.open_session({"id": "s-history", "title": "History Session"}, is_new=False)
            real_app.client.list_messages.assert_called_once_with("s-history")
            print_history.assert_called_once_with(
                messages,
                limit=None,
                show_subagent_details=False,
            )

    def test_picker_live_search_filtering(self) -> None:
        """测试选择器支持实时搜索与过滤。"""
        from cli.picker import HistoryMessageResumer, SessionPicker, SubagentPicker

        mock_sessions = [
            {"id": "sess-alpha", "title": "Alpha Analysis"},
            {"id": "sess-beta", "title": "Beta Build"},
            {"id": "sess-gamma", "title": "Gamma Graph"},
        ]

        # 1. SessionPicker 搜索测试
        sp = SessionPicker(mock_sessions)
        self.assertEqual(len(sp._get_filtered()), 3)
        sp.search_query = "alpha"
        self.assertEqual(len(sp._get_filtered()), 1)
        self.assertEqual(sp._get_filtered()[0]["id"], "sess-alpha")
        sp.search_query = "nonexistent"
        self.assertEqual(len(sp._get_filtered()), 0)
        formatted = sp._get_formatted_text()
        self.assertTrue(len(formatted) > 0)

        # 2. SubagentPicker 搜索测试
        sap = SubagentPicker(
            main_session=mock_sessions[0],
            subagents=[mock_sessions[1]],
            current_session_id="sess-alpha",
        )
        self.assertEqual(len(sap._get_filtered()), 2)
        sap.search_query = "beta"
        self.assertEqual(len(sap._get_filtered()), 1)
        self.assertEqual(sap._get_filtered()[0]["id"], "sess-beta")

        # 3. HistoryMessageResumer 搜索测试
        msgs = [
            {"role": "human", "content": "How to deploy docker?", "seq": 1},
            {"role": "assistant", "content": "Use docker compose", "seq": 2},
            {"role": "human", "content": "Explain python asyncio", "seq": 3},
            {"role": "assistant", "content": "Asyncio is event loop based", "seq": 4},
        ]
        resumer = HistoryMessageResumer(msgs)
        self.assertEqual(len(resumer.turns), 2)
        resumer.search_query = "docker"
        self.assertEqual(len(resumer._get_filtered()), 1)
        self.assertIn("docker", resumer._get_filtered()[0]["ask"]["content"])
        formatted = resumer._get_formatted_text()
        self.assertTrue(len(formatted) > 0)


    def test_tool_printing_and_history_list(self) -> None:
        """测试工具调用渲染与会话历史打印。"""
        from cli.ui import (
            print_history_list,
            print_tool_call_start,
            print_tool_result_end,
        )

        # 工具调用开始
        print_tool_call_start("execute_python", {"code": "print(123)"})
        print_tool_call_start("search_mcp", {"mcp": "exa", "query": "latest AI"})
        print_tool_call_start("send_message", {"recipient": "coder", "message": "do task"})

        # 工具调用结束
        print_tool_result_end('{"stdout": "123\\n", "status": "ok"}')

        # 历史记录列表打印（限制展示最近条数）
        msgs = [
            {"role": "human", "content": f"msg {i}", "seq": i}
            for i in range(1, 20)
        ]
        print_history_list(msgs, limit=5)
        print_history_list(msgs, limit=None)

    def test_history_groups_tool_calls_and_results_under_agent(self) -> None:
        """历史中的工具活动应归属 Agent，并按调用 ID 展示简短结果。"""
        import io

        from rich.console import Console

        from cli.ui import print_history_list

        messages = [
            {"role": "human", "content": "Inspect the project"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-first",
                        "name": "execute_python",
                        "arguments": {"code": "print('FIRST-CALL')"},
                    },
                    {
                        "id": "call-second",
                        "name": "execute_python",
                        "arguments": {"code": "print('SECOND-CALL')"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-second",
                "tool_name": "execute_python",
                "content": '{"stdout": "RESULT-SECOND\\n", "status": "ok"}',
            },
            {
                "role": "tool",
                "tool_call_id": "call-first",
                "tool_name": "execute_python",
                "content": '{"stdout": "RESULT-FIRST\\n", "status": "ok"}',
            },
            {"role": "assistant", "content": "Inspection complete."},
        ]
        output = io.StringIO()
        history_console = Console(
            file=output,
            color_system=None,
            force_terminal=False,
            width=100,
        )

        with patch("cli.ui.console", history_console):
            print_history_list(messages)

        rendered = output.getvalue()
        first_agent = rendered.index("AgentDesk:")
        first_call = rendered.index("FIRST-CALL")
        first_result = rendered.index("RESULT-FIRST")
        second_call = rendered.index("SECOND-CALL")
        second_result = rendered.index("RESULT-SECOND")
        self.assertLess(first_agent, first_call)
        self.assertLess(first_call, first_result)
        self.assertLess(first_result, second_call)
        self.assertLess(second_call, second_result)
        self.assertIn("RESULT-FIRST\n\n● Tool", rendered)
        self.assertNotIn("Calling execute_python", rendered)

    def test_subagent_history_is_compact_by_default(self) -> None:
        """主会话只展示子代理汇报摘要，不让长报告淹没上下文。"""
        from cli.ui import print_message

        report = "**Important finding** about `send_message` " + ("detail " * 80)
        with patch("cli.ui.console.print") as mock_print:
            print_message(
                {
                    "role": "human",
                    "content": report,
                    "metadata": {"source": "agent", "name": "researcher"},
                },
                show_subagent_details=False,
            )

        rendered = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("Subagent · researcher", rendered)
        self.assertIn("Important finding", rendered)
        self.assertNotIn("**", rendered)
        self.assertNotIn("`send_message`", rendered)
        self.assertNotIn(report, rendered)

    def test_prompt_and_session_history_auto_load(self) -> None:
        """测试简洁 Prompt 与切换会话时自动加载完整历史。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        self.assertIn("›", app.get_prompt_text().value)

        # 切换会话后 Prompt 依旧保持简洁 ›
        app.current_session = {"id": "sess-main", "title": "Refactoring Work"}
        self.assertEqual(app.get_prompt_text().value, "<prompt>› </prompt>")

        messages = [
            {"role": "human", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        app.client.list_messages = MagicMock(return_value=messages)
        with patch("cli.ui.print_history_list") as print_history:
            app.open_session({"id": "sess-main", "title": "Refactoring Work"})
        app.client.list_messages.assert_called_once_with("sess-main")
        print_history.assert_called_once_with(
            messages,
            limit=None,
            show_subagent_details=False,
        )

    def test_clear_reloads_complete_session_history(self) -> None:
        """测试 /clear 重绘时会自动恢复完整历史。"""
        import cli.commands as command_module
        from cli.commands import handle_clear

        app = MagicMock()
        app.current_session = {"id": "sess-clear", "title": "Clear Test"}
        messages = [{"role": "human", "content": "Keep me", "seq": 1}]
        app.client.list_messages.return_value = messages
        with patch.object(command_module.ui, "clear_screen") as clear_screen, patch.object(
            command_module.ui, "print_history_list"
        ) as print_history:
            handle_clear(app, "")
        clear_screen.assert_called_once_with()
        app.print_header.assert_called_once_with()
        app.client.list_messages.assert_called_once_with("sess-clear")
        print_history.assert_called_once_with(
            messages,
            limit=None,
            show_subagent_details=False,
        )

    def test_clear_screen_and_unanswered_history(self) -> None:
        """测试 clear_screen 以及末尾未回复问题的明确标记。"""
        import io
        from unittest.mock import patch

        from cli.ui import clear_screen, print_history_list

        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            clear_screen()
            self.assertIn("\033[2J\033[3J\033[H", fake_stdout.getvalue())

        # 测试末尾人类提问没有助手回复时的标记
        history = [
            {"role": "human", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "human", "content": "Question 2 without answer"},
        ]
        print_history_list(history)

    def test_tool_call_start_new_tools(self) -> None:
        """测试 load_skill, wait_for_replies, list_subagents 工具预览。"""
        from cli.ui import print_tool_call_start

        print_tool_call_start("load_skill", {"name": "test_skill"})
        print_tool_call_start("wait_for_replies", {"subagents": ["coder"]})
        print_tool_call_start("list_subagents", {})

















    def test_markdown_streamer_degrades_when_live_is_owned_by_another_turn(self) -> None:
        """并发 turn 抢不到 Rich Live 时仍会在收尾输出完整文本。"""
        import io

        from rich.console import Console
        from rich.errors import LiveError

        from cli.streamer import MarkdownStreamer

        output = io.StringIO()
        console = Console(file=output, width=60)
        streamer = MarkdownStreamer(console, live_window=2, min_delay=0.0)
        with patch("cli.streamer.Live", side_effect=LiveError("busy")):
            streamer.update("first\n\nsecond")
            streamer.finish("first\n\nsecond")
        self.assertIn("first", output.getvalue())
        self.assertIn("second", output.getvalue())

    def test_markdown_streamer_does_not_stop_other_live_and_can_reuse(self) -> None:
        """次要流收尾不能清掉主流 Live，且 streamer 可重新使用。"""
        import io

        from rich.console import Console
        from rich.live import Live
        from rich.text import Text

        from cli.streamer import MarkdownStreamer

        output = io.StringIO()
        console = Console(file=output, width=60)
        owner = Live(Text("owner"), console=console, auto_refresh=False)
        owner.start()
        try:
            secondary = MarkdownStreamer(console, live_window=2, min_delay=0.0)
            secondary.update("secondary")
            secondary.finish("secondary")
            self.assertIs(console._live, owner)
        finally:
            owner.stop()

        secondary.update("fresh")
        secondary.finish("fresh")
        self.assertIn("fresh", output.getvalue())

    def test_markdown_streamer_code_block_incremental_no_truncation(self) -> None:
        """流式输出未闭合代码块时，不得提前刷新半截代码行导致输出被截断。"""
        import io

        from rich.console import Console
        from rich.text import Text

        from cli.streamer import MarkdownStreamer

        output = io.StringIO()
        console = Console(file=output, width=80, force_terminal=True)
        streamer = MarkdownStreamer(console, live_window=1, live_enabled=False, min_delay=0.0)

        full_doc = (
            "针对测试目标的压力测试命令：\n\n"
            "**SYN Flood**\n"
            "```bash\n"
            "hping3 -S -p 80 --flood --rand-source 8.208.87.162\n"
            "```\n\n"
            "**TCP Connect**\n"
            "```bash\n"
            "nping --tcp-connect -p 80 --rate 1000 --count 0 8.208.87.162\n"
            "```\n\n"
            "**Python Script**\n"
            "```python\n"
            "TARGET = \"8.208.87.162\"\n"
            "PORT = 80\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
            "```\n\n"
            "**Slowloris**\n"
            "```bash\nslowloris 8.208.87.162 -p 80 -s 500\n```\n"
        )

        accum = ""
        # Feed document in tiny 2-character increments to simulate fine-grained SSE chunks
        for i in range(0, len(full_doc), 2):
            accum = full_doc[: i + 2]
            streamer.update(accum)
        streamer.finish(full_doc)

        result = Text.from_ansi(output.getvalue()).plain
        self.assertIn("hping3 -S -p 80 --flood --rand-source 8.208.87.162", result)
        self.assertIn("nping --tcp-connect -p 80 --rate 1000 --count 0 8.208.87.162", result)
        self.assertIn("TARGET = \"8.208.87.162\"", result)
        self.assertIn("s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)", result)
        self.assertIn("slowloris 8.208.87.162 -p 80 -s 500", result)

    def test_markdown_streamer_table_and_heading_incremental(self) -> None:
        """流式输出表格与标题时，未闭合的结构不得提前拆散渲染。"""
        import io

        from rich.console import Console
        from rich.text import Text

        from cli.streamer import MarkdownStreamer

        output = io.StringIO()
        console = Console(file=output, width=80, force_terminal=True)
        streamer = MarkdownStreamer(console, live_window=1, live_enabled=False, min_delay=0.0)

        table_doc = (
            "# Main Title\n\n"
            "| Service | Port | Status |\n"
            "| :--- | :--- | :--- |\n"
            "| HTTP | 80 | Active |\n"
            "| HTTPS | 443 | Active |\n\n"
            "All services tested.\n"
        )

        accum = ""
        for i in range(0, len(table_doc), 3):
            accum = table_doc[: i + 3]
            streamer.update(accum)
        streamer.finish(table_doc)

        result = Text.from_ansi(output.getvalue()).plain
        self.assertIn("Main Title", result)
        self.assertIn("HTTP", result)
        self.assertIn("HTTPS", result)
        self.assertIn("All services tested.", result)


    def test_cli_live_renderers_disable_stdout_and_stderr_redirects(self) -> None:
        """Rich animations must not replace prompt_toolkit's global stdout proxy."""
        import io

        from rich.console import Console

        import cli.ui as ui
        from cli.streamer import MarkdownStreamer

        output = io.StringIO()
        console = Console(file=output, width=60)

        with patch("cli.ui.Live", return_value=MagicMock()) as status_live:
            ui.make_status("Thinking...", target_console=console)
        self.assertIs(False, status_live.call_args.kwargs["redirect_stdout"])
        self.assertIs(False, status_live.call_args.kwargs["redirect_stderr"])

        with patch("cli.streamer.Live", return_value=MagicMock()) as markdown_live:
            streamer = MarkdownStreamer(console)
            streamer.start()
            streamer.finish()
        self.assertIs(False, markdown_live.call_args.kwargs["redirect_stdout"])
        self.assertIs(False, markdown_live.call_args.kwargs["redirect_stderr"])
        self.assertFalse(ui.LIVE_RENDER_LOCK.locked())

    def test_cli_live_renderers_share_and_release_one_nonblocking_lease(self) -> None:
        """A second animation buffers until the current Live lifecycle ends."""
        import io

        from rich.console import Console
        from rich.errors import LiveError

        import cli.ui as ui
        from cli.streamer import MarkdownStreamer

        output = io.StringIO()
        console = Console(file=output, width=60, force_terminal=True)
        status = ui.make_status("Thinking...", target_console=console)
        secondary = MarkdownStreamer(console, min_delay=0.0)

        try:
            status.start()
            secondary.start()
            self.assertIsNone(secondary.live)

            secondary.finish()
            status.stop()

            secondary.start()
            self.assertIsNotNone(secondary.live)

            blocked_status = ui.make_status("Still thinking...", target_console=console)
            with self.assertRaises(LiveError):
                blocked_status.start()
            secondary.finish()

            blocked_status.start()
            blocked_status.stop()
        finally:
            status.stop()
            secondary.finish()

        self.assertFalse(ui.LIVE_RENDER_LOCK.locked())

    def test_live_render_lease_is_released_when_start_fails(self) -> None:
        """Constructor/start failures cannot permanently disable animations."""
        import io

        from rich.console import Console

        import cli.ui as ui
        from cli.streamer import MarkdownStreamer

        output = io.StringIO()
        console = Console(file=output, width=60)

        broken_status_live = MagicMock()
        broken_status_live.start.side_effect = RuntimeError("status failed")
        with patch("cli.ui.Live", return_value=broken_status_live):
            status = ui.make_status("Thinking...", target_console=console)
            with self.assertRaisesRegex(RuntimeError, "status failed"):
                status.start()
        self.assertFalse(ui.LIVE_RENDER_LOCK.locked())

        broken_markdown_live = MagicMock()
        broken_markdown_live.start.side_effect = RuntimeError("stream failed")
        with patch("cli.streamer.Live", return_value=broken_markdown_live):
            streamer = MarkdownStreamer(console)
            with self.assertRaisesRegex(RuntimeError, "stream failed"):
                streamer.start()
        self.assertFalse(ui.LIVE_RENDER_LOCK.locked())

    def test_safe_status_cleans_up_a_partially_started_live(self) -> None:
        """A Live failure after claiming its Console cannot poison later output."""
        import io

        from rich.console import Console

        import cli.ui as ui

        output = io.StringIO()
        console = Console(file=output, width=60)

        class PartialLive:
            def __init__(self) -> None:
                self.console = console
                self.stop_called = False

            def start(self) -> None:
                self.console._live = self
                raise RuntimeError("partial start")

            def stop(self) -> None:
                self.stop_called = True
                self.console._live = None

        partial_live = PartialLive()
        with patch("cli.ui.Live", return_value=partial_live):
            status = ui.make_status("Thinking...", target_console=console)
            with self.assertRaisesRegex(RuntimeError, "partial start"):
                status.start()

        self.assertTrue(partial_live.stop_called)
        self.assertIsNone(console._live)
        self.assertFalse(ui.LIVE_RENDER_LOCK.locked())

    def test_live_render_leases_survive_base_exceptions(self) -> None:
        """KeyboardInterrupt during start/update must release the shared lease."""
        import io

        from rich.console import Console

        import cli.ui as ui
        from cli.streamer import MarkdownStreamer

        output = io.StringIO()
        console = Console(file=output, width=60)

        broken_status_live = MagicMock()
        broken_status_live.start.side_effect = KeyboardInterrupt()
        with patch("cli.ui.Live", return_value=broken_status_live):
            status = ui.make_status("Thinking...", target_console=console)
            with self.assertRaises(KeyboardInterrupt):
                status.start()
        self.assertFalse(ui.LIVE_RENDER_LOCK.locked())

        streamer = MarkdownStreamer(console, min_delay=0.0)
        with patch.object(streamer, "_render_to_lines", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                streamer.update("content")
        self.assertFalse(ui.LIVE_RENDER_LOCK.locked())

    def test_safe_status_does_not_restore_prompt_toolkit_stdout_proxy(self) -> None:
        """Stopping Live after patch_stdout exits must preserve the real stdout."""
        import io
        import sys

        from prompt_toolkit.patch_stdout import patch_stdout
        from rich.console import Console

        import cli.ui as ui

        original_stdout = sys.stdout
        output = io.StringIO()
        console = Console(file=output, width=60, force_terminal=True)
        status = ui.make_status("Thinking...", target_console=console)

        with patch_stdout(raw=True):
            prompt_stdout = sys.stdout
            status.start()
            self.assertIs(sys.stdout, prompt_stdout)

        self.assertIs(sys.stdout, original_stdout)
        status.stop()
        self.assertIs(sys.stdout, original_stdout)
        self.assertFalse(ui.LIVE_RENDER_LOCK.locked())

    def test_stream_setup_failure_cannot_leave_a_status_running(self) -> None:
        """The initial spinner starts only after stream registration succeeds."""
        import cli.ui as ui
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-setup", "title": "Setup"}
        app._request_context.session_id = "sess-setup"
        app._request_context.request_id = "request-setup"

        with patch.object(
            app,
            "_register_stream",
            side_effect=RuntimeError("registration failed"),
        ), patch("cli.app.ui.make_status") as make_status:
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                app._ask_agent_stream("question")

        make_status.assert_not_called()
        self.assertFalse(ui.LIVE_RENDER_LOCK.locked())
        self.assertNotIn("request-setup", app._stream_states)


    def test_resolve_session_and_title_parsing(self) -> None:
        """测试 session 查找大小写不敏感与标题去引号。"""
        from cli.commands import _parse_new_argument, _resolve_session

        sessions = [
            {"id": "b9d23fe2-D3A1-4327-B604-3D1C7C732247", "title": "Probe Session"},
            {"id": "abc-123", "title": "Alpha Project"},
        ]
        # 大写 / 小写 ID 前缀查询
        self.assertEqual(_resolve_session(sessions, "b9d23fe2"), sessions[0])
        self.assertEqual(_resolve_session(sessions, "B9D23FE2"), sessions[0])
        # 1-based 索引查询
        self.assertEqual(_resolve_session(sessions, "1"), sessions[0])
        self.assertEqual(_resolve_session(sessions, "2"), sessions[1])

        # 标题引号去除解析
        title, tools = _parse_new_argument('"My Quoted Title" --tools a,b')
        self.assertEqual(title, "My Quoted Title")
        self.assertEqual(tools, ["a", "b"])

    def test_session_commands_use_main_session_numbering_and_busy_guard(self) -> None:
        """会话编号与 picker 一致，运行中不会切换/删除当前会话。"""
        from cli.commands import handle_delete, handle_new, handle_resume

        busy = MagicMock()
        busy.active_turn_count = 1
        handle_new(busy, "ignored")
        busy.reset_to_new_session.assert_not_called()

        app = MagicMock()
        app.active_turn_count = 0
        app.current_session = {"id": "main1", "title": "Main 1"}
        app.client.list_sessions.return_value = [
            {"id": "main1", "title": "Main 1", "main_session_id": None},
            {"id": "sub1", "title": "Sub", "main_session_id": "main1"},
            {"id": "main2", "title": "Main 2", "main_session_id": None},
        ]
        handle_resume(app, "2")
        app.open_session.assert_called_once_with(
            {"id": "main2", "title": "Main 2", "main_session_id": None}
        )

        app.reset_mock()
        app.active_turn_count = 1
        handle_delete(app, "1")
        app.client.delete_session.assert_not_called()

    def test_completer_subagent_suggestions(self) -> None:
        """测试 /subagent 命令动态补全子代理名称。"""
        mock_sessions = [
            {"id": "s1", "title": "Main Session", "main_session_id": None},
            {"id": "s2", "title": "s1_subagent_coder", "main_session_id": "s1"},
            {"id": "s3", "title": "s1_subagent_reviewer", "main_session_id": "s1"},
        ]
        completer = SlashCompleter(get_sessions_fn=lambda: mock_sessions)
        doc = Document(text="/subagent co")
        completions = list(completer.get_completions(doc, None))
        values = [c.text for c in completions]
        self.assertIn("coder", values)

    def test_completer_subagent_suggestions_are_scoped_to_current_owner(self) -> None:
        """子代理补全不能把其它主会话的 worker 暴露给当前 picker。"""
        mock_sessions = [
            {"id": "main-1", "title": "Main 1", "main_session_id": None},
            {"id": "main-2", "title": "Main 2", "main_session_id": None},
            {"id": "sub-1", "title": "main-1_subagent_coder", "main_session_id": "main-1"},
            {"id": "sub-2", "title": "main-2_subagent_coder", "main_session_id": "main-2"},
        ]
        completer = SlashCompleter(
            get_sessions_fn=lambda: mock_sessions,
            get_current_session_fn=lambda: mock_sessions[0],
        )
        completions = list(completer.get_completions(Document(text="/subagent co"), None))
        self.assertEqual(["coder"], [completion.text for completion in completions])

    def test_empty_conversational_history_notice(self) -> None:
        """测试历史只有系统/工具消息时的提示。"""
        import io
        from unittest.mock import patch

        from cli.ui import print_history_list

        msgs = [
            {"role": "tool", "content": "tool output", "metadata": {}},
            {"role": "system", "content": "system prompt", "metadata": {}},
        ]
        with patch("sys.stdout", new=io.StringIO()):
            # 执行不报错且优雅输出提示
            print_history_list(msgs)

    def test_delete_cancelled_on_keyboard_interrupt(self) -> None:
        """测试删除会话被取消时不抛异常。"""
        from unittest.mock import patch

        from cli.app import AnnaCliApp
        from cli.commands import handle_delete

        app = AnnaCliApp()
        app.client.list_sessions = MagicMock(return_value=[{"id": "s1", "title": "Test"}])
        with patch("cli.ui.console.input", side_effect=KeyboardInterrupt):
            # 取消删除应当安全返回
            handle_delete(app, "1")

    def test_make_title_from_question(self) -> None:
        """测试直接从用户提问生成标题。"""
        from cli.app import make_title_from_question

        # 直接作为标题
        self.assertEqual("hi", make_title_from_question("hi"))
        self.assertEqual("hi?", make_title_from_question("hi?"))
        self.assertEqual("你好", make_title_from_question("你好"))
        self.assertEqual(
            "请帮我写一个快速排序算法",
            make_title_from_question("请帮我写一个快速排序算法"),
        )

        # 超长文本截断
        long_q = "Very long prompt that contains way too many words and characters exceeding limit"
        short_title = make_title_from_question(long_q)
        self.assertTrue(len(short_title) <= 36)
        self.assertTrue(short_title.endswith("…"))

    def test_lazy_session_creation(self) -> None:
        """测试会话延迟创建（仅在输入首条消息时创建）。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app._ask_agent_stream = MagicMock()
        app.client.list_sessions = MagicMock(return_value=[])
        app.client.create_session = MagicMock(
            return_value={"id": "sess-1", "title": "hi?"}
        )

        # 初始状态未在后端创建任何 session
        self.assertIsNone(app.current_session)

        # 发送第一条提问：延迟创建 session，标题直接采用 "hi?"
        app.ask_agent("hi?")
        app.client.create_session.assert_called_once_with(title="hi?", allowed_tools=None)
        self.assertIsNotNone(app.current_session)
        self.assertEqual(app.current_session["title"], "hi?")

        # 后续提问不再重复创建
        app.ask_agent("second question")
        app.client.create_session.assert_called_once()

    def test_ask_user_picker_structure(self) -> None:
        """测试 AskUserPicker 的结构、预选和文本渲染。"""
        from cli.picker import AskUserPicker

        picker = AskUserPicker(
            options=["PostgreSQL", "SQLite", "DuckDB"],
            recommended="SQLite",
        )
        self.assertEqual(4, len(picker.items))
        # 预选推荐项 SQLite (index 1)
        self.assertEqual(1, picker.selected_index)
        self.assertTrue(picker.items[1]["is_recommended"])
        self.assertEqual(picker.items[3]["type"], "custom")

        formatted = picker._get_formatted_text()
        text_content = "".join(t[1] for t in formatted)
        self.assertIn("PostgreSQL", text_content)
        self.assertIn("SQLite", text_content)
        self.assertIn("(Recommended)", text_content)
        self.assertIn("Type your own answer...", text_content)
        self.assertIn("[1]", text_content)
        self.assertIn("[2]", text_content)
        self.assertIn("[3]", text_content)
        self.assertIn("[4]", text_content)
        # 验证不包含多余的底部按键提示
        self.assertNotIn("navigate", text_content)

    def test_ask_user_picker_non_interactive_fallback(self) -> None:
        """测试非 TTY 环境下 AskUserPicker 安全回退。"""
        from cli.picker import AskUserPicker

        picker = AskUserPicker(options=["A", "B"], recommended="B")
        with patch("sys.stdin.isatty", return_value=False):
            self.assertEqual("B", picker.run())

        picker2 = AskUserPicker(options=["A", "B"], recommended=None)
        with patch("sys.stdin.isatty", return_value=False):
            self.assertEqual("A", picker2.run())

    def test_ask_user_picker_interactive_keys(self) -> None:
        """测试 AskUserPicker 的交互按键、自由输入与快捷键行为。"""
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from cli.picker import AskUserPicker

        # 1. 直接回车：选中推荐项 (SQLite)
        with create_pipe_input() as pipe:
            pipe.send_text("\r")
            picker = AskUserPicker(options=["PostgreSQL", "SQLite"], recommended="SQLite")
            res = picker.run(input=pipe, output=DummyOutput())
            self.assertEqual("SQLite", res)

        # 2. 上下键移动光标后回车：从 index 0 向下移动选中 SQLite
        with create_pipe_input() as pipe:
            pipe.send_text("\x1b[B\r")
            picker = AskUserPicker(options=["PostgreSQL", "SQLite"], recommended=None)
            res = picker.run(input=pipe, output=DummyOutput())
            self.assertEqual("SQLite", res)

        # 3. 向下移动到自定义输入项，输入自定义内容后回车
        with create_pipe_input() as pipe:
            pipe.send_text("\x1b[B\x1b[BTornado\r")
            picker = AskUserPicker(options=["PostgreSQL", "SQLite"], recommended=None)
            res = picker.run(input=pipe, output=DummyOutput())
            self.assertEqual("Tornado", res)

        # 4. 直接键入字符自动激活自由输入并支持退格修改
        with create_pipe_input() as pipe:
            pipe.send_text("SanicX\x7f\r")
            picker = AskUserPicker(options=["PostgreSQL", "SQLite"], recommended=None)
            res = picker.run(input=pipe, output=DummyOutput())
            self.assertEqual("Sanic", res)

        # 5. 按 Esc 取消
        with create_pipe_input() as pipe:
            pipe.send_text("\x1b")
            picker = AskUserPicker(options=["PostgreSQL", "SQLite"])
            res = picker.run(input=pipe, output=DummyOutput())
            self.assertIsNone(res)

        # 6. Ctrl+C 随时取消并立即退出
        with create_pipe_input() as pipe:
            pipe.send_text("\x03")
            picker = AskUserPicker(options=["PostgreSQL", "SQLite"])
            res = picker.run(input=pipe, output=DummyOutput())
            self.assertIsNone(res)

        # 7. 自定义输入支持左移光标插入字符
        with create_pipe_input() as pipe:
            pipe.send_text("ab\x1b[Dc\r")
            picker = AskUserPicker(options=["PostgreSQL", "SQLite"])
            res = picker.run(input=pipe, output=DummyOutput())
            self.assertEqual("acb", res)

    def test_ask_user_picker_cursor_positioning(self) -> None:
        """测试 AskUserPicker 光标定位与可见性：输入时不漂移至 (0,0)。"""
        from prompt_toolkit.application import Application
        from prompt_toolkit.application.current import create_app_session
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.layout import Layout
        from prompt_toolkit.output import DummyOutput

        from cli.picker import AskUserPicker

        picker = AskUserPicker(options=["PostgreSQL", "SQLite"])
        content_control = FormattedTextControl(picker._get_formatted_text)
        window = Window(
            content=content_control,
            height=len(picker.items),
            dont_extend_height=True,
            always_hide_cursor=Condition(lambda: picker.selected_index != picker.custom_index),
        )
        layout = Layout(HSplit([window]))

        with create_app_session(output=DummyOutput()) as session:
            app = Application(layout=layout, output=DummyOutput())
            session.app = app

            # 1. 选中首项选项时：隐藏终端光标
            picker.selected_index = 0
            app.render_counter += 1
            app.renderer.render(app, app.layout)
            self.assertFalse(app.renderer._last_screen.show_cursor)

            # 2. 移动到自定义输入项（尚无输入）：光标显示在第 2 行第 7 列
            picker.selected_index = picker.custom_index
            picker.custom_text = ""
            picker.custom_cursor_idx = 0
            app.render_counter += 1
            app.renderer.render(app, app.layout)
            s2 = app.renderer._last_screen
            pos2 = s2.cursor_positions.get(window)
            self.assertTrue(s2.show_cursor)
            self.assertEqual((2, 7), (pos2.y, pos2.x))

            # 3. 键入内容后：光标正确跟在文本后，绝不会漂移至 (0, 0)
            picker.custom_text = "FastAPI"
            picker.custom_cursor_idx = 7
            app.render_counter += 1
            app.renderer.render(app, app.layout)
            s3 = app.renderer._last_screen
            pos3 = s3.cursor_positions.get(window)
            self.assertTrue(s3.show_cursor)
            self.assertEqual(2, pos3.y)
            self.assertEqual(14, pos3.x)  # 7 + len("FastAPI")
            self.assertNotEqual((0, 0), (pos3.y, pos3.x))

    def test_ui_ask_user_handling(self) -> None:
        """测试 UI 层对 ask_user 的渲染与特判。"""
        import io

        from cli.ui import (
            format_tool_result_preview,
            print_clarification_question,
            print_message,
            print_tool_call_start,
        )

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_clarification_question("Which database?")
            self.assertIn("Clarification", fake_out.getvalue())
            self.assertIn("Which database?", fake_out.getvalue())

        # print_tool_call_start 针对 ask_user 静默
        with patch("cli.ui.console.print") as mock_print:
            print_tool_call_start("ask_user", {"question": "Q"})
            mock_print.assert_not_called()

        # format_tool_result_preview 格式化
        preview = format_tool_result_preview(
            '{"ok": true, "question": "Pick DB", "options": ["A", "B"]}'
        )
        self.assertIn("Clarification: Pick DB (2 options)", preview)

        # print_message 展示 "Clarification requested"
        with patch("cli.ui.console.print") as mock_print:
            print_message({
                "role": "assistant",
                "tool_calls": [{"name": "ask_user"}],
                "content": "",
            })
            printed_texts = [str(call[0][0]) for call in mock_print.call_args_list if call[0]]
            self.assertTrue(any("Clarification requested" in t for t in printed_texts))

        # 后台澄清面板也必须保留选项和推荐项，避免用户只能盲输答案。
        with patch("cli.ui.console.print") as mock_print:
            print_clarification_question(
                "Which database?",
                options=["PostgreSQL", "SQLite"],
                recommended="PostgreSQL",
            )
            panel = mock_print.call_args.args[0]
            rendered = "\n".join(str(item) for item in panel.renderable.renderables)
            self.assertIn("PostgreSQL", rendered)
            self.assertIn("SQLite", rendered)

        # Malformed tool payloads must degrade to a preview instead of
        # terminating the background stream's renderer.
        from cli.ui import _render_tool_content, print_tool_call_start
        print_tool_call_start("execute_python", {"code": ""})
        print_tool_call_start("search_mcp", {"mcp": None, "query": None})
        print_tool_call_start("search_memory", {"query": 123})
        print_tool_call_start("load_skill", {"name": 123})
        self.assertIn("3", format_tool_result_preview('{"stdout": 3}'))
        self.assertIsNotNone(_render_tool_content("execute_python", '{"stdout": 3}'))

    def test_app_ask_user_loop(self) -> None:
        """测试 AgentCliApp 在 ask_user 澄清提问时的多轮对话循环。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-123", "title": "Test"}

        # 模拟第一轮触发 ask_user，第二轮正常回复
        payload = {
            "question": "Which db?",
            "options": ["PostgreSQL", "SQLite"],
            "recommended": "PostgreSQL",
        }

        stream_calls: list[str] = []

        def mock_stream(q: str):
            stream_calls.append(q)
            if len(stream_calls) == 1:
                return payload
            return None

        app._ask_agent_stream = MagicMock(side_effect=mock_stream)
        # 模拟用户在 picker 中选择了 PostgreSQL
        app._prompt_ask_user = MagicMock(return_value="PostgreSQL")

        with patch("cli.ui.print_user_prompt") as mock_user_prompt:
            app.ask_agent("Initial question", print_prompt=False)

            # 验证第一轮发送 Initial question，第二轮自动把用户选择的 PostgreSQL 送入流
            self.assertEqual(["Initial question", "PostgreSQL"], stream_calls)
            # 自动打印了选择的选项 prompt: > PostgreSQL
            mock_user_prompt.assert_called_once_with("PostgreSQL")

    def test_queued_input_promotes_at_tool_boundary_without_polling(self) -> None:
        """排队输入只在活跃流的 tool_result 边界提升，不误认活跃回答。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-queue", "title": "Queue"}
        app._request_context.session_id = "sess-queue"
        app._request_context.request_id = "active-1"
        app._register_stream("sess-queue", "active-1")
        app._set_stream_state("active-1", "active")
        app._mark_queued_turn("sess-queue", "queued-1", "follow-up while running")

        app.client.ask_stream = MagicMock(return_value=[
            {"event": "tool_result", "data": {"name": "execute_python", "content": "done"}},
            {"event": "message_end", "data": {"message": {"content": "active answer"}}},
        ])
        with patch("cli.app.MarkdownStreamer") as streamer_cls, \
                patch("cli.app.ui.print_user_prompt"), \
                patch("cli.app.ui.print_queued_promotion") as print_promotion:
            streamer_cls.return_value = MagicMock()
            app._ask_agent_stream("active question")

        # When tool boundary is reached, queued turns are promoted with their question
        # text so they join the active conversation.
        print_promotion.assert_called_once_with(
            1, questions=["follow-up while running"]
        )
        self.assertEqual(0, app.queued_turn_count)

    def test_queued_ack_reports_persisted_waiting_state(self) -> None:
        """202 acknowledgement explains persistence and the next boundary."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-queue-status", "title": "Queue"}
        app._register_stream("sess-queue-status", "active-1")
        app._set_stream_state("active-1", "active")
        app._request_context.session_id = "sess-queue-status"
        app._request_context.request_id = "queued-1"
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "queued", "data": {"session_id": "sess-queue-status"}},
        ])

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.print_queued_status") as print_status:
            app._ask_agent_stream("follow-up")

        print_status.assert_not_called()
        self.assertEqual(1, app.queued_turn_count)

    def test_queued_status_is_compact_and_explains_the_tool_boundary(self) -> None:
        """The acknowledgement stays readable beside concurrent output."""
        from cli.ui import print_queued_status

        with patch("cli.ui.console.print") as print_line:
            print_queued_status("waiting")

        rendered = str(print_line.call_args.args[0])
        self.assertIn("queued", rendered)
        self.assertIn("saved", rendered)
        self.assertIn("next tool to finish", rendered)
        self.assertFalse(rendered.startswith("\n"))
        self.assertFalse(rendered.endswith("\n"))

    def test_queued_ack_updates_state_inside_render_lock(self) -> None:
        """Queue acknowledgement cannot print waiting after a promotion."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-queue-lock", "title": "Queue"}
        app._register_stream("sess-queue-lock", "active-1")
        app._set_stream_state("active-1", "active")
        app._request_context.session_id = "sess-queue-lock"
        app._request_context.request_id = "queued-1"
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "queued", "data": {"session_id": "sess-queue-lock"}},
        ])

        def mark_queued(*_args: object) -> str:
            self.assertTrue(app._render_lock.locked())
            return "waiting"

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch.object(app, "_mark_queued_turn", side_effect=mark_queued), \
                patch("cli.app.ui.print_queued_status") as print_status:
            app._ask_agent_stream("follow-up")

        print_status.assert_not_called()

    def test_late_queued_ack_waits_for_next_observed_tool_result(self) -> None:
        """A tool result before the 202 acknowledgement is not causal evidence."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-late", "title": "Late"}
        app._register_stream("sess-late", "active-1")
        app._register_stream("sess-late", "queued-1")
        app._set_stream_state("active-1", "active")
        app._mark_stream_boundary("sess-late", "active-1", terminal=True)
        app._unregister_stream("sess-late", "active-1")
        app._set_stream_state("queued-1", "queued")
        app._request_context.session_id = "sess-late"
        app._request_context.request_id = "queued-1"
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "queued", "data": {"session_id": "sess-late"}},
        ])

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.print_queued_status") as print_status, \
                patch("cli.app.ui.print_queued_promotion") as print_promotion:
            app._ask_agent_stream("late follow-up")

        print_status.assert_not_called()
        print_promotion.assert_not_called()
        self.assertEqual(1, app.queued_turn_count)

        app._register_stream("sess-late", "next-active")
        app._set_stream_state("next-active", "active")
        promoted = app._mark_stream_boundary("sess-late", "next-active")
        self.assertEqual(1, len(promoted))
        self.assertEqual(0, app.queued_turn_count)

    def test_queued_ack_during_earlier_tool_render_waits_for_next_boundary(self) -> None:
        """A delayed 202 cannot be consumed by the tool result already rendering."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-render-race", "title": "Race"}
        app._request_context.session_id = "sess-render-race"
        app._request_context.request_id = "active-1"
        app._register_stream("sess-render-race", "active-1")
        app._set_stream_state("active-1", "active")
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "tool_result", "data": {"name": "execute_python", "content": "done"}},
            {"event": "message_end", "data": {"message": {"content": "answer"}}},
        ])

        def acknowledge_during_render(_content: str) -> None:
            app._register_stream("sess-render-race", "queued-1")
            app._mark_queued_turn("sess-render-race", "queued-1", "late follow-up")

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.make_status", return_value=MagicMock()), \
                patch(
                    "cli.app.ui.print_tool_result_end",
                    side_effect=acknowledge_during_render,
                ), \
                patch("cli.app.ui.print_queued_promotion") as print_promotion:
            app._ask_agent_stream("active question")

        print_promotion.assert_not_called()
        self.assertEqual(1, app.queued_turn_count)

        app._register_stream("sess-render-race", "next-active")
        app._set_stream_state("next-active", "active")
        self.assertEqual(
            1,
            len(app._mark_stream_boundary("sess-render-race", "next-active")),
        )
        self.assertEqual(0, app.queued_turn_count)

    def test_queued_ack_is_recorded_before_status_cleanup(self) -> None:
        """A later tool boundary can consume a 202 while its spinner is stopping."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-ack-race", "title": "Race"}
        app._request_context.session_id = "sess-ack-race"
        app._request_context.request_id = "queued-1"
        app._register_stream("sess-ack-race", "queued-1")
        app._register_stream("sess-ack-race", "active-1")
        app._set_stream_state("active-1", "active")
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "queued", "data": {"session_id": "sess-ack-race"}},
        ])

        promoted: list[dict[str, object]] = []
        status = MagicMock()

        def observe_later_boundary() -> None:
            promoted.extend(
                app._mark_stream_boundary("sess-ack-race", "active-1")
            )

        status.stop.side_effect = observe_later_boundary
        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.make_status", return_value=status):
            app._ask_agent_stream("queued follow-up")

        self.assertEqual(1, len(promoted))
        self.assertEqual(0, app.queued_turn_count)

    def test_ask_user_boundary_does_not_promote_queued_message(self) -> None:
        """ask_user ends the graph, so its tool result cannot consume queued input."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-ask-queue", "title": "Ask"}
        app._register_stream("sess-ask-queue", "active-1")
        app._set_stream_state("active-1", "active")
        app._mark_queued_turn("sess-ask-queue", "queued-1", "follow-up")
        app._request_context.session_id = "sess-ask-queue"
        app._request_context.request_id = "active-1"
        app.client.ask_stream = MagicMock(return_value=[
            {
                "event": "tool_call",
                "data": {
                    "name": "ask_user",
                    "arguments": {"question": "Continue?", "options": ["yes", "no"]},
                },
            },
            {
                "event": "tool_result",
                "data": {"name": "ask_user", "content": '{"ok": true}'},
            },
            {
                "event": "message_end",
                "data": {"message": {"id": "m-ask", "role": "assistant", "content": "Continue?"}},
            },
        ])

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.print_queued_promotion") as print_promotion:
            app._ask_agent_stream("active question")

        print_promotion.assert_not_called()
        self.assertEqual(1, app.queued_turn_count)

    def test_message_end_does_not_promote_queued_message(self) -> None:
        """A final answer is not the requested tool-result context boundary."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-final-queue", "title": "Final"}
        app._register_stream("sess-final-queue", "active-1")
        app._set_stream_state("active-1", "active")
        app._mark_queued_turn("sess-final-queue", "queued-1", "follow-up")
        app._request_context.session_id = "sess-final-queue"
        app._request_context.request_id = "active-1"
        app.client.ask_stream = MagicMock(return_value=[
            {
                "event": "message_end",
                "data": {
                    "message": {
                        "id": "m-final",
                        "role": "assistant",
                        "content": "done",
                    }
                },
            },
        ])

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.print_queued_promotion") as print_promotion:
            app._ask_agent_stream("active question")

        print_promotion.assert_not_called()
        self.assertEqual(1, app.queued_turn_count)

    def test_late_queued_ack_after_ask_user_is_not_promoted(self) -> None:
        """A delayed 202 cannot treat ask_user completion as a context reload."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-ask-late", "title": "Ask"}
        app._register_stream("sess-ask-late", "active-1")
        app._register_stream("sess-ask-late", "queued-1")
        app._set_stream_state("active-1", "active")

        app._mark_stream_boundary(
            "sess-ask-late",
            "active-1",
            promote_queued=False,
        )
        app._mark_stream_boundary(
            "sess-ask-late",
            "active-1",
            terminal=True,
            promote_queued=False,
        )
        app._unregister_stream("sess-ask-late", "active-1")
        app._set_stream_state("queued-1", "queued")

        state = app._mark_queued_turn(
            "sess-ask-late",
            "queued-1",
            "late follow-up",
        )

        self.assertEqual("external", state)
        self.assertEqual(1, app.queued_turn_count)

    def test_queued_ask_user_continuation_tracks_each_persisted_input(self) -> None:
        """Each 202 stays visible until a later observed ordinary tool result."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-ask-reuse", "title": "Ask"}
        app._register_stream("sess-ask-reuse", "ask-owner")
        app._register_stream("sess-ask-reuse", "queued-1")
        app._set_stream_state("ask-owner", "active")
        app._set_stream_state("queued-1", "queued")
        self.assertEqual(
            "waiting",
            app._mark_queued_turn(
                "sess-ask-reuse",
                "queued-1",
                "follow-up",
            ),
        )
        app._unregister_stream("sess-ask-reuse", "queued-1")
        app._mark_stream_boundary(
            "sess-ask-reuse",
            "ask-owner",
            terminal=True,
            promote_queued=False,
        )
        app._unregister_stream("sess-ask-reuse", "ask-owner")
        self.assertEqual(1, app.queued_turn_count)

        app._register_stream("sess-ask-reuse", "ask-owner")
        app._set_stream_state("ask-owner", "queued")
        self.assertEqual(
            "external",
            app._mark_queued_turn(
                "sess-ask-reuse",
                "ask-owner",
                "clarification answer",
            ),
        )
        app._unregister_stream("sess-ask-reuse", "ask-owner")

        self.assertEqual(2, app.queued_turn_count)

    def test_late_queued_ack_does_not_use_request_start_boundary(self) -> None:
        """202 只等待其确认之后真正观察到的普通 tool_result。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-queue", "title": "Queue"}
        app._register_stream("sess-queue", "active-1")
        app._register_stream("sess-queue", "queued-1")
        app._set_stream_state("active-1", "active")

        with patch("cli.app.ui.print_user_prompt") as print_prompt:
            app._mark_stream_boundary("sess-queue", "active-1", terminal=True)
            app._unregister_stream("sess-queue", "active-1")
            app._set_stream_state("queued-1", "queued")
            app._mark_queued_turn("sess-queue", "queued-1", "late follow-up")

        print_prompt.assert_not_called()
        self.assertEqual(1, app.queued_turn_count)

    def test_late_queued_ack_after_failed_stream_remains_visible(self) -> None:
        """活动流失败也不能否定随后由 202 确认的持久化消息。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-error", "title": "Error"}
        app._register_stream("sess-error", "active-1")
        app._register_stream("sess-error", "queued-1")
        app._set_stream_state("active-1", "active")
        app._set_stream_state("queued-1", "queued")
        app._mark_queued_turn("sess-error", "queued-1", "follow-up")
        self.assertEqual(1, app.queued_turn_count)

        app._fail_stream_state("sess-error", "active-1")
        app._unregister_stream("sess-error", "active-1")
        app._mark_queued_turn("sess-error", "queued-1", "late follow-up")

        self.assertEqual(1, app.queued_turn_count)

    def test_same_request_failure_cannot_erase_a_queued_ack(self) -> None:
        """A failure before or after 202 is not evidence that persistence vanished."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-same-error", "title": "Error"}
        app._register_stream("sess-same-error", "queued-1")
        app._fail_stream_state("sess-same-error", "queued-1")

        self.assertEqual(
            "external",
            app._mark_queued_turn("sess-same-error", "queued-1", "late ack"),
        )
        app._fail_stream_state("sess-same-error", "queued-1")
        self.assertEqual(1, app.queued_turn_count)

    def test_terminal_boundary_cannot_promote_queued_messages(self) -> None:
        """terminal=True remains conservative even if a caller omits the flag."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-terminal", "title": "Terminal"}
        app._register_stream("sess-terminal", "active-1")
        app._set_stream_state("active-1", "active")
        app._mark_queued_turn("sess-terminal", "queued-1", "follow-up")

        promoted = app._mark_stream_boundary(
            "sess-terminal",
            "active-1",
            terminal=True,
        )

        self.assertEqual([], promoted)
        self.assertEqual(1, app.queued_turn_count)

    def test_legacy_queued_attribute_error_is_not_a_persistence_ack(self) -> None:
        """Only a real queued response may claim that the message was persisted."""
        from cli.app import AnnaCliApp
        from cli.ports import AgentApiError

        app = AnnaCliApp()
        app.current_session = {"id": "sess-error-ack", "title": "Error"}
        app._request_context.session_id = "sess-error-ack"
        app._request_context.request_id = "request-1"
        app.client.ask_stream = MagicMock(return_value=[
            {
                "event": "error",
                "data": {"message": "'Queued' object has no attribute 'message'"},
            },
        ])

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.make_status", return_value=MagicMock()), \
                patch("cli.app.ui.print_queued_status") as print_status:
            with self.assertRaisesRegex(AgentApiError, "Queued"):
                app._ask_agent_stream("follow-up")

        print_status.assert_not_called()
        self.assertEqual(0, app.queued_turn_count)

    def test_failed_stream_does_not_erase_sibling_stream_state(self) -> None:
        """One failed request cannot cancel another local stream's queue."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-shared", "title": "Shared"}
        app._register_stream("sess-shared", "stream-a")
        app._register_stream("sess-shared", "stream-b")
        app._set_stream_state("stream-a", "active")
        app._set_stream_state("stream-b", "active")
        app._mark_queued_turn("sess-shared", "queued-a", "follow-up")

        app._fail_stream_state("sess-shared", "stream-a")

        self.assertEqual("failed", app._stream_states["stream-a"]["state"])
        self.assertEqual("active", app._stream_states["stream-b"]["state"])
        self.assertEqual(1, app.queued_turn_count)

    def test_external_queued_ack_remains_visible_without_observed_boundary(self) -> None:
        """另一客户端的边界不可见时，CLI 不猜测消息已经被消费。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-external", "title": "External"}
        app._register_stream("sess-external", "queued-1")
        app._set_stream_state("queued-1", "queued")
        app._mark_queued_turn("sess-external", "queued-1", "follow-up")

        self.assertEqual(1, app.queued_turn_count)

    def test_external_queued_worker_follows_persisted_progress_to_final_answer(self) -> None:
        """接手其它客户端的 run 时，202 worker 应通过历史同步后续进度。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-follow", "title": "Follow"}
        queued_human = {
            "seq": 10,
            "role": "human",
            "content": "follow-up",
            "metadata": {"source": "user", "kind": "user"},
        }
        tool_call = {
            "seq": 11,
            "role": "assistant",
            "content": "Continuing with the queued input.",
            "tool_calls": [{"name": "wait_for_replies"}],
            "metadata": {"kind": "assistant_tool_call"},
        }
        tool_result = {
            "seq": 12,
            "role": "tool",
            "content": '{"ok": true, "waited_seconds": 1}',
            "tool_name": "wait_for_replies",
            "metadata": {"kind": "tool_result"},
        }
        final_answer = {
            "seq": 13,
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [],
            "metadata": {"kind": "assistant_answer"},
        }
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "queued", "data": {"session_id": "sess-follow"}},
        ])
        app.client.list_messages = MagicMock(side_effect=[
            [queued_human],
            [queued_human, tool_call, tool_result],
            [queued_human, tool_call, tool_result, final_answer],
        ])

        with patch("cli.app._QUEUE_WATCH_INTERVAL_SECONDS", 0.0), \
                patch("cli.app._QUEUE_WATCH_IDLE_TIMEOUT_SECONDS", 0.1), \
                patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.print_queued_status") as print_status, \
                patch("cli.app.ui.print_tool_result_end") as print_tool_result, \
                patch("cli.app.ui.print_queued_promotion") as print_promotion, \
                patch("cli.app.ui.print_message") as print_message, \
                patch("cli.app.ui.print_queued_deferred") as print_deferred, \
                patch("cli.app.ui.print_queue_watch_stopped") as print_stopped:
            app._run_background_question("follow-up", "sess-follow", "queued-1")

        print_status.assert_not_called()
        print_tool_result.assert_called_once_with(tool_result["content"])
        print_promotion.assert_called_once_with(1, questions=["follow-up"])
        self.assertEqual(2, print_message.call_count)
        print_deferred.assert_not_called()
        print_stopped.assert_not_called()
        self.assertEqual(0, app.queued_turn_count)
        self.assertEqual({}, app._queue_watch_owners)
        self.assertTrue(all(
            call.kwargs.get("timeout") == 5.0
            for call in app.client.list_messages.call_args_list
        ))

    def test_external_queue_watcher_reports_repeated_poll_failure(self) -> None:
        """History polling failures must become visible instead of looking hung."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-poll-fail", "title": "Poll failure"}
        app._mark_queued_turn("sess-poll-fail", "queued-1", "follow-up")
        app.client.list_messages = MagicMock(side_effect=RuntimeError("backend unavailable"))

        with patch("cli.app._QUEUE_WATCH_INTERVAL_SECONDS", 0.0), \
                patch("cli.app._QUEUE_WATCH_MAX_POLL_FAILURES", 3), \
                patch("cli.app.ui.print_queue_watch_stopped") as print_stopped:
            app._watch_external_queue("sess-poll-fail", "queued-1")

        self.assertEqual(3, app.client.list_messages.call_count)
        print_stopped.assert_called_once_with("History refresh failed 3 times.")
        self.assertEqual(1, app.queued_turn_count)
        self.assertEqual({}, app._queue_watch_owners)

    def test_external_queue_watcher_shutdown_releases_owner_without_warning(self) -> None:
        """CLI shutdown cancels polling quietly and releases the per-session owner."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-watch-stop", "title": "Stop"}
        app._mark_queued_turn("sess-watch-stop", "queued-1", "follow-up")
        app.client.list_messages = MagicMock()
        app._shutdown_event.set()

        with patch("cli.app.ui.print_queue_watch_stopped") as print_stopped:
            app._watch_external_queue("sess-watch-stop", "queued-1")

        app.client.list_messages.assert_not_called()
        print_stopped.assert_not_called()
        self.assertEqual({}, app._queue_watch_owners)

    def test_external_queue_terminal_before_tool_boundary_stays_queued(self) -> None:
        """A final answer without a later tool result cannot claim queue consumption."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-no-boundary", "title": "No boundary"}
        app._mark_queued_turn("sess-no-boundary", "queued-1", "follow-up")
        queued_human = {
            "seq": 20,
            "role": "human",
            "content": "follow-up",
            "metadata": {"source": "user", "kind": "user"},
        }
        final_answer = {
            "seq": 21,
            "role": "assistant",
            "content": "The active turn ended before another tool call.",
            "tool_calls": [],
            "metadata": {"kind": "assistant_answer"},
        }
        app.client.list_messages = MagicMock(side_effect=[
            [queued_human],
            [queued_human, final_answer],
        ])

        with patch("cli.app._QUEUE_WATCH_INTERVAL_SECONDS", 0.0), \
                patch("cli.app.ui.print_message") as print_message, \
                patch("cli.app.ui.print_queued_promotion") as print_promotion, \
                patch("cli.app.ui.print_queued_deferred") as print_deferred, \
                patch("cli.app.ui.print_queue_watch_stopped") as print_stopped:
            app._watch_external_queue("sess-no-boundary", "queued-1")

        print_message.assert_called_once_with(
            final_answer,
            show_subagent_details=False,
        )
        print_promotion.assert_not_called()
        print_deferred.assert_called_once_with(1)
        print_stopped.assert_not_called()
        self.assertEqual(1, app.queued_turn_count)
        self.assertEqual({}, app._queue_watch_owners)

    def test_identical_external_messages_bind_to_distinct_persisted_rows(self) -> None:
        """Repeated text must map newest-to-newest without reusing an old history row."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-identical", "title": "Identical"}
        app._mark_queued_turn("sess-identical", "queued-1", "same message")
        app._mark_queued_turn("sess-identical", "queued-2", "same message")
        messages = [
            {
                "seq": seq,
                "role": "human",
                "content": "same message",
                "metadata": {"source": "user", "kind": "user"},
            }
            for seq in (3, 10, 11)
        ]

        app._bind_queued_message_seqs("sess-identical", messages)

        queued = app._queued_turns["sess-identical"]
        self.assertEqual([10, 11], [item["message_seq"] for item in queued])
        promoted = app._promote_queued_through("sess-identical", 11)
        self.assertEqual(["queued-1"], [item["request_id"] for item in promoted])
        self.assertEqual(1, app.queued_turn_count)

    def test_multiple_external_queued_acks_track_each_persisted_input(self) -> None:
        """多个本地 202 在看见下一次工具结果前都保持 queued。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-external-many", "title": "External"}
        app._register_stream("sess-external-many", "queued-1")
        app._register_stream("sess-external-many", "queued-2")

        app._set_stream_state("queued-1", "queued")
        first_state = app._mark_queued_turn(
            "sess-external-many",
            "queued-1",
            "first follow-up",
        )
        app._unregister_stream("sess-external-many", "queued-1")

        app._set_stream_state("queued-2", "queued")
        second_state = app._mark_queued_turn(
            "sess-external-many",
            "queued-2",
            "second follow-up",
        )
        app._unregister_stream("sess-external-many", "queued-2")

        self.assertEqual("waiting", first_state)
        self.assertEqual("external", second_state)
        self.assertEqual(2, app.queued_turn_count)

    def test_switch_session_clears_local_queued_state(self) -> None:
        """切换会话时丢弃旧会话的本地 queued 展示状态。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-a", "title": "A"}
        app._register_stream("sess-a", "active-1")
        app._set_stream_state("active-1", "active")
        app._mark_queued_turn("sess-a", "queued-1", "queued elsewhere")

        app.client.list_messages = MagicMock(return_value=[
            {"role": "human", "content": "queued elsewhere", "seq": 1}
        ])
        with patch("cli.app.ui.print_history_list"), \
                patch("cli.app.ui.print_user_prompt") as print_prompt:
            app.open_session({"id": "sess-b", "title": "B"})

        print_prompt.assert_not_called()
        self.assertNotIn("sess-a", app._queued_turns)

    def test_message_end_renders_new_content_after_tool_boundary(self) -> None:
        """工具前后的两个文本片段都应显示，同时避免重复旧片段。"""
        from unittest.mock import call

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-tool", "title": "Tool"}
        app._request_context.session_id = "sess-tool"
        app._request_context.request_id = "active-1"
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "text_delta", "data": {"content": "before"}},
            {"event": "tool_call", "data": {"name": "execute_python", "arguments": {}}},
            {"event": "tool_result", "data": {"name": "execute_python", "content": "done"}},
            {"event": "message_end", "data": {"message": {
                "id": "m-1", "role": "assistant", "content": "after"
            }}},
        ])
        with patch("cli.app.MarkdownStreamer") as streamer_cls:
            streamer = streamer_cls.return_value
            app._ask_agent_stream("question")

        self.assertEqual([call("before"), call("after")], streamer.finish.call_args_list)

    def test_message_end_splits_queued_followup_suffix(self) -> None:
        """同一 SSE 中连续两条回复不能被拼成一条 assistant 文本。"""
        from unittest.mock import call

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-queue", "title": "Queue"}
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "text_delta", "data": {"content": "FIRST"}},
            {"event": "text_delta", "data": {"content": "SECOND"}},
            {"event": "message_end", "data": {"message": {
                "id": "m-2", "role": "assistant", "content": "SECOND"
            }}},
        ])
        with patch("cli.app.MarkdownStreamer") as streamer_cls:
            first_streamer = MagicMock()
            second_streamer = MagicMock()
            streamer_cls.side_effect = [first_streamer, second_streamer]
            app._ask_agent_stream("question")

        self.assertEqual([call("FIRST")], first_streamer.finish.call_args_list)
        self.assertEqual([call("SECOND")], second_streamer.finish.call_args_list)

    def test_message_end_trailing_whitespace_is_not_rendered_twice(self) -> None:
        """Delta and final envelope whitespace differences are one answer."""
        from unittest.mock import call

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-whitespace", "title": "Whitespace"}
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "text_delta", "data": {"content": "Hello\n"}},
            {"event": "message_end", "data": {"message": {
                "id": "m-w", "role": "assistant", "content": "Hello"
            }}},
        ])
        with patch("cli.app.MarkdownStreamer") as streamer_cls:
            streamer = streamer_cls.return_value
            app._ask_agent_stream("question")

        self.assertEqual([call("Hello\n")], streamer.finish.call_args_list)

    def test_message_end_queued_suffix_with_trailing_whitespace_is_split_once(self) -> None:
        """A queued answer suffix with a newline still renders as one segment."""
        from unittest.mock import call

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-whitespace-queue", "title": "Whitespace"}
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "text_delta", "data": {"content": "FIRST\nSECOND\n"}},
            {"event": "message_end", "data": {"message": {
                "id": "m-wq", "role": "assistant", "content": "SECOND"
            }}},
        ])
        with patch("cli.app.MarkdownStreamer") as streamer_cls:
            first_streamer = MagicMock()
            second_streamer = MagicMock()
            streamer_cls.side_effect = [first_streamer, second_streamer]
            app._ask_agent_stream("question")

        self.assertEqual([call("FIRST\n")], first_streamer.finish.call_args_list)
        self.assertEqual([call("SECOND")], second_streamer.finish.call_args_list)

    def test_thinking_indicator_positions_above_prompt_like_codex(self) -> None:
        """Thinking 状态显示在 Prompt 上方（参考 Codex 布局），并支持 esc 中断与队列展示。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-queue", "title": "Queue"}
        self.assertEqual("<prompt>› </prompt>", app.get_prompt_text().value)

        app._register_stream("sess-queue", "active-1")
        app._set_stream_state("active-1", "active")
        app._mark_queued_turn("sess-queue", "queued-1", "follow-up")
        prompt_with_queue = app.get_prompt_text().value
        self.assertIn("Messages to be submitted after next tool call", prompt_with_queue)
        self.assertIn("follow-up", prompt_with_queue)

        worker = MagicMock()
        with app._turn_threads_lock:
            app._turn_threads.add(worker)
        try:
            prompt_text = app.get_prompt_text().value
            self.assertIn("Working", prompt_text)
            self.assertIn("follow-up", prompt_text)

            toolbar_text = "".join(text for _, text in app.get_status_toolbar())
            self.assertIn(app.model_name, toolbar_text)

            # When streaming text output starts, Working status remains clear
            app._set_stream_text_streaming("active-1", True)
            self.assertTrue(app.is_streaming_text)
            prompt_streaming = app.get_prompt_text().value
            self.assertIn("Working", prompt_streaming)

            # When streaming stops, Working continues while turn is active
            app._set_stream_text_streaming("active-1", False)
            self.assertFalse(app.is_streaming_text)
            self.assertIn("Working", app.get_prompt_text().value)

            # When a specific tool is running, the prompt displays the tool name
            app._set_stream_activity("active-1", "Running execute_bash...")
            self.assertIn("Running execute_bash...", app.get_prompt_text().value)
        finally:
            with app._turn_threads_lock:
                app._turn_threads.discard(worker)

    def test_live_running_indicator_and_tool_activity_stream_lifecycle(self) -> None:
        """运行状态实时感知：工具执行展示工具名，流式输出展示生成中，空闲时恢复纯净模型徽标。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-lifecycle", "title": "Lifecycle"}
        app._request_context.session_id = "sess-lifecycle"
        app._request_context.request_id = "req-lifecycle"
        app._request_context.bound = True

        events = [
            {"event": "tool_call", "data": {"name": "execute_bash", "arguments": {"cmd": "ls"}}},
            {"event": "tool_result", "data": {"name": "execute_bash", "content": "file1.txt"}},
            {"event": "text_delta", "data": {"content": "Found file1."}},
            {"event": "message_end", "data": {"message": {"content": "Found file1."}}},
        ]
        app.client.ask_stream = MagicMock(return_value=events)

        activities_observed: list[str | None] = []
        prompts_observed: list[str] = []

        worker = MagicMock()
        with app._turn_threads_lock:
            app._turn_threads.add(worker)

        original_set_activity = app._set_stream_activity

        def tracking_set_activity(request_id: str, activity: str | None) -> None:
            original_set_activity(request_id, activity)
            activities_observed.append(activity)
            prompts_observed.append(app.get_prompt_text().value)

        app._set_stream_activity = tracking_set_activity

        with patch("cli.app.MarkdownStreamer"), patch("cli.app.ui.console"):
            app._ask_agent_stream("test lifecycle")

        with app._turn_threads_lock:
            app._turn_threads.discard(worker)

        # 检查活动状态流变：开始 Working -> 工具执行 Running execute_bash... ->
        # 结果后 Working -> 生成 Generating... -> 结束 None
        self.assertIn("Working...", activities_observed)
        self.assertIn("Running execute_bash...", activities_observed)
        self.assertIn("Generating...", activities_observed)
        self.assertEqual(activities_observed[-1], None)

        # 检查工具运行时的 prompt 显示具体工具名称
        tool_prompts = [p for p in prompts_observed if "Running execute_bash..." in p]
        self.assertTrue(len(tool_prompts) > 0)
        self.assertIn("Working", tool_prompts[0])

        # 检查空闲时底部工具栏展示模型名
        toolbar_text = "".join(text for _, text in app.get_status_toolbar())
        self.assertIn(app.model_name, toolbar_text)
        self.assertEqual("", app.get_activity_text().value)

    def test_background_stream_incrementally_renders_without_live_over_prompt(self) -> None:
        """后台正文逐步刷新，但不能用 Rich Live 覆盖仍在编辑的 prompt。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-visual", "title": "Visual"}
        app._request_context.session_id = "sess-visual"
        app._request_context.request_id = "request-visual"
        app._request_context.bound = True
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "text_delta", "data": {"content": "clean output"}},
            {"event": "message_end", "data": {"message": {"content": "clean output"}}},
        ])

        prompts_during_stream: list[str] = []
        worker = MagicMock()
        with app._turn_threads_lock:
            app._turn_threads.add(worker)

        def capture_prompt(_content: str) -> None:
            prompts_during_stream.append(app.get_prompt_text().value)

        with patch("cli.app.MarkdownStreamer") as streamer_cls, \
                patch("cli.app.ui.make_status"):
            streamer_cls.return_value.update.side_effect = capture_prompt
            app._ask_agent_stream("question")

        with app._turn_threads_lock:
            app._turn_threads.discard(worker)

        self.assertTrue(streamer_cls.call_args_list)
        self.assertTrue(all(
            call.kwargs.get("live_enabled") is False
            for call in streamer_cls.call_args_list
        ))
        self.assertTrue(all(
            call.kwargs.get("live_window") == 1
            for call in streamer_cls.call_args_list
        ))
        streamer_cls.return_value.update.assert_called_once_with("clean output")
        streamer_cls.return_value.finish.assert_called_once_with("clean output")
        self.assertTrue(len(prompts_during_stream) > 0)
        self.assertTrue(all("Working" in p for p in prompts_during_stream))
        self.assertTrue(all("›" in p for p in prompts_during_stream))

    def test_submit_background_question_returns_before_network_finishes(self) -> None:
        """提交只启动 worker，不等待后端响应，输入框可立即进入下一轮。"""
        import threading
        import time

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-background", "title": "Background"}
        entered = threading.Event()
        release = threading.Event()

        def blocked_ask(_question: str, print_prompt: bool = False) -> None:
            self.assertFalse(print_prompt)
            entered.set()
            release.wait(timeout=2)

        app.ask_agent = blocked_ask
        started = time.monotonic()
        with patch("cli.app.ui.print_user_prompt") as print_user:
            app._submit_background_question("keep the prompt available")
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        print_user.assert_called_once_with("keep the prompt available")
        self.assertTrue(entered.wait(timeout=1))
        self.assertEqual(1, app.active_turn_count)
        release.set()

        deadline = time.monotonic() + 2
        while app.active_turn_count and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(0, app.active_turn_count)

    def test_submit_while_running_queues_and_promotes_on_context_update(self) -> None:
        """正在运行中输入只提示入队，不直接打入会话；等待 context updated 时才正式渲染入对话。"""
        import threading
        import time

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-live-queue", "title": "Queue Test"}
        entered = threading.Event()
        release = threading.Event()

        def blocked_ask(_question: str, print_prompt: bool = False) -> None:
            entered.set()
            release.wait(timeout=2)

        app.ask_agent = blocked_ask
        with patch("cli.app.ui.print_user_prompt") as print_user, \
                patch("cli.app.ui.print_queued_input") as print_queued:
            app._submit_background_question("First question")
            print_user.assert_called_once_with("First question")
            print_queued.assert_not_called()

        self.assertTrue(entered.wait(timeout=1))
        self.assertEqual(1, app.active_turn_count)

        with patch("cli.app.ui.print_user_prompt") as print_user, \
                patch("cli.app.ui.print_queued_input") as print_queued:
            app._submit_background_question("Queued question while running")
            print_user.assert_not_called()
            print_queued.assert_called_once_with("Queued question while running")

        with patch("cli.app.ui.print_user_prompt") as print_user, \
                patch("cli.app.ui.console.print"):
            import cli.ui as ui
            ui.print_queued_promotion(1, questions=["Queued question while running"])
            print_user.assert_called_once_with("Queued question while running")

        release.set()
        deadline = time.monotonic() + 2
        while app.active_turn_count and time.monotonic() < deadline:
            time.sleep(0.005)

    def test_queued_question_executes_as_sequential_followup_turn(self) -> None:
        """排队的问题在前一轮结束后作为后续轮次按序执行，并正确打印用户提问。"""
        import threading
        import time

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-seq-queue", "title": "Sequential Queue"}

        turn1_entered = threading.Event()
        turn1_release = threading.Event()
        turn2_entered = threading.Event()
        turn2_release = threading.Event()
        ask_order: list[str] = []

        def tracked_ask(question: str, print_prompt: bool = False) -> None:
            ask_order.append(question)
            if question == "Q1":
                turn1_entered.set()
                turn1_release.wait(timeout=2)
            elif question == "Q2":
                turn2_entered.set()
                turn2_release.wait(timeout=2)

        app.ask_agent = tracked_ask
        user_prompts: list[str] = []
        queued_inputs: list[str] = []

        with patch("cli.app.ui.print_user_prompt", side_effect=user_prompts.append), \
                patch("cli.app.ui.print_queued_input", side_effect=queued_inputs.append):
            # Submit Q1
            app._submit_background_question("Q1")
            self.assertTrue(turn1_entered.wait(timeout=1))
            self.assertEqual(["Q1"], user_prompts)
            self.assertEqual([], queued_inputs)

            # Submit Q2 while Q1 is running
            app._submit_background_question("Q2")
            self.assertEqual(["Q1"], user_prompts)
            self.assertEqual(["Q2"], queued_inputs)
            self.assertEqual(1, app.queued_turn_count)

            # Complete Q1; Q2 should start as sequential follow-up
            turn1_release.set()
            self.assertTrue(turn2_entered.wait(timeout=1))
            self.assertEqual(["Q1", "Q2"], user_prompts)
            self.assertEqual(0, app.queued_turn_count)

            # Complete Q2
            turn2_release.set()
            deadline = time.monotonic() + 2
            while app.active_turn_count and time.monotonic() < deadline:
                time.sleep(0.005)

        self.assertEqual(0, app.active_turn_count)
        self.assertEqual(["Q1", "Q2"], ask_order)

    def test_multiturn_stream_suffix_does_not_duplicate_text_or_invert_queue(self) -> None:
        """多轮合并流式输出不重复打印后缀文本，不后置倒装渲染用户已回答的问题。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.model_name = "Qwen 3.8 Flash"
        session_id = "sess-multiturn-order"
        app.current_session = {"id": session_id, "title": "Order Test"}
        app._request_context.session_id = session_id
        app._request_context.request_id = "req-main"

        # Simulate queued turns entered while the turn was running
        app._mark_queued_turn(session_id, "req-q1", "but")
        app._mark_queued_turn(session_id, "req-q2", "wait")
        app._mark_queued_turn(session_id, "req-q3", "oh")
        self.assertEqual(3, app.queued_turn_count)

        part1 = "You're welcome! If anything comes up, just let me know. 😊"
        part2 = "Take your time — I'm here whenever you're ready. 🙂"
        part3 = "Is there something you'd like help with? I can answer questions."

        app.client.ask_stream = MagicMock(return_value=[
            {"event": "text_delta", "data": {"content": f"{part1}\n"}},
            {"event": "text_delta", "data": {"content": f"{part2}\n"}},
            {"event": "text_delta", "data": {"content": part3}},
            # message_end contains only the final envelope (part3)
            {"event": "message_end", "data": {"message": {"content": part3}}},
        ])

        streamer_instances = []

        def fake_streamer(*args, **kwargs):
            m = MagicMock()
            streamer_instances.append(m)
            return m

        with patch("cli.app.MarkdownStreamer", side_effect=fake_streamer), \
                patch("cli.app.ui.print_queued_promotion") as mock_promotion:
            app._ask_agent_stream("ok thanks")

        # 1. Prefix is flushed on first streamer, suffix on second streamer
        self.assertEqual(2, len(streamer_instances))
        streamer_instances[0].finish.assert_called_once_with(f"{part1}\n{part2}\n")
        streamer_instances[1].finish.assert_called_once_with(part3)
        # 2. Queued items answered in this stream are retired cleanly
        self.assertEqual(0, app.queued_turn_count)
        # 3. Queued promotion is NOT called upside-down after the final answer
        mock_promotion.assert_not_called()

    def test_response_label_includes_active_model(self) -> None:
        """AgentDesk 回复标题包含当前模型展示名。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.model_name = "GLM 5.3 Flash"
        app.current_session = {"id": "sess-model", "title": "Model Test"}
        app._request_context.session_id = "sess-model"
        app._request_context.request_id = "req-model"
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "text_delta", "data": {"content": "Hello world"}},
            {"event": "message_end", "data": {"message": {"content": "Hello world"}}},
        ])

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.console.print") as mock_print:
            app._ask_agent_stream("test")
            printed = [str(call[0][0]) for call in mock_print.call_args_list if call[0]]
            self.assertTrue(any("GLM 5.3 Flash" in line for line in printed))

    def test_ask_user_waiters_keep_visual_fifo_order(self) -> None:
        """并发澄清问题的显示顺序必须和主 prompt 的回答队列一致。"""
        import threading
        import time

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        printed: list[str] = []
        results: dict[str, str | None] = {}
        payloads = {
            "A": {"question": "Question A", "options": ["a"]},
            "B": {"question": "Question B", "options": ["b"]},
        }

        def ask(label: str) -> None:
            results[label] = app._prompt_ask_user(payloads[label])

        with patch(
            "cli.app.ui.print_clarification_question",
            side_effect=lambda question, **_kwargs: printed.append(question),
        ):
            workers = [threading.Thread(target=ask, args=(label,)) for label in ("A", "B")]
            for worker in workers:
                worker.start()

            deadline = time.time() + 2.0
            while time.time() < deadline:
                with app._ask_user_lock:
                    if len(app._pending_ask_users) == 2:
                        pending = list(app._pending_ask_users)
                        break
                time.sleep(0.005)
            else:
                self.fail("ask_user waiters were not enqueued")

            self.assertEqual([], printed)
            self.assertEqual(
                {"Question A", "Question B"},
                {request["payload"]["question"] for request in pending},
            )
            for index, request in enumerate(pending, 1):
                request["answer"] = f"answer-{index}"
                request["event"].set()

            for worker in workers:
                worker.join(timeout=2.0)

        self.assertEqual(2, len(results))
        self.assertCountEqual(results.values(), {"answer-1", "answer-2"})

    def test_ask_user_numeric_answer_maps_to_choice(self) -> None:
        """主 prompt 的数字回答应选择对应的 ask_user 选项。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        request = app._enqueue_ask_user_request(
            {"question": "Which database?", "options": ["PostgreSQL", "SQLite"]}
        )
        self.assertTrue(app._answer_pending_ask_user("2"))
        self.assertEqual("SQLite", request["answer"])
        self.assertTrue(request["event"].is_set())

    def test_permission_picker_renders_below_input_and_uses_arrow_keys(self) -> None:
        """Permission stays in the prompt toolbar and Up/Down changes Enter's choice."""
        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        self.assertTrue(app.session_prompt.layout.current_window.dont_extend_height())
        self.assertIsNone(app.session_prompt.bottom_toolbar)
        with patch("cli.app.ui.print_permission_request") as legacy_panel:
            request = app._enqueue_permission_request(
                {
                    "permission_id": "perm-picker",
                    "session_id": "sess-picker",
                    "name": "execute_python",
                    "arguments": {"code": "import time\nprint(time.time())"},
                },
                owner_id="turn-picker",
            )

        legacy_panel.assert_not_called()
        self.assertEqual(app.get_permission_toolbar, app.session_prompt.bottom_toolbar)
        self.assertTrue(app._should_hide_prompt_input())
        self.assertTrue(app.session_prompt.layout.current_window.always_hide_cursor())
        self.assertEqual("", app.get_prompt_text().value)

        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("Permission required", toolbar)
        self.assertIn("execute_python", toolbar)
        self.assertIn("› Allow once", toolbar)
        self.assertIn("Always allow tool 'execute_python'", toolbar)
        self.assertIn("Reject", toolbar)
        self.assertNotIn("print(time.time())", toolbar)

        bindings = app.session_prompt.key_bindings.bindings
        down = next(binding for binding in bindings if binding.keys == (Keys.Down,))
        enter = next(binding for binding in bindings if binding.keys == (Keys.ControlM,))
        event = MagicMock()
        event.current_buffer.text = ""
        down.handler(event)
        enter.handler(event)

        self.assertEqual(1, request["selected_index"])
        self.assertEqual("always", event.current_buffer.text)
        event.current_buffer.validate_and_handle.assert_called_once_with()

        event.current_buffer.text = ""
        down.handler(event)
        enter.handler(event)
        self.assertEqual(2, request["selected_index"])
        self.assertEqual("reject", event.current_buffer.text)

        draft_event = MagicMock()
        draft_event.current_buffer.text = "keep this draft"
        enter.handler(draft_event)
        self.assertEqual("keep this draft", app._prompt_draft)
        self.assertEqual("reject", draft_event.current_buffer.text)
        draft_event.current_buffer.validate_and_handle.assert_called_once_with()

        request["state"] = "submitting"
        blocked_event = MagicMock()
        blocked_event.current_buffer.text = "still editing"
        enter.handler(blocked_event)
        blocked_event.current_buffer.validate_and_handle.assert_not_called()

        app._cancel_ask_user_request(request)
        self.assertIsNone(app.session_prompt.bottom_toolbar)
        self.assertEqual(8, app.session_prompt.reserve_space_for_menu)

    def test_permission_picker_number_shortcuts_and_no_duplicate_bash_command(self) -> None:
        """权限选择支持数字快捷键(1/2/3), 且不重复展示命令内容。"""
        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        request = app._enqueue_permission_request(
            {
                "permission_id": "perm-bash",
                "session_id": "sess-bash",
                "name": "bash",
                "arguments": {"command": "echo 'system info'"},
                "targets": ["echo 'system info'"],
            },
            owner_id="turn-bash",
        )

        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("Permission required: bash", toolbar)
        # 不作为单独的命令行再次重复打印
        self.assertNotIn("$ echo 'system info'", toolbar)
        # 选项中包含作用域预览
        self.assertIn("Always allow 'echo 'system info''", toolbar)

        bindings = app.session_prompt.key_bindings.bindings
        any_key = next(binding for binding in bindings if binding.keys == (Keys.Any,))

        # 按 '2' 选择第2项 (always)
        event2 = MagicMock()
        event2.data = "2"
        any_key.handler(event2)
        self.assertEqual(1, request["selected_index"])

        # 按 '3' 选择第3项 (reject)
        event3 = MagicMock()
        event3.data = "3"
        any_key.handler(event3)
        self.assertEqual(2, request["selected_index"])

        # 按 '1' 选择第1项 (once)
        event1 = MagicMock()
        event1.data = "1"
        any_key.handler(event1)
        self.assertEqual(0, request["selected_index"])

        app._cancel_ask_user_request(request)

    def test_ask_user_picker_toolbar_highlight_and_enter_confirm(self) -> None:
        """ask_user 与权限共用底部选择条:箭头移动高亮,空 Enter 提交高亮项。"""
        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question"):
            request = app._enqueue_ask_user_request(
                {
                    "question": "Which database?",
                    "options": ["PostgreSQL", "SQLite"],
                    "recommended": "SQLite",
                }
            )
        self.assertEqual(1, request.get("selected_index"))
        self.assertEqual(app.get_permission_toolbar, app.session_prompt.bottom_toolbar)
        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("Clarification", toolbar)
        self.assertIn("Which database?", toolbar)
        self.assertIn("\u203a 2. SQLite", toolbar)
        self.assertIn("Enter confirm", toolbar)

        bindings = app.session_prompt.key_bindings.bindings
        up = next(binding for binding in bindings if binding.keys == (Keys.Up,))
        enter = next(binding for binding in bindings if binding.keys == (Keys.ControlM,))
        event = MagicMock()
        event.current_buffer.text = ""
        up.handler(event)
        self.assertEqual(0, request["selected_index"])
        enter.handler(event)
        self.assertEqual("PostgreSQL", event.current_buffer.text)
        event.current_buffer.validate_and_handle.assert_called_once_with()
        self.assertEqual("pending", request["state"])

        self.assertTrue(app._answer_pending_ask_user("PostgreSQL"))
        self.assertEqual("PostgreSQL", request["answer"])
        self.assertTrue(request["event"].is_set())
        self.assertIsNone(app.session_prompt.bottom_toolbar)

    def test_out_of_range_number_keeps_ask_user_waiter(self) -> None:
        """越界编号不发送给模型:提示重选,等待者保持原位。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question"), patch("cli.app.ui.console.print"):
            request = app._enqueue_ask_user_request(
                {"question": "Pick", "options": ["A-one", "B-two"]}
            )
            self.assertTrue(app._answer_pending_ask_user("123"))
        self.assertEqual("pending", request["state"])
        self.assertIsNone(request["answer"])
        self.assertTrue(app._answer_pending_ask_user("2"))
        self.assertEqual("B-two", request["answer"])

    def test_permission_reply_does_not_block_prompt_thread(self) -> None:
        """权限确认的 HTTP 往返必须在后台进行。"""
        import threading
        import time

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-perm", "title": "Permission"}
        entered = threading.Event()
        release = threading.Event()

        def delayed_reply(*_args: object, **_kwargs: object) -> dict[str, str]:
            entered.set()
            release.wait(timeout=2)
            return {
                "status": "accepted",
                "permission_id": "perm-1",
                "decision": "once",
            }

        app.client.reply_permission = MagicMock(side_effect=delayed_reply)
        with patch("cli.app.ui.print_permission_request"):
            request = app._enqueue_permission_request(
                {
                    "permission_id": "perm-1",
                    "session_id": "sess-perm",
                    "name": "execute_python",
                    "arguments": {"code": "print(1)"},
                },
                owner_id="turn-1",
            )
        started = time.monotonic()
        self.assertTrue(app._answer_pending_ask_user("1"))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(entered.wait(timeout=1))
        release.set()
        self.assertTrue(request["event"].wait(timeout=2))
        self.assertEqual("once", request["answer"])
        self.assertFalse(app._has_pending_ask_users())
        app.client.reply_permission.assert_called_once_with("perm-1", "allow", scope=None)

    def test_permission_reply_allow_project_and_reject(self) -> None:
        """测试 3 个选项的解析与派发：1->once, 2->always, 3->reject。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-perm", "title": "Permission"}
        app.client.reply_permission = MagicMock(return_value={"status": "accepted"})

        # 选项 2: always（无 targets，回填 [""]）
        with patch("cli.app.ui.print_permission_request"):
            req2 = app._enqueue_permission_request(
                {"permission_id": "perm-2", "name": "edit_file"},
                owner_id="turn-2",
            )
        self.assertTrue(app._answer_pending_ask_user("2"))
        self.assertTrue(req2["event"].wait(timeout=2))
        self.assertEqual("always", req2["answer"])
        app.client.reply_permission.assert_called_with("perm-2", "allow", scope=[""])

        # 别名: p / always / project / allow_project
        for alias in ("p", "always", "project", "allow_project"):
            with patch("cli.app.ui.print_permission_request"):
                req = app._enqueue_permission_request(
                    {"permission_id": f"perm-{alias}", "name": "edit_file"},
                    owner_id=f"turn-{alias}",
                )
            self.assertTrue(app._answer_pending_ask_user(alias))
            self.assertTrue(req["event"].wait(timeout=2))
            self.assertEqual("always", req["answer"])
            app.client.reply_permission.assert_called_with(f"perm-{alias}", "allow", scope=[""])

        # targets: 单段回填 [target]
        with patch("cli.app.ui.print_permission_request"):
            req_single = app._enqueue_permission_request(
                {"permission_id": "perm-single", "name": "bash", "targets": ["git status"]},
                owner_id="turn-single",
            )
        self.assertTrue(app._answer_pending_ask_user("2"))
        self.assertTrue(req_single["event"].wait(timeout=2))
        app.client.reply_permission.assert_called_with("perm-single", "allow", scope=["git status"])

        # targets: 多段回填 list(targets)
        with patch("cli.app.ui.print_permission_request"):
            req_multi = app._enqueue_permission_request(
                {
                    "permission_id": "perm-multi",
                    "name": "bash",
                    "targets": ["git status", "git commit"],
                },
                owner_id="turn-multi",
            )
        self.assertTrue(app._answer_pending_ask_user("2"))
        self.assertTrue(req_multi["event"].wait(timeout=2))
        app.client.reply_permission.assert_called_with(
            "perm-multi", "allow", scope=["git status", "git commit"]
        )

        # 选项 3: reject
        with patch("cli.app.ui.print_permission_request"):
            req3 = app._enqueue_permission_request(
                {"permission_id": "perm-3", "name": "run_command"},
                owner_id="turn-3",
            )
        self.assertTrue(app._answer_pending_ask_user("3"))
        self.assertTrue(req3["event"].wait(timeout=2))
        self.assertEqual("reject", req3["answer"])
        app.client.reply_permission.assert_called_with("perm-3", "reject", scope=None)

        # 别名: n / no / deny / reject
        for alias in ("n", "no", "deny", "reject"):
            with patch("cli.app.ui.print_permission_request"):
                req = app._enqueue_permission_request(
                    {"permission_id": f"perm-{alias}", "name": "run_command"},
                    owner_id=f"turn-{alias}",
                )
            self.assertTrue(app._answer_pending_ask_user(alias))
            self.assertTrue(req["event"].wait(timeout=2))
            self.assertEqual("reject", req["answer"])
            app.client.reply_permission.assert_called_with(f"perm-{alias}", "reject", scope=None)

        # 无效输入
        with (
            patch("cli.app.ui.print_permission_request"),
            patch("cli.app.ui.console.print") as mock_print,
        ):
            req_invalid = app._enqueue_permission_request(
                {"permission_id": "perm-inv", "name": "run_command"},
                owner_id="turn-inv",
            )
            self.assertTrue(app._answer_pending_ask_user("invalid_choice"))
            self.assertFalse(req_invalid["event"].is_set())
            mock_print.assert_called_once()
            self.assertIn(
                "Please answer 1 to allow once, 2 to always allow, or 3 to reject.",
                mock_print.call_args[0][0],
            )
            app._cancel_ask_user_request(req_invalid)

    def test_print_permission_request_renders_three_options(self) -> None:
        """验证 ui.print_permission_request 渲染 3 个选项。"""
        import cli.ui as ui

        with patch.object(ui.console, "print") as mock_print:
            ui.print_permission_request("test_tool", {"arg": "val"})
            mock_print.assert_called_once()
            panel = mock_print.call_args[0][0]
            rendered_texts = [str(r) for r in panel.renderable.renderables]
            combined = "\n".join(rendered_texts)
            self.assertIn("1. Allow once", combined)
            self.assertIn("2. Always allow tool 'test_tool'", combined)
            self.assertIn("3. Reject", combined)

    def test_permission_sse_event_is_reserved_for_the_main_prompt(self) -> None:
        """permission_request 必须进入输入 FIFO，并保持原 SSE 继续等待。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-perm", "title": "Permission"}
        app._request_context.session_id = "sess-perm"
        app._request_context.request_id = "turn-perm"
        payload = {
            "permission_id": "perm-1",
            "session_id": "sess-perm",
            "tool_call_id": "call-1",
            "name": "execute_python",
            "arguments": {"code": "print(1)", "timeout_seconds": 5},
        }
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "permission_request", "data": payload},
            {
                "event": "message_end",
                "data": {
                    "message": {
                        "id": "m-1",
                        "session_id": "sess-perm",
                        "role": "assistant",
                        "content": "Done.",
                    }
                },
            },
        ])

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch.object(app, "_enqueue_permission_request") as enqueue:
            app._ask_agent_stream("run it")

        enqueue.assert_called_once_with(payload, owner_id="turn-perm")

    def test_expired_permission_reply_does_not_requeue_waiter(self) -> None:
        """已过期的 permission 不应永久拦截后续输入。"""
        from cli.app import AnnaCliApp
        from cli.ports import AgentApiError

        app = AnnaCliApp()
        app.current_session = {"id": "sess-perm", "title": "Permission"}
        app.client.reply_permission = MagicMock(
            side_effect=AgentApiError("expired", status_code=409)
        )
        with patch("cli.app.ui.print_permission_request"), patch("cli.app.ui.console.print"):
            request = app._enqueue_permission_request(
                {"permission_id": "perm-2", "session_id": "sess-perm"},
                owner_id="turn-2",
            )
            self.assertTrue(app._answer_pending_ask_user("1"))
            self.assertTrue(request["event"].wait(timeout=2))
        self.assertFalse(app._has_pending_ask_users())
        self.assertIsNone(request["answer"])

    def test_ask_user_owner_cleanup_wakes_abandoned_waiter(self) -> None:
        """worker 异常退出时必须移除并唤醒其 clarification waiter。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        request = app._enqueue_ask_user_request(
            {"question": "Continue?", "options": ["yes", "no"]},
            owner_id="request-1",
        )
        app._cancel_ask_user_requests_for_owner("request-1")

        self.assertFalse(app._has_pending_ask_users())
        self.assertTrue(request["event"].is_set())
        self.assertIsNone(request["answer"])

    def test_ask_user_handoff_is_reserved_at_tool_call(self) -> None:
        """工具调用到 worker 等待之间，下一条输入也应归给澄清问题。"""
        import threading
        import time

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-ask", "title": "Ask"}
        ask_events = [
            {
                "event": "tool_call",
                "data": {
                    "name": "ask_user",
                    "arguments": {
                        "question": "Which runtime?",
                        "options": ["asyncio", "trio"],
                    },
                },
            },
            {
                "event": "tool_result",
                "data": {
                    "name": "ask_user",
                    "content": '{"ok": true, "question": "Which runtime?", '
                    '"options": ["asyncio", "trio"]}',
                },
            },
        ]
        app.client.ask_stream = MagicMock(side_effect=[ask_events, []])
        worker_error: list[Exception] = []

        def run() -> None:
            try:
                app.ask_agent("start")
            except Exception as exc:  # pragma: no cover - failure is asserted below
                worker_error.append(exc)

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.make_status", return_value=MagicMock()), \
                patch("cli.app.ui.print_clarification_question") as print_question:
            worker = threading.Thread(target=run)
            worker.start()
            deadline = time.time() + 2.0
            while time.time() < deadline:
                with app._ask_user_lock:
                    if app._pending_ask_users:
                        request = app._pending_ask_users[0]
                        break
                time.sleep(0.005)
            else:
                self.fail("ask_user reservation was not visible to the prompt loop")

            request["answer"] = "asyncio"
            request["event"].set()
            worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual([], worker_error)
        print_question.assert_not_called()
        self.assertEqual("Which runtime?", request["payload"]["question"])
        self.assertEqual(["asyncio", "trio"], request["payload"]["options"])

    def test_stream_error_keeps_persisted_queued_prompt_state(self) -> None:
        """活动流错误不能伪造一个未观察到的工具结果边界。"""
        from cli.app import AnnaCliApp
        from cli.ports import AgentApiError

        app = AnnaCliApp()
        app.current_session = {"id": "sess-error", "title": "Error"}
        app._request_context.session_id = "sess-error"
        app._request_context.request_id = "active-error"
        app._register_stream("sess-error", "active-error")
        app._set_stream_state("active-error", "active")
        app._mark_queued_turn("sess-error", "queued-error", "follow-up")
        app.client.ask_stream = MagicMock(return_value=[
            {"event": "error", "data": {"message": "provider failed"}},
        ])

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()):
            with self.assertRaises(AgentApiError):
                app._ask_agent_stream("active question")

        self.assertEqual(1, app.queued_turn_count)

    def test_ask_agent_stream_no_duplicate_content_on_message_end(self) -> None:
        """测试 _ask_agent_stream 在触发 tool_call 后不会在 message_end 再次重复打印 content。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-test", "title": "Test"}

        events = [
            {"event": "text_delta", "data": {"content": "请问你的技术偏好是什么？"}},
            {
                "event": "tool_call",
                "data": {"name": "ask_user", "arguments": {"question": "Q", "options": ["A", "B"]}},
            },
            {"event": "tool_result", "data": {"name": "ask_user", "content": '{"ok": true}'}},
            {"event": "message_end", "data": {"message": {"content": "请问你的技术偏好是什么？"}}},
        ]
        app.client.ask_stream = MagicMock(return_value=events)

        with patch("cli.app.MarkdownStreamer") as MockStreamer:
            mock_streamer_inst = MagicMock()
            MockStreamer.return_value = mock_streamer_inst

            payload = app._ask_agent_stream("hello")
            self.assertIsNotNone(payload)
            self.assertEqual(["A", "B"], payload.get("options"))

            # finish 应该只在 tool_call 前调用一次，不会在 message_end 重复调用
            self.assertEqual(1, mock_streamer_inst.finish.call_count)
            mock_streamer_inst.finish.assert_called_once_with("请问你的技术偏好是什么？")


class SessionCheckoutTests(TestCase):
    """open_session 的驾驶权签出:主会话签、子会话不签、被占不切换。"""

    def _app_with_mock_client(self):
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.client = MagicMock()
        return app

    def test_open_session_refuses_switch_when_held_elsewhere(self) -> None:
        from cli.ports import AgentApiError

        app = self._app_with_mock_client()
        app.client.checkout.side_effect = AgentApiError("held", status_code=409)
        app.open_session({"id": "m1", "title": "M", "main_session_id": None})
        app.client.checkout.assert_called_once_with("m1")
        self.assertIsNone(app.current_session)
        self.assertIsNone(app._held_session_id)

    def test_main_session_checks_out_and_switching_releases_previous(self) -> None:
        app = self._app_with_mock_client()
        app.open_session({"id": "m1", "title": "A", "main_session_id": None})
        app.client.checkout.assert_called_once_with("m1")
        self.assertEqual("m1", app._held_session_id)

        app.open_session({"id": "m2", "title": "B", "main_session_id": None})
        app.client.release.assert_called_once_with("m1")
        self.assertEqual("m2", app._held_session_id)

    def test_subagent_view_does_not_checkout_and_keeps_parent(self) -> None:
        app = self._app_with_mock_client()
        app.open_session({"id": "m1", "title": "M", "main_session_id": None})
        app.client.reset_mock()
        app.open_session({"id": "s1", "title": "S", "main_session_id": "m1"})
        app.client.checkout.assert_not_called()
        app.client.release.assert_not_called()
        self.assertEqual("s1", app.current_session["id"])
        self.assertEqual("m1", app._held_session_id)

    def test_delete_targets_current_session_only(self) -> None:
        from cli.commands import handle_delete

        app = MagicMock()
        app.active_turn_count = 0
        app._has_pending_ask_users.return_value = False
        app.current_session = None
        handle_delete(app, "1")
        app.client.delete_session.assert_not_called()

    def test_skip_pending_ask_user_cancels_waiter_without_backend_call(self) -> None:
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question"), patch("cli.app.ui.console.print"):
            request = app._enqueue_ask_user_request(
                {
                    "question": "Which framework?",
                    "options": ["FastAPI", "Flask"],
                    "recommended": "FastAPI",
                }
            )
        self.assertTrue(app._has_pending_ask_users())
        self.assertIsNotNone(app.session_prompt.bottom_toolbar)

        with patch("cli.app.ui.console.print") as mock_print:
            skipped = app._skip_pending_ask_user()

        self.assertTrue(skipped)
        self.assertFalse(app._has_pending_ask_users())
        self.assertIsNone(app.session_prompt.bottom_toolbar)
        self.assertEqual("cancelled", request["state"])
        self.assertIsNone(request["answer"])
        self.assertTrue(request["event"].is_set())
        printed = [str(call[0][0]) for call in mock_print.call_args_list if call[0]]
        self.assertTrue(any("Clarification paused" in text for text in printed))

    def test_handle_skip_command_invokes_skip_pending(self) -> None:
        from cli.commands import handle_skip

        app = MagicMock()
        app._skip_pending_ask_user.return_value = True
        handle_skip(app, "")
        app._skip_pending_ask_user.assert_called_once_with()

    def test_escape_key_skips_when_buffer_empty(self) -> None:
        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question"), patch("cli.app.ui.console.print"):
            app._enqueue_ask_user_request(
                {"question": "Pick", "options": ["A", "B"]}
            )

        bindings = app.session_prompt.key_bindings.bindings
        escape_binding = next(binding for binding in bindings if binding.keys == (Keys.Escape,))

        event = MagicMock()
        event.current_buffer.text = ""
        with patch.object(app, "_skip_pending_ask_user") as mock_skip:
            escape_binding.handler(event)
            mock_skip.assert_called_once_with()

    def test_print_message_renders_ask_user_header_without_duplicate_options(self) -> None:
        """print_message 仅展示 Clarification requested，避免与底部选择器冲突。"""
        from cli.ui import print_message

        with patch("cli.ui.console.print") as mock_print:
            print_message({
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "ask_user",
                        "arguments": {
                            "question": "What should we do next?",
                            "options": ["Explore code", "Run tests"],
                            "recommended": "Explore code",
                        },
                    }
                ],
                "content": "",
            })
        printed_texts = "\n".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
        self.assertIn("Clarification requested", printed_texts)
        self.assertNotIn("1. Explore code", printed_texts)
        self.assertNotIn("Type your own answer", printed_texts)

    def test_open_session_does_not_restore_unanswered_ask_user(self) -> None:
        """加载历史会话时不重新恢复历史中的clarification（只在实时对话中展示）。"""
        import json

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        messages = [
            {"role": "human", "content": "hi?"},
            {
                "role": "assistant",
                "content": "Sure, here's a question:",
                "tool_calls": [
                    {
                        "name": "ask_user",
                        "arguments": {
                            "question": "Pick one",
                            "options": ["Opt A", "Opt B"],
                            "recommended": "Opt A",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_name": "ask_user",
                "content": json.dumps({
                    "ok": True,
                    "question": "Pick one",
                    "options": ["Opt A", "Opt B"],
                    "recommended": "Opt A",
                }),
            },
        ]
        app.client.list_messages = MagicMock(return_value=messages)
        app.client.checkout = MagicMock()
        with patch("cli.app.ui.print_history_list"), patch("cli.app.ui.console.print"):
            app.open_session({"id": "sess-unanswered", "title": "Test"})

        self.assertFalse(app._has_pending_ask_users())
        self.assertIsNone(app.session_prompt.bottom_toolbar)

    def test_ask_user_hides_input_and_cursor_on_predefined_options(self) -> None:
        """选择预置选项时隐藏输入行和光标，防止与选项列表抢光标。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question"):
            request = app._enqueue_ask_user_request(
                {"question": "Pick framework", "options": ["FastAPI", "Django"]}
            )

        self.assertTrue(app._should_hide_prompt_input())
        self.assertTrue(app.session_prompt.layout.current_window.always_hide_cursor())
        self.assertEqual("", app.get_prompt_text().value)
        self.assertEqual("", app.get_activity_text().value)

        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("1. FastAPI", toolbar)
        self.assertIn("2. Django", toolbar)
        self.assertIn("3. Type your own answer...", toolbar)
        app._cancel_ask_user_request(request)

    def test_ask_user_custom_option_types_directly_in_toolbar(self) -> None:
        """切换到'Type your own answer...'时直接在toolbar输入，上方输入框始终保持隐藏。"""
        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question"):
            request = app._enqueue_ask_user_request(
                {"question": "Pick", "options": ["Option A", "Option B"]}
            )

        bindings = app.session_prompt.key_bindings.bindings
        down = next(binding for binding in bindings if binding.keys == (Keys.Down,))
        up = next(binding for binding in bindings if binding.keys == (Keys.Up,))
        any_key = next(binding for binding in bindings if binding.keys == (Keys.Any,))
        backspace = next(
            binding
            for binding in bindings
            if binding.keys in ((Keys.ControlH,), (Keys.Backspace,))
        )
        event = MagicMock()
        event.current_buffer.text = ""

        # Down from 0 -> 1 (Option B)
        down.handler(event)
        self.assertTrue(app._should_hide_prompt_input())

        # Down from 1 -> 2 (Type your own answer...)
        down.handler(event)
        self.assertEqual(2, request["selected_index"])
        self.assertTrue(app._should_hide_prompt_input())
        self.assertEqual("", app.get_prompt_text().value)

        # Toolbar shows cursor before placeholder on option 3
        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("3. █ Type your own answer...", toolbar)

        # Type custom text directly into option 3
        event_char = MagicMock()
        event_char.data = "h"
        any_key.handler(event_char)
        event_char.data = "i"
        any_key.handler(event_char)
        self.assertEqual("hi", app._ask_user_custom_draft)
        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("3. hi█", toolbar)

        # Backspace removes a character
        backspace.handler(event)
        self.assertEqual("h", app._ask_user_custom_draft)
        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("3. h█", toolbar)

        # Move Up back to option B; draft is preserved
        up.handler(event)
        self.assertEqual(1, request["selected_index"])
        self.assertEqual("h", app._ask_user_custom_draft)
        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("\u203a 2. Option B", toolbar)
        self.assertIn("3. h", toolbar)

        # Down back to custom option restores focus and cursor
        down.handler(event)
        self.assertEqual(2, request["selected_index"])
        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("\u203a 3. h█", toolbar)
        app._cancel_ask_user_request(request)

    def test_ask_user_typing_character_jumps_to_custom_answer(self) -> None:
        """预置选项状态下直接输入字符，自动跳转到自定义输入并填入字符。"""
        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question"):
            request = app._enqueue_ask_user_request(
                {"question": "Pick", "options": ["A", "B"]}
            )

        bindings = app.session_prompt.key_bindings.bindings
        any_key = next(binding for binding in bindings if binding.keys == (Keys.Any,))

        # Typing '2' jumps to option 2
        event2 = MagicMock()
        event2.data = "2"
        event2.current_buffer.text = ""
        any_key.handler(event2)
        self.assertEqual(1, request["selected_index"])
        self.assertTrue(app._should_hide_prompt_input())

        # Typing '3' jumps to custom option
        event3 = MagicMock()
        event3.data = "3"
        event3.current_buffer.text = ""
        any_key.handler(event3)
        self.assertEqual(2, request["selected_index"])
        self.assertTrue(app._should_hide_prompt_input())

        # Reset to option 0
        request["selected_index"] = 0
        self.assertTrue(app._should_hide_prompt_input())

        # Typing a letter 'x' jumps to custom and inserts 'x'
        event_char = MagicMock()
        event_char.data = "x"
        event_char.current_buffer.text = ""
        any_key.handler(event_char)
        self.assertEqual(2, request["selected_index"])
        self.assertEqual("x", app._ask_user_custom_draft)
        self.assertTrue(app._should_hide_prompt_input())
        toolbar = "".join(text for _style, text in app.get_permission_toolbar())
        self.assertIn("3. x█", toolbar)
        app._cancel_ask_user_request(request)

    def test_ask_user_enter_on_custom_option_logic(self) -> None:
        """自定义选项为空时回车不提交；有内容时提交内容。"""
        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question"):
            request = app._enqueue_ask_user_request(
                {"question": "Pick", "options": ["A", "B"]}
            )
        request["selected_index"] = 2  # custom option

        bindings = app.session_prompt.key_bindings.bindings
        enter = next(binding for binding in bindings if binding.keys == (Keys.ControlM,))

        # Empty draft -> does not validate/submit
        empty_event = MagicMock()
        empty_event.current_buffer.text = ""
        enter.handler(empty_event)
        empty_event.current_buffer.validate_and_handle.assert_not_called()

        # Non-empty draft -> validates and submits
        app._ask_user_custom_draft = "My Own Answer"
        typed_event = MagicMock()
        typed_event.current_buffer.text = ""
        enter.handler(typed_event)
        self.assertEqual("My Own Answer", typed_event.current_buffer.text)
        typed_event.current_buffer.validate_and_handle.assert_called_once_with()

        self.assertTrue(app._answer_pending_ask_user("My Own Answer"))
        self.assertEqual("My Own Answer", request["answer"])
        self.assertEqual("", app._ask_user_custom_draft)

    def test_ask_user_numeric_selection_of_custom_option_prompts_user(self) -> None:
        """输入对应'Type your own answer...'的数字时提示用户输入，而不是作为最终答案发送。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question"):
            request = app._enqueue_ask_user_request(
                {"question": "Pick", "options": ["A", "B"]}
            )

        with patch("cli.app.ui.console.print") as mock_print:
            handled = app._answer_pending_ask_user("3")
            self.assertTrue(handled)

        self.assertEqual(2, request["selected_index"])
        self.assertEqual("pending", request["state"])
        self.assertIsNone(request["answer"])
        printed = [str(call[0][0]) for call in mock_print.call_args_list if call[0]]
        self.assertTrue(any("Please type your answer" in text for text in printed))
        app._cancel_ask_user_request(request)

    def test_esc_interrupts_active_streams(self) -> None:
        """运行中按 Esc 键能够中断当前活动的数据流。"""
        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.client.abort_stream = MagicMock()
        app._register_stream("sess-esc", "req-esc")

        worker = MagicMock()
        with app._turn_threads_lock:
            app._turn_threads.add(worker)

        try:
            bindings = app.session_prompt.key_bindings.bindings
            esc_binding = next(b for b in bindings if b.keys == (Keys.Escape,))
            event = MagicMock()
            event.current_buffer.text = ""
            with patch("cli.app.ui.console.print") as mock_print:
                esc_binding.handler(event)
            app.client.abort_stream.assert_called_once_with("req-esc")
            printed = [str(call[0][0]) for call in mock_print.call_args_list if call[0]]
            self.assertTrue(any("interrupted" in text for text in printed))
        finally:
            with app._turn_threads_lock:
                app._turn_threads.discard(worker)
            app._unregister_stream("sess-esc", "req-esc")

    def test_queued_turn_clean_transition_between_answers(self) -> None:
        """队列问题在流式响应中平滑过渡，不会产生粘连与重复打印。"""
        from unittest.mock import call

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-trans", "title": "Transition"}
        app._request_context.request_id = "req-trans"
        app._register_stream("sess-trans", "req-trans")
        app._set_stream_state("req-trans", "active")
        # 队列中添加问题
        app._mark_queued_turn("sess-trans", "req-queued", "no")

        events = [
            {"event": "text_delta", "data": {"content": "First answer."}},
            {"event": "text_delta", "data": {"content": "Second answer."}},
            {"event": "message_end", "data": {"message": {"content": "Second answer."}}},
        ]
        app.client.ask_stream = MagicMock(return_value=events)

        with patch("cli.app.MarkdownStreamer") as streamer_cls, \
                patch("cli.app.time.monotonic", side_effect=[1.0, 1.05, 1.3, 1.35, 1.4]):
            first_streamer = MagicMock()
            second_streamer = MagicMock()
            streamer_cls.side_effect = [first_streamer, second_streamer]
            app._ask_agent_stream("initial")

        # 第一个回答 finish
        self.assertEqual([call("First answer.")], first_streamer.finish.call_args_list)
        # 第二个回答 finish
        self.assertEqual([call("Second answer.")], second_streamer.finish.call_args_list)
        # 队列在过渡后被消费
        self.assertEqual(0, app.queued_turn_count)

    def test_summary_detection_and_compaction_hint(self) -> None:
        """Context compaction summaries are split from real answers and display hint."""
        import cli.ui as ui

        summary = (
            "GOALS\n"
            "- User asked to understand the repo.\n\n"
            "DECISIONS\n"
            "- Did not run test suite.\n\n"
            "FACTS\n"
            "- Repo is AgentDesk.\n"
        )
        answer = "Here is the real answer."
        self.assertTrue(ui.is_summary_text(summary))
        self.assertFalse(ui.is_summary_text(answer))
        self.assertFalse(ui.is_summary_text(""))

        combined = summary + "\n\n" + answer
        sum_part, rem_part = ui.split_summary_prefix(combined)
        self.assertEqual(summary.rstrip(), sum_part)
        self.assertEqual(answer, rem_part)

        with patch("cli.ui.console.print") as print_mock:
            ui.print_context_compacted()
            rendered = str(print_mock.call_args.args[0])
            self.assertIn("context compacted", rendered)

    def test_summary_suppressed_in_message_and_watch_queue(self) -> None:
        """Summary messages are suppressed in print_message and show hint in queue watcher."""
        import cli.ui as ui
        from cli.app import AnnaCliApp

        summary_msg = {
            "seq": 10,
            "role": "assistant",
            "content": "GOALS\n- Goal\nDECISIONS\n- Dec\nFACTS\n- Fact",
            "metadata": '{"kind": "summary"}',
        }

        with patch("cli.ui.console.print") as print_mock:
            ui.print_message(summary_msg)
            print_mock.assert_not_called()

        app = AnnaCliApp()
        app.current_session = {"id": "sess-summary-watch", "title": "Watch"}
        app._mark_queued_turn("sess-summary-watch", "req-1", "question")
        queued_human = {
            "seq": 9,
            "role": "human",
            "content": "question",
            "metadata": {"source": "user", "kind": "user"},
        }
        terminal_answer = {
            "seq": 11,
            "role": "assistant",
            "content": "Final answer.",
            "tool_calls": [],
            "metadata": {"kind": "assistant_answer"},
        }
        app.client.list_messages = MagicMock(side_effect=[
            [queued_human],
            [queued_human, summary_msg, terminal_answer],
        ])

        with patch("cli.app._QUEUE_WATCH_INTERVAL_SECONDS", 0.0), \
                patch("cli.app.ui.print_context_compacted") as print_compact, \
                patch("cli.app.ui.print_message") as print_msg, \
                patch("cli.app.ui.print_queue_watch_stopped"):
            app._watch_external_queue("sess-summary-watch", "req-1")

        print_compact.assert_called_once()
        print_msg.assert_called_once_with(terminal_answer, show_subagent_details=False)

    def test_ask_stream_renders_answer_deltas_directly(self) -> None:
        """ask_stream renders text deltas directly without frontend summary parsing."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-stream-answer", "title": "Stream"}

        delta1 = "Hello "
        delta2 = "world!"
        answer = "Hello world!"

        events = [
            {"event": "text_delta", "data": {"content": delta1}},
            {"event": "text_delta", "data": {"content": delta2}},
            {"event": "message_end", "data": {"message": {"content": answer, "role": "assistant"}}},
        ]
        app.client.ask_stream = MagicMock(return_value=events)

        with patch("cli.app.MarkdownStreamer") as streamer_cls:
            streamer = MagicMock()
            streamer_cls.return_value = streamer
            app._ask_agent_stream("test")

        streamer.update.assert_any_call("Hello ")
        streamer.update.assert_any_call("Hello world!")
        streamer.finish.assert_called_once_with(answer)

    def test_ask_user_renders_clarification_question_when_answered(self) -> None:
        """ask_user does not render static box while choosing, but renders once answered."""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        with patch("cli.app.ui.print_clarification_question") as mock_render:
            request = app._enqueue_ask_user_request(
                {"question": "Which database?", "options": ["PostgreSQL", "SQLite"]}
            )
            mock_render.assert_not_called()

            self.assertTrue(app._answer_pending_ask_user("2"))
            mock_render.assert_called_once_with(
                "Which database?",
                options=["PostgreSQL", "SQLite"],
                recommended=None,
                answer="SQLite",
            )
            self.assertEqual("SQLite", request["answer"])
            self.assertTrue(request.get("rendered"))

        # Test custom answer rendering
        with patch("cli.app.ui.print_clarification_question") as mock_render2:
            req_custom = app._enqueue_ask_user_request(
                {"question": "Framework?", "options": ["React", "Vue"]}
            )
            self.assertTrue(app._answer_pending_ask_user("Svelte"))
            mock_render2.assert_called_once_with(
                "Framework?",
                options=["React", "Vue"],
                recommended=None,
                answer="Svelte",
            )
            self.assertEqual("Svelte", req_custom["answer"])

    def test_empty_enter_ignored_in_normal_prompt_to_prevent_flicker(self) -> None:
        """Pressing Enter on an empty line in normal prompt does not submit or flicker."""
        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        # When no picker is active, the non-picker enter binding is active
        active_enter = [
            b for b in app.session_prompt.key_bindings.bindings
            if b.keys == (Keys.ControlM,) and b.filter()
        ]
        self.assertTrue(bool(active_enter))
        normal_enter = active_enter[-1]

        event = MagicMock()
        event.current_buffer.text = ""
        normal_enter.handler(event)
        event.current_buffer.validate_and_handle.assert_not_called()

        event.current_buffer.text = "   "
        normal_enter.handler(event)
        event.current_buffer.validate_and_handle.assert_not_called()

        event.current_buffer.text = "hello"
        normal_enter.handler(event)
        event.current_buffer.validate_and_handle.assert_called_once()

    def test_print_clarification_question_with_preset_and_custom_answer(self) -> None:
        """print_clarification_question marks selected preset option or shows custom answer."""
        from cli.ui import print_clarification_question

        # 1. Preset option selected
        with patch("cli.ui.console.print") as mock_print:
            print_clarification_question(
                "Pick DB",
                options=["PostgreSQL", "SQLite"],
                answer="SQLite",
            )
            panel = mock_print.call_args.args[0]
            rendered = "\n".join(str(item) for item in panel.renderable.renderables)
            self.assertIn("› 2. SQLite", rendered)
            self.assertIn("  1. PostgreSQL", rendered)
            self.assertNotIn("Type your own answer...", rendered)

        # 2. Custom answer typed
        with patch("cli.ui.console.print") as mock_print:
            print_clarification_question(
                "Pick DB",
                options=["PostgreSQL", "SQLite"],
                answer="MyCustomDB",
            )
            panel = mock_print.call_args.args[0]
            rendered = "\n".join(str(item) for item in panel.renderable.renderables)
            self.assertIn("› 3. MyCustomDB", rendered)
            self.assertIn("  1. PostgreSQL", rendered)
            self.assertIn("  2. SQLite", rendered)
            self.assertNotIn("Type your own answer...", rendered)


class ModelCommandAndPickerTests(TestCase):
    """Smoke and integration tests for model listing, switching, and autocompletion."""

    def test_ui_print_models_table(self) -> None:
        from cli.ui import print_models_table

        with patch("cli.ui.console.print") as mock_print:
            print_models_table([])
            mock_print.assert_called_once()
            self.assertIn("No models configured", str(mock_print.call_args[0][0]))

        models = [
            {"id": "glm-5.3-flash", "name": "GLM 5.3 Flash"},
            {"id": "gpt-4o", "name": "GPT-4o"},
        ]
        with patch("cli.ui.console.print") as mock_print:
            print_models_table(models, current_model_id="gpt-4o")
            table = mock_print.call_args_list[0][0][0]
            self.assertEqual(len(table.rows), 2)

    def test_model_picker_filtering_and_formatting(self) -> None:
        from cli.picker import ModelPicker

        models = [
            {"id": "glm-5.3-flash", "name": "GLM 5.3 Flash"},
            {"id": "deepseek-chat", "name": "DeepSeek Chat"},
            {"id": "claude-3-5-sonnet", "name": "Claude 3.5 Sonnet"},
        ]
        picker = ModelPicker(models, current_model_id="deepseek-chat")
        self.assertEqual(picker.selected_index, 1)
        self.assertEqual(len(picker._get_filtered()), 3)

        picker.search_query = "claude"
        filtered = picker._get_filtered()
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], "claude-3-5-sonnet")

        picker.search_query = "1"
        filtered = picker._get_filtered()
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], "glm-5.3-flash")

        formatted = picker._get_formatted_text()
        self.assertTrue(len(formatted) > 0)

    def test_model_picker_run_erase_when_done(self) -> None:
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from cli.picker import ModelPicker

        models = [
            {"id": "m1", "name": "Model 1"},
            {"id": "m2", "name": "Model 2"},
        ]
        picker = ModelPicker(models)

        with create_pipe_input() as inp:
            inp.send_text("\r")
            res = picker.run(input=inp, output=DummyOutput())
            self.assertEqual(res, models[0])

    def test_resolve_model(self) -> None:
        from cli.commands import _resolve_model

        models = [
            {"id": "glm-5.3-flash", "name": "GLM 5.3 Flash"},
            {"id": "gpt-4o", "name": "OpenAI GPT-4o"},
        ]
        # By 1-based index
        self.assertEqual(_resolve_model(models, "1"), models[0])
        self.assertEqual(_resolve_model(models, "2"), models[1])
        self.assertIsNone(_resolve_model(models, "3"))

        # By exact ID or prefix
        self.assertEqual(_resolve_model(models, "glm-5.3-flash"), models[0])
        self.assertEqual(_resolve_model(models, "gpt"), models[1])

        # By name substring
        self.assertEqual(_resolve_model(models, "openai"), models[1])
        self.assertIsNone(_resolve_model(models, "nonexistent"))

    def test_handle_models_command(self) -> None:
        from cli.commands import handle_models

        app = MagicMock()
        app.client.list_models.return_value = [
            {"id": "glm-5.3-flash", "name": "GLM 5.3 Flash"}
        ]
        app.model_id = "glm-5.3-flash"

        with patch("cli.ui.print_models_table") as mock_print_table:
            handle_models(app, "")
            mock_print_table.assert_called_once_with(
                app.client.list_models.return_value,
                current_model_id="glm-5.3-flash",
            )

    def test_handle_model_command_rejection_when_turn_running(self) -> None:
        from cli.commands import handle_model

        app = MagicMock()
        app.active_turn_count = 1

        with patch("cli.ui.console.print") as mock_print:
            handle_model(app, "gpt-4o")
            app.client.list_models.assert_not_called()
            mock_print.assert_called_once()
            self.assertIn(
                "Cannot switch model while a turn is running",
                str(mock_print.call_args[0][0]),
            )

    def test_handle_model_command_direct_and_index(self) -> None:
        from cli.commands import handle_model

        app = MagicMock()
        app.active_turn_count = 0
        app._has_live_waiters.return_value = False
        app.client.list_models.return_value = [
            {"id": "glm-5.3-flash", "name": "GLM 5.3 Flash"},
            {"id": "gpt-4o", "name": "GPT-4o"},
        ]

        # Switch by name/prefix
        handle_model(app, "gpt-4o")
        app.select_model.assert_called_with("gpt-4o")

        # Switch by index
        handle_model(app, "1")
        app.select_model.assert_called_with("glm-5.3-flash")

        # Unknown model
        with patch("cli.ui.console.print") as mock_print:
            handle_model(app, "unknown-xyz")
            mock_print.assert_called_once()
            self.assertIn(
                'Model not found: "unknown-xyz"',
                str(mock_print.call_args[0][0]),
            )

    def test_handle_model_command_opens_picker(self) -> None:
        from cli.commands import handle_model

        app = MagicMock()
        app.active_turn_count = 0
        app._has_live_waiters.return_value = False
        models = [{"id": "m1", "name": "Model 1"}]
        app.client.list_models.return_value = models

        with patch("cli.commands.ModelPicker") as mock_picker_cls:
            mock_picker = mock_picker_cls.return_value
            mock_picker.run.return_value = models[0]
            handle_model(app, "")
            mock_picker_cls.assert_called_once()
            app.select_model.assert_called_with("m1")

    def test_slash_completer_model_suggestions(self) -> None:
        from cli.completer import SLASH_COMMANDS, SlashCompleter

        commands = [cmd for cmd, _ in SLASH_COMMANDS]
        self.assertIn("/model", commands)
        self.assertIn("/models", commands)

        models = [
            {"id": "glm-5.3-flash", "name": "GLM 5.3 Flash"},
            {"id": "gpt-4o", "name": "GPT-4o"},
        ]
        completer = SlashCompleter(get_models_fn=lambda: models)

        # /model prefix with space triggers completions
        doc = Document(text="/model ")
        completions = list(completer.get_completions(doc, None))
        values = [c.text for c in completions]
        self.assertIn("glm-5.3-flash", values)
        self.assertIn("gpt-4o", values)

        # /model gp filters to gpt-4o
        doc2 = Document(text="/model gp")
        completions2 = list(completer.get_completions(doc2, None))
        values2 = [c.text for c in completions2]
        self.assertEqual(values2, ["gpt-4o"])

    def test_app_select_model_and_safe_list_models(self) -> None:
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        mock_client = MagicMock()
        mock_client.list_models.return_value = [{"id": "m1", "name": "Model 1"}]
        mock_client.select_model.return_value = {"id": "m1", "name": "Model 1"}
        mock_client.get_current_model.return_value = {"id": "m1", "name": "Model 1"}
        app.client = mock_client

        # _safe_list_models
        self.assertEqual(app._safe_list_models(), [{"id": "m1", "name": "Model 1"}])

        # select_model
        with patch("cli.ui.console.print") as mock_print:
            app.select_model("m1")
            mock_client.select_model.assert_called_with("m1")
            self.assertEqual(app.model_id, "m1")
            self.assertEqual(app.model_name, "Model 1")
            mock_print.assert_called_once()
            self.assertIn("Switched model to:", str(mock_print.call_args[0][0]))

        # auto_init_session syncs model
        app.model_id = None
        app.model_name = None
        app.auto_init_session()
        self.assertEqual(app.model_id, "m1")
        self.assertEqual(app.model_name, "Model 1")

    def test_local_client_model_methods(self) -> None:
        import asyncio
        from typing import Any

        from cli.local_client import LocalApiClient

        client = LocalApiClient()
        fake_catalog = MagicMock()
        info1 = MagicMock()
        info1.model_id = "glm-flash"
        info1.name = "GLM Flash"
        fake_catalog.list_models.return_value = [info1]
        fake_catalog.current_id = "glm-flash"
        profile = MagicMock()
        profile.name = "GLM Flash"
        fake_catalog.current_profile.return_value = profile
        del fake_catalog.current_entry
        fake_catalog.select_model.return_value = "glm-flash"

        def _fake_call(_iface: Any, fn: Any, **_kw: Any) -> Any:
            return asyncio.run(fn(fake_catalog))

        with patch.object(client, "_call", side_effect=_fake_call):
            models = client.list_models()
            self.assertEqual(models, [{"id": "glm-flash", "name": "GLM Flash"}])

            cur = client.get_current_model()
            self.assertEqual(cur, {"id": "glm-flash", "name": "GLM Flash"})

            chosen = client.select_model("glm-flash")
            self.assertEqual(chosen, {"id": "glm-flash", "name": "GLM Flash"})

    def test_silence_background_loggers(self) -> None:
        import io
        import logging
        import os
        import sys
        import warnings

        from cli.ui import silence_background_loggers

        silence_background_loggers()

        self.assertEqual(str(os.environ.get("MEM0_TELEMETRY")).lower(), "false")
        self.assertEqual(os.environ.get("LANGCHAIN_OPENAI_TCP_KEEPALIVE"), "0")

        for name in (
            "mem0",
            "agent.infrastructure.memory",
            "posthog",
            "langchain_openai",
            "mcp.client.streamable_http",
        ):
            logger = logging.getLogger(name)
            self.assertFalse(logger.propagate)
            self.assertTrue(any(isinstance(h, logging.NullHandler) for h in logger.handlers))

        captured = io.StringIO()
        with patch.object(sys, "stderr", captured):
            child = logging.getLogger("mem0.memory.main")
            child.error("LLM extraction failed (async): Error code: 401")
            agent_mem = logging.getLogger("agent.infrastructure.memory.logging")
            agent_mem.error("[Mem0] [MEMORY_ADD_FAILED] status=ERROR", exc_info=False)
            posthog_log = logging.getLogger("posthog")
            posthog_log.warning("[PostHog] Multiple active PostHog clients detected")
            langchain_log = logging.getLogger("langchain_openai")
            langchain_log.warning("langchain-openai injected a custom httpx transport")

        self.assertEqual(captured.getvalue(), "")

        # Normal warnings are rendered in yellow on console
        with patch("cli.ui.console.print") as mock_print:
            warnings.warn("Legitimate user warning", UserWarning)
            mock_print.assert_called_once()
            self.assertIn("yellow", str(mock_print.call_args[0][0]))
            self.assertIn("Legitimate user warning", str(mock_print.call_args[0][0]))

        # agent.infrastructure.mcp warnings are silenced from the console (handled via UI hint)
        from cli.ui import _CliLogHandler

        handler = _CliLogHandler()
        record = logging.LogRecord(
            name="agent.infrastructure.mcp.registry",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="MCP server 'bad' failed to connect",
            args=(),
            exc_info=None,
        )
        with patch("cli.ui.console.print") as mock_print:
            handler.emit(record)
            mock_print.assert_not_called()

    def test_codex_style_banner_prompt_and_toolbar(self) -> None:
        """Codex 风格对齐测试：欢迎横幅包含模型及工作区路径，Prompt区展示Working，
        底部工具栏常驻模型与目录。
        """
        import io

        from rich.console import Console

        from cli.app import AnnaCliApp
        from cli.ui import print_banner

        # 1. 测试欢迎横幅
        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=True, color_system="truecolor")
        with patch("cli.ui.console", test_console):
            print_banner(
                base_url="http://127.0.0.1:8000/api",
                session_title="Codex Test",
                session_id="s12345678",
                model_name="Qwen 3.8 Flash",
            )
        banner_output = buf.getvalue()
        self.assertIn("AgentDesk", banner_output)
        self.assertIn("model:", banner_output)
        self.assertIn("Qwen 3.8 Flash", banner_output)
        self.assertIn("/model", banner_output)
        self.assertIn("directory:", banner_output)
        self.assertIn("Tip:", banner_output)

        # 2. 测试状态工具栏（附着在输入框下方，不在屏幕最底部）
        app = AnnaCliApp()
        app.model_name = "Qwen 3.8 Flash"
        toolbar_fragments = app.get_status_toolbar()
        toolbar_text = "".join(text for _, text in toolbar_fragments)
        self.assertIn("Qwen 3.8 Flash", toolbar_text)
        self.assertIn(" · ", toolbar_text)
        self.assertIsNone(app.session_prompt.bottom_toolbar)
        alt = app.session_prompt.layout.container.children[0].alternative_content
        has_tb = any(
            "bottom-toolbar" in getattr(getattr(ch, "content", None), "style", "")
            for ch in alt.content.children
        )
        self.assertTrue(has_tb)

        # 3. 测试 Working 与排队消息
        app.current_session = {"id": "sess-codex", "title": "Codex UI"}
        worker = MagicMock()
        with app._turn_threads_lock:
            app._turn_threads.add(worker)
        try:
            # 单纯运行时展示带有转圈动画帧的 Working (0s)
            prompt_html = app.get_prompt_text().value
            self.assertIn("Working", prompt_html)
            from cli.app import _THINKING_FRAMES
            self.assertTrue(any(frame in prompt_html for frame in _THINKING_FRAMES))
            self.assertIn("› ", prompt_html)

            # 运行中有消息排队时，展示排队消息列表
            app._mark_queued_turn("sess-codex", "req-q1", "first follow-up")
            app._mark_queued_turn("sess-codex", "req-q2", "second follow-up")
            prompt_queued_html = app.get_prompt_text().value
            self.assertIn("Working", prompt_queued_html)
            self.assertIn("Messages to be submitted after next tool call", prompt_queued_html)
            self.assertIn("↳", prompt_queued_html)
            self.assertIn("first follow-up", prompt_queued_html)
            self.assertIn("second follow-up", prompt_queued_html)

            # 消费第一个排队消息时，该消息移出排队区，剩下第二个
            retired = app._retire_queued_turn("sess-codex", request_id="req-q1")
            self.assertEqual(1, len(retired))
            self.assertEqual("first follow-up", retired[0]["question"])
            prompt_step_html = app.get_prompt_text().value
            self.assertNotIn("first follow-up", prompt_step_html)
            self.assertIn("second follow-up", prompt_step_html)
        finally:
            with app._turn_threads_lock:
                app._turn_threads.discard(worker)
            app._discard_queued_turns("sess-codex")

        # 结束运行时清空提示，恢复干净的 ›
        self.assertEqual("<prompt>› </prompt>", app.get_prompt_text().value)

    def test_status_toolbar_紧贴输入框_dynamic_buffer_height(self) -> None:
        """模型与工作目录紧贴输入框下方，无 completions 时不预留空白行。"""
        from prompt_toolkit.buffer import CompletionState
        from prompt_toolkit.completion import Completion
        from prompt_toolkit.document import Document

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.model_name = "Qwen 3.8 Flash"

        # 1. 默认无补全状态：高度为 Dimension(preferred=0, min=0)，dont_extend_height 为 True
        h = app.session_prompt.layout.current_window.height()
        self.assertEqual(0, h.min)
        self.assertTrue(app.session_prompt.layout.current_window.dont_extend_height())

        # 2. 状态栏紧贴：get_status_toolbar 返回模型名和工作目录
        toolbar = app.get_status_toolbar()
        self.assertTrue(any("Qwen 3.8 Flash" in text for _, text in toolbar))
        self.assertFalse(any(text == "" for _, text in toolbar))

        # 3. 补全激活时：根据候选词数量动态设定高度，不超过预留行数
        doc = Document("/")
        app.session_prompt.default_buffer.complete_state = CompletionState(
            original_document=doc,
            completions=[Completion("/help"), Completion("/model")],
        )
        h_comp = app.session_prompt.layout.current_window.height()
        self.assertEqual(3, h_comp.min)

    def test_input_box_card_style_and_dynamic_placeholder(self) -> None:
        from cli.app import CLI_STYLE, AnnaCliApp
        from cli.theme import PROMPT_SYMBOL

        app = AnnaCliApp()

        # 1. 样式表包含提示符与占位符样式（无斜体、无多余全屏底色块）
        style_rules = dict(CLI_STYLE.style_rules)
        self.assertIn("prompt", style_rules)
        self.assertIn("placeholder", style_rules)
        self.assertNotIn("italic", style_rules["placeholder"].lower())
        self.assertNotIn("bg:", style_rules["placeholder"].lower())
        self.assertIn("toolbar-model", style_rules)
        self.assertIn("toolbar-cwd", style_rules)

        # 2. 空闲状态下的 placeholder 提示
        idle_ph = app.get_placeholder_text().value
        self.assertIn("Ask AgentDesk anything", idle_ph)

        # 3. 运行状态下的 placeholder 提示为排队引导
        worker = MagicMock()
        with app._turn_threads_lock:
            app._turn_threads.add(worker)
        try:
            active_ph = app.get_placeholder_text().value
            self.assertIn("queue", active_ph.lower())
        finally:
            with app._turn_threads_lock:
                app._turn_threads.discard(worker)

        # 4. 验证 Prompt Chevron 与状态栏格式（首项为空行留白，避免紧贴拥挤）
        self.assertEqual("› ", PROMPT_SYMBOL)
        self.assertEqual("<prompt>› </prompt>", app.get_prompt_text().value)
        status_toolbar = app.get_status_toolbar()
        self.assertEqual(6, len(status_toolbar))
        self.assertEqual(("", "\n"), status_toolbar[0])
        self.assertEqual("class:toolbar-model", status_toolbar[1][0])
        self.assertEqual(" · ", status_toolbar[2][1])
        self.assertEqual("class:toolbar-cwd", status_toolbar[3][0])
        self.assertEqual(" · ", status_toolbar[4][1])
        self.assertEqual("class:toolbar-mode-default", status_toolbar[5][0])
        self.assertEqual("[default]", status_toolbar[5][1])
        app.client.close()

    def test_slash_typing_hides_status_toolbar_to_prevent_jitter(self) -> None:
        """测试输入 / 或激活补全菜单时自动隐藏 status toolbar，避免上下晃动。"""
        from unittest.mock import PropertyMock, patch

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        try:
            # 1. 空闲状态：展示状态栏
            self.assertTrue(bool(app.get_status_toolbar()))

            # 2. 输入 / 时：状态栏自动隐藏，留出空间给补全且不抖动
            buffer_type = type(app.session_prompt.default_buffer)
            with patch.object(
                buffer_type, "text", new_callable=PropertyMock, return_value="/"
            ):
                self.assertEqual([], app.get_status_toolbar())

            # 3. 输入命令前缀 /model 时：保持隐藏
            with patch.object(
                buffer_type, "text", new_callable=PropertyMock, return_value="/model"
            ):
                self.assertEqual([], app.get_status_toolbar())

            # 4. 补全菜单处于激活状态时：保持隐藏
            mock_complete = MagicMock()
            mock_complete.completions = [MagicMock()]
            app.session_prompt.default_buffer.complete_state = mock_complete
            try:
                self.assertEqual([], app.get_status_toolbar())
            finally:
                app.session_prompt.default_buffer.complete_state = None

            # 5. 普通非命令输入时：状态栏依然显示
            with patch.object(
                buffer_type, "text", new_callable=PropertyMock, return_value="hello"
            ):
                self.assertTrue(bool(app.get_status_toolbar()))
        finally:
            app.client.close()

    def test_is_slash_command_distinguishes_paths_from_commands(self) -> None:
        """测试斜杠指令与文件路径/普通文本的精准识别。"""
        from cli.app import is_slash_command

        # 1. 已注册命令
        is_cmd, name, arg = is_slash_command("/help")
        self.assertTrue(is_cmd)
        self.assertEqual("help", name)

        is_cmd, name, arg = is_slash_command("/model gpt-4o")
        self.assertTrue(is_cmd)
        self.assertEqual("model", name)
        self.assertEqual("gpt-4o", arg)

        # 2. 文件与目录绝对路径（不应被当作命令拦截）
        is_cmd, _, _ = is_slash_command("/home/anna/code/projects/Agent/image.png 怎么看")
        self.assertFalse(is_cmd)

        is_cmd, _, _ = is_slash_command("/home/anna/code/projects/Agent/image.png")
        self.assertFalse(is_cmd)

        is_cmd, _, _ = is_slash_command("/tmp")
        self.assertFalse(is_cmd)

        is_cmd, _, _ = is_slash_command("/etc/hosts")
        self.assertFalse(is_cmd)

        is_cmd, _, _ = is_slash_command("/script.py")
        self.assertFalse(is_cmd)

        # 3. 中文等非命令字符
        is_cmd, _, _ = is_slash_command("/这是一个问题")
        self.assertFalse(is_cmd)

        # 4. 未知命令（格式为命令但不存在）
        is_cmd, name, _ = is_slash_command("/notarealcommand")
        self.assertTrue(is_cmd)
        self.assertEqual("notarealcommand", name)

        # 5. 单独的 /
        is_cmd, name, _ = is_slash_command("/")
        self.assertTrue(is_cmd)
        self.assertEqual("", name)

    def test_permission_mode_display_and_cycling(self) -> None:
        """测试权限 mode 状态栏展示与 Shift+Tab 循环切换逻辑。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        try:
            # 1. 默认状态为 default
            self.assertEqual("default", app.permission_mode)
            toolbar = app.get_status_toolbar()
            self.assertEqual("class:toolbar-mode-default", toolbar[5][0])
            self.assertEqual("[default]", toolbar[5][1])

            # 2. 模拟 Shift+Tab 循环切换：default -> dont_ask -> bypass -> default
            next_mode = app.cycle_permission_mode()
            self.assertEqual("dont_ask", next_mode)
            self.assertEqual("dont_ask", app.permission_mode)
            toolbar = app.get_status_toolbar()
            self.assertEqual("class:toolbar-mode-dont_ask", toolbar[5][0])
            self.assertEqual("[dont_ask]", toolbar[5][1])

            next_mode = app.cycle_permission_mode()
            self.assertEqual("bypass", next_mode)
            self.assertEqual("bypass", app.permission_mode)
            toolbar = app.get_status_toolbar()
            self.assertEqual("class:toolbar-mode-bypass", toolbar[5][0])
            self.assertEqual("[bypass]", toolbar[5][1])

            next_mode = app.cycle_permission_mode()
            self.assertEqual("default", next_mode)
            self.assertEqual("default", app.permission_mode)
            toolbar = app.get_status_toolbar()
            self.assertEqual("class:toolbar-mode-default", toolbar[5][0])
            self.assertEqual("[default]", toolbar[5][1])
        finally:
            app.client.close()

    def test_handle_mode_command(self) -> None:
        """测试 /mode 命令查看与切换 mode。"""
        from cli.app import AnnaCliApp
        from cli.commands import handle_mode

        app = AnnaCliApp()
        try:
            # 查看当前 mode
            with patch("cli.ui.console.print") as print_mock:
                handle_mode(app, "")
                rendered = str(print_mock.call_args.args[0])
                self.assertIn("Current permission mode", rendered)
                self.assertIn("default", rendered)

            # 切换为 dont_ask
            with patch("cli.ui.console.print") as print_mock:
                handle_mode(app, "dont_ask")
                self.assertEqual("dont_ask", app.permission_mode)
                rendered = str(print_mock.call_args.args[0])
                self.assertIn("Permission mode set to", rendered)
                self.assertIn("dont_ask", rendered)

            # 非法 mode
            with patch("cli.ui.console.print") as print_mock:
                handle_mode(app, "invalid_mode")
                rendered = str(print_mock.call_args.args[0])
                self.assertIn("Invalid mode", rendered)

            # 切换为 bypass
            handle_mode(app, "bypass")
            self.assertEqual("bypass", app.permission_mode)
        finally:
            app.client.close()

    def test_slash_completer_mode_suggestions(self) -> None:
        """测试 /mode 命令参数补全。"""
        from prompt_toolkit.document import Document

        from cli.completer import SLASH_COMMANDS, SlashCompleter

        commands = [cmd for cmd, _ in SLASH_COMMANDS]
        self.assertIn("/mode", commands)

        completer = SlashCompleter()
        doc = Document(text="/mode ")
        completions = [c.text for c in completer.get_completions(doc, None)]
        self.assertIn("default", completions)
        self.assertIn("dont_ask", completions)
        self.assertIn("bypass", completions)

        doc2 = Document(text="/mode do")
        completions2 = [c.text for c in completer.get_completions(doc2, None)]
        self.assertEqual(["dont_ask"], completions2)

    def test_ask_user_request_stream_event_enqueues_and_replies(self) -> None:
        """ask_user_request 在流中挂起，由 reply_ask_user 唤醒后继续同一流至 message_end。"""
        import time

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-interruption", "title": "Interruption"}
        app.client.reply_ask_user = MagicMock(return_value={"status": "accepted"})

        def stream_gen():
            yield {"event": "text_delta", "data": {"content": "Thinking..."}}
            yield {
                "event": "ask_user_request",
                "data": {
                    "interruption_id": "intr-123",
                    "session_id": "sess-interruption",
                    "question": "Which database?",
                    "options": ["PostgreSQL", "SQLite"],
                    "recommended": "PostgreSQL",
                },
            }
            # 中断期间挂起等待用户答复；答复后才产出工具结果与最终消息
            with patch("cli.app.ui.print_clarification_question"):
                handled = app._answer_pending_ask_user("1")
            assert handled
            yield {
                "event": "tool_result",
                "data": {
                    "name": "ask_user",
                    "content": '{"ok": true, "answer": "PostgreSQL"}',
                },
            }
            yield {"event": "text_delta", "data": {"content": "Using PostgreSQL."}}
            yield {
                "event": "message_end",
                "data": {
                    "message": {
                        "id": "m1",
                        "session_id": "sess-interruption",
                        "role": "assistant",
                        "content": "Using PostgreSQL.",
                    }
                },
            }

        app.client.ask_stream = MagicMock(return_value=stream_gen())

        with patch("cli.app.MarkdownStreamer", return_value=MagicMock()), \
                patch("cli.app.ui.make_status", return_value=MagicMock()):
            result = app._ask_agent_stream("Start")

        self.assertIsNone(result)

        # 等待后台分发线程完成
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if app.client.reply_ask_user.called:
                break
            time.sleep(0.01)

        app.client.reply_ask_user.assert_called_once_with("intr-123", "PostgreSQL")

    def test_skip_pending_ask_user_with_interruption_id_dispatches_declined_answer(self) -> None:
        """带 interruption_id 的 ask_user 跳过时，向后端分发拒绝选择的答复。"""
        import time

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-skip", "title": "Skip"}
        app.client.reply_ask_user = MagicMock(return_value={"status": "accepted"})

        app._enqueue_ask_user_request(
            {
                "interruption_id": "intr-skip-999",
                "session_id": "sess-skip",
                "question": "Choose framework?",
                "options": ["FastAPI", "Django"],
            }
        )
        self.assertTrue(app._has_pending_ask_users())

        with patch("cli.app.ui.print_clarification_question"):
            skipped = app._skip_pending_ask_user()

        self.assertTrue(skipped)
        self.assertFalse(app._has_pending_ask_users())

        # 等待后台分发线程完成
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if app.client.reply_ask_user.called:
                break
            time.sleep(0.01)

        app.client.reply_ask_user.assert_called_once_with(
            "intr-skip-999",
            "User declined to make a choice. Please state your assumptions and proceed.",
        )

    def test_escape_cancels_running_turn_when_no_pending_waiters(self) -> None:
        """无 pending ask_user/permission 时，按下 ESC 取消在途轮次并中止流。"""
        import threading

        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-esc-1", "title": "EscCancel"}
        app.client.cancel = MagicMock()
        app.client.abort_stream = MagicMock()

        dummy_thread = threading.current_thread()
        app._turn_threads.add(dummy_thread)
        app._stream_states["req-esc-1"] = {"session_id": "sess-esc-1", "state": "active"}

        try:
            escape = next(
                b for b in app.session_prompt.key_bindings.bindings if b.keys == (Keys.Escape,)
            )
            event = MagicMock()
            event.current_buffer.text = ""
            with patch("cli.app.ui.console.print"):
                escape.handler(event)

            app.client.cancel.assert_called_once_with("sess-esc-1")
            app.client.abort_stream.assert_called_once_with("req-esc-1")
        finally:
            app._turn_threads.discard(dummy_thread)

    def test_escape_skips_clarification_instead_of_cancelling_when_ask_user_pending(self) -> None:
        """有 pending ask_user 时，按下 ESC 只跳过/放弃澄清，绝不取消整轮。"""
        import threading
        import time

        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-esc-2", "title": "EscAskUser"}
        app.client.cancel = MagicMock()
        app.client.reply_ask_user = MagicMock(return_value={"status": "accepted"})

        dummy_thread = threading.current_thread()
        app._turn_threads.add(dummy_thread)
        app._enqueue_ask_user_request(
            {
                "interruption_id": "intr-esc-2",
                "session_id": "sess-esc-2",
                "question": "Which db?",
                "options": ["PostgreSQL", "SQLite"],
            }
        )

        try:
            escape = next(
                b for b in app.session_prompt.key_bindings.bindings if b.keys == (Keys.Escape,)
            )
            event = MagicMock()
            event.current_buffer.text = ""
            with patch("cli.app.ui.print_clarification_question"), \
                    patch("cli.app.ui.console.print"):
                escape.handler(event)

            app.client.cancel.assert_not_called()

            # 等待后台分发线程完成
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if app.client.reply_ask_user.called:
                    break
                time.sleep(0.01)
            app.client.reply_ask_user.assert_called_once()
        finally:
            app._turn_threads.discard(dummy_thread)

    def test_escape_does_not_cancel_turn_when_permission_pending(self) -> None:
        """有 pending permission 时，按下 ESC 不触发整轮取消。"""
        import threading

        from prompt_toolkit.keys import Keys

        from cli.app import AnnaCliApp

        app = AnnaCliApp()
        app.current_session = {"id": "sess-esc-3", "title": "EscPerm"}
        app.client.cancel = MagicMock()

        dummy_thread = threading.current_thread()
        app._turn_threads.add(dummy_thread)
        app._enqueue_permission_request(
            {"permission_id": "perm-esc-3", "tool_name": "bash"},
            owner_id="req-esc-3",
        )

        try:
            escape = next(
                b for b in app.session_prompt.key_bindings.bindings if b.keys == (Keys.Escape,)
            )
            event = MagicMock()
            event.current_buffer.text = ""
            with patch("cli.app.ui.console.print"):
                escape.handler(event)

            app.client.cancel.assert_not_called()
        finally:
            app._turn_threads.discard(dummy_thread)

    def test_handle_skip_cancels_running_turn_when_no_clarification_pending(self) -> None:
        """无澄清提问时，/cancel 或 /skip 可取消在途运行轮次。"""
        import threading

        from cli.app import AnnaCliApp
        from cli.commands import handle_skip

        app = AnnaCliApp()
        app.current_session = {"id": "sess-cmd-cancel", "title": "CmdCancel"}
        app.client.cancel = MagicMock()
        app.client.abort_stream = MagicMock()

        dummy_thread = threading.current_thread()
        app._turn_threads.add(dummy_thread)

        try:
            with patch("cli.app.ui.console.print"):
                handle_skip(app, "")

            app.client.cancel.assert_called_once_with("sess-cmd-cancel")
        finally:
            app._turn_threads.discard(dummy_thread)

    def test_print_failed_mcps_hint(self) -> None:
        """启动时若发现 MCP 处于 failed 状态，在控制台紧凑提示。"""
        from cli.app import AnnaCliApp

        app = AnnaCliApp()

        # 1. 0 个失败：静默，不打印
        app.client.list_mcps = MagicMock(return_value=[{"id": "py", "state": "connected"}])
        with patch("cli.app.ui.console.print") as mock_print:
            app._print_failed_mcps_hint()
            mock_print.assert_not_called()

        # 2. 单个失败：精准提示对应重连命令
        app.client.list_mcps = MagicMock(
            return_value=[
                {"id": "exa-search", "state": "failed", "error": "timeout"},
                {"id": "python", "state": "connected", "error": None},
            ]
        )
        with patch("cli.app.ui.console.print") as mock_print:
            app._print_failed_mcps_hint()
            mock_print.assert_called_once()
            output = str(mock_print.call_args[0][0])
            self.assertIn("exa-search", output)
            self.assertIn("/mcp", output)

        # 3. 2~3 个失败：汇总聚合为单行，指引使用 /mcp
        app.client.list_mcps = MagicMock(
            return_value=[
                {"id": "exa-search", "state": "failed"},
                {"id": "github", "state": "failed"},
                {"id": "python", "state": "connected"},
            ]
        )
        with patch("cli.app.ui.console.print") as mock_print:
            app._print_failed_mcps_hint()
            mock_print.assert_called_once()
            output = str(mock_print.call_args[0][0])
            self.assertIn("2 MCP servers ('exa-search', 'github') failed to connect", output)
            self.assertIn("/mcp to view or reconnect", output)

        # 4. >3 个失败：截断并显示剩余数量
        app.client.list_mcps = MagicMock(
            return_value=[
                {"id": "s1", "state": "failed"},
                {"id": "s2", "state": "failed"},
                {"id": "s3", "state": "failed"},
                {"id": "s4", "state": "failed"},
            ]
        )
        with patch("cli.app.ui.console.print") as mock_print:
            app._print_failed_mcps_hint()
            output = str(mock_print.call_args[0][0])
            expected = "4 MCP servers ('s1', 's2', 's3', and 1 more) failed to connect"
            self.assertIn(expected, output)

    def test_completer_suggests_mcps_and_retry(self) -> None:
        """测试 /mcps 与 /mcp 的子命令补全及 Server ID 补全。"""
        from cli.completer import SlashCompleter

        mcps = [
            {"id": "exa-search", "state": "failed"},
            {"id": "python", "state": "connected"},
        ]
        completer = SlashCompleter(get_mcps_fn=lambda: mcps)

        # 1. /mcps 提示 retry
        doc = Document(text="/mcps ")
        completions = list(completer.get_completions(doc, None))
        self.assertTrue(any(c.text == "retry" for c in completions))

        # 2. /mcp 同样提示 retry
        doc = Document(text="/mcp ")
        completions = list(completer.get_completions(doc, None))
        self.assertTrue(any(c.text == "retry" for c in completions))

        # 3. /mcps retry 提示 Server ID
        doc = Document(text="/mcps retry ")
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        self.assertIn("exa-search", texts)
        self.assertIn("python", texts)

        # 4. /mcps retry ex 前缀过滤
        doc = Document(text="/mcps retry ex")
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        self.assertEqual(texts, ["exa-search"])

    def test_handle_mcps_smart_retry(self) -> None:
        """测试 /mcps retry 无参数时自动重试所有 failed 的 MCP。"""
        from cli.app import AnnaCliApp
        from cli.commands import handle_mcps

        app = AnnaCliApp()
        app.client.list_mcps = MagicMock(
            return_value=[
                {"id": "exa-search", "state": "failed"},
                {"id": "python", "state": "connected"},
            ]
        )
        app.client.retry_mcp = MagicMock(
            return_value={"id": "exa-search", "state": "connected", "tool_count": 3}
        )

        with patch("cli.commands._print_mcps_retry_summary") as mock_summary:
            # /mcps retry 无参数：自动挑选 failed 的 exa-search 重试
            handle_mcps(app, "retry")
            app.client.retry_mcp.assert_called_once_with("exa-search")
            mock_summary.assert_called_once()
            retried_map = mock_summary.call_args[0][0]
            self.assertIn("exa-search", retried_map)
            self.assertEqual(retried_map["exa-search"]["state"], "connected")

    def test_mcp_picker_filtering_and_formatting(self) -> None:
        """测试 McpPicker 的过滤、高亮与失败项优先选择。"""
        from cli.picker import McpPicker

        mcps = [
            {"id": "python", "state": "connected", "tool_count": 5},
            {
                "id": "exa-search",
                "state": "failed",
                "tool_count": 0,
                "error": "Connection timeout\nDetails...",
            },
            {"id": "filesystem", "state": "disabled", "tool_count": 0},
        ]
        # 自动高亮第一个 failed 项
        picker = McpPicker(mcps)
        self.assertEqual(picker.selected_index, 1)
        self.assertEqual(len(picker._get_filtered()), 3)

        # 搜索过滤
        picker.search_query = "exa"
        filtered = picker._get_filtered()
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], "exa-search")

        # 格式化文本包含错误摘要
        formatted = picker._get_formatted_text()
        text_content = "".join(t[1] for t in formatted)
        self.assertIn("exa-search", text_content)
        self.assertIn("FAILED", text_content)
        self.assertIn("Err: Connection timeout", text_content)

    def test_mcp_picker_in_place_retry_and_spinner(self) -> None:
        """测试 McpPicker 原地重试、转轮动画与状态就地更新。"""
        import threading
        from typing import Any

        from cli.picker import McpPicker

        mcps = [
            {"id": "exa-search", "state": "failed", "tool_count": 0},
            {"id": "python", "state": "connected", "tool_count": 5},
        ]
        gate = threading.Event()

        def fake_retry(sid: str) -> dict[str, Any]:
            gate.wait(timeout=2.0)
            return {"id": sid, "state": "connected", "tool_count": 3}

        picker = McpPicker(mcps, retry_fn=fake_retry)

        # 触发重试
        picker._trigger_retry("exa-search")
        self.assertIn("exa-search", picker._retrying_servers)

        # 重试中状态呈现 RETRYING 转轮
        formatted = picker._get_formatted_text()
        text_content = "".join(t[1] for t in formatted)
        self.assertIn("RETRYING", text_content)
        self.assertIn("Connecting…", text_content)

        # 释放栅栏，等待后台重试完成
        gate.set()
        picker.wait_retries(timeout=3.0)
        self.assertNotIn("exa-search", picker._retrying_servers)
        self.assertEqual(picker.mcps[0]["state"], "connected")
        self.assertEqual(picker.mcps[0]["tool_count"], 3)

        # 完成后状态变为 CONNECTED
        formatted_done = picker._get_formatted_text()
        text_done = "".join(t[1] for t in formatted_done)
        self.assertIn("CONNECTED", text_done)
        self.assertIn("✓ Reconnected", text_done)

    def test_mcp_picker_run_interactive_keys(self) -> None:
        """测试 McpPicker 的回车选择与 Esc 取消。"""
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from cli.picker import McpPicker

        mcps = [
            {"id": "exa-search", "state": "failed"},
            {"id": "python", "state": "connected"},
        ]
        # 无 retry_fn 时，Enter 退出并返回结果
        picker = McpPicker(mcps)
        with create_pipe_input() as inp:
            inp.send_text("\r")
            res = picker.run(input=inp, output=DummyOutput())
            self.assertEqual(res, mcps[0])

        # Esc 取消直接退出
        picker_cancel = McpPicker(mcps)
        with create_pipe_input() as inp:
            inp.send_text("\x1b")
            res = picker_cancel.run(input=inp, output=DummyOutput())
            self.assertIsNone(res)

    def test_handle_mcps_opens_picker_and_reconnects(self) -> None:
        """测试 /mcp 无参数时打开 McpPicker 并注入 retry_fn。"""
        from cli.app import AnnaCliApp
        from cli.commands import handle_mcps

        app = AnnaCliApp()
        mcps = [
            {"id": "exa-search", "state": "failed"},
            {"id": "python", "state": "connected"},
        ]
        app.client.list_mcps = MagicMock(return_value=mcps)
        app.client.retry_mcp = MagicMock(
            return_value={"id": "exa-search", "state": "connected", "tool_count": 3}
        )

        with (
            patch("cli.commands.McpPicker") as mock_picker_cls,
            patch("cli.commands._print_mcps_retry_summary") as mock_summary,
        ):
            mock_picker = mock_picker_cls.return_value
            mock_picker.retried_results = {
                "exa-search": {"id": "exa-search", "state": "connected", "tool_count": 3}
            }

            handle_mcps(app, "")

            mock_picker_cls.assert_called_once_with(mcps, retry_fn=app.client.retry_mcp)
            mock_picker.run.assert_called_once()
            mock_summary.assert_called_once_with(mock_picker.retried_results)







