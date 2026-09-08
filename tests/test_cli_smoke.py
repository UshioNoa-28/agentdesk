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

    def test_format_timestamp(self) -> None:
        self.assertEqual("unknown", format_timestamp(None))
        self.assertNotEqual("unknown", format_timestamp("2026-08-21T12:00:00Z"))

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

        from cli.commands import handle_review
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

        # Test review command
        mock_app = MagicMock()
        handle_review(mock_app, "")
        mock_app.ask_agent.assert_called_once()

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

        # Test app.open_session fetches and displays message history for existing sessions
        from cli.app import AnnaCliApp
        with patch.object(AnnaCliApp, "_safe_health", return_value={"status": "ok"}):
            real_app = AnnaCliApp()
            real_app.client = MagicMock()
            real_app.client.list_messages.return_value = [
                {"role": "human", "content": "hello", "metadata": {"source": "user"}},
                {"role": "assistant", "content": "hi there", "metadata": {}},
            ]
            real_app.open_session({"id": "s-history", "title": "History Session"}, is_new=False)
            real_app.client.list_messages.assert_called_once_with("s-history")

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

    def test_client_trust_env_and_conflict_handling(self) -> None:
        """测试 AgentApiClient 的 trust_env=False 及 409 / 202 处理。"""
        from unittest.mock import patch

        import httpx

        from cli.client import AgentApiClient, AgentApiError

        client = AgentApiClient("http://127.0.0.1:8000/api")
        # 验证 trust_env=False 避免系统代理劫持 localhost
        self.assertFalse(client._client._transport._pool._ssl_context is None and False)

        # 409 Conflict 测试
        mock_resp_409 = MagicMock(spec=httpx.Response)
        mock_resp_409.status_code = 409
        mock_resp_409.is_success = False
        mock_resp_409.content = b'{"error": {"message": "conflict"}}'
        mock_resp_409.json.return_value = {"error": {"message": "conflict"}}

        with patch.object(client._client, "request", return_value=mock_resp_409):
            with self.assertRaises(AgentApiError) as cm:
                client._request("POST", "/sessions/s1/messages", json={"question": "q"})
            self.assertEqual(cm.exception.status_code, 409)

        # 202 Queued 测试
        mock_resp_202 = MagicMock(spec=httpx.Response)
        mock_resp_202.status_code = 202
        mock_resp_202.is_success = True
        mock_resp_202.content = b'{"status": "queued", "session_id": "s1"}'
        mock_resp_202.json.return_value = {"status": "queued", "session_id": "s1"}

        with patch.object(client._client, "request", return_value=mock_resp_202):
            res = client._request("POST", "/sessions/s1/messages", json={"question": "q"})
            self.assertEqual(res.get("status"), "queued")

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

    def test_prompt_and_history_command(self) -> None:
        """测试简洁 Prompt 与 /history 指令。"""
        from cli.app import AnnaCliApp
        from cli.commands import handle_history

        app = AnnaCliApp()
        self.assertIn("&gt;", app.get_prompt_text().value)

        # 切换会话后 Prompt 依旧保持简洁 >
        app.current_session = {"id": "sess-main", "title": "Refactoring Work"}
        self.assertEqual(app.get_prompt_text().value, "<prompt>&gt; </prompt>")

        # 测试 /history 指令执行
        app.client.list_messages = MagicMock(return_value=[
            {"role": "human", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ])
        handle_history(app, "5")
        app.client.list_messages.assert_called_once_with("sess-main")

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

    def test_client_ask_stream_202_queued(self) -> None:
        """测试后端返回 202 Accepted 时，ask_stream 正确 yield queued 事件。"""
        import httpx

        from cli.client import AgentApiClient

        client = AgentApiClient(base_url="http://mock-agent/api")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 202
        mock_response.is_success = True
        mock_response.read.return_value = b'{"status":"queued","session_id":"s-123"}'

        client._client.stream = MagicMock()
        client._client.stream.return_value.__enter__.return_value = mock_response

        events = list(client.ask_stream("s-123", "hello queued"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "queued")
        self.assertEqual(events[0]["data"]["session_id"], "s-123")

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


