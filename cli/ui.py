"""Refined Terminal UI components styled in clean & minimal dark theme."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.markdown import Heading, Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from cli.theme import (
    AMBER,
    BG_CARD,
    BORDER,
    BORDER_FOCUS,
    CYAN,
    CYAN_BRIGHT,
    DIM,
    DIM_DARK,
    FG,
    FG_MUTED,
    GREEN,
    RED,
    WHITE,
    WHITE_SOFT,
    YELLOW,
)

cli_theme = Theme({
    "markdown.h1": f"bold {WHITE}",
    "markdown.h2": f"bold {CYAN}",
    "markdown.h3": f"bold {CYAN_BRIGHT}",
    "markdown.h4": f"bold {WHITE}",
    "markdown.strong": f"bold {WHITE}",
    "markdown.emph": f"italic {FG_MUTED}",
    "markdown.code": f"bold {CYAN} on {BG_CARD}",
    "markdown.item.bullet": f"bold {CYAN}",
    "markdown.link": f"underline {CYAN}",
})

console = Console(theme=cli_theme)


# ---------------------------------------------------------------------------
# Custom Markdown Heading Renderer (left-aligned, compact)
# ---------------------------------------------------------------------------
class CleanHeading(Heading):
    """Left-aligned and compact Markdown heading."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text
        text.justify = "left"
        if self.tag == "h1":
            text.stylize(f"bold {WHITE}")
        elif self.tag == "h2":
            text.stylize(f"bold {CYAN}")
        elif self.tag == "h3":
            text.stylize(f"bold {CYAN_BRIGHT}")
        else:
            text.stylize(f"bold {WHITE}")
        yield text


Markdown.elements["heading_open"] = CleanHeading

MAX_TOOL_CONTENT_CHARS = 4_000
MAX_TOOL_ARGS_CHARS = 2_000

CODEX_TIPS = [
    "Tip: Type / for commands",
    "Tip: Use /resume to switch sessions",
    "Tip: Use /compact to reduce context size",
    "Tip: Use /sessions to list conversation history",
]


def get_clean_cwd() -> str:
    """Get clean current working directory with $HOME replaced by ~."""
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        return "~" + cwd[len(home):]
    return cwd


def get_current_model() -> str:
    """Get currently configured model name."""
    try:
        from agent.infrastructure.settings import ChatModelSettings
        return ChatModelSettings().openai_model
    except Exception:
        return os.environ.get("OPENAI_MODEL") or "glm-5.3-flash"


def format_relative_time(ts_str: str | None) -> str:
    """Format relative time (e.g. 1d ago, 2h ago, 5m ago, just now)."""
    if not ts_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 0 or seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        if days < 30:
            return f"{days}d ago"
        months = days // 30
        if months < 12:
            return f"{months}mo ago"
        return f"{days // 365}y ago"
    except Exception:
        return ts_str[:10]


def format_timestamp(ts_str: str | None) -> str:
    """Format ISO timestamp string."""
    if not ts_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d %H:%M")
    except Exception:
        return ts_str[:16]


def clear_screen() -> None:
    """Clear terminal viewport and scrollback buffer completely."""
    import sys
    sys.stdout.write("\033[2J\033[3J\033[H")
    sys.stdout.flush()


def print_banner(
    base_url: str = "",
    session_title: str | None = None,
    session_id: str | None = None,
    health: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> None:
    """Print minimal startup banner and Codex-style tip."""
    import random
    model = model_name or get_current_model()
    directory = get_clean_cwd()

    content = Text()
    content.append("AgentDesk\n", style=f"bold {WHITE}")
    if session_title or session_id:
        s_text = session_title or "Untitled"
        sid_short = f" ({session_id[:8]})" if session_id else ""
        content.append("Session:   ", style=DIM)
        content.append(f"{s_text}{sid_short}\n", style=WHITE)
    content.append("Model:     ", style=DIM)
    content.append(f"{model}\n", style=WHITE)
    content.append("Directory: ", style=DIM)
    content.append(f"{directory}", style=WHITE)

    panel = Panel(
        content,
        border_style=BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )
    console.print(panel)
    if health is not None and health.get("status") != "ok":
        console.print(f"[{RED}]● Backend status: {health.get('status')} ({base_url})[/{RED}]")
    elif health is None and base_url:
        console.print(f"[{YELLOW}]● Backend service unreachable ({base_url})[/{YELLOW}]")
    tip = random.choice(CODEX_TIPS)
    console.print(f"[{DIM}]{tip}[/{DIM}]\n")


def print_user_prompt(text: str) -> None:
    """Render user prompt."""
    console.print(f"\n[bold {CYAN}]>[/bold {CYAN}] [bold {WHITE}]{text}[/bold {WHITE}]")


def print_error_message(error: str) -> None:
    """Render error message."""
    console.print(f"[{RED}]Error:[/{RED}] [{RED}]{escape(str(error))}[/{RED}]\n")


def print_health(health: dict[str, Any], base_url: str) -> None:
    """Render backend health check status."""
    status = health.get("status") or "unknown"
    if status == "ok":
        body = f"[{GREEN}]Backend service online[/{GREEN}] [{DIM}]({base_url})[/{DIM}]"
    else:
        body = f"[{RED}]Backend service unhealthy[/{RED}] [{DIM}]status={status} ({base_url})[/{DIM}]"
    console.print(body)
    console.print()


def print_sessions_table(
    sessions: list[dict[str, Any]],
    current_session_id: str | None = None,
) -> None:
    """Render sessions list as a clean Rich table."""
    if not sessions:
        console.print(f"[{DIM}](No sessions found. Use /new to create one)[/{DIM}]\n")
        return

    table = Table(
        title=f"[bold {WHITE}]Sessions[/bold {WHITE}]",
        border_style=BORDER,
        box=box.ROUNDED,
        row_styles=["", DIM],
        title_justify="left",
    )
    table.add_column("Idx", justify="right", style=CYAN, width=5)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Type", justify="center", width=9)
    table.add_column("Title", style=WHITE, min_width=20)
    table.add_column("Updated", style=DIM, width=13)
    table.add_column("ID", style=DIM, width=10)

    for idx, s in enumerate(sessions, 1):
        is_current = s.get("id") == current_session_id
        status_tag = f"[{GREEN}]current[/{GREEN}]" if is_current else f"[{DIM}]-[/{DIM}]"
        kind_tag = f"[{AMBER}]subagent[/{AMBER}]" if s.get("main_session_id") else f"[{CYAN}]main[/{CYAN}]"
        title = s.get("title") or "Untitled"
        if is_current:
            title = f"[bold {CYAN}]{title}[/bold {CYAN}]"
        table.add_row(
            f"[{idx}]",
            status_tag,
            kind_tag,
            title,
            format_timestamp(s.get("updated_at")),
            (s.get("id") or "")[:8],
        )

    console.print(table)
    console.print()


def print_session_detail(session: dict[str, Any], message_count: int) -> None:
    """Render current session details."""
    table = Table(
        title=f"[bold {WHITE}]Session Details[/bold {WHITE}]",
        border_style=BORDER,
        box=box.ROUNDED,
        show_header=False,
        title_justify="left",
        row_styles=["", DIM],
        expand=True,
    )
    table.add_column("Field", style=f"bold {CYAN}", width=18)
    table.add_column("Value", style=WHITE)

    main_id = session.get("main_session_id")
    parent_id = session.get("parent_session_id")
    tools = session.get("allowed_tools") or []

    table.add_row("Title", session.get("title") or "Untitled")
    table.add_row("Session ID", session.get("id", ""))
    table.add_row("Type", "subagent" if main_id else "main session")
    if main_id:
        table.add_row("Main Session", main_id)
    if parent_id:
        table.add_row("Parent Session", parent_id)
    table.add_row("User ID", session.get("user_id") or "default_user")
    table.add_row("Messages", str(message_count))
    table.add_row("Allowed Tools", ", ".join(tools) if tools else "(none)")
    table.add_row("Created", format_timestamp(session.get("created_at")))
    table.add_row("Updated", format_timestamp(session.get("updated_at")))

    console.print(table)
    console.print()


def print_mcps_table(mcps: list[dict[str, Any]]) -> None:
    """Render MCP server status table."""
    if not mcps:
        console.print(f"[{DIM}](No MCP servers configured)[/{DIM}]\n")
        return

    table = Table(
        title=f"[bold {WHITE}]MCP Servers[/bold {WHITE}]",
        border_style=BORDER,
        box=box.ROUNDED,
        row_styles=["", DIM],
        title_justify="left",
        expand=True,
    )
    table.add_column("Server ID", style=f"bold {CYAN}", width=18)
    table.add_column("Status", justify="center", width=14)
    table.add_column("Tools", justify="right", width=8)
    table.add_column("Description / Error", style=FG_MUTED)

    for m in mcps:
        state = (m.get("state") or "unknown").upper()
        if state == "CONNECTED":
            state_tag = f"[{GREEN}]CONNECTED[/{GREEN}]"
        elif state == "FAILED":
            state_tag = f"[{RED}]FAILED[/{RED}]"
        elif state == "DISABLED":
            state_tag = f"[{DIM}]DISABLED[/{DIM}]"
        else:
            state_tag = f"[{YELLOW}]{state}[/{YELLOW}]"

        tool_count = str(m.get("tool_count", 0))
        desc = m.get("description") or ""
        if m.get("error"):
            desc += f"\n[{RED}]Error: {m['error']}[/{RED}]"

        table.add_row(m.get("id", ""), state_tag, tool_count, desc)

    console.print(table)
    console.print()


def print_skills_table(skills: list[dict[str, Any]]) -> None:
    """Render skills catalog table."""
    if not skills:
        console.print(f"[{DIM}](No skills configured)[/{DIM}]\n")
        return

    table = Table(
        title=f"[bold {WHITE}]Skills Catalog[/bold {WHITE}]",
        border_style=BORDER,
        box=box.ROUNDED,
        row_styles=["", DIM],
        title_justify="left",
        expand=True,
    )
    table.add_column("Skill", style=f"bold {CYAN}", width=22)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Description", style=FG_MUTED)
    table.add_column("Path", style=DIM)

    for s in skills:
        state = (s.get("state") or "unknown").upper()
        state_tag = f"[{GREEN}]LOADED[/{GREEN}]" if state == "LOADED" else f"[{RED}]FAILED[/{RED}]"
        table.add_row(
            s.get("name", ""),
            state_tag,
            s.get("description") or "",
            s.get("path") or "",
        )

    console.print(table)
    console.print()


def print_history_list(messages: list[dict[str, Any]], limit: int | None = None) -> None:
    """Render conversation history messages with optional limit."""
    if not messages:
        return

    valid_messages = [
        m for m in messages
        if m.get("role") not in ("system", "tool")
        and (m.get("metadata") or {}).get("kind") not in ("summary", "maintenance", "system")
    ]
    if not valid_messages:
        console.print(f"[{DIM}](No conversational messages in this session)[/{DIM}]\n")
        return

    if limit is not None and len(valid_messages) > limit:
        hidden = len(valid_messages) - limit
        console.print(f"[{DIM}]... ({hidden} earlier messages hidden · Type /history to view all)[/{DIM}]\n")
        to_show = valid_messages[-limit:]
    else:
        to_show = valid_messages

    for i, msg in enumerate(to_show):
        is_last = (i == len(to_show) - 1)
        print_message(msg, is_last_in_history=is_last)
    console.print()


def print_message(message: dict[str, Any], is_last_in_history: bool = False) -> None:
    """Render a single message from history."""
    role = message.get("role")
    content = message.get("content", "")
    metadata = message.get("metadata") or {}
    kind = metadata.get("kind")

    if role in ("system", "tool") or kind in ("summary", "maintenance", "system"):
        return

    if role == "human":
        if metadata.get("source") == "agent":
            sender = metadata.get("name") or "subagent"
            console.print(f"\n[bold {CYAN}][Subagent: {escape(sender)}][/bold {CYAN}]")
            right_pad = max(6, int(console.width * 0.12))
            console.print(Padding(Markdown(content, code_theme="one-dark"), (0, right_pad, 0, 0)))
        else:
            console.print(f"\n[bold {CYAN}]You:[/] [bold {WHITE}]{escape(content)}[/bold {WHITE}]")
            if is_last_in_history:
                console.print(f"  [{DIM}](No response recorded for this question)[/{DIM}]")
        return

    if role == "assistant":
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", str(call))
                if not name and isinstance(call, dict) and "function" in call:
                    name = call["function"].get("name", "")
                if name:
                    if name == "ask_user":
                        console.print(f"[{DIM}]  Clarification requested[/{DIM}]")
                    else:
                        console.print(f"[{DIM}]  Calling [bold {CYAN}]{escape(name)}[/bold {CYAN}]...[/{DIM}]")

        if content and content.strip():
            console.print(f"\n[bold {GREEN}]AgentDesk:[/] ")
            right_pad = max(6, int(console.width * 0.12))
            console.print(Padding(Markdown(content, code_theme="one-dark"), (0, right_pad, 0, 0)))
            usage = metadata.get("usage") or {}
            if usage:
                console.print(
                    f"[{DIM}]tokens: {usage.get('input_tokens', 0):,} in · {usage.get('output_tokens', 0):,} out "
                    f"· {usage.get('total_tokens', 0):,} total[/{DIM}]"
                )


def format_tool_result_preview(content: str, max_len: int = 90) -> str:
    """Extract clean, meaningful content from tool results, stripping status/envelope metadata."""
    if not content:
        return ""

    raw = content.strip()
    parsed: Any = None
    if raw.startswith(("{", "[")):
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None

    result_text = ""
    if isinstance(parsed, dict):
        # 1. Error case
        if "error" in parsed and (parsed.get("ok") is False or parsed.get("status") in ("error", "failed")):
            result_text = f"Error: {parsed['error']}"
        # 2. Python stdout / stderr
        elif "stdout" in parsed or "stderr" in parsed:
            stdout = (parsed.get("stdout") or "").strip()
            stderr = (parsed.get("stderr") or "").strip()
            result_text = stdout or stderr or "(completed)"
        # 2.5 Ask user clarification result
        elif "options" in parsed and "question" in parsed:
            opts = parsed.get("options") or []
            q = parsed.get("question") or ""
            result_text = f"Clarification: {q} ({len(opts)} options)"
        # 3. Primary payload fields
        elif any(k in parsed for k in ("result", "data", "summary", "content")):
            for key in ("result", "data", "summary", "content"):
                if key in parsed and parsed[key]:
                    val = parsed[key]
                    result_text = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
                    break
        # 4. Collection / search result fields
        elif "tools" in parsed and isinstance(parsed["tools"], list):
            items = [x.get("name", str(x)) if isinstance(x, dict) else str(x) for x in parsed["tools"]]
            result_text = ", ".join(items[:4])
            if len(items) > 4:
                result_text += f" (+{len(items)-4} more)"
        elif "memories" in parsed and isinstance(parsed["memories"], list):
            items = [x.get("content", str(x)) if isinstance(x, dict) else str(x) for x in parsed["memories"]]
            result_text = "; ".join(items[:2])
        # 5. Generic dictionary: strip metadata envelope keys (ok, status, code, etc.)
        else:
            filtered = {
                k: v for k, v in parsed.items()
                if k not in ("ok", "status", "code", "success", "session_id", "caller_session_id")
            }
            if filtered:
                parts = []
                for k, v in filtered.items():
                    if isinstance(v, (str, int, float, bool)):
                        parts.append(str(v))
                    else:
                        parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
                result_text = " · ".join(parts)
            else:
                result_text = "(done)"

    elif isinstance(parsed, list):
        items = [x.get("name", str(x)) if isinstance(x, dict) else str(x) for x in parsed]
        result_text = ", ".join(items[:4])
        if len(parsed) > 4:
            result_text += f" (+{len(parsed)-4} more)"
    else:
        result_text = raw

    cleaned = " ".join(result_text.split())
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len - 3] + "..."
    return cleaned


def _render_tool_call(name: str, args: Any) -> None:
    """Render tool execution card."""
    parsed_args: dict[str, Any] = {}
    if isinstance(args, str):
        try:
            parsed_args = json.loads(args, strict=False)
        except Exception:
            parsed_args = {}
    elif isinstance(args, dict):
        parsed_args = args

    if name == "execute_python" and "code" in parsed_args:
        code = str(parsed_args["code"]).strip()
        syntax = Syntax(code, "python", theme="one-dark", line_numbers=False, padding=(0, 1))
        panel = Panel(
            syntax,
            title=f"[bold {CYAN}]Python Execution[/bold {CYAN}]",
            title_align="left",
            border_style=BORDER,
            box=box.ROUNDED,
            padding=(0, 1),
        )
        console.print(panel)
        return

    if name == "send_message" and "message" in parsed_args:
        recipient = parsed_args.get("recipient", "agent")
        msg_text = str(parsed_args["message"]).strip()
        title = f"[bold {CYAN}]send_message -> {recipient}[/bold {CYAN}]"
        panel = Panel(
            Markdown(msg_text, code_theme="one-dark"),
            title=title,
            title_align="left",
            border_style=BORDER,
            box=box.ROUNDED,
            padding=(0, 1),
        )
        console.print(panel)
        return

    if name == "define_subagent":
        sub_name = parsed_args.get("name", "subagent")
        role = parsed_args.get("role", "")
        console.print(f"  [bold {CYAN}]define_subagent:[/bold {CYAN}] [bold {WHITE}]{sub_name}[/bold {WHITE}] [{DIM}]({role})[/{DIM}]")
        return

    arg_summary = _format_arguments(parsed_args or args)
    console.print(f"  [bold {CYAN}]Calling tool: {name}[/bold {CYAN}] [{DIM}]{arg_summary}[/{DIM}]")


def print_tool_call_start(name: str, args: Any) -> None:
    """Render clean tool call start indicator."""
    if name == "ask_user":
        return
    arg_snippet = ""
    parsed: Any = None
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except Exception:
            parsed = args
    elif isinstance(args, dict):
        parsed = args

    if name == "execute_python":
        if isinstance(parsed, dict) and "code" in parsed:
            lines = str(parsed["code"]).strip().splitlines()
            preview = lines[0][:40] + ("..." if len(lines) > 1 or len(lines[0]) > 40 else "")
            arg_snippet = f"code: {preview}"
        else:
            arg_snippet = "running script"
    elif name == "send_message":
        if isinstance(parsed, dict):
            recipient = parsed.get("recipient", "subagent")
            arg_snippet = f"-> {recipient}"
    elif name == "define_subagent":
        if isinstance(parsed, dict):
            sub = parsed.get("name", "subagent")
            arg_snippet = f"name={sub}"
    elif name == "search_mcp":
        if isinstance(parsed, dict):
            mcp = parsed.get("mcp", "")
            q = parsed.get("query", "")
            arg_snippet = f'{mcp} "{q[:30]}"'
    elif name == "execute_mcp":
        if isinstance(parsed, dict):
            mcp = parsed.get("mcp", "")
            tool = parsed.get("tool", "")
            arg_snippet = f"{mcp}/{tool}"
    elif name == "search_memory":
        if isinstance(parsed, dict):
            q = parsed.get("query", "")
            arg_snippet = f'query="{q[:30]}"'
    elif name in ("load_skill", "search_skill"):
        if isinstance(parsed, dict):
            sname = parsed.get("name") or parsed.get("query") or ""
            arg_snippet = f'name="{sname[:30]}"'
    elif name == "wait_for_replies":
        if isinstance(parsed, dict):
            subs = parsed.get("subagents") or []
            arg_snippet = f"subagents={subs}"
    elif name == "list_subagents":
        arg_snippet = ""
    elif parsed:
        arg_snippet = _format_arguments(parsed)

    if arg_snippet:
        console.print(f"\n[bold {CYAN}]● Tool:[/] [bold {WHITE}]{escape(name)}[/] [{DIM}]({escape(arg_snippet)})[/]")
    else:
        console.print(f"\n[bold {CYAN}]● Tool:[/] [bold {WHITE}]{escape(name)}[/]")


def print_tool_result_end(content: str) -> None:
    """Render clean tool result summary line."""
    preview = format_tool_result_preview(content)
    if preview:
        console.print(f"  [{DIM}]└─ {escape(preview)}[/{DIM}]")


def print_clarification_question(question: str) -> None:
    """Render clarifying question panel for ask_user."""
    console.print(
        Panel(
            Markdown(question, code_theme="one-dark"),
            title=f"[bold {CYAN}]Clarification[/bold {CYAN}]",
            title_align="left",
            border_style=BORDER_FOCUS,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _render_tool_content(tool_name: str, content: str) -> Any:
    """Render tool result content."""
    stripped = content.strip()
    parsed = None
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped, strict=False)
        except Exception:
            parsed = None

    if tool_name == "execute_python" and isinstance(parsed, dict):
        stdout = parsed.get("stdout", "")
        stderr = parsed.get("stderr", "")
        ok = parsed.get("ok", True)
        exit_code = parsed.get("exit_code", 0)

        parts: list[Any] = []
        if stdout:
            title = f"[{GREEN}]output (exit 0)[/{GREEN}]" if ok else f"[{RED}]output (exit {exit_code})[/{RED}]"
            parts.append(Panel(
                Syntax(stdout.rstrip(), "text", theme="one-dark", padding=(0, 1)),
                title=title,
                title_align="left",
                border_style=BORDER,
                box=box.ROUNDED,
                padding=(0, 1),
            ))
        if stderr:
            parts.append(Panel(
                Text(f"{stderr.rstrip()}", style=RED),
                title=f"[{RED}]stderr[/{RED}]",
                title_align="left",
                border_style=RED,
                box=box.ROUNDED,
                padding=(0, 1),
            ))
        if not stdout and not stderr:
            tag = f"[{GREEN}]Success (exit 0)[/{GREEN}]" if ok else f"[{RED}]Failed (exit {exit_code})[/{RED}]"
            parts.append(Text.from_markup(f"  {tag}"))
        return Group(*parts) if parts else Text("")

    if tool_name == "send_message" and isinstance(parsed, dict):
        recipient = parsed.get("recipient", "agent")
        return Text.from_markup(f"  [{GREEN}]Message delivered to {recipient}[/{GREEN}]")

    if parsed is not None:
        pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
        truncated, was_truncated = _truncate(pretty, MAX_TOOL_CONTENT_CHARS)
        renderable: Any = Syntax(truncated, "json", theme="one-dark", word_wrap=True)
        if was_truncated:
            return Group(renderable, Text.from_markup(f"[{DIM}]... (truncated)[/{DIM}]"))
        return renderable

    truncated, was_truncated = _truncate(content, MAX_TOOL_CONTENT_CHARS)
    renderable = Markdown(truncated, code_theme="one-dark")
    if was_truncated:
        return Group(renderable, Text.from_markup(f"[{DIM}]... (truncated)[/{DIM}]"))
    return renderable


def print_compact_result(result: dict[str, Any] | None) -> None:
    """Render context compaction outcome."""
    if not isinstance(result, dict):
        console.print(f"[{DIM}]Context compaction skipped: empty or invalid response from server[/{DIM}]\n")
        return

    status = result.get("status")
    if status == "compacted":
        before = result.get("estimated_tokens_before", 0)
        after = result.get("estimated_tokens_after", 0)
        kind = result.get("kind") or "summary"
        diff_text = f"({before - after:,} tokens reduced)" if before > after else "(checkpoint summary generated)"
        console.print(
            f"[{GREEN}]Context compacted ({kind})[/{GREEN}] "
            f"[{DIM}]Tokens: {before:,} -> {after:,} {diff_text}[/{DIM}]\n"
        )
    else:
        reason = result.get("message") or "nothing to compact"
        console.print(f"[{DIM}]Context compaction skipped: {reason}[/{DIM}]\n")


def _format_arguments(args: Any) -> str:
    """Format arguments for preview."""
    if isinstance(args, str):
        try:
            parsed = json.loads(args, strict=False)
            text = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            text = args
    elif isinstance(args, (dict, list)):
        text = json.dumps(args, ensure_ascii=False)
    else:
        text = str(args)

    truncated, _ = _truncate(text, MAX_TOOL_ARGS_CHARS)
    return truncated


def _truncate(content: str, limit: int) -> tuple[str, bool]:
    """Truncate long text content."""
    if len(content) <= limit:
        return content, False
    return content[:limit], True