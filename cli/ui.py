"""Refined Terminal UI components styled in clean & minimal dark theme."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import warnings
from datetime import datetime, timezone
from typing import Any, Sequence

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.errors import LiveError
from rich.live import Live
from rich.markdown import Heading, Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
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
    YELLOW,
)

cli_theme = Theme(
    {
        "markdown.h1": f"bold {WHITE}",
        "markdown.h2": f"bold {CYAN}",
        "markdown.h3": f"bold {CYAN_BRIGHT}",
        "markdown.h4": f"bold {WHITE}",
        "markdown.strong": f"bold {WHITE}",
        "markdown.emph": f"italic {FG_MUTED}",
        "markdown.code": f"bold {CYAN} on {BG_CARD}",
        "markdown.item.bullet": f"bold {CYAN}",
        "markdown.link": f"underline {CYAN}",
    }
)

console = Console(theme=cli_theme)
LIVE_RENDER_LOCK = threading.Lock()

_SILENCED_LOGGERS = (
    "mem0",
    "agent.infrastructure.memory",
    "agent.infrastructure.memory.logging",
    "posthog",
    "langchain_openai",
    "langchain_core",
    "langchain",
    "httpx",
    "httpcore",
    "mcp.client.streamable_http",
    "mcp.client.sse",
)


class _CliLogHandler(logging.Handler):
    """Log handler that renders warnings in yellow and keeps noise off the console."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            name = record.name or ""
            silenced_prefixes = (
                "mem0",
                "agent.infrastructure.memory",
                "posthog",
                "httpx",
                "httpcore",
                "langchain",
                "mcp.client",
                "agent.infrastructure.mcp",
            )
            if name.startswith(silenced_prefixes):
                return
            msg = record.getMessage()
            noise_keywords = (
                "PostHog",
                "http_socket_options",
                "Mem0",
                "mem0",
                "MEMORY_ADD",
                "LLM extraction failed",
                "invocation cancelled",
                "invocation canceled",
            )
            if any(k in msg for k in noise_keywords):
                return
            if record.levelno == logging.WARNING:
                console.print(f"[yellow]{msg}[/yellow]")
            elif record.levelno >= logging.ERROR:
                console.print(f"[red]{msg}[/red]")
        except Exception:
            pass


def silence_background_loggers() -> None:
    """Silence noisy background loggers and set clean environment defaults."""
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")

    for name in _SILENCED_LOGGERS:
        log = logging.getLogger(name)
        log.handlers.clear()
        log.addHandler(logging.NullHandler())
        log.propagate = False

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*PostHog.*")
    warnings.filterwarnings("ignore", message=".*http_socket_options.*")

    def _format_warning_yellow(
        message: Any,
        category: type,
        filename: str,
        lineno: int,
        file: Any = None,
        line: Any = None,
    ) -> None:
        msg = str(message)
        if "PostHog" in msg or "http_socket_options" in msg or "transport" in msg:
            return
        if issubclass(category, (DeprecationWarning, PendingDeprecationWarning, ImportWarning)):
            return
        console.print(f"[yellow]Warning: {msg}[/yellow]")

    warnings.showwarning = _format_warning_yellow

    root = logging.getLogger()
    if not any(isinstance(h, _CliLogHandler) for h in root.handlers):
        root.addHandler(_CliLogHandler())


silence_background_loggers()


class SafeStatus:
    """Rich status display that never replaces process-wide stdout/stderr."""

    def __init__(self, message: str, *, target_console: Console | None = None) -> None:
        self._lease_acquired = False
        self._started = False
        self._live = Live(
            Spinner("dots", message, style="status.spinner"),
            console=target_console or console,
            refresh_per_second=12.5,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )

    def start(self) -> None:
        if self._started:
            return
        if not LIVE_RENDER_LOCK.acquire(blocking=False):
            raise LiveError("Another CLI live display is active")
        self._lease_acquired = True
        try:
            self._live.start()
            self._started = True
        except BaseException:
            try:
                if getattr(self._live.console, "_live", None) is self._live:
                    self._live.stop()
            except BaseException:
                pass
            finally:
                self._started = False
                self._release_lease()
            raise

    def stop(self) -> None:
        if not self._started:
            self._release_lease()
            return
        try:
            self._live.stop()
        finally:
            self._started = False
            self._release_lease()

    def _release_lease(self) -> None:
        if self._lease_acquired:
            self._lease_acquired = False
            LIVE_RENDER_LOCK.release()


def make_status(message: str, *, target_console: Console | None = None) -> SafeStatus:
    """Create a status renderer safe to use beside prompt_toolkit."""
    return SafeStatus(message, target_console=target_console)


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
    "Type / for commands, /model to switch, Shift+Tab to toggle mode",
    "Use Shift+Tab to toggle permission mode (default, dont_ask, bypass)",
    "Use /model to switch LLM provider or model",
    "Use /resume to switch sessions",
    "Use /compact to reduce context size",
    "Use /sessions to list conversation history",
]


def get_clean_cwd() -> str:
    """Get clean current working directory with $HOME replaced by ~."""
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        return "~" + cwd[len(home) :]
    return cwd


def get_current_model() -> str:
    """Get currently configured model display name from settings.json."""
    try:
        from agent.infrastructure.model import ModelCatalog
        from agent.infrastructure.settings import AgentRuntimeSettings

        catalog = ModelCatalog.from_root(AgentRuntimeSettings().workspace_start)
        return catalog.current_profile().name
    except Exception:
        return "glm-5.3-flash"


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
    permission_mode: str | None = None,
) -> None:
    """Print minimal startup banner and Codex-style tip."""
    model = model_name or get_current_model()
    directory = get_clean_cwd()

    content = Text()
    if session_title or session_id:
        s_text = session_title or "Untitled"
        sid_short = f" ({session_id[:8]})" if session_id else ""
        content.append("session:   ", style=DIM)
        content.append(f"{s_text}{sid_short}\n", style=f"bold {CYAN_BRIGHT}")
    content.append("model:     ", style=DIM)
    content.append(f"{model}\n", style=WHITE)
    content.append("directory: ", style=DIM)
    content.append(f"{directory}\n\n", style=FG_MUTED)

    content.append("Tip: ", style=f"bold {WHITE}")
    content.append("Type ", style=DIM)
    content.append("/", style=CYAN)
    content.append(" for commands, ", style=DIM)
    content.append("/model", style=CYAN)
    content.append(" to switch, ", style=DIM)
    content.append("Shift+Tab", style=CYAN)
    content.append(" to toggle mode", style=DIM)

    from cli import __version__

    panel_title = (
        f"[{DIM}]>_[/{DIM}] "
        f"[bold {WHITE}]AgentDesk[/bold {WHITE}] "
        f"[{DIM}](v{__version__})[/{DIM}]"
    )
    panel = Panel(
        content,
        title=panel_title,
        title_align="left",
        border_style=BORDER,
        box=box.ROUNDED,
        padding=(1, 2),
        expand=False,
    )
    console.print(panel)
    if health is not None and health.get("status") != "ok":
        console.print(f"[{RED}]● Runtime status: {health.get('status')} ({base_url})[/{RED}]")
    elif health is None and base_url:
        console.print(f"[{YELLOW}]● Runtime failed to start ({base_url})[/{YELLOW}]")
    console.print()


def print_user_prompt(text: str) -> None:
    """Render user prompt."""
    console.print(f"\n[bold {CYAN}]You[/bold {CYAN}]  [{WHITE}]{escape(text)}[/{WHITE}]")


def print_error_message(error: str) -> None:
    """Render error message."""
    console.print(f"[{RED}]Error:[/{RED}] [{RED}]{escape(str(error))}[/{RED}]\n")


def print_health(health: dict[str, Any], base_url: str) -> None:
    """Render the embedded runtime's health status."""
    status = health.get("status") or "unknown"
    if status == "ok":
        body = f"[{GREEN}]Runtime online[/{GREEN}] [{DIM}]({base_url})[/{DIM}]"
    else:
        body = f"[{RED}]Runtime unhealthy[/{RED}] [{DIM}]status={status} ({base_url})[/{DIM}]"
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
        box=box.SIMPLE_HEAD,
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
        kind_tag = (
            f"[{AMBER}]subagent[/{AMBER}]" if s.get("main_session_id") else f"[{CYAN}]main[/{CYAN}]"
        )
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
        box=box.SIMPLE_HEAD,
        show_header=False,
        title_justify="left",
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
        box=box.SIMPLE_HEAD,
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
        err = m.get("error")
        if err:
            first_err = err.split("\n")[0].strip()
            if len(first_err) > 80:
                first_err = first_err[:78] + "…"
            desc = f"[{RED}]Error: {first_err}[/{RED}]"
        else:
            desc = m.get("description") or ""

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
        box=box.SIMPLE_HEAD,
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


def print_memories_table(memories: list[dict[str, Any]]) -> None:
    """Render long-term memories catalog table."""
    if not memories:
        console.print(
            f"[{DIM}](No memories stored yet. "
            f"The agent records memories automatically via 'remember')[/{DIM}]\n"
        )
        return

    table = Table(
        title=f"[bold {WHITE}]Long-Term Memories[/bold {WHITE}]",
        border_style=BORDER,
        box=box.SIMPLE_HEAD,
        title_justify="left",
        expand=True,
    )
    table.add_column("Layer", justify="center", width=11)
    table.add_column("Title", style=f"bold {WHITE}", min_width=20)
    table.add_column("Description", style=FG_MUTED)

    for m in memories:
        layer = str(m.get("layer") or "").lower()
        if layer == "user":
            layer_tag = f"[{CYAN}]user[/{CYAN}]"
        elif layer == "project":
            layer_tag = f"[{AMBER}]project[/{AMBER}]"
        else:
            layer_tag = f"[{DIM}]{layer}[/{DIM}]"

        title = str(m.get("title") or "")
        description = str(m.get("description") or "")
        table.add_row(layer_tag, title, description)

    console.print(table)
    console.print(
        f"[{DIM}]Tip: /memory <title> to view note; /memory user or project to filter.[/{DIM}]\n"
    )


def print_memory_note(layer: str, title: str, content: str) -> None:
    """Render a single memory note in a panel."""
    layer_style = CYAN if layer.lower() == "user" else AMBER
    title_text = (
        f"[{layer_style} bold]{layer}[/{layer_style} bold] · "
        f"[bold {WHITE}]{title}[/bold {WHITE}]"
    )
    console.print(
        Panel(
            content.strip(),
            title=title_text,
            border_style=BORDER,
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print()


def print_models_table(
    models: list[dict[str, Any]], current_model_id: str | None = None
) -> None:
    """Render configured models table."""
    if not models:
        console.print(f"[{DIM}](No models configured)[/{DIM}]\n")
        return

    table = Table(
        title=f"[bold {WHITE}]Configured Models[/bold {WHITE}]",
        border_style=BORDER,
        box=box.SIMPLE_HEAD,
        title_justify="left",
        expand=True,
    )
    table.add_column("#", style=DIM, width=4, justify="right")
    table.add_column("Model ID", style=f"bold {CYAN}", width=24)
    table.add_column("Name", style=WHITE, width=28)
    table.add_column("Status", justify="center", width=12)

    for idx, m in enumerate(models, 1):
        mid = m.get("id") or m.get("model_id") or ""
        name = m.get("name") or mid
        is_active = mid == current_model_id or (current_model_id is None and idx == 1)
        status_tag = (
            f"[bold {GREEN}]● Active[/bold {GREEN}]" if is_active else f"[{DIM}]-[/{DIM}]"
        )
        table.add_row(str(idx), mid, name, status_tag)

    console.print(table)
    console.print()


def message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    """Safely extract metadata dictionary from a message dictionary or JSON string."""
    meta = message.get("metadata")
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str) and meta.strip():
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def is_summary_text(text: Any) -> bool:
    """Check whether text is a context compaction summary.

    Summaries are generated internally by ContextCompactor using the policy in
    agent.prompt.summary. They are structured with standard section headings:
    GOALS, DECISIONS, FACTS, TOOL_ACTIVITY, FAILURES_AND_UNCERTAINTY, PENDING_WORK.
    """
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if s.startswith("<conversation_summary>"):
        return True
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if not lines:
        return False
    first = lines[0].lstrip("#* \t").rstrip(":* \t").upper()
    if first == "GOALS":
        summary_headings = {
            "DECISIONS",
            "FACTS",
            "TOOL_ACTIVITY",
            "FAILURES_AND_UNCERTAINTY",
            "PENDING_WORK",
        }
        if any(h in s for h in summary_headings) or (
            len(lines) > 1 and lines[1].startswith(("-", "*", "•"))
        ):
            return True
    return False


def split_summary_prefix(text: str) -> tuple[str | None, str]:
    """If text begins with a context compaction summary, split into (summary, remainder)."""
    if not isinstance(text, str) or not text.strip():
        return None, text
    s = text.lstrip()
    if not is_summary_text(s):
        return None, text
    summary_headings = {
        "GOALS",
        "DECISIONS",
        "FACTS",
        "TOOL_ACTIVITY",
        "FAILURES_AND_UNCERTAINTY",
        "PENDING_WORK",
    }
    lines = text.splitlines(keepends=True)
    summary_line_idx = 0
    in_summary = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = stripped.lstrip("#* \t").rstrip(":* \t").upper()
        if cleaned in summary_headings:
            in_summary = True
            summary_line_idx = i + 1
        elif in_summary and stripped.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
            summary_line_idx = i + 1
        else:
            if in_summary:
                break
    if summary_line_idx > 0:
        summary_part = "".join(lines[:summary_line_idx]).rstrip()
        remainder_part = "".join(lines[summary_line_idx:]).lstrip("\n")
        return summary_part, remainder_part
    return None, text


def print_history_list(
    messages: list[dict[str, Any]],
    limit: int | None = None,
    *,
    show_subagent_details: bool = False,
) -> None:
    """Render conversation history, keeping subagent reports compact by default."""
    if not messages:
        return

    valid_messages = [
        m
        for m in messages
        if m.get("role") != "system"
        and message_metadata(m).get("kind") not in ("summary", "maintenance", "system")
        and not is_summary_text(str(m.get("content") or ""))
    ]
    conversational_indexes = [
        index
        for index, message in enumerate(valid_messages)
        if message.get("role") in ("human", "assistant")
    ]
    if not conversational_indexes:
        console.print(f"[{DIM}](No conversational messages in this session)[/{DIM}]\n")
        return

    if limit is not None and len(conversational_indexes) > limit:
        hidden = len(conversational_indexes) - limit
        console.print(f"[{DIM}]... ({hidden} earlier messages hidden)[/{DIM}]\n")
        to_show = valid_messages[conversational_indexes[-limit] :]
    else:
        to_show = valid_messages

    index = 0
    rendered_count = 0
    while index < len(to_show):
        msg = to_show[index]
        role = msg.get("role")
        if role == "tool":
            # Tool results are rendered only with their owning assistant call.
            index += 1
            continue

        next_index = index + 1
        tool_results: list[dict[str, Any]] = []
        if role == "assistant" and msg.get("tool_calls"):
            while next_index < len(to_show) and to_show[next_index].get("role") == "tool":
                tool_results.append(to_show[next_index])
                next_index += 1

        is_last = role == "human" and not any(
            later.get("role") == "assistant" for later in to_show[next_index:]
        )
        # A separator starts each new user/subagent turn without adding a
        # horizontal rule between the question and its answer.
        if rendered_count and role == "human":
            console.print(Rule(style=DIM_DARK))
        print_message(
            msg,
            is_last_in_history=is_last,
            show_subagent_details=show_subagent_details,
            leading_newline=rendered_count > 0,
            tool_results=tool_results,
        )
        rendered_count += 1
        index = next_index
    console.print()


def _compact_markdown_preview(content: Any) -> str:
    """Render Markdown syntax away before building a width-aware summary."""
    max_len = max(60, min(160, console.width - 8))
    output = io.StringIO()
    preview_console = Console(
        file=output,
        color_system=None,
        force_terminal=False,
        width=max(320, max_len * 2),
    )
    preview_console.print(Markdown(str(content), code_theme="one-dark"))
    preview = " ".join(output.getvalue().split())
    if len(preview) > max_len:
        preview = preview[: max_len - 3].rstrip() + "..."
    return preview or "(empty report)"


def _tool_call_details(call: Any) -> tuple[str, str, Any]:
    """Normalize persisted and OpenAI-shaped tool calls for history rendering."""
    if isinstance(call, dict):
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        call_id = str(call.get("id") or "")
        name = str(call.get("name") or function.get("name") or "")
        arguments = call.get("arguments", function.get("arguments", {}))
        return call_id, name, arguments
    return (
        str(getattr(call, "id", "") or ""),
        str(getattr(call, "name", "") or ""),
        getattr(call, "arguments", {}),
    )


def _take_tool_result(
    remaining: list[dict[str, Any]],
    *,
    call_id: str,
    tool_name: str,
) -> dict[str, Any] | None:
    """Remove the persisted result belonging to one displayed tool call."""
    for index, result in enumerate(remaining):
        if call_id and str(result.get("tool_call_id") or "") == call_id:
            return remaining.pop(index)
    for index, result in enumerate(remaining):
        if tool_name and str(result.get("tool_name") or "") == tool_name:
            return remaining.pop(index)
    return remaining.pop(0) if remaining else None


def print_message(
    message: dict[str, Any],
    is_last_in_history: bool = False,
    *,
    show_subagent_details: bool = True,
    leading_newline: bool = True,
    tool_results: list[dict[str, Any]] | None = None,
) -> None:
    """Render one complete message from history."""
    role = message.get("role")
    content = message.get("content", "")
    metadata = message_metadata(message)
    kind = metadata.get("kind")

    if (
        role in ("system", "tool")
        or kind in ("summary", "maintenance", "system")
        or is_summary_text(str(content or ""))
    ):
        return

    if role == "human":
        if metadata.get("source") == "agent":
            sender = metadata.get("name") or "subagent"
            prefix = "\n" if leading_newline else ""
            if not show_subagent_details:
                preview = _compact_markdown_preview(content)
                console.print(f"{prefix}[bold {AMBER}]Subagent · {escape(sender)}[/bold {AMBER}]")
                console.print(f"  [{FG_MUTED}]└─ {escape(preview)}[/{FG_MUTED}]")
                return

            console.print(f"{prefix}[bold {AMBER}]Subagent · {escape(sender)}[/bold {AMBER}]")
            right_pad = 2
            console.print(Padding(Markdown(content, code_theme="one-dark"), (0, right_pad, 0, 0)))
        else:
            prefix = "\n" if leading_newline else ""
            console.print(f"{prefix}[bold {CYAN}]You[/bold {CYAN}]  [{FG}]{escape(content)}[/{FG}]")
            if is_last_in_history:
                console.print(f"  [{DIM}](No response recorded for this question)[/{DIM}]")
        return

    if role == "assistant":
        tool_calls = message.get("tool_calls") or []
        has_content = bool(content and str(content).strip())
        if not has_content and not tool_calls:
            return

        prefix = "\n" if leading_newline else ""
        model = metadata.get("model") or message.get("model")
        model_label = f" [dim]({model})[/]" if model else ""
        console.print(f"{prefix}[bold {GREEN}]AgentDesk[/]{model_label}: ")
        if has_content:
            right_pad = 2
            console.print(
                Padding(Markdown(str(content), code_theme="one-dark"), (0, right_pad, 0, 0))
            )

        remaining_results = list(tool_results or [])
        rendered_tool = False
        for call in tool_calls:
            call_id, name, arguments = _tool_call_details(call)
            if not name:
                continue
            if name == "ask_user":
                gap = "\n" if has_content or rendered_tool else ""
                console.print(f"{gap}[{DIM}]  Clarification requested[/{DIM}]")
                result = _take_tool_result(
                    remaining_results,
                    call_id=call_id,
                    tool_name=name,
                )
                if result is not None:
                    print_tool_result_end(str(result.get("content") or ""))
            else:
                print_tool_call_start(
                    name,
                    arguments,
                    leading_gap=has_content or rendered_tool,
                )
                result = _take_tool_result(
                    remaining_results,
                    call_id=call_id,
                    tool_name=name,
                )
                if result is not None:
                    print_tool_result_end(str(result.get("content") or ""))
            rendered_tool = True

        usage = metadata.get("usage") or {}
        if usage and os.environ.get("ANNA_CLI_SHOW_USAGE", "").lower() in {"1", "true", "yes"}:
            console.print(
                f"[{DIM}]tokens: {usage.get('input_tokens', 0):,} in · "
                f"{usage.get('output_tokens', 0):,} out "
                f"· {usage.get('total_tokens', 0):,} total[/{DIM}]"
            )


def format_tool_result_preview(content: str, max_len: int = 120) -> str:
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
        if "error" in parsed and (
            parsed.get("ok") is False or parsed.get("status") in ("error", "failed")
        ):
            result_text = f"Error: {parsed['error']}"
        # 1.5 Process non-zero exit code (e.g. bash / scripts)
        elif "exit_code" in parsed and parsed.get("exit_code") not in (0, None):
            code = parsed["exit_code"]
            stderr = str(parsed.get("stderr") or "").strip()
            stdout = str(parsed.get("stdout") or "").strip()
            result_text = f"exit {code}: {stderr or stdout or 'failed'}"
        # 2. Python stdout / stderr
        elif "stdout" in parsed or "stderr" in parsed:
            stdout = str(parsed.get("stdout") or "").strip()
            stderr = str(parsed.get("stderr") or "").strip()
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
                    result_text = (
                        val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
                    )
                    break
        # 4. Collection / search result fields
        elif "tools" in parsed and isinstance(parsed["tools"], list):
            items = [
                x.get("name", str(x)) if isinstance(x, dict) else str(x) for x in parsed["tools"]
            ]
            result_text = ", ".join(items[:4])
            if len(items) > 4:
                result_text += f" (+{len(items) - 4} more)"
        elif "memories" in parsed and isinstance(parsed["memories"], list):
            items = [
                x.get("content", str(x)) if isinstance(x, dict) else str(x)
                for x in parsed["memories"]
            ]
            result_text = "; ".join(items[:2])
        # 5. Generic dictionary: strip metadata envelope keys (ok, status, code, etc.)
        else:
            filtered = {
                k: v
                for k, v in parsed.items()
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
            result_text += f" (+{len(parsed) - 4} more)"
    else:
        result_text = raw

    cleaned = " ".join(result_text.split())
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3] + "..."
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
            box=box.MINIMAL,
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
            box=box.MINIMAL,
            padding=(0, 1),
        )
        console.print(panel)
        return

    if name == "define_subagent":
        sub_name = parsed_args.get("name", "subagent")
        role = parsed_args.get("role", "")
        console.print(
            f"  [bold {CYAN}]define_subagent:[/bold {CYAN}] "
            f"[bold {WHITE}]{sub_name}[/bold {WHITE}] [{DIM}]({role})[/{DIM}]"
        )
        return

    arg_summary = _format_arguments(parsed_args or args)
    console.print(f"  [bold {CYAN}]Calling tool: {name}[/bold {CYAN}] [{DIM}]{arg_summary}[/{DIM}]")


def print_tool_call_start(name: str, args: Any, *, leading_gap: bool = True) -> None:
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
            if lines:
                preview = lines[0][:40] + ("..." if len(lines) > 1 or len(lines[0]) > 40 else "")
            else:
                preview = "(empty)"
            arg_snippet = f"code: {preview}"
        else:
            arg_snippet = "running script"
    elif name == "bash":
        if isinstance(parsed, dict):
            cmd = str(parsed.get("command") or "").strip()
            if len(cmd) > 55:
                cmd = cmd[:52] + "..."
            arg_snippet = f"$ {cmd}"
    elif name in ("read_file", "edit_file"):
        if isinstance(parsed, dict):
            arg_snippet = str(parsed.get("path") or "")
    elif name == "search_text":
        if isinstance(parsed, dict):
            q = str(parsed.get("query") or "")
            path = str(parsed.get("path") or "")
            target_str = f" in {path}" if path else ""
            arg_snippet = f'"{q[:25]}"{target_str}'
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
            mcp = str(parsed.get("mcp") or "")
            q = str(parsed.get("query") or "")
            arg_snippet = f'{mcp} "{q[:30]}"'
    elif name == "execute_mcp":
        if isinstance(parsed, dict):
            mcp = str(parsed.get("mcp") or "")
            tool = str(parsed.get("tool") or "")
            arg_snippet = f"{mcp}/{tool}"
    elif name == "search_memory":
        if isinstance(parsed, dict):
            q = str(parsed.get("query") or "")
            arg_snippet = f'query="{q[:30]}"'
    elif name in ("load_skill", "search_skill"):
        if isinstance(parsed, dict):
            sname = str(parsed.get("name") or parsed.get("query") or "")
            arg_snippet = f'name="{sname[:30]}"'
    elif name == "wait_for_replies":
        if isinstance(parsed, dict):
            subs = parsed.get("subagents") or []
            arg_snippet = f"subagents={subs}"
    elif name == "list_subagents":
        arg_snippet = ""
    elif parsed:
        arg_snippet = _format_arguments(parsed)

    prefix = "\n" if leading_gap else ""
    if arg_snippet:
        console.print(
            f"{prefix}[bold {AMBER}]● Tool[/bold {AMBER}]  "
            f"[bold {WHITE}]{escape(name)}[/] [{DIM}]({escape(arg_snippet)})[/]"
        )
    else:
        console.print(
            f"{prefix}[bold {AMBER}]● Tool[/bold {AMBER}]  [bold {WHITE}]{escape(name)}[/]"
        )


def print_tool_result_end(content: str) -> None:
    """Render clean tool result summary line."""
    preview = format_tool_result_preview(content)
    if preview:
        console.print(f"  [{DIM}]└─ {escape(preview)}[/{DIM}]")


def print_queued_input(content: str) -> None:
    """Acknowledge that input was queued while a turn is running."""
    preview = content.strip().replace("\n", " ")
    if len(preview) > 60:
        preview = preview[:57] + "..."
    console.print(f"[{CYAN}]↳ queued[/{CYAN}] [{DIM}]{escape(preview)}[/{DIM}]")


def print_queued_status(state: str) -> None:
    """Explain what happened to a message accepted while a turn was running.

    The submitted line has already been echoed by prompt_toolkit.  Keep this
    status separate from the normal user-message renderer so a queue update
    cannot print the same question a second time.
    """
    if state == "promoted":
        detail = "saved and added to the conversation."
    elif state == "external":
        detail = "saved; following the active runtime run."
    elif state == "dropped":
        detail = "saved; waiting for a future tool update."
    else:
        detail = "saved; waiting for the next tool to finish."
    console.print(f"[{CYAN}]↳ queued[/{CYAN}] [{DIM}]{detail}[/{DIM}]")


def print_queued_promotion(
    count: int = 1,
    questions: Sequence[str] | list[str] | None = None,
) -> None:
    """Announce that queued messages are now visible to the active context and join conversation."""
    safe_count = max(1, int(count))
    noun = "message" if safe_count == 1 else "messages"
    console.print(
        f"[{GREEN}]✓ context updated[/{GREEN}] [{DIM}]{safe_count} queued {noun} added.[/{DIM}]"
    )
    if questions:
        for q in questions:
            clean = str(q).strip()
            if clean:
                print_user_prompt(clean)


def print_context_compacted() -> None:
    """Announce that background context compaction has generated a continuation summary."""
    console.print(f"[{GREEN}]✓ context compacted[/{GREEN}]")


def print_queued_deferred(count: int = 1) -> None:
    """Explain that a completed turn did not reach a tool boundary for queued input."""
    safe_count = max(1, int(count))
    noun = "message" if safe_count == 1 else "messages"
    console.print(
        f"[{CYAN}]↳ queued[/{CYAN}] "
        f"[{DIM}]{safe_count} {noun} remain saved for the next turn.[/{DIM}]"
    )


def print_queue_watch_stopped(reason: str | None = None) -> None:
    """Make loss of observable runtime activity explicit without dropping persistence state."""
    detail = f"{reason} " if reason else ""
    console.print(
        f"[{AMBER}]● runtime activity unavailable[/{AMBER}] "
        f"[{DIM}]{escape(detail)}queued messages remain saved; "
        f"/resume refreshes persisted history.[/{DIM}]"
    )


def print_clarification_question(
    question: str,
    options: list[Any] | None = None,
    recommended: Any = None,
    *,
    answer: str | None = None,
) -> None:
    """Render an ask_user question together with its available choices."""
    renderables: list[Any] = [Markdown(question, code_theme="one-dark")]
    clean_options = [str(option) for option in (options or []) if str(option).strip()]
    clean_answer = str(answer).strip() if answer is not None else None

    if clean_options:
        choice_text = Text("\n")
        selected_found = False
        items: list[tuple[str, str]] = []

        for index, option in enumerate(clean_options, 1):
            suffix = (
                " (recommended)" if recommended is not None and option == str(recommended) else ""
            )
            if clean_answer is not None and option == clean_answer:
                selected_found = True
                items.append((f"› {index}. {option}{suffix}", f"bold {CYAN}"))
            elif clean_answer is not None:
                items.append((f"  {index}. {option}{suffix}", FG_MUTED))
            else:
                items.append((f"{index}. {option}{suffix}", FG_MUTED))

        if clean_answer is not None:
            if not selected_found:
                items.append(
                    (f"› {len(clean_options) + 1}. {clean_answer}", f"bold {CYAN}")
                )
        else:
            items.append((f"{len(clean_options) + 1}. Type your own answer...", FG_MUTED))

        for i, (line, style) in enumerate(items):
            if i > 0:
                choice_text.append("\n")
            choice_text.append(line, style=style)

        renderables.append(choice_text)
    elif clean_answer:
        renderables.append(Text(f"\n› {clean_answer}", style=f"bold {CYAN}"))

    console.print(
        Panel(
            Group(*renderables),
            title=f"[bold {CYAN}]Clarification[/bold {CYAN}]",
            title_align="left",
            border_style=BORDER_FOCUS,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_permission_request(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    targets: Sequence[str] = (),
) -> None:
    """Render a one-time tool permission request for the interactive prompt."""
    args = arguments if isinstance(arguments, dict) else {}
    if not targets:
        always_label = f"Always allow tool '{tool_name}'"
    elif len(targets) == 1:
        target_preview = targets[0]
        if len(target_preview) > 60:
            target_preview = target_preview[:57] + "..."
        always_label = f"Always allow '{target_preview}'"
    else:
        segments_preview = " && ".join(targets)
        if len(segments_preview) > 60:
            segments_preview = segments_preview[:57] + "..."
        always_label = f"Always allow all: {segments_preview}"

    renderables: list[Any] = [
        Text(f"Allow {tool_name} to run?", style=f"bold {WHITE}"),
        Text(
            f"\n1. Allow once\n2. {always_label}\n3. Reject",
            style=FG_MUTED,
        ),
    ]
    code = args.get("code")
    if isinstance(code, str) and code.strip():
        preview = code.strip()
        if len(preview) > 1_200:
            preview = preview[:1_197].rstrip() + "..."
        renderables.insert(
            1,
            Panel(
                Syntax(preview, "python", theme="one-dark", line_numbers=False),
                title=f"[{DIM}]code[/{DIM}]",
                title_align="left",
                border_style=BORDER,
                box=box.MINIMAL,
                padding=(0, 1),
            ),
        )
    console.print(
        Panel(
            Group(*renderables),
            title=f"[bold {AMBER}]Permission required[/bold {AMBER}]",
            title_align="left",
            border_style=AMBER,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _render_tool_content(tool_name: str, content: str) -> Any:
    """Render tool result content."""
    content = str(content or "")
    stripped = content.strip()
    parsed = None
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped, strict=False)
        except Exception:
            parsed = None

    if tool_name == "execute_python" and isinstance(parsed, dict):
        stdout = str(parsed.get("stdout") or "")
        stderr = str(parsed.get("stderr") or "")
        ok = parsed.get("ok", True)
        exit_code = parsed.get("exit_code", 0)

        parts: list[Any] = []
        if stdout:
            title = (
                f"[{GREEN}]output (exit 0)[/{GREEN}]"
                if ok
                else f"[{RED}]output (exit {exit_code})[/{RED}]"
            )
            parts.append(
                Panel(
                    Syntax(stdout.rstrip(), "text", theme="one-dark", padding=(0, 1)),
                    title=title,
                    title_align="left",
                    border_style=BORDER,
                    box=box.MINIMAL,
                    padding=(0, 1),
                )
            )
        if stderr:
            parts.append(
                Panel(
                    Text(f"{stderr.rstrip()}", style=RED),
                    title=f"[{RED}]stderr[/{RED}]",
                    title_align="left",
                    border_style=RED,
                    box=box.MINIMAL,
                    padding=(0, 1),
                )
            )
        if not stdout and not stderr:
            tag = (
                f"[{GREEN}]Success (exit 0)[/{GREEN}]"
                if ok
                else f"[{RED}]Failed (exit {exit_code})[/{RED}]"
            )
            parts.append(Text.from_markup(f"  {tag}"))
        return Group(*parts) if parts else Text("")

    if tool_name == "send_message" and isinstance(parsed, dict):
        recipient = str(parsed.get("recipient") or "agent")
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
        console.print(
            f"[{DIM}]Context compaction skipped: empty or invalid response from server[/{DIM}]\n"
        )
        return

    status = result.get("status")
    if status == "compacted":
        console.print(f"[{GREEN}]Context compacted.[/{GREEN}]\n")
    else:
        console.print(f"[{DIM}]Context unchanged.[/{DIM}]\n")


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
