"""Main application: clean & minimal interactive terminal CLI for AgentDesk."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import threading
import time
from collections import deque
from typing import Any
from uuid import uuid4

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition, is_done, to_filter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import ConditionalContainer, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
from rich.errors import LiveError

import cli.ui as ui
from cli.commands import COMMAND_MAP
from cli.completer import SlashCompleter
from cli.ports import AgentApiError, AgentClientPort
from cli.streamer import MarkdownStreamer
from cli.theme import (
    AMBER,
    BG_CARD,
    BG_SELECTION,
    CYAN,
    CYAN_BRIGHT,
    DIM,
    FG,
    FG_MUTED,
    GREEN,
    RED,
    WHITE,
)

logger = logging.getLogger(__name__)

CLI_STYLE = Style.from_dict(
    {
        "prompt": f"bold {CYAN}",
        "prompt-session": f"{CYAN_BRIGHT}",
        "prompt-bracket": f"{DIM}",
        "placeholder": f"{DIM}",
        "activity": f"{FG_MUTED}",
        "activity-running": f"{CYAN}",
        "activity-queued": f"{CYAN}",
        "activity-bullet": f"bold {WHITE}",
        "activity-name": f"bold {WHITE}",
        "activity-detail": f"{FG_MUTED}",
        "toolbar-model": "bold #cbd5e1",
        "toolbar-cwd": f"{FG_MUTED}",
        "toolbar-mode-default": f"{FG_MUTED}",
        "toolbar-mode-dont_ask": f"bold {AMBER}",
        "toolbar-mode-bypass": f"bold {RED}",
        "bottom-toolbar": "noreverse",
        "bottom-toolbar.text": f"noreverse {DIM}",
        "permission-title": f"bold {AMBER}",
        "permission-tool": f"bold {WHITE}",
        "permission-detail": f"{FG_MUTED}",
        "permission-choice": f"{FG_MUTED}",
        "permission-choice-selected": f"bold {CYAN_BRIGHT}",
        "permission-submitting": f"{CYAN}",
        "completion-menu": f"bg:{BG_CARD} {FG_MUTED}",
        "completion-menu.completion": f"{FG}",
        "completion-menu.completion.current": f"bg:{BG_SELECTION} bold {WHITE}",
        "completion-menu.meta": f"bg:{BG_CARD} {DIM}",
        "completion-menu.meta.current": f"bg:{BG_SELECTION} {CYAN}",
    }
)

_QUEUE_WATCH_INTERVAL_SECONDS = 1.0
_QUEUE_WATCH_IDLE_TIMEOUT_SECONDS = 900.0
_QUEUE_WATCH_MAX_POLL_FAILURES = 3
_PROMPT_REFRESH_INTERVAL_SECONDS = 0.1
_THINKING_FRAME_INTERVAL_SECONDS = 0.1
_COMPLETION_MENU_RESERVED_ROWS = 8
_THINKING_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_PERMISSION_CHOICES = (
    ("once", "Allow once"),
    ("always", "Always allow"),
    ("reject", "Reject"),
)


def _permission_choices_for_request(
    request: dict[str, Any] | None,
) -> list[tuple[str, str, list[str] | None]]:
    if not request:
        return [
            ("once", "Allow once", None),
            ("always", "Always allow", [""]),
            ("reject", "Reject", None),
        ]
    payload = dict(request.get("payload") or {})
    tool_name = " ".join(str(payload.get("name") or "tool").split())
    raw_targets = payload.get("targets")
    targets = tuple(str(t) for t in raw_targets) if raw_targets else ()

    if not targets:
        always_label = f"Always allow tool '{tool_name}'"
        always_scope: list[str] | None = [""]
    elif len(targets) == 1:
        target_preview = targets[0]
        if len(target_preview) > 60:
            target_preview = target_preview[:57] + "..."
        always_label = f"Always allow '{target_preview}'"
        always_scope = [targets[0]]
    else:
        segments_preview = " && ".join(targets)
        if len(segments_preview) > 60:
            segments_preview = segments_preview[:57] + "..."
        always_label = f"Always allow all: {segments_preview}"
        always_scope = list(targets)

    return [
        ("once", "Allow once", None),
        ("always", always_label, always_scope),
        ("reject", "Reject", None),
    ]


def generate_fork_title(parent_title: str, existing_titles: set[str]) -> str:
    """Generate title for a forked session."""
    clean = parent_title.strip()
    match = re.match(r"^(.*?)(?:\s+\((?:fork\s+)?(\d+)\))?$", clean)
    base = match.group(1).strip() if match else clean
    if not base:
        base = "Session"

    counter = 1
    candidate = f"{base} (fork {counter})"
    while candidate in existing_titles:
        counter += 1
        candidate = f"{base} (fork {counter})"
    return candidate


def make_title_from_question(question: str) -> str:
    """Generate a clean, readable session title directly from the user question."""
    first_line = ""
    for line in question.strip().splitlines():
        line = line.strip()
        if line and not line.startswith("```"):
            first_line = line
            break

    if not first_line:
        first_line = question.strip()

    clean = " ".join(first_line.split()).lstrip(">#*- \t")
    if not clean:
        return "Chat"

    # Truncate at around 12-15 characters
    if len(clean) > 15:
        clean = clean[:14].rstrip() + "…"
    return clean


def is_slash_command(user_input: str) -> tuple[bool, str, str]:
    """Parse slash input into (is_command, cmd_name, cmd_arg).

    Distinguishes slash commands (e.g. /help, /model) from filesystem paths
    (e.g. /home/user/file.png, /etc/hosts, /tmp) and natural text.
    """
    clean = user_input.strip()
    if not clean.startswith("/"):
        return False, "", ""
    if clean == "/":
        return True, "", ""

    parts = clean[1:].split(maxsplit=1)
    candidate = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    # 1. Registered slash commands are always commands
    if candidate.lower() in COMMAND_MAP:
        return True, candidate.lower(), arg

    # 2. Check if the token is a path or non-command text:
    # 2a. Contains path separators (e.g. /home/anna/..., /a/b.py)
    if "/" in candidate:
        return False, "", ""

    first_token = clean.split()[0]
    # 2b. Existing path on filesystem (e.g. /tmp, /etc, /var, /home)
    try:
        if os.path.exists(os.path.expanduser(first_token)):
            return False, "", ""
    except Exception:
        pass

    # 2c. Has file extension (e.g. /script.py, /image.png, /data.json)
    if re.search(r"\.[a-zA-Z0-9_-]+$", first_token):
        return False, "", ""

    # 2d. Contains non-command characters (CJK/Chinese, punctuation, symbols)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", candidate):
        return False, "", ""

    # Otherwise, it looks like an unknown/misspelled slash command (e.g. /helpp)
    return True, candidate.lower(), arg


class AnnaCliApp:
    """AgentDesk interactive terminal CLI application."""

    def __init__(self) -> None:
        ui.silence_background_loggers()
        try:
            from cli.local_client import LocalApiClient
        except ImportError as exc:
            # CLI 需要完整后端栈(alembic/dishka/langchain...),这些只装在
            # 项目 venv;conda/系统 python 给不出,早停并给出可执行指引。
            raise SystemExit(
                f"agentdesk needs the project virtualenv ({exc}).\n"
                "Run: uv run agentdesk  (or .venv/bin/agentdesk)"
            ) from exc

        self.client: AgentClientPort = LocalApiClient()
        self.base_url = "embedded runtime"
        self.current_session: dict[str, Any] | None = None
        # 本窗口当前持有的主会话签出句柄(驾驶权)。
        self._held_session_id: str | None = None
        self._pending_title: str | None = None
        self._pending_tools: list[str] | None = None
        self.running = True
        self.model_name = ui.get_current_model()
        self.model_id: str | None = None
        try:
            from agent.infrastructure.model import ModelCatalog
            from agent.infrastructure.settings import AgentRuntimeSettings

            catalog = ModelCatalog.from_root(AgentRuntimeSettings().workspace_start)
            self.model_id = catalog.current_id
        except Exception:
            self.model_id = None
        self._health: dict[str, Any] | None = None

        self._last_interrupt_time: float = 0.0
        self._turn_start_time: float = 0.0
        # The interactive prompt must remain usable while an Agent turn is
        # streaming.  Each worker owns one request; the set is only used for
        # lifecycle/status bookkeeping and is protected because workers finish
        # outside the prompt thread.
        self._turn_threads: set[threading.Thread] = set()
        self._turn_threads_lock = threading.RLock()
        self._permission_reply_threads: set[threading.Thread] = set()
        self._request_context = threading.local()
        self._prompt_thread_id = threading.get_ident()
        self._ask_user_lock = threading.Lock()
        self._pending_ask_users: deque[dict[str, Any]] = deque()
        self._shutdown_event = threading.Event()
        # Queue and stream state share one re-entrant lock.  A queue response
        # can race the terminal event of the active stream, so checking the
        # two structures under separate locks would leave a small lost-update
        # window.
        self._turn_state_lock = threading.RLock()
        self._queued_turns: dict[str, list[dict[str, Any]]] = {}
        self._queued_turns_lock = self._turn_state_lock
        self._stream_states: dict[str, dict[str, Any]] = {}
        self._stream_states_lock = self._turn_state_lock
        self._streaming_requests: set[str] = set()
        self._queue_watch_owners: dict[str, str] = {}
        self._render_lock = threading.Lock()
        self._prompt_draft = ""
        self._ask_user_custom_draft = ""
        self._permission_mode: str = "default"

        # Keybindings: standard Ctrl+C & Esc line clearing
        kb = KeyBindings()

        @kb.add("escape")
        def _on_escape(event: Any) -> None:
            buf = event.current_buffer
            if self._ask_user_custom_draft:
                self._ask_user_custom_draft = ""
                event.app.invalidate()
            elif buf.text:
                buf.text = ""
                buf.cursor_position = 0
            elif self._has_pending_ask_user_request():
                self._skip_pending_ask_user()
                event.app.invalidate()
            elif self._has_pending_permission_request():
                event.app.invalidate()
            elif self.active_turn_count > 0:
                self.cancel_active_turns()
                ui.console.print("\n[yellow]> Request interrupted[/yellow]\n")
                event.app.invalidate()

        @kb.add("escape", "enter")
        def _on_alt_enter(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _on_ctrl_c(event: Any) -> None:
            buf = event.current_buffer
            if self._ask_user_custom_draft:
                self._ask_user_custom_draft = ""
                event.app.invalidate()
            elif buf.text:
                buf.text = ""
                buf.cursor_position = 0
            elif self._has_pending_ask_users():
                self._skip_pending_ask_user()
                event.app.invalidate()
            else:
                event.app.exit(exception=KeyboardInterrupt())

        @kb.add("s-tab")
        def _on_shift_tab(event: Any) -> None:
            self.cycle_permission_mode()
            event.app.invalidate()

        @kb.add(Keys.Any, filter=Condition(self._should_hide_prompt_input))
        def _on_ask_user_key(event: Any) -> None:
            char = event.data
            if not char or not char.isprintable():
                return
            with self._ask_user_lock:
                request = self._pending_picker_locked()
                if request is None:
                    return
                if request.get("kind") == "permission":
                    choices = _permission_choices_for_request(request)
                    if char.isdigit() and 1 <= int(char) <= len(choices):
                        request["selected_index"] = int(char) - 1
                        event.app.invalidate()
                    elif char.lower() in {"y", "o"}:
                        request["selected_index"] = 0
                        event.app.invalidate()
                    elif char.lower() in {"a", "p"}:
                        request["selected_index"] = 1
                        event.app.invalidate()
                    elif char.lower() in {"n", "r"}:
                        request["selected_index"] = 2
                        event.app.invalidate()
                    return
                if request.get("kind") != "ask_user":
                    return
                options = (request.get("payload") or {}).get("options") or []
                if not (isinstance(options, list) and len(options) >= 2):
                    return
                total = len(options) + 1
                selected = int(request.get("selected_index") or 0) % total
                is_custom = selected == len(options)

                if is_custom:
                    self._ask_user_custom_draft += char
                    event.app.invalidate()
                    return

                if char.isdigit() and 1 <= int(char) <= total:
                    target_index = int(char) - 1
                    request["selected_index"] = target_index
                    event.app.invalidate()
                    return

                # Any other printable character jumps to custom option and starts typing into it
                request["selected_index"] = len(options)
                self._ask_user_custom_draft += char
                event.app.invalidate()

        @kb.add("backspace", filter=Condition(self._should_hide_prompt_input))
        @kb.add("c-h", filter=Condition(self._should_hide_prompt_input))
        def _on_ask_user_backspace(event: Any) -> None:
            with self._ask_user_lock:
                request = self._pending_picker_locked()
                if request is None or request.get("kind") != "ask_user":
                    return
                options = (request.get("payload") or {}).get("options") or []
                total = len(options) + 1
                selected = int(request.get("selected_index") or 0) % total
                if selected == len(options) and self._ask_user_custom_draft:
                    self._ask_user_custom_draft = self._ask_user_custom_draft[:-1]
                    event.app.invalidate()

        @kb.add("c-u", filter=Condition(self._should_hide_prompt_input))
        def _on_ask_user_clear(event: Any) -> None:
            with self._ask_user_lock:
                request = self._pending_picker_locked()
                if request is None or request.get("kind") != "ask_user":
                    return
                options = (request.get("payload") or {}).get("options") or []
                total = len(options) + 1
                selected = int(request.get("selected_index") or 0) % total
                if selected == len(options) and self._ask_user_custom_draft:
                    self._ask_user_custom_draft = ""
                    event.app.invalidate()

        @kb.add("up", filter=Condition(self._picker_active))
        def _on_picker_up(event: Any) -> None:
            self._move_picker_selection(-1)
            event.app.invalidate()

        @kb.add("down", filter=Condition(self._picker_active))
        def _on_picker_down(event: Any) -> None:
            self._move_picker_selection(1)
            event.app.invalidate()

        @kb.add("enter", filter=Condition(self._picker_input_owned))
        def _on_picker_enter(event: Any) -> None:
            buf = event.current_buffer
            if self._permission_enter_event(buf):
                return
            if not self._picker_active():
                event.app.invalidate()
                return

            with self._ask_user_lock:
                request = self._pending_picker_locked()
                if request is not None and request.get("kind") == "ask_user":
                    options = (request.get("payload") or {}).get("options") or []
                    total = len(options) + 1
                    selected = int(request.get("selected_index") or 0) % total
                    if selected < len(options):
                        choice = str(options[selected])
                        buf.text = choice
                        buf.cursor_position = len(choice)
                        buf.validate_and_handle()
                        return
                    draft = self._ask_user_custom_draft.strip()
                    if draft:
                        self._ask_user_custom_draft = ""
                        buf.text = draft
                        buf.cursor_position = len(draft)
                        buf.validate_and_handle()
                    return

            if not buf.text.strip():
                choice = self._picker_selected_value()
                if choice is None:
                    return
                buf.text = choice
                buf.cursor_position = len(choice)
            buf.validate_and_handle()

        @kb.add("enter", filter=Condition(lambda: not self._picker_input_owned()))
        def _on_enter(event: Any) -> None:
            buf = event.current_buffer
            if not buf.text.strip():
                return
            buf.validate_and_handle()

        history_path = os.path.expanduser("~/.anna_cli_history")
        self.session_prompt = PromptSession(
            history=FileHistory(history_path),
            completer=SlashCompleter(
                get_sessions_fn=self._safe_list_sessions,
                get_mcps_fn=self._safe_list_mcps,
                get_current_session_fn=lambda: self.current_session,
                get_models_fn=self._safe_list_models,
            ),
            key_bindings=kb,
            style=CLI_STYLE,
            rprompt=self.get_activity_text,
            placeholder=self.get_placeholder_text,
            # Dynamic session suggestions use REST, so run completion away
            # from the prompt thread while keeping the menu automatic.
            complete_while_typing=True,
            complete_in_thread=True,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=_COMPLETION_MENU_RESERVED_ROWS,
            refresh_interval=_PROMPT_REFRESH_INTERVAL_SECONDS,
            erase_when_done=True,
        )
        # PromptSession normally stretches its input window when a bottom
        # toolbar is present. Keep this REPL compact so permission choices sit
        # immediately under the editable row.
        self.session_prompt.layout.current_window.dont_extend_height = to_filter(True)
        self.session_prompt.layout.current_window.always_hide_cursor = Condition(
            self._should_hide_prompt_input
        )

        def _get_buffer_control_height() -> Dimension:
            buff = self.session_prompt.default_buffer
            if (
                buff.complete_state is not None
                and buff.complete_state.completions
            ):
                count = len(buff.complete_state.completions)
                max_space = (
                    self.session_prompt.reserve_space_for_menu
                    or _COMPLETION_MENU_RESERVED_ROWS
                )
                return Dimension(min=min(count + 1, max_space))
            return Dimension()

        self.session_prompt.layout.current_window.height = _get_buffer_control_height

        alt = self.session_prompt.layout.container.children[0].alternative_content
        status_toolbar_window = ConditionalContainer(
            Window(
                FormattedTextControl(self.get_status_toolbar),
                style="class:bottom-toolbar",
                dont_extend_height=True,
                height=2,
            ),
            filter=~is_done & Condition(lambda: bool(self.get_status_toolbar())),
        )
        alt.content.children.append(status_toolbar_window)

    def _safe_list_sessions(self) -> list[dict[str, Any]]:
        """Safely fetch sessions list."""
        try:
            return self.client.list_sessions(timeout=3.0)
        except TypeError as exc:
            # Keep compatibility with small embedders/test doubles that still
            # expose the pre-timeout method signature.
            if "timeout" not in str(exc):
                return []
            try:
                return self.client.list_sessions()
            except Exception:
                return []
        except Exception:
            return []

    def _safe_list_models(self) -> list[dict[str, Any]]:
        """Safely fetch models list."""
        try:
            return self.client.list_models()
        except Exception:
            return []

    def _safe_health(self) -> dict[str, Any] | None:
        """Probe the embedded runtime on startup."""
        try:
            return self.client.health(timeout=5.0)
        except TypeError as exc:
            if "timeout" not in str(exc):
                return None
            try:
                return self.client.health()
            except Exception:
                return None
    def _safe_list_mcps(self) -> list[dict[str, Any]]:
        """Safely fetch MCP servers list."""
        try:
            return self.client.list_mcps()
        except Exception:
            return []

    def _print_failed_mcps_hint(self) -> None:
        """Render a helpful, non-intrusive reconnect tip if any MCP server failed during startup."""
        mcps = self._safe_list_mcps()
        failed = [m.get("id") for m in mcps if m.get("state") == "failed" and m.get("id")]
        if not failed:
            return

        if len(failed) == 1:
            server_id = failed[0]
            ui.console.print(
                f"[yellow]Tip: MCP server '{server_id}' failed to connect. "
                "Type /mcp to view or reconnect.[/yellow]\n"
            )
        elif len(failed) <= 3:
            servers_str = ", ".join(f"'{s}'" for s in failed)
            ui.console.print(
                f"[yellow]Tip: {len(failed)} MCP servers ({servers_str}) failed to connect. "
                "Type /mcp to view or reconnect.[/yellow]\n"
            )
        else:
            servers_str = ", ".join(f"'{s}'" for s in failed[:3])
            rem = len(failed) - 3
            ui.console.print(
                f"[yellow]Tip: {len(failed)} MCP servers ({servers_str}, and {rem} more) "
                "failed to connect. Type /mcp to view or reconnect.[/yellow]\n"
            )

    def print_header(self) -> None:
        """Print startup information banner."""
        title = (
            self.current_session.get("title")
            if self.current_session
            else (self._pending_title or None)
        )
        sid = self.current_session.get("id") if self.current_session else None
        ui.print_banner(
            base_url=self.base_url,
            session_title=title,
            session_id=sid,
            health=self._health,
            model_name=self.model_name,
        )

    def reset_to_new_session(
        self,
        pending_title: str | None = None,
        pending_tools: list[str] | None = None,
    ) -> None:
        """Reset to a clean pending new session state without creating a DB record yet."""
        if self.active_turn_count or self._has_live_waiters():
            ui.console.print(
                "[yellow]Cannot switch sessions while a turn is running. "
                "Wait for the current response to finish.[/yellow]\n"
            )
            return
        with self._queued_turns_lock:
            self._queued_turns.clear()
            self._stream_states.clear()
        with self._ask_user_lock:
            self._pending_ask_users = deque(
                p for p in self._pending_ask_users if not p.get("standalone")
            )
            self._ask_user_custom_draft = ""
        self._refresh_permission_toolbar()
        if self._held_session_id:
            self.client.release(self._held_session_id)
            self._held_session_id = None
        self.current_session = None
        self._pending_title = pending_title
        self._pending_tools = pending_tools

    def auto_init_session(self) -> None:
        """Start with a clean initial page; session is created on the first question."""
        self.reset_to_new_session()
        try:
            current = self.client.get_current_model()
            if current and current.get("id"):
                self.model_id = current.get("id")
                if current.get("name"):
                    self.model_name = current.get("name")
        except Exception:
            pass

    def select_model(self, model_id: str) -> None:
        """Switch active model for the embedded runtime."""
        if self.active_turn_count or self._has_live_waiters():
            ui.console.print(
                "[yellow]Cannot switch model while a turn is running. "
                "Wait for the current response to finish.[/yellow]\n"
            )
            return
        try:
            result = self.client.select_model(model_id)
            self.model_id = result.get("id") or model_id
            self.model_name = result.get("name") or self.model_id
            ui.console.print(
                f"[green]Switched model to:[/green] [white]{self.model_name}[/white] "
                f"[dim]({self.model_id})[/dim]\n"
            )
            self._invalidate_prompt()
        except Exception as exc:
            ui.print_error_message(f"Failed to switch model: {exc}")

    def open_session(self, session: dict[str, Any] | None, is_new: bool = False) -> None:
        """Switch sessions and render history with compact subagent reports."""
        if self.active_turn_count or self._has_live_waiters():
            ui.console.print(
                "[yellow]Cannot switch sessions while a turn is running. "
                "Wait for the current response to finish.[/yellow]\n"
            )
            return
        if not session or not session.get("id"):
            ui.print_error_message("无法打开会话：会话不存在或未成功创建。")
            return

        # 驾驶权签出：只有主会话需要独占。子会话只能经已签出主会话下的
        # /subagents 到达，保护沿父子关系传递，无需单独签出。
        if session.get("main_session_id") is None and session["id"] != self._held_session_id:
            try:
                self.client.checkout(session["id"])
            except AgentApiError as exc:
                ui.print_error_message(f"Cannot open session: {exc}")
                return
            if self._held_session_id:
                self.client.release(self._held_session_id)
            self._held_session_id = session["id"]

        previous_session_id = self.current_session.get("id") if self.current_session else None
        if previous_session_id and previous_session_id != session["id"]:
            # Queued entries are only local presentation state.  Once the
            # user leaves a session, keeping them would make an unrelated
            # session show stale counts when the prompt is redrawn.
            self._discard_queued_turns(previous_session_id)

        with self._ask_user_lock:
            self._pending_ask_users = deque(
                p for p in self._pending_ask_users if not p.get("standalone")
            )
            self._ask_user_custom_draft = ""
        self._refresh_permission_toolbar()

        self.current_session = session
        self._pending_title = None
        self._pending_tools = None
        title = session.get("title") or "Untitled"
        sid = (session.get("id") or "")[:8]
        if is_new:
            ui.console.print(
                f"[green]Created session:[/green] [white]{title}[/white] [dim]({sid})[/dim]\n"
            )
        else:
            ui.console.print(
                f"[green]Switched to session:[/green] [white]{title}[/white] [dim]({sid})[/dim]\n"
            )

        if is_new:
            return

        try:
            messages = self.client.list_messages(session["id"])
            ui.print_history_list(
                messages,
                limit=None,
                show_subagent_details=False,
            )
        except Exception as exc:
            ui.print_error_message(f"Failed to load session history: {exc}")
        # History is authoritative for a reopened session.  Drop all local
        # presentation-only entries, including ones that never reached a
        # locally observed boundary.
        self._discard_queued_turns(session["id"])

    def rewind_and_resume(self) -> None:
        """Select a past turn in the current session and fork a new session."""
        if self.active_turn_count or self._has_live_waiters():
            ui.console.print(
                "[yellow]Cannot rewind while a turn is running. "
                "Wait for the current response to finish.[/yellow]\n"
            )
            return
        if not self.current_session:
            ui.console.print("[dim]No active session.[/dim]\n")
            return

        try:
            messages = self.client.list_messages(self.current_session["id"])
        except Exception as exc:
            ui.print_error_message(f"Failed to read session messages: {exc}")
            return

        user_asks = [
            m
            for m in messages
            if m.get("role") == "human" and ui.message_metadata(m).get("source") != "agent"
        ]
        if not user_asks:
            ui.console.print("[dim](No user question history found in this session)[/dim]\n")
            return

        from cli.picker import HistoryMessageResumer

        resumer = HistoryMessageResumer(messages)
        selected = resumer.run()
        if not selected:
            return

        seq = selected["seq"]
        all_sessions = []
        try:
            all_sessions = self.client.list_sessions()
        except Exception:
            pass
        existing_titles = {s.get("title") for s in all_sessions if s.get("title")}
        parent_title = self.current_session.get("title") or "Session"
        new_title = generate_fork_title(parent_title, existing_titles)

        try:
            status: Any = None
            try:
                status = ui.make_status(f"[cyan]Forking new session from step {seq}...[/cyan]")
                status.start()
            except LiveError:
                status = None
            try:
                new_session = self.client.resume_session(
                    session_id=self.current_session["id"],
                    title=new_title,
                    parent_last_seq=seq,
                )
            finally:
                if status is not None:
                    try:
                        status.stop()
                    except Exception:
                        pass
            self.open_session(new_session)
            ui.console.print(
                f"[green]Forked new session from step {seq}:[/green] [white]{new_title}[/white]\n"
            )
        except Exception as exc:
            ui.print_error_message(f"Failed to fork session: {exc}")

    def _get_unique_title(self, base_title: str, exclude_id: str | None = None) -> str:
        """Ensure title is unique among existing sessions."""
        all_sessions = self._safe_list_sessions()
        existing = {s.get("title") for s in all_sessions if s.get("id") != exclude_id}
        if base_title not in existing:
            return base_title
        counter = 2
        while f"{base_title} ({counter})" in existing:
            counter += 1
        return f"{base_title} ({counter})"

    def ask_agent(self, question: str, print_prompt: bool = False) -> None:
        """Send question to Agent and render response."""
        clean_q = question.strip()
        if not clean_q:
            return

        bound_session_id = getattr(self._request_context, "session_id", None)
        bound_to_worker = getattr(self._request_context, "bound", False)
        if not bound_to_worker:
            if not self._ensure_session(clean_q):
                return

            # A direct command call uses the current session.  Background
            # workers set a bound identity below and never overwrite it.
            self._request_context.session_id = self.current_session["id"]
            self._request_context.request_id = uuid4().hex
            bound_session_id = self.current_session["id"]
        elif not bound_session_id:
            raise AgentApiError("Background turn has no bound session")

        should_print = print_prompt
        current_question: str | None = question

        while current_question:
            if should_print:
                ui.print_user_prompt(current_question)

            try:
                ask_user_payload = self._ask_agent_stream(current_question)
            except AgentApiError as exc:
                msg = str(exc)
                if "No streaming chunk received for" in msg:
                    msg = (
                        "Upstream model provider timed out (streaming silence cap "
                        "exceeded). Please retry."
                    )
                ui.print_error_message(f"Request failed: {msg}")
                break
            except Exception as exc:
                ui.print_error_message(f"Error occurred: {exc}")
                break

            if not ask_user_payload:
                break

            user_choice = self._prompt_ask_user(ask_user_payload)
            if not user_choice:
                break

            current_question = user_choice
            should_print = True

    def _ensure_session(self, question: str) -> bool:
        """Create the lazy session once and report failures to the user."""
        if self.current_session:
            return True

        candidate_title = self._pending_title or make_title_from_question(question)
        try:
            unique_title = self._get_unique_title(candidate_title)
            self.current_session = self.client.create_session(
                title=unique_title,
                allowed_tools=self._pending_tools,
            )
            self.client.checkout(self.current_session["id"])
            self._held_session_id = self.current_session["id"]
            self._pending_title = None
            self._pending_tools = None
            return True
        except Exception as exc:
            ui.print_error_message(f"Failed to create session: {exc}")
            return False

    def _mark_queued_turn(self, session_id: str, request_id: str, question: str) -> str:
        """Track an accepted input until the active stream reaches a boundary.

        The runtime persists queued input immediately, but does not associate a
        later assistant answer with that request.  This state is presentation
        only; it must never be used to attribute another turn's answer.

        A 202 acknowledgement is the first point where this CLI knows the
        message was persisted.  It deliberately does not infer consumption
        from tool results that arrived before that acknowledgement.
        """
        with self._queued_turns_lock:
            if self._shutdown_event.is_set():
                return "dropped"
            stream = self._stream_states.get(request_id)
            if stream is not None:
                stream["state"] = "queued"
            has_local_stream = any(
                stream_id != request_id
                and item.get("session_id") == session_id
                for stream_id, item in self._stream_states.items()
            ) or self.active_turn_count > 0
            queue_state = "waiting" if has_local_stream else "external"
            for queued_item in self._queued_turns.get(session_id, []):
                if queued_item.get("request_id") == request_id:
                    return queue_state
            self._queued_turns.setdefault(session_id, []).append(
                {
                    "request_id": request_id,
                    "question": question,
                    "message_seq": None,
                }
            )
            for stream_id, item in self._stream_states.items():
                if stream_id != request_id and item.get("session_id") == session_id:
                    item.setdefault("queued_questions", []).append(question)
        self._invalidate_prompt()
        return queue_state

    def _claim_queue_watch(self, session_id: str, request_id: str) -> bool:
        """Allow one queued worker to follow a session whose live stream is elsewhere."""
        with self._turn_state_lock:
            if session_id in self._queue_watch_owners:
                return False
            self._queue_watch_owners[session_id] = request_id
            return True

    def _release_queue_watch(self, session_id: str, request_id: str) -> None:
        with self._turn_state_lock:
            if self._queue_watch_owners.get(session_id) == request_id:
                self._queue_watch_owners.pop(session_id, None)

    @staticmethod
    def _message_seq(message: dict[str, Any]) -> int | None:
        value = message.get("seq")
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _bind_queued_message_seqs(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Match 202 acknowledgements to their already-persisted human rows."""
        candidates: dict[str, list[int]] = {}
        for message in messages:
            metadata = ui.message_metadata(message)
            seq = self._message_seq(message)
            if seq is None or message.get("role") != "human" or metadata.get("source") != "user":
                continue
            content = str(message.get("content") or "").strip()
            candidates.setdefault(content, []).append(seq)
        for seqs in candidates.values():
            seqs.sort(reverse=True)

        with self._turn_state_lock:
            queued = self._queued_turns.get(session_id, [])
            used = {
                item.get("message_seq")
                for item in queued
                if isinstance(item.get("message_seq"), int)
            }
            for item in reversed(queued):
                if isinstance(item.get("message_seq"), int):
                    continue
                question = str(item.get("question") or "").strip()
                match = next(
                    (seq for seq in candidates.get(question, []) if seq not in used),
                    None,
                )
                if match is not None:
                    item["message_seq"] = match
                    used.add(match)

    def _queued_message_seq(self, session_id: str, request_id: str) -> int | None:
        with self._turn_state_lock:
            for item in self._queued_turns.get(session_id, []):
                if item.get("request_id") == request_id:
                    seq = item.get("message_seq")
                    return seq if isinstance(seq, int) else None
        return None

    def _promote_queued_through(
        self,
        session_id: str,
        boundary_seq: int,
    ) -> list[dict[str, Any]]:
        """Retire only queued rows that precede a persisted tool boundary."""
        with self._turn_state_lock:
            queued = self._queued_turns.get(session_id, [])
            promoted = [
                item
                for item in queued
                if isinstance(item.get("message_seq"), int) and item["message_seq"] < boundary_seq
            ]
            if not promoted:
                return []
            promoted_ids = {item["request_id"] for item in promoted}
            remaining = [item for item in queued if item.get("request_id") not in promoted_ids]
            if remaining:
                self._queued_turns[session_id] = remaining
            else:
                self._queued_turns.pop(session_id, None)
        self._invalidate_prompt()
        return promoted

    def _watch_external_queue(self, session_id: str, request_id: str) -> None:
        """Follow persisted progress when another worker owns the live stream."""
        if not self._claim_queue_watch(session_id, request_id):
            return

        cursor: int | None = None
        last_activity = time.monotonic()
        consecutive_poll_failures = 0
        try:
            while not self._shutdown_event.is_set():
                try:
                    messages = self.client.list_messages(session_id, timeout=5.0)
                except Exception:
                    consecutive_poll_failures += 1
                    if consecutive_poll_failures >= _QUEUE_WATCH_MAX_POLL_FAILURES:
                        ui.print_queue_watch_stopped(
                            f"History refresh failed {consecutive_poll_failures} times."
                        )
                        return
                    messages = []
                else:
                    consecutive_poll_failures = 0

                if messages:
                    self._bind_queued_message_seqs(session_id, messages)
                    if cursor is None:
                        cursor = self._queued_message_seq(session_id, request_id)

                new_messages = []
                if cursor is not None:
                    new_messages = sorted(
                        (
                            message
                            for message in messages
                            if (self._message_seq(message) or -1) > cursor
                        ),
                        key=lambda message: self._message_seq(message) or -1,
                    )

                terminal_seen = False
                if new_messages:
                    with self._render_lock:
                        for message in new_messages:
                            seq = self._message_seq(message)
                            if seq is None:
                                continue
                            cursor = max(cursor or seq, seq)
                            role = message.get("role")
                            metadata = ui.message_metadata(message)
                            kind = metadata.get("kind")
                            content = str(message.get("content") or "")
                            if role == "tool":
                                ui.print_tool_result_end(str(message.get("content") or ""))
                                if message.get("tool_name") != "ask_user":
                                    promoted = self._promote_queued_through(session_id, seq)
                                    if promoted:
                                        questions = [
                                            str(p.get("question") or "").strip()
                                            for p in promoted
                                            if p.get("question")
                                        ]
                                        ui.print_queued_promotion(
                                            len(promoted),
                                            questions=questions,
                                        )
                            elif role == "human" and metadata.get("source") == "agent":
                                ui.print_message(
                                    message,
                                    show_subagent_details=False,
                                )
                            elif role == "assistant":
                                if kind == "summary" or ui.is_summary_text(content):
                                    ui.print_context_compacted()
                                else:
                                    ui.print_message(
                                        message,
                                        show_subagent_details=False,
                                    )
                                    terminal_seen = terminal_seen or kind == "assistant_answer"
                    last_activity = time.monotonic()
                    self._invalidate_prompt()

                if terminal_seen:
                    if self.queued_turn_count:
                        ui.print_queued_deferred(self.queued_turn_count)
                    return

                if time.monotonic() - last_activity >= _QUEUE_WATCH_IDLE_TIMEOUT_SECONDS:
                    idle_minutes = max(1, round(_QUEUE_WATCH_IDLE_TIMEOUT_SECONDS / 60))
                    ui.print_queue_watch_stopped(
                        f"No new persisted activity was observed for {idle_minutes} minutes."
                    )
                    return
                self._shutdown_event.wait(_QUEUE_WATCH_INTERVAL_SECONDS)
        finally:
            self._release_queue_watch(session_id, request_id)

    def _discard_queued_turns(self, session_id: str | None = None) -> None:
        """Drop local queued presentation state without rendering it."""
        with self._queued_turns_lock:
            if session_id is None:
                removed = bool(self._queued_turns)
                self._queued_turns.clear()
            else:
                removed = self._queued_turns.pop(session_id, None) is not None
        if removed:
            self._invalidate_prompt()

    def _retire_queued_turn(
        self,
        session_id: str,
        *,
        request_id: str | None = None,
        question: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retire a specific queued turn when it executes or is cancelled."""
        with self._queued_turns_lock:
            turns = self._queued_turns.get(session_id, [])
            if not turns:
                return []
            retired: list[dict[str, Any]] = []
            remaining: list[dict[str, Any]] = []
            for item in turns:
                match = (
                    (request_id and item.get("request_id") == request_id)
                    or (question and item.get("question") == question)
                )
                if match and not retired:
                    retired.append(item)
                else:
                    remaining.append(item)
            if retired:
                if remaining:
                    self._queued_turns[session_id] = remaining
                else:
                    self._queued_turns.pop(session_id, None)
                self._invalidate_prompt()
            return retired

    def _register_stream(self, session_id: str, request_id: str) -> None:
        with self._stream_states_lock:
            # ask_user follow-ups reuse their worker request ID, but each stream
            # unregisters before the next one starts.
            if request_id in self._stream_states:
                return
            self._stream_states[request_id] = {
                "session_id": session_id,
                "state": "pending",
                "activity": "Working...",
            }

    def _set_stream_state(self, request_id: str, state: str) -> None:
        with self._stream_states_lock:
            item = self._stream_states.get(request_id)
            if item is not None:
                item["state"] = state

    def _set_stream_text_streaming(self, request_id: str, streaming: bool) -> None:
        with self._turn_state_lock:
            if streaming:
                changed = request_id not in self._streaming_requests
                self._streaming_requests.add(request_id)
            else:
                changed = request_id in self._streaming_requests
                self._streaming_requests.discard(request_id)
        if changed:
            self._invalidate_prompt()

    @property
    def is_streaming_text(self) -> bool:
        """Return whether any stream is actively streaming response text."""
        with self._turn_state_lock:
            return bool(self._streaming_requests)

    def _set_stream_activity(self, request_id: str, activity: str | None) -> None:
        with self._stream_states_lock:
            item = self._stream_states.get(request_id)
            if item is not None:
                item["activity"] = activity
        self._invalidate_prompt()

    @property
    def current_activity(self) -> str | None:
        """Return the most specific active activity description across running streams."""
        if self.is_streaming_text:
            return "Generating..."
        with self._stream_states_lock:
            for item in self._stream_states.values():
                act = item.get("activity")
                if act:
                    return str(act)
        if self.active_turn_count > 0:
            return "Working..."
        return None

    def _unregister_stream(self, session_id: str, request_id: str) -> None:
        with self._stream_states_lock:
            removed = self._stream_states.pop(request_id, None) is not None
            changed = request_id in self._streaming_requests
            self._streaming_requests.discard(request_id)
        if removed or changed:
            self._invalidate_prompt()

    def cancel_active_turns(self) -> None:
        """Cancel all running agent turns on the backend and abort client streams."""
        with self._stream_states_lock:
            sessions = {
                state["session_id"]
                for state in self._stream_states.values()
                if state.get("session_id")
            }
        if not sessions and self.current_session and self.current_session.get("id"):
            sessions.add(self.current_session["id"])

        for session_id in sessions:
            try:
                self.client.cancel(session_id)
            except Exception:
                pass
            self._discard_queued_turns(session_id)
        self.abort_active_streams()

    def abort_active_streams(self) -> None:
        """Abort all running streams."""
        with self._stream_states_lock:
            req_ids = list(self._stream_states.keys())
        for req_id in req_ids:
            try:
                self.client.abort_stream(req_id)
            except Exception:
                pass

    def _fail_stream_state(self, session_id: str, request_id: str) -> None:
        """Record a transport failure without changing persisted queue markers."""
        with self._stream_states_lock:
            failed = self._stream_states.get(request_id)
            if failed is not None and failed.get("session_id") == session_id:
                if failed.get("state") not in {"terminal", "failed"}:
                    failed["state"] = "failed"

    def _mark_stream_boundary(
        self,
        session_id: str,
        request_id: str,
        *,
        terminal: bool = False,
        promote_queued: bool = True,
    ) -> list[dict[str, Any]]:
        """Record a stream boundary and promote only on ordinary tool results."""
        promoted: list[dict[str, Any]] = []
        with self._stream_states_lock:
            item = self._stream_states.get(request_id)
            if item is None or item.get("state") == "queued":
                return []
            item["state"] = "terminal" if terminal else "active"
            # ask_user and message_end do not prove that the workflow re-read
            # persisted input.  Only an ordinary tool_result is the queue
            # presentation boundary exposed by the current stream contract.
            if terminal or not promote_queued:
                return []
            promoted = self._queued_turns.pop(session_id, [])
        if promoted:
            self._invalidate_prompt()
        return promoted

    def _invalidate_prompt(self) -> None:
        """Ask prompt_toolkit to redraw its callable prompt when state changes."""
        try:
            prompt_app = self.session_prompt.app
            if prompt_app is not None:
                prompt_app.invalidate()
        except Exception:
            # PromptSession has no active Application during unit tests and
            # before ``run`` starts; state updates must remain best effort.
            pass

    def _pending_permission_locked(self) -> dict[str, Any] | None:
        """Return the permission currently owning the main input."""
        if not self._pending_ask_users:
            return None
        request = self._pending_ask_users[0]
        if request.get("cancelled") or request.get("kind") != "permission":
            return None
        return request

    def _pending_picker_locked(self) -> dict[str, Any] | None:
        """Oldest waiter that owns arrow keys and Enter (permission or ask_user)."""
        if not self._pending_ask_users:
            return None
        request = self._pending_ask_users[0]
        if request.get("cancelled"):
            return None
        kind = request.get("kind")
        if kind == "permission":
            return request
        if kind == "ask_user":
            options = (request.get("payload") or {}).get("options")
            if isinstance(options, list) and len(options) >= 2:
                return request
        return None

    def _picker_active(self) -> bool:
        """Whether arrow keys control the highlighted choice of the head waiter."""
        with self._ask_user_lock:
            request = self._pending_picker_locked()
            return request is not None and request.get("state") == "pending"

    def _picker_input_owned(self) -> bool:
        """Whether a picker waiter currently owns Enter on the main input."""
        with self._ask_user_lock:
            return self._pending_picker_locked() is not None

    def _permission_enter_event(self, buf: Any) -> bool:
        """Permission-specific Enter handling; returns True when consumed."""
        with self._ask_user_lock:
            request = self._pending_permission_locked()
            is_permission = request is not None
        if not is_permission:
            return False
        raw = buf.text.strip().lower()
        permission_answers = {
            "1",
            "2",
            "3",
            "y",
            "yes",
            "p",
            "n",
            "no",
            "allow",
            "once",
            "always",
            "project",
            "allow_project",
            "deny",
            "reject",
        }
        if not self._permission_picker_state():
            return True  # submitting: swallow Enter without validating
        if raw and raw not in permission_answers:
            self._prompt_draft = buf.text
        if not raw or raw not in permission_answers:
            decision = self._selected_permission_decision()
            if decision is not None:
                buf.text = decision
                buf.cursor_position = len(decision)
        buf.validate_and_handle()
        return True

    def _permission_picker_state(self) -> bool:
        with self._ask_user_lock:
            request = self._pending_permission_locked()
            return request is not None and request.get("state") == "pending"

    def _picker_selected_value(self) -> str | None:
        """Highlighted option text of an ask_user waiter."""
        with self._ask_user_lock:
            request = self._pending_picker_locked()
            if request is None or request.get("kind") != "ask_user":
                return None
            return self._picker_choice_locked(request)

    def _picker_choice_count_locked(self, request: dict[str, Any]) -> int:
        if request.get("kind") == "permission":
            return len(_permission_choices_for_request(request))
        options = (request.get("payload") or {}).get("options") or []
        if isinstance(options, list) and len(options) >= 2:
            return len(options) + 1
        return max(1, len(options))

    def _picker_choice_locked(self, request: dict[str, Any]) -> str | None:
        selected = int(request.get("selected_index") or 0)
        if request.get("kind") == "permission":
            choices = _permission_choices_for_request(request)
            return choices[selected % len(choices)][0]
        options = (request.get("payload") or {}).get("options") or []
        if not options:
            return None
        total = len(options) + 1
        index = selected % total
        if index == len(options):
            return None
        return str(options[index])

    def _move_picker_selection(self, offset: int) -> tuple[bool, bool]:
        """Move the highlighted choice of the head waiter by one row."""
        with self._ask_user_lock:
            request = self._pending_picker_locked()
            if request is None or request.get("state") != "pending":
                return (False, False)
            count = self._picker_choice_count_locked(request)
            current = int(request.get("selected_index") or 0)
            new_index = (current + offset) % count
            request["selected_index"] = new_index
            self._invalidate_prompt()
            if request.get("kind") == "ask_user":
                options = (request.get("payload") or {}).get("options") or []
                total = len(options) + 1
                was_custom = (current % total) == len(options)
                now_custom = (new_index % total) == len(options)
                return (was_custom, now_custom)
            return (False, False)

    def _move_permission_selection(self, offset: int) -> None:
        """Move the active permission selection by one row."""
        self._move_picker_selection(offset)

    def _selected_permission_decision(self) -> str | None:
        """Return the decision selected in the permission toolbar."""
        with self._ask_user_lock:
            request = self._pending_permission_locked()
            if request is None or request.get("state") != "pending":
                return None
            choices = _permission_choices_for_request(request)
            selected = int(request.get("selected_index") or 0) % len(choices)
            return choices[selected][0]

    @staticmethod
    def _permission_argument_preview(payload: dict[str, Any]) -> str:
        """Build a compact, single-line preview for an informed decision."""
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict) or not arguments:
            return ""
        code = arguments.get("code")
        if isinstance(code, str) and code.strip():
            lines = [line.strip() for line in code.strip().splitlines() if line.strip()]
            preview = lines[0] if lines else ""
            if len(lines) > 1:
                preview += f"  (+{len(lines) - 1} lines)"
            detail = f"code  {preview}"
        else:
            detail = json.dumps(arguments, ensure_ascii=False, separators=(", ", ": "))
        detail = " ".join(detail.split())
        return detail if len(detail) <= 110 else detail[:107].rstrip() + "..."

    def get_permission_toolbar(self) -> list[tuple[str, str]]:
        """Render the head waiter's chooser (permission or ask_user) below the input."""
        with self._ask_user_lock:
            request = self._pending_picker_locked()
            if request is None:
                return []
            if request.get("kind") == "ask_user":
                return self._ask_user_toolbar_locked(request)
            payload = dict(request.get("payload") or {})
            state = str(request.get("state") or "pending")
            choices = _permission_choices_for_request(request)
            selected = int(request.get("selected_index") or 0) % len(choices)
            decision = str(request.get("decision") or choices[selected][0])

        payload = dict(request.get("payload") or {})
        tool_name = " ".join(str(payload.get("name") or "tool").split())
        fragments: list[tuple[str, str]] = [
            ("class:permission-title", " Permission required: "),
            ("class:permission-tool", f"{tool_name}\n"),
        ]

        if state == "submitting":
            label_map = {item[0]: item[1] for item in choices}
            label = label_map.get(decision, decision)
            fragments.append(("class:permission-submitting", f" Sending {label}..."))
            return fragments

        for index, (_value, label, _scope) in enumerate(choices):
            marker = "›" if index == selected else " "
            style = (
                "class:permission-choice-selected"
                if index == selected
                else "class:permission-choice"
            )
            fragments.append((style, f" {marker} {label}\n"))
        fragments.append((
            "class:permission-detail",
            "\n [↑/↓ select · Enter confirm · 1/2/3 to choose]",
        ))
        return fragments

    def _ask_user_toolbar_locked(self, request: dict[str, Any]) -> list[tuple[str, str]]:
        payload = request.get("payload") or {}
        options = [str(option) for option in (payload.get("options") or [])]
        question = str(payload.get("question") or "Please provide more details.")
        fragments: list[tuple[str, str]] = [
            ("class:permission-title", " Clarification  "),
            ("class:permission-tool", f"{question}\n"),
        ]
        if not options:
            fragments.append(("class:permission-detail", " type your answer + Enter"))
            return fragments
        total = len(options) + 1
        selected = int(request.get("selected_index") or 0) % total
        for index, option in enumerate(options):
            marker = "›" if index == selected else " "
            style = (
                "class:permission-choice-selected"
                if index == selected
                else "class:permission-choice"
            )
            fragments.append((style, f" {marker} {index + 1}. {option}\n"))
        is_custom_selected = selected == len(options)
        custom_marker = "›" if is_custom_selected else " "
        custom_style = (
            "class:permission-choice-selected"
            if is_custom_selected
            else "class:permission-choice"
        )
        draft = self._ask_user_custom_draft
        cursor_str = "█" if is_custom_selected else ""
        if draft:
            fragments.append(
                (
                    custom_style,
                    f" {custom_marker} {len(options) + 1}. {draft}",
                )
            )
            if cursor_str:
                fragments.append(("class:permission-tool", cursor_str))
            fragments.append(("", "\n"))
        else:
            fragments.append(
                (
                    custom_style,
                    f" {custom_marker} {len(options) + 1}. ",
                )
            )
            if cursor_str:
                fragments.append(("class:permission-tool", f"{cursor_str} "))
            fragments.append(
                (
                    "class:permission-detail",
                    "Type your own answer...",
                )
            )
            fragments.append(("", "\n"))

        hint = (
            f"\n↑/↓ select · Enter confirm · Esc or /skip to pause · "
            f"or type 1-{total} / free text"
        )
        fragments.append(("class:permission-detail", hint))
        return fragments

    def _refresh_permission_toolbar(self) -> None:
        """Show the permission toolbar when a picker is active."""
        with self._ask_user_lock:
            visible = self._pending_picker_locked() is not None
        self.session_prompt.bottom_toolbar = (
            self.get_permission_toolbar if visible else None
        )
        self.session_prompt.reserve_space_for_menu = (
            0 if visible else _COMPLETION_MENU_RESERVED_ROWS
        )
        self._invalidate_prompt()

    def get_status_toolbar(self) -> list[tuple[str, str]]:
        """Codex-style status toolbar displaying model and clean cwd directly below prompt."""
        with self._ask_user_lock:
            if self._pending_picker_locked() is not None:
                return []
        try:
            buf = getattr(self.session_prompt, "default_buffer", None)
            if buf is not None:
                buf_text = buf.text.lstrip()
                if buf_text.startswith("/"):
                    return []
                if (
                    buf.complete_state is not None
                    and buf.complete_state.completions
                ):
                    return []
        except Exception:
            pass
        raw_model = (
            getattr(self, "model_name", None)
            or getattr(self, "model_id", None)
            or ""
        )
        model_name = str(raw_model).strip() or "default"
        directory = ui.get_clean_cwd()
        mode = self.permission_mode
        mode_style = (
            "class:toolbar-mode-bypass"
            if mode == "bypass"
            else "class:toolbar-mode-dont_ask"
            if mode == "dont_ask"
            else "class:toolbar-mode-default"
        )
        return [
            ("", "\n"),
            ("class:toolbar-model", f"{model_name}"),
            ("class:bottom-toolbar.text", " · "),
            ("class:toolbar-cwd", f"{directory}"),
            ("class:bottom-toolbar.text", " · "),
            (mode_style, f"[{mode}]"),
        ]

    @property
    def permission_mode(self) -> str:
        mode = getattr(self, "_permission_mode", None)
        if mode:
            return mode
        try:
            fn = getattr(self.client, "get_permission_mode", None)
            if callable(fn):
                res = fn()
                if isinstance(res, str) and res:
                    self._permission_mode = res
                    return res
        except Exception:
            pass
        return "default"

    def cycle_permission_mode(self) -> str:
        """Cycle permission mode: default -> dont_ask -> bypass -> default."""
        modes = ("default", "dont_ask", "bypass")
        current = self.permission_mode
        try:
            idx = modes.index(current)
            next_mode = modes[(idx + 1) % len(modes)]
        except ValueError:
            next_mode = "default"

        try:
            fn = getattr(self.client, "set_permission_mode", None)
            if callable(fn):
                res = fn(next_mode)
                if isinstance(res, str) and res:
                    next_mode = res
        except Exception as exc:
            logger.warning("Failed to update permission mode: %s", exc)

        self._permission_mode = next_mode
        self._invalidate_prompt()
        return next_mode

    def get_queued_questions(self) -> list[str]:
        """Return list of queued questions waiting for context incorporation."""
        with self._queued_turns_lock:
            session_id = (self.current_session or {}).get("id")
            if not session_id:
                items = [item for turns in self._queued_turns.values() for item in turns]
            else:
                items = self._queued_turns.get(session_id, [])
            return [
                str(item.get("question") or "").strip()
                for item in items
                if item.get("question")
            ]

    @property
    def queued_turn_count(self) -> int:
        """Return the number of accepted inputs awaiting presentation."""
        with self._queued_turns_lock:
            if self.current_session and self.current_session.get("id"):
                return len(self._queued_turns.get(self.current_session["id"], []))
            return sum(len(items) for items in self._queued_turns.values())

    def _enqueue_ask_user_request(
        self,
        payload: dict[str, Any],
        *,
        owner_id: str | None = None,
        standalone: bool = False,
        render: bool = False,
    ) -> dict[str, Any]:
        """Reserve a background clarification for the prompt toolbar."""
        request: dict[str, Any] = {
            "kind": "ask_user",
            "payload": payload,
            "event": threading.Event(),
            "answer": None,
            "owner_id": owner_id,
            "standalone": standalone,
            "state": "pending",
            "cancelled": False,
            "rendered": render,
        }
        question = str(payload.get("question") or "Please provide more details.")
        options = payload.get("options")
        recommended = payload.get("recommended")
        if isinstance(options, list) and recommended in options:
            request["selected_index"] = options.index(recommended)
        # Enqueue and render while holding both locks so visible order and
        # answer-routing order cannot diverge across concurrent workers.
        with self._ask_user_lock, self._render_lock:
            if self._shutdown_event.is_set():
                request["event"].set()
                return request
            self._pending_ask_users.append(request)
            if render:
                options = payload.get("options")
                ui.print_clarification_question(
                    question,
                    options=options if isinstance(options, list) else None,
                    recommended=payload.get("recommended"),
                )
        self._refresh_permission_toolbar()
        return request

    def _cancel_ask_user_request(self, request: dict[str, Any]) -> None:
        """Remove an abandoned clarification reservation and wake its worker."""
        removed = False
        with self._ask_user_lock:
            for index, pending in enumerate(self._pending_ask_users):
                if pending is request:
                    del self._pending_ask_users[index]
                    removed = True
                    break
            request["cancelled"] = True
            request["state"] = "cancelled"
            if not request["event"].is_set():
                request["answer"] = None
                request["event"].set()
        if removed:
            self._ask_user_custom_draft = ""
            self._refresh_permission_toolbar()

    def _enqueue_permission_request(
        self,
        payload: dict[str, Any],
        *,
        owner_id: str,
    ) -> dict[str, Any]:
        """Reserve a permission decision in the same FIFO as ask_user input."""
        request: dict[str, Any] = {
            "kind": "permission",
            "payload": payload,
            "event": threading.Event(),
            "answer": None,
            "owner_id": owner_id,
            "state": "pending",
            "cancelled": False,
            "selected_index": 0,
        }
        with self._ask_user_lock:
            if self._shutdown_event.is_set():
                request["event"].set()
                return request
            self._pending_ask_users.append(request)
        self._refresh_permission_toolbar()
        return request

    def _cancel_ask_user_requests_for_owner(
        self,
        owner_id: str,
        *,
        kinds: set[str] | None = None,
    ) -> None:
        """Cancel every clarification reservation owned by a worker stream."""
        removed = False
        with self._ask_user_lock:
            retained: deque[dict[str, Any]] = deque()
            for request in self._pending_ask_users:
                if request.get("owner_id") != owner_id or (
                    kinds is not None and request.get("kind") not in kinds
                ):
                    retained.append(request)
                    continue
                removed = True
                request["cancelled"] = True
                request["state"] = "cancelled"
                request["answer"] = None
                request["event"].set()
            self._pending_ask_users = retained
        if removed:
            self._refresh_permission_toolbar()

    def _remove_pending_request_locked(self, request: dict[str, Any]) -> None:
        """Remove one request by identity while holding ``_ask_user_lock``."""
        self._pending_ask_users = deque(
            pending for pending in self._pending_ask_users if pending is not request
        )

    def _dispatch_permission_reply(
        self,
        request: dict[str, Any],
        decision: str,
    ) -> None:
        """Send a permission acknowledgement without blocking the prompt."""
        with self._ask_user_lock:
            if request.get("cancelled") or request.get("state") != "submitting":
                return

        def deliver() -> None:
            error: Exception | None = None
            try:
                payload = request.get("payload") or {}
                permission_id = str(payload.get("permission_id") or "")
                choices = _permission_choices_for_request(request)

                norm = decision.strip().lower()
                if norm in {"once", "allow", "1", "y", "yes"}:
                    api_decision = "allow"
                    scope = None
                elif norm in {"always", "allow_project", "2", "p", "project"}:
                    api_decision = "allow"
                    always_choice = next((c for c in choices if c[0] == "always"), None)
                    scope = always_choice[2] if always_choice else [""]
                elif norm in {"reject", "3", "n", "no", "deny"}:
                    api_decision = "reject"
                    scope = None
                else:
                    api_decision = norm
                    scope = None

                self.client.reply_permission(
                    permission_id,
                    api_decision,
                    scope=scope,
                )
            except Exception as exc:
                error = exc

            should_report = False
            stale = False
            with self._ask_user_lock:
                active = not request.get("cancelled") and not self._shutdown_event.is_set()
                if active and error is None:
                    self._remove_pending_request_locked(request)
                    request["state"] = "answered"
                    request["answer"] = decision
                    request["event"].set()
                elif active:
                    if isinstance(error, AgentApiError) and error.status_code in {404, 409}:
                        # The broker may have timed out or been answered by a
                        # different client.  This request is terminal and
                        # must not keep intercepting every subsequent input.
                        self._remove_pending_request_locked(request)
                        request["state"] = "cancelled"
                        request["answer"] = None
                        request["event"].set()
                        stale = True
                    else:
                        request["state"] = "pending"
                        if not any(item is request for item in self._pending_ask_users):
                            self._pending_ask_users.appendleft(request)
                        should_report = True
            self._refresh_permission_toolbar()
            if should_report and error is not None:
                ui.print_error_message(f"Permission response failed: {error}")
            elif stale and error is not None:
                ui.console.print(
                    "[yellow]Permission request expired or was already answered.[/yellow]\n"
                )
            with self._turn_threads_lock:
                self._permission_reply_threads.discard(threading.current_thread())

        worker = threading.Thread(
            target=deliver,
            name="agentdesk-permission-reply",
            daemon=True,
        )
        with self._turn_threads_lock:
            self._permission_reply_threads.add(worker)
        try:
            worker.start()
        except Exception as exc:
            with self._turn_threads_lock:
                self._permission_reply_threads.discard(worker)
            with self._ask_user_lock:
                if not request.get("cancelled"):
                    request["state"] = "pending"
            self._refresh_permission_toolbar()
            ui.print_error_message(f"Permission response failed: {exc}")

    def _dispatch_ask_user_reply(
        self,
        request: dict[str, Any],
        answer: str,
    ) -> None:
        """Send an ask_user clarification answer to the backend without blocking the prompt."""
        payload = request.get("payload") or {}
        interruption_id = str(payload.get("interruption_id") or "")
        if not interruption_id:
            with self._ask_user_lock:
                self._remove_pending_request_locked(request)
                request["state"] = "answered"
                request["answer"] = answer
                request["event"].set()
            self._refresh_permission_toolbar()
            return

        def deliver() -> None:
            error: Exception | None = None
            try:
                self.client.reply_ask_user(interruption_id, answer)
            except Exception as exc:
                error = exc

            should_report = False
            stale = False
            with self._ask_user_lock:
                active = not request.get("cancelled") and not self._shutdown_event.is_set()
                if active and error is None:
                    self._remove_pending_request_locked(request)
                    request["state"] = "answered"
                    request["answer"] = answer
                    request["event"].set()
                elif active:
                    if isinstance(error, AgentApiError) and error.status_code in {404, 409}:
                        self._remove_pending_request_locked(request)
                        request["state"] = "cancelled"
                        request["answer"] = None
                        request["event"].set()
                        stale = True
                    else:
                        request["state"] = "pending"
                        if not any(item is request for item in self._pending_ask_users):
                            self._pending_ask_users.appendleft(request)
                        should_report = True
            self._refresh_permission_toolbar()
            if should_report and error is not None:
                ui.print_error_message(f"Clarification response failed: {error}")
            elif stale and error is not None:
                ui.console.print(
                    "[yellow]Clarification request expired or was already answered.[/yellow]\n"
                )
            with self._turn_threads_lock:
                self._permission_reply_threads.discard(threading.current_thread())

        worker = threading.Thread(
            target=deliver,
            name="agentdesk-ask-user-reply",
            daemon=True,
        )
        with self._turn_threads_lock:
            self._permission_reply_threads.add(worker)
        try:
            worker.start()
        except Exception as exc:
            with self._turn_threads_lock:
                self._permission_reply_threads.discard(worker)
            with self._ask_user_lock:
                if not request.get("cancelled"):
                    request["state"] = "pending"
            self._refresh_permission_toolbar()
            ui.print_error_message(f"Clarification response failed: {exc}")

    def _prompt_ask_user(self, payload: dict[str, Any]) -> str | None:
        """Prompt user for multiple-choice selection cleanly without redundant headers.

        Returns:
            The chosen option string, or None if skipped/cancelled.
        """
        reserved = getattr(self._request_context, "pending_ask_user", None)
        options = payload.get("options") or []
        recommended = payload.get("recommended")

        if not options or not isinstance(options, list):
            if reserved is not None:
                self._request_context.pending_ask_user = None
                self._cancel_ask_user_request(reserved)
            return None

        # A background turn cannot open a second prompt_toolkit application.
        # Hand the clarification to the main input loop and wait for its next
        # line instead; the worker then resumes the same ask_user cycle.
        if threading.get_ident() != self._prompt_thread_id:
            request = reserved
            if request is not None:
                self._request_context.pending_ask_user = None
                request["payload"] = payload
            else:
                request = self._enqueue_ask_user_request(payload)
            request["event"].wait()
            return request["answer"]

        # Launch inline choice picker directly without redundant clarification panel
        from cli.picker import AskUserPicker

        picker = AskUserPicker(options=options, recommended=recommended)
        chosen = picker.run()

        # Handle cancellation / skip
        if not chosen:
            ui.console.print(f"[{DIM}](Clarification skipped)[/{DIM}]\n")
            return None

        question = str(payload.get("question") or "Please provide more details.")
        with self._render_lock:
            ui.print_clarification_question(
                question,
                options=options,
                recommended=recommended,
                answer=chosen,
            )

        return chosen

    def _has_pending_ask_user_request(self) -> bool:
        """Whether an ask_user clarification is waiting for user response."""
        with self._ask_user_lock:
            return any(
                p.get("kind") == "ask_user"
                and not p.get("cancelled")
                and p.get("state") == "pending"
                for p in self._pending_ask_users
            )

    def _has_pending_permission_request(self) -> bool:
        """Whether a permission approval is waiting for user response."""
        with self._ask_user_lock:
            return any(
                p.get("kind") == "permission"
                and not p.get("cancelled")
                and p.get("state") == "pending"
                for p in self._pending_ask_users
            )

    def _has_pending_ask_users(self) -> bool:
        """Whether a worker is waiting for an answer in the main prompt."""
        with self._ask_user_lock:
            return bool(self._pending_ask_users)

    def _has_live_waiters(self) -> bool:
        """Return whether an interactive waiter owns the prompt for a live turn."""
        with self._ask_user_lock:
            return any(not p.get("standalone") for p in self._pending_ask_users)

    @staticmethod
    def _find_unanswered_ask_user(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find pending ask_user payload if the session ends with an unanswered clarification."""
        if not messages:
            return None

        seen_ask_user = False
        payload: dict[str, Any] | None = None

        for message in reversed(messages):
            role = message.get("role")
            if role == "human":
                return None

            if role == "tool":
                if message.get("tool_name") == "ask_user":
                    seen_ask_user = True
                    raw = message.get("content")
                    if isinstance(raw, str) and raw.startswith("{"):
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict) and parsed.get("options"):
                                payload = parsed
                        except Exception:
                            pass
                elif not seen_ask_user:
                    return None

            elif role == "assistant":
                tool_calls = message.get("tool_calls") or []
                for call in tool_calls:
                    name = (
                        call.get("name")
                        if isinstance(call, dict)
                        else getattr(call, "name", "")
                    )
                    if name == "ask_user":
                        seen_ask_user = True
                        if not payload:
                            args = (
                                call.get("arguments")
                                if isinstance(call, dict)
                                else getattr(call, "arguments", {})
                            )
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {}
                            if isinstance(args, dict) and args.get("options"):
                                payload = args
                        break
                if seen_ask_user:
                    return payload
                return None

        return payload if seen_ask_user else None

    def _answer_pending_ask_user(self, answer: str) -> bool:
        """Deliver one input line to the oldest clarification or permission waiter."""
        decision: str | None = None
        permission_request: dict[str, Any] | None = None
        ask_user_request: dict[str, Any] | None = None
        ask_user_answer: str | None = None
        standalone_answer: str | None = None
        wait_for_permission = False
        invalid_permission = False
        invalid_ask_user_range: int | None = None
        custom_prompt_needed: int | None = None
        with self._ask_user_lock:
            # Drop cancelled entries that may have been marked while a
            # producer was waking its worker.  The identity check and state
            # transition happen under this same lock, so cleanup cannot make
            # the current input target a different waiter halfway through.
            while self._pending_ask_users and self._pending_ask_users[0].get("cancelled"):
                stale = self._pending_ask_users.popleft()
                stale["state"] = "cancelled"
                stale["answer"] = None
                stale["event"].set()
            if not self._pending_ask_users:
                return False
            request = self._pending_ask_users[0]

            if request.get("kind") == "permission":
                if request.get("state") == "submitting":
                    wait_for_permission = True
                else:
                    raw = answer.strip().lower()
                    if raw in {"1", "y", "yes", "allow", "once"}:
                        decision = "once"
                    elif raw in {"2", "p", "always", "project", "allow_project"}:
                        decision = "always"
                    elif raw in {"3", "n", "no", "deny", "reject"}:
                        decision = "reject"
                    else:
                        invalid_permission = True
                    if decision is not None and request.get("state") == "pending":
                        request["state"] = "submitting"
                        request["decision"] = decision
                        permission_request = request
            else:
                self._pending_ask_users.popleft()
                clean_answer = answer.strip()
                options = request.get("payload", {}).get("options")
                total_choices = (
                    len(options) + 1
                    if isinstance(options, list) and len(options) >= 2
                    else (len(options) if isinstance(options, list) else 0)
                )
                if (
                    isinstance(options, list)
                    and len(options) >= 2
                    and clean_answer.isdigit()
                    and int(clean_answer) == total_choices
                ):
                    request["selected_index"] = len(options)
                    self._pending_ask_users.appendleft(request)
                    custom_prompt_needed = total_choices
                elif (
                    isinstance(options, list)
                    and options
                    and clean_answer.isdigit()
                    and not (1 <= int(clean_answer) <= total_choices)
                ):
                    # 越界编号不该被当成自由文本发给模型;放回队首并要求重选。
                    self._pending_ask_users.appendleft(request)
                    invalid_ask_user_range = total_choices
                else:
                    if isinstance(options, list) and clean_answer.isdigit():
                        choice_index = int(clean_answer) - 1
                        if 0 <= choice_index < len(options):
                            clean_answer = str(options[choice_index])
                    has_interruption = bool((request.get("payload") or {}).get("interruption_id"))
                    request["state"] = "submitting" if has_interruption else "answered"
                    request["answer"] = clean_answer
                    if not request.get("rendered"):
                        payload = dict(request.get("payload") or {})
                        q_text = str(payload.get("question") or "Please provide more details.")
                        opts = payload.get("options")
                        with self._render_lock:
                            ui.print_clarification_question(
                                q_text,
                                options=opts if isinstance(opts, list) else None,
                                recommended=payload.get("recommended"),
                                answer=clean_answer,
                            )
                        request["rendered"] = True
                    request["event"].set()
                    self._ask_user_custom_draft = ""
                    ask_user_request = request
                    ask_user_answer = clean_answer
                    if request.get("standalone"):
                        standalone_answer = clean_answer

        self._refresh_permission_toolbar()
        if custom_prompt_needed is not None:
            ui.console.print(
                f"[yellow]Option {custom_prompt_needed} is \"Type your own answer...\". "
                "Please type your answer.[/yellow]\n"
            )
            return True
        if wait_for_permission:
            ui.console.print("[dim]Waiting for the permission response to be accepted...[/dim]\n")
            return True
        if invalid_permission:
            ui.console.print(
                "[yellow]Please answer 1 to allow once, "
                "2 to always allow, or 3 to reject.[/yellow]\n"
            )
            return True
        if invalid_ask_user_range is not None:
            ui.console.print(
                f"[yellow]Choose a number between 1 and {invalid_ask_user_range}, "
                "or type a free-form answer.[/yellow]\n"
            )
            return True
        if permission_request is not None and decision is not None:
            self._dispatch_permission_reply(permission_request, decision)
        if ask_user_request is not None and ask_user_answer is not None:
            self._dispatch_ask_user_reply(ask_user_request, ask_user_answer)
        if standalone_answer is not None:
            self._submit_background_question(standalone_answer)
        return True

    def _skip_pending_ask_user(self) -> bool:
        """Skip the oldest pending clarification."""
        self._ask_user_custom_draft = ""
        target: dict[str, Any] | None = None
        with self._ask_user_lock:
            for request in self._pending_ask_users:
                if request.get("kind") == "ask_user" and not request.get("cancelled"):
                    target = request
                    break
        if target is None:
            return False

        payload = dict(target.get("payload") or {})
        interruption_id = str(payload.get("interruption_id") or "")
        if interruption_id:
            skip_answer = (
                "User declined to make a choice. Please state your assumptions and proceed."
            )
            target["state"] = "submitting"
            target["answer"] = skip_answer
            if not target.get("rendered"):
                q_text = str(payload.get("question") or "Please provide more details.")
                opts = payload.get("options")
                with self._render_lock:
                    ui.print_clarification_question(
                        q_text,
                        options=opts if isinstance(opts, list) else None,
                        recommended=payload.get("recommended"),
                        answer="Declined to choose",
                    )
                target["rendered"] = True
            self._dispatch_ask_user_reply(target, skip_answer)
            ui.console.print(f"[{DIM}](Clarification skipped: declined to choose)[/{DIM}]\n")
        else:
            self._cancel_ask_user_request(target)
            ui.console.print(f"[{DIM}](Clarification paused)[/{DIM}]\n")
        return True

    def _ask_agent_stream(self, question: str) -> dict[str, Any] | None:
        """Stream response tokens formatted as Markdown and handle tool calls and queued events.

        Returns ask_user payload dictionary if an ask_user tool call was intercepted, else None.
        """
        console = ui.console
        status: Any = None
        background_turn = bool(getattr(self._request_context, "bound", False))

        def make_streamer() -> MarkdownStreamer:
            # Rich Live and prompt_toolkit both repaint terminal rows. Background
            # turns keep the prompt editable, so their Markdown is flushed above
            # it without opening another live display.
            return MarkdownStreamer(
                console,
                live_window=1 if background_turn else 5,
                live_enabled=not background_turn,
            )

        streamer = make_streamer()
        full_text = ""
        last_rendered_text = ""
        response_label_printed = False
        ask_user_payload: dict[str, Any] | None = None
        ask_user_tool_seen = False
        terminal_event_seen = False
        handoff_ready = False
        permission_request_seen = False
        ask_user_request_seen = False

        session_id = getattr(self._request_context, "session_id", None)
        if not session_id:
            if not self.current_session:
                raise AgentApiError("No active session")
            session_id = self.current_session["id"]
        request_id = getattr(self._request_context, "request_id", None) or uuid4().hex
        if self._shutdown_event.is_set():
            return None

        prompt_rendered = False

        def ensure_prompt_rendered() -> None:
            nonlocal prompt_rendered
            if prompt_rendered:
                return
            prompt_rendered = True
            retired = self._retire_queued_turn(
                session_id,
                request_id=request_id,
                question=question,
            )
            if retired:
                with self._render_lock:
                    ui.print_user_prompt(question)

        def ensure_response_label() -> None:
            nonlocal response_label_printed
            if not response_label_printed:
                ensure_prompt_rendered()
                model_label = (
                    f" [dim]({self.model_name})[/]"
                    if getattr(self, "model_name", None)
                    else ""
                )
                console.print(f"\n[bold {GREEN}]AgentDesk[/]{model_label}:")
                response_label_printed = True

        def finish_text(text: str) -> None:
            if not text:
                return
            ensure_response_label()
            streamer.finish(text)

        def stop_status() -> None:
            nonlocal status
            if status is not None:
                try:
                    status.stop()
                except Exception:
                    pass
                status = None

        def start_status(msg: str) -> None:
            nonlocal status
            self._set_stream_activity(request_id, msg)
            stop_status()
            if background_turn:
                return
            status = ui.make_status(
                f"[cyan]{msg}[/cyan]",
                target_console=console,
            )
            try:
                status.start()
            except LiveError:
                # Rich permits only one Live display per Console.  A queued
                # worker must remain alive when another turn owns that display.
                status = None

        try:
            self._register_stream(session_id, request_id)
            start_status("Working...")
            for event in self.client.ask_stream(
                session_id,
                question,
                request_id=request_id,
            ):
                if self._shutdown_event.is_set():
                    self.client.abort_stream(request_id)
                    return None
                if not isinstance(event, dict):
                    continue
                name = event.get("event")
                raw_data = event.get("data")
                data = raw_data if isinstance(raw_data, dict) else {}

                if name in {
                    "text_delta",
                    "tool_call",
                    "permission_request",
                    "message_end",
                }:
                    self._set_stream_state(request_id, "active")

                if name == "tool_call":
                    ensure_prompt_rendered()
                    self._set_stream_text_streaming(request_id, False)
                    stop_status()
                    if full_text:
                        finish_text(full_text)
                        last_rendered_text = full_text
                        full_text = ""
                        streamer = make_streamer()
                    t_name = data.get("name") or "tool"
                    raw_args = data.get("arguments") or {}
                    if t_name == "ask_user":
                        ask_user_tool_seen = True
                        self._set_stream_activity(request_id, "Waiting for clarification...")
                        if isinstance(raw_args, str):
                            try:
                                parsed_args = json.loads(raw_args)
                                ask_user_payload = (
                                    dict(parsed_args) if isinstance(parsed_args, dict) else None
                                )
                            except Exception:
                                ask_user_payload = {"question": raw_args, "options": []}
                        elif isinstance(raw_args, dict):
                            ask_user_payload = dict(raw_args)
                        if (
                            ask_user_payload
                            and isinstance(ask_user_payload.get("options"), list)
                            and ask_user_payload.get("options")
                            and threading.get_ident() != self._prompt_thread_id
                        ):
                            previous_request = getattr(
                                self._request_context,
                                "pending_ask_user",
                                None,
                            )
                            if previous_request is not None:
                                # Keep one reservation per stream.  If a
                                # malformed model response contains multiple
                                # ask_user calls, update the existing waiter
                                # instead of orphaning the first visible one.
                                with self._ask_user_lock:
                                    if previous_request["event"].is_set():
                                        previous_request = None
                                    else:
                                        previous_request["payload"] = ask_user_payload
                            if previous_request is None:
                                self._request_context.pending_ask_user = (
                                    self._enqueue_ask_user_request(
                                        ask_user_payload,
                                        owner_id=request_id,
                                    )
                                )
                    else:
                        ui.print_tool_call_start(t_name, raw_args)
                        start_status(f"Running {t_name}...")

                elif name == "tool_result":
                    t_name = data.get("name") or ""
                    is_ask_user_result = t_name == "ask_user" or (ask_user_tool_seen and not t_name)
                    raw_content = str(data.get("content", ""))
                    # Queue acknowledgements and ordinary tool results use the
                    # same render lock for both state and output. Whichever
                    # event obtains it first defines the CLI's observed order.
                    with self._render_lock:
                        promoted = self._mark_stream_boundary(
                            session_id,
                            request_id,
                            promote_queued=not is_ask_user_result,
                        )
                        stop_status()
                        if is_ask_user_result:
                            if raw_content.startswith("{"):
                                try:
                                    res_obj = json.loads(raw_content)
                                    if isinstance(res_obj, dict) and res_obj.get("ok"):
                                        if not ask_user_payload:
                                            ask_user_payload = res_obj
                                        else:
                                            for k in ("options", "question", "recommended"):
                                                if k in res_obj and not ask_user_payload.get(k):
                                                    ask_user_payload[k] = res_obj[k]
                                except Exception:
                                    pass
                        else:
                            # Keep the visible tool result before the context
                            # update that this boundary made possible.
                            ui.print_tool_result_end(raw_content)
                            if promoted:
                                questions = [
                                    str(p.get("question") or "").strip()
                                    for p in promoted
                                    if p.get("question")
                                ]
                                ui.print_queued_promotion(
                                    len(promoted),
                                    questions=questions,
                                )
                    if not is_ask_user_result:
                        start_status("Working...")
                    else:
                        self._set_stream_activity(request_id, "Working...")

                elif name == "text_delta":
                    delta = data.get("content", "")
                    if isinstance(delta, str) and delta:
                        full_text += delta
                        stop_status()
                        ensure_response_label()
                        self._set_stream_text_streaming(request_id, True)
                        self._set_stream_activity(request_id, "Generating...")
                        streamer.update(full_text)

                elif name == "permission_request":
                    self._set_stream_text_streaming(request_id, False)
                    self._set_stream_activity(request_id, "Waiting for permission...")
                    stop_status()
                    permission_request_seen = True
                    permission_id = str(data.get("permission_id") or "").strip()
                    permission_session_id = str(data.get("session_id") or session_id).strip()
                    if not permission_id or permission_session_id != session_id:
                        raise AgentApiError("permission_request 事件缺少有效的会话或权限 ID")
                    self._enqueue_permission_request(
                        {
                            **data,
                            "permission_id": permission_id,
                            "session_id": permission_session_id,
                        },
                        owner_id=request_id,
                    )

                elif name == "ask_user_request":
                    self._set_stream_text_streaming(request_id, False)
                    self._set_stream_activity(request_id, "Waiting for clarification...")
                    stop_status()
                    ask_user_request_seen = True
                    interruption_id = str(data.get("interruption_id") or "").strip()
                    req_session_id = str(data.get("session_id") or session_id).strip()
                    if not interruption_id or req_session_id != session_id:
                        raise AgentApiError("ask_user_request 事件缺少有效的会话或中断 ID")
                    existing = getattr(self._request_context, "pending_ask_user", None)
                    if existing is not None and not existing["event"].is_set():
                        with self._ask_user_lock:
                            existing["payload"] = data
                    else:
                        self._request_context.pending_ask_user = (
                            self._enqueue_ask_user_request(
                                data,
                                owner_id=request_id,
                            )
                        )

                elif name == "message_end":
                    ensure_prompt_rendered()
                    self._set_stream_text_streaming(request_id, False)
                    self._set_stream_activity(request_id, None)
                    stop_status()
                    raw_message = data.get("message")
                    if not isinstance(raw_message, dict):
                        raise AgentApiError("message_end 事件缺少有效的 message payload")
                    msg = raw_message
                    raw_content = str(msg.get("content") or "")
                    content = raw_content.strip()

                    if full_text:
                        full_normalized = full_text.strip()
                        if content and full_normalized == content:
                            finish_text(full_text)
                            last_rendered_text = full_text
                            full_text = ""
                        elif content and full_normalized.endswith(content):
                            suffix_start = full_text.rfind(content)
                            prefix = full_text[:suffix_start] if suffix_start >= 0 else full_text
                            if prefix:
                                finish_text(prefix)
                                last_rendered_text = prefix
                            full_text = ""
                            streamer = make_streamer()
                            finish_text(raw_content)
                            last_rendered_text = raw_content
                            with self._queued_turns_lock:
                                self._queued_turns.pop(session_id, None)
                            self._invalidate_prompt()
                        else:
                            finish_text(full_text)
                            last_rendered_text = full_text
                            full_text = ""
                        if content and content != str(last_rendered_text).strip() and not full_text:
                            if str(last_rendered_text).strip() != content:
                                finish_text(raw_content)
                                last_rendered_text = raw_content
                    elif content and content != last_rendered_text:
                        finish_text(raw_content)
                        last_rendered_text = raw_content
                    console.print()
                    # message_end closes this request but does not prove that
                    # persisted follow-ups were re-read.  Keep their queued
                    # presentation state until an ordinary tool_result arrives.
                    self._mark_stream_boundary(
                        session_id,
                        request_id,
                        terminal=True,
                        promote_queued=False,
                    )
                    terminal_event_seen = True
                    metadata = msg.get("metadata")
                    usage = metadata.get("usage") if isinstance(metadata, dict) else {}
                    usage = usage if isinstance(usage, dict) else {}
                    if usage and os.environ.get("ANNA_CLI_SHOW_USAGE", "").lower() in {
                        "1",
                        "true",
                        "yes",
                    }:
                        console.print(
                            f"[{DIM}]tokens: {usage.get('input_tokens', 0):,} in · "
                            f"{usage.get('output_tokens', 0):,} out · "
                            f"{usage.get('total_tokens', 0):,} total[/{DIM}]\n"
                        )

                elif name == "queued":
                    self._set_stream_text_streaming(request_id, False)
                    self._set_stream_activity(request_id, None)
                    with self._render_lock:
                        queue_state = self._mark_queued_turn(
                            session_id,
                            request_id,
                            question,
                        )
                        stop_status()
                        if full_text:
                            finish_text(full_text)
                            full_text = ""
                        terminal_event_seen = True
                        if queue_state == "external":
                            self._request_context.watch_external_queue = True

                elif name == "error":
                    ensure_prompt_rendered()
                    self._set_stream_text_streaming(request_id, False)
                    self._set_stream_activity(request_id, None)
                    stop_status()
                    if full_text:
                        finish_text(full_text)
                        full_text = ""
                    raw_msg = str(data.get("message", "Execution failed"))
                    if "No streaming chunk received for" in raw_msg:
                        raw_msg = (
                            "Upstream model provider timed out (streaming silence "
                            "cap exceeded). Please retry."
                        )
                    raise AgentApiError(raw_msg, status_code=502, details=data)

            handoff_ready = bool(ask_user_payload and not ask_user_request_seen)
            return ask_user_payload if handoff_ready else None

        except KeyboardInterrupt:
            stop_status()
            self._set_stream_activity(request_id, None)
            if full_text:
                finish_text(full_text)
                full_text = ""
            try:
                self.client.cancel(session_id)
            except Exception:
                pass
            self.client.abort_stream(request_id)
            now = time.time()
            if now - self._last_interrupt_time < 1.2:
                self.running = False
                return None
            self._last_interrupt_time = now
            console.print("\n[yellow]> Request interrupted[/yellow]\n")
            return None
        finally:
            self._set_stream_text_streaming(request_id, False)
            self._set_stream_activity(request_id, None)
            stop_status()
            if full_text:
                finish_text(full_text)
            # A direct caller may consume a stream without the background
            # worker cleanup path.  Permission waiters are only valid for
            # this stream, while ask_user reservations may intentionally be
            # handed back to the prompt loop for a follow-up turn.
            if permission_request_seen:
                self._cancel_ask_user_requests_for_owner(
                    request_id,
                    kinds={"permission"},
                )
            if ask_user_request_seen and not terminal_event_seen:
                self._cancel_ask_user_requests_for_owner(
                    request_id,
                    kinds={"ask_user"},
                )
            # A transport/error/cancel path has no reliable tool boundary. It
            # may mark this stream failed, but a separately acknowledged queue
            # marker remains until a later ordinary tool result is observed.
            with self._stream_states_lock:
                stream_state = self._stream_states.get(request_id, {}).get("state")
            pending_ask_user = getattr(self._request_context, "pending_ask_user", None)
            if pending_ask_user is not None and not handoff_ready:
                self._request_context.pending_ask_user = None
                self._cancel_ask_user_request(pending_ask_user)
            if not terminal_event_seen and not handoff_ready and stream_state != "queued":
                self._fail_stream_state(session_id, request_id)
                self._cancel_ask_user_requests_for_owner(request_id)
            self._unregister_stream(session_id, request_id)

    def _should_hide_prompt_input(self) -> bool:
        """Whether the prompt input box and cursor should be hidden."""
        with self._ask_user_lock:
            request = self._pending_picker_locked()
            if request is None:
                return False
            return True

    def get_prompt_text(self) -> HTML:
        """Keep editable prompt stable with Codex-style Working and Queue indicators."""
        if self._should_hide_prompt_input():
            return HTML("")

        blocks: list[str] = []

        if self.active_turn_count > 0:
            if self._turn_start_time <= 0:
                self._turn_start_time = time.monotonic()
            elapsed = max(0, int(time.monotonic() - self._turn_start_time))
            activity = self.current_activity or ""
            if (
                activity
                and not activity.startswith("Working")
                and not activity.startswith("Thinking")
                and not activity.startswith("Generating")
            ):
                detail_str = f"({elapsed}s · {html.escape(activity)})"
            else:
                detail_str = f"({elapsed}s)"

            frame_index = int(
                time.monotonic() / _THINKING_FRAME_INTERVAL_SECONDS
            ) % len(_THINKING_FRAMES)
            frame = _THINKING_FRAMES[frame_index]

            blocks.append(
                f"<activity-running>{frame}</activity-running> "
                f"<activity-name>Working</activity-name> "
                f"<activity-detail>{detail_str}</activity-detail>"
            )

        queued_questions = self.get_queued_questions()
        if queued_questions:
            queue_lines = [
                "<activity-bullet>•</activity-bullet> "
                "<activity-name>Messages to be submitted after next tool call</activity-name>"
            ]
            for q in queued_questions:
                clean_q = html.escape(q.replace("\n", " ").strip())
                if len(clean_q) > 80:
                    clean_q = clean_q[:77] + "..."
                queue_lines.append(
                    f"  <activity-queued>↳</activity-queued> "
                    f"<activity-detail>{clean_q}</activity-detail>"
                )
            blocks.append("\n".join(queue_lines))

        if blocks:
            return HTML("\n\n".join(blocks) + "\n\n<prompt>› </prompt>")

        return HTML("<prompt>› </prompt>")

    def get_activity_text(self) -> HTML:
        """Clean right margin (status and model are rendered in prompt and bottom toolbar)."""
        return HTML("")

    def get_placeholder_text(self) -> HTML:
        """Dynamic placeholder text inside the input box."""
        if self._should_hide_prompt_input():
            return HTML("")
        with self._ask_user_lock:
            if self._pending_picker_locked() is not None:
                return HTML("")
        if self.active_turn_count > 0:
            return HTML(
                "<placeholder>Type a message to queue after this turn...</placeholder>"
            )
        return HTML(
            "<placeholder>Ask AgentDesk anything, or / for commands</placeholder>"
        )

    @property
    def active_turn_count(self) -> int:
        """Return the number of background turns still producing output."""
        with self._turn_threads_lock:
            # Count registered workers, including the short window between
            # ``Thread`` construction and ``start()``.  This closes a race in
            # which /resume or /new could switch sessions before a submitted
            # request had begun its network call.
            return len(self._turn_threads)

    def _run_background_question(
        self,
        question: str,
        session_id: str,
        request_id: str,
        prior_workers: list[threading.Thread] | None = None,
    ) -> None:
        """Run one question without blocking the interactive prompt."""
        try:
            if self._shutdown_event.is_set():
                return
            for pw in (prior_workers or []):
                while pw.is_alive() and not self._shutdown_event.is_set():
                    pw.join(timeout=0.1)
            if self._shutdown_event.is_set() or not self.running:
                return
            if self.current_session and self.current_session.get("id") != session_id:
                return

            if prior_workers:
                with self._queued_turns_lock:
                    queued_items = self._queued_turns.get(session_id, [])
                    still_queued = any(
                        item.get("request_id") == request_id for item in queued_items
                    )
                if not still_queued:
                    return
                self._retire_queued_turn(session_id, request_id=request_id)
                with self._turn_threads_lock:
                    self._turn_start_time = time.monotonic()
                with self._render_lock:
                    ui.print_user_prompt(question)
                self._invalidate_prompt()

            self._request_context.session_id = session_id
            self._request_context.request_id = request_id
            self._request_context.bound = True
            self._request_context.watch_external_queue = False
            self.ask_agent(question, print_prompt=False)
            if getattr(self._request_context, "watch_external_queue", False):
                self._watch_external_queue(session_id, request_id)
        except Exception as exc:
            # Exceptions in a worker must be visible; otherwise the prompt
            # appears healthy while a request silently disappeared.
            if not self._shutdown_event.is_set():
                ui.print_error_message(f"Error occurred: {exc}")
        finally:
            self._request_context.bound = False
            self._request_context.request_id = None
            self._request_context.watch_external_queue = False
            pending_ask_user = getattr(self._request_context, "pending_ask_user", None)
            self._request_context.pending_ask_user = None
            if pending_ask_user is not None:
                self._cancel_ask_user_request(pending_ask_user)
            self._cancel_ask_user_requests_for_owner(request_id)
            self._retire_queued_turn(session_id, request_id=request_id)
            with self._turn_threads_lock:
                self._turn_threads.discard(threading.current_thread())
                if not self._turn_threads:
                    self._turn_start_time = 0.0
            self._invalidate_prompt()

    def _submit_background_question(self, question: str) -> None:
        """Start a question and immediately return control to prompt_toolkit."""
        clean_q = question.strip()
        if not clean_q:
            return
        if self._shutdown_event.is_set() or not self.running:
            return

        # Session creation is intentionally kept on the prompt thread.  This
        # makes the first two rapid messages deterministic: both workers share
        # the same newly-created session instead of racing to create one.
        if not self.current_session:
            if not self._ensure_session(clean_q):
                return
        session_id = self.current_session["id"]
        request_id = uuid4().hex

        with self._turn_threads_lock:
            prior_workers = [
                t for t in self._turn_threads
                if t is not threading.current_thread() and t.is_alive()
            ]
            is_turn_running = len(prior_workers) > 0
            if not self._turn_threads:
                self._turn_start_time = time.monotonic()

        worker = threading.Thread(
            target=self._run_background_question,
            args=(clean_q, session_id, request_id, prior_workers if is_turn_running else None),
            name="agentdesk-turn",
            daemon=True,
        )
        with self._turn_threads_lock:
            self._turn_threads.add(worker)

        if is_turn_running:
            self._mark_queued_turn(session_id, request_id, clean_q)
            with self._render_lock:
                ui.print_queued_input(clean_q)
        else:
            with self._render_lock:
                ui.print_user_prompt(clean_q)

        self._invalidate_prompt()
        # The worker owns all network I/O.  Starting it is the only work the
        # prompt thread performs after a user submits a line, so the next
        # editable prompt can appear immediately while this request streams.
        try:
            worker.start()
        except Exception as exc:
            with self._turn_threads_lock:
                self._turn_threads.discard(worker)
                if not self._turn_threads:
                    self._turn_start_time = 0.0
            if is_turn_running:
                self._retire_queued_turn(session_id, request_id=request_id)
            self._invalidate_prompt()
            ui.print_error_message(f"Unable to start background request: {exc}")

    def _stop_background_turns(self) -> None:
        """Stop transport and give detached turns a short cleanup window."""
        self._shutdown_event.set()
        # Mark the client closed before joining workers.  This prevents a
        # worker that was between its shutdown check and stream registration
        # from opening a new request after the abort snapshot.
        self.client.close()
        with self._queued_turns_lock:
            self._queued_turns.clear()
            self._queue_watch_owners.clear()
        with self._ask_user_lock:
            pending = list(self._pending_ask_users)
            self._pending_ask_users.clear()
            for request in pending:
                request["cancelled"] = True
                request["state"] = "cancelled"
                request["answer"] = None
                request["event"].set()
        self.session_prompt.bottom_toolbar = None
        with self._turn_threads_lock:
            self._turn_start_time = 0.0
            workers = list(self._turn_threads)
        for worker in workers:
            worker.join(timeout=1.0)
        with self._turn_threads_lock:
            permission_workers = list(self._permission_reply_threads)
        for worker in permission_workers:
            worker.join(timeout=1.0)

    def run(self) -> None:
        """Start the primary interactive loop and always release resources."""
        try:
            self._run_loop()
        except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
            pass
        finally:
            self.running = False
            try:
                self._stop_background_turns()
            except Exception:
                pass
            try:
                self.client.close()
            except Exception:
                pass

    def _run_loop(self) -> None:
        """Run initialization and the prompt loop; ``run`` owns cleanup."""
        self._prompt_thread_id = threading.get_ident()
        self._health = self._safe_health()
        self.auto_init_session()
        self.print_header()
        self._print_failed_mcps_hint()

        while self.running:
            try:
                # Background stream/tool output is routed above the editable
                # prompt instead of corrupting the user's current line.
                prompt_default = self._prompt_draft
                self._prompt_draft = ""
                with patch_stdout(raw=True):
                    user_input = self.session_prompt.prompt(
                        self.get_prompt_text,
                        default=prompt_default,
                        placeholder=self.get_placeholder_text,
                    ).strip()
            except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
                break
            except Exception as exc:
                # Prompt/session failures must still pass through the normal
                # shutdown path below so streams and permission waiters do
                # not outlive the interactive loop.
                ui.print_error_message(f"Input loop failed: {exc}")
                break

            if not user_input:
                continue

            # Handle slash commands vs filesystem paths / natural text
            is_cmd, cmd_name, cmd_arg = is_slash_command(user_input)
            if is_cmd:
                if not cmd_name:
                    ui.print_error_message("Type /help for commands.")
                    continue
                handler = COMMAND_MAP.get(cmd_name)
                if handler:
                    try:
                        handler(self, cmd_arg)
                    except AgentApiError as exc:
                        ui.print_error_message(f"Request failed: {exc}")
                    except Exception as exc:
                        ui.print_error_message(f"Command execution failed: {exc}")
                elif self._answer_pending_ask_user(user_input):
                    # Unknown slash-prefixed text can still be a valid free
                    # form clarification answer (for example /usr/bin).
                    pass
                else:
                    ui.print_error_message(
                        f"Unknown command: /{cmd_name}. Type /help for commands."
                    )
                continue

            if self._answer_pending_ask_user(user_input):
                # Ordinary text answers the oldest waiting clarification.  A
                # slash command above always retains its command semantics.
                continue

            # Standard user chat is rendered as ``You`` by the submit path.
            self._submit_background_question(user_input)

    def run_repl(self) -> None:
        """Compatibility entry point."""
        self.run()
