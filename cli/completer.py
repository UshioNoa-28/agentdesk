"""Slash command autocompleter for the AgentDesk CLI."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

SLASH_COMMANDS = [
    ("/new", "Start a new session"),
    ("/resume", "Resume or switch session"),
    ("/sessions", "List all sessions"),
    ("/session", "Show current session details"),
    ("/subagent", "Manage subagent sessions"),
    ("/rename", "Rename current session"),
    ("/delete", "Delete a session"),
    ("/compact", "Compact session context"),
    ("/health", "Check backend health status"),
    ("/mcps", "List or retry MCP servers"),
    ("/skills", "List available skills"),
    ("/history", "View message history"),
    ("/review", "Review workspace git diff"),
    ("/clear", "Clear screen and reload"),
    ("/help", "Show help and commands"),
    ("/quit", "Exit the CLI"),
    ("/exit", "Exit the CLI"),
]


class SlashCompleter(Completer):
    """Autocompleter for slash commands."""

    def __init__(
        self,
        get_sessions_fn: Callable[[], list[dict[str, Any]]] | None = None,
        get_mcps_fn: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.get_sessions_fn = get_sessions_fn
        self.get_mcps_fn = get_mcps_fn

    def get_completions(self, document: Document, complete_event: Any) -> Iterable[Completion]:
        text = document.text_before_cursor

        # 1. Root slash command completions (e.g. /, /re, /co)
        if text.startswith("/") and " " not in text:
            prefix = text.lower()
            for cmd, desc in SLASH_COMMANDS:
                if cmd.lower().startswith(prefix):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=f"{cmd:<12}",
                        display_meta=desc,
                    )
            return

        # 2. Dynamic session index suggestions for /resume and /delete
        if text.startswith("/resume ") or text.startswith("/delete "):
            command, _, arg = text.partition(" ")
            arg = arg.strip().lower()
            try:
                sessions = self.get_sessions_fn() if self.get_sessions_fn else []
                for idx, s in enumerate(sessions, 1):
                    sid = s.get("id", "")
                    title = s.get("title", "")
                    num = str(idx)
                    if not arg or num.startswith(arg) or sid.lower().startswith(arg) or title.lower().startswith(arg):
                        yield Completion(
                            num,
                            start_position=-len(text[len(command) + 1 :]),
                            display=f"[{num}] {title}",
                            display_meta=f"ID: {sid[:8]}...",
                        )
            except Exception:
                pass
            return

        # 3. Dynamic subagent suggestions for /subagent, /subagents, /sub
        if text.startswith("/subagent ") or text.startswith("/subagents ") or text.startswith("/sub "):
            command, _, arg = text.partition(" ")
            arg = arg.strip().lower()
            try:
                sessions = self.get_sessions_fn() if self.get_sessions_fn else []
                subs = [s for s in sessions if s.get("main_session_id")]
                for s in subs:
                    title = s.get("title", "")
                    sid = s.get("id", "")
                    display_name = title.rsplit("_subagent_", 1)[-1] if "_subagent_" in title else title
                    if not arg or arg in display_name.lower() or arg in sid.lower():
                        yield Completion(
                            display_name,
                            start_position=-len(text[len(command) + 1 :]),
                            display=display_name,
                            display_meta=f"Subagent ({sid[:8]})",
                        )
            except Exception:
                pass
            return
