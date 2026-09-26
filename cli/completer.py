"""Slash command autocompleter for the AgentDesk CLI."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

SLASH_COMMANDS = [
    ("/new", "Start a new session"),
    ("/resume", "Resume or switch session"),
    ("/rewind", "Fork from an earlier turn"),
    ("/rename", "Rename current session"),
    ("/delete", "Delete the current session"),
    ("/compact", "Compact session context"),
    ("/memory", "List or inspect long-term memories"),
    ("/mcp", "List or retry MCP servers"),
    ("/mcps", "List or retry MCP servers"),
    ("/skills", "List available skills"),
    ("/model", "Switch model (picker or ID)"),
    ("/models", "List configured models"),
    ("/mode", "Switch permission mode (default, dont_ask, bypass)"),
    ("/subagents", "Manage subagent sessions"),
    ("/sessions", "List all sessions"),
    ("/session", "Show current session details"),
    ("/clear", "Clear screen and reload history"),
    ("/cancel", "Cancel running turn or clarification"),
    ("/help", "Show help and available commands"),
    ("/exit", "Exit the CLI"),
]


class SlashCompleter(Completer):
    """Autocompleter for slash commands."""

    def __init__(
        self,
        get_sessions_fn: Callable[[], list[dict[str, Any]]] | None = None,
        get_mcps_fn: Callable[[], list[dict[str, Any]]] | None = None,
        get_current_session_fn: Callable[[], dict[str, Any] | None] | None = None,
        get_models_fn: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.get_sessions_fn = get_sessions_fn
        self.get_mcps_fn = get_mcps_fn
        self.get_current_session_fn = get_current_session_fn
        self.get_models_fn = get_models_fn

    def get_completions(self, document: Document, complete_event: Any) -> Iterable[Completion]:
        text = document.text_before_cursor

        # 1. Root slash command completions (e.g. /, /re, /co)
        if text.startswith("/") and " " not in text:
            if "/" in text[1:]:
                return
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

        # 2. Dynamic session index suggestions.  /resume's picker numbers
        # only main sessions (falling back to all sessions if none exist).
        # /delete takes no argument: it only ever targets the current session.
        if text.startswith("/resume "):
            command, _, arg = text.partition(" ")
            arg = arg.strip().lower()
            try:
                sessions = self.get_sessions_fn() if self.get_sessions_fn else []
                if command == "/resume":
                    mains = [s for s in sessions if not s.get("main_session_id")]
                    sessions = mains or sessions
                for idx, s in enumerate(sessions, 1):
                    sid = s.get("id", "")
                    title = s.get("title", "")
                    num = str(idx)
                    if (
                        not arg
                        or num.startswith(arg)
                        or sid.lower().startswith(arg)
                        or title.lower().startswith(arg)
                    ):
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
        if (
            text.startswith("/subagent ")
            or text.startswith("/subagents ")
            or text.startswith("/sub ")
        ):
            command, _, arg = text.partition(" ")
            arg = arg.strip().lower()
            try:
                sessions = self.get_sessions_fn() if self.get_sessions_fn else []
                subs = [s for s in sessions if s.get("main_session_id")]
                if self.get_current_session_fn is not None:
                    current = self.get_current_session_fn()
                    if current:
                        owner_id = current.get("main_session_id") or current.get("id")
                        subs = [s for s in subs if s.get("main_session_id") == owner_id]
                for s in subs:
                    title = s.get("title", "")
                    sid = s.get("id", "")
                    display_name = (
                        title.rsplit("_subagent_", 1)[-1] if "_subagent_" in title else title
                    )
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

        # 4. Dynamic model suggestions for /model
        if text.startswith("/model "):
            command, _, arg = text.partition(" ")
            arg = arg.strip().lower()
            try:
                models = self.get_models_fn() if self.get_models_fn else []
                for idx, m in enumerate(models, 1):
                    mid = m.get("id") or m.get("model_id") or ""
                    name = m.get("name") or mid
                    num = str(idx)
                    if (
                        not arg
                        or num.startswith(arg)
                        or mid.lower().startswith(arg)
                        or name.lower().startswith(arg)
                    ):
                        yield Completion(
                            mid,
                            start_position=-len(text[len(command) + 1 :]),
                            display=f"[{num}] {mid}",
                            display_meta=name,
                        )
            except Exception:
                pass
            return

        # 5. Permission mode suggestions for /mode
        if text.startswith("/mode "):
            _, _, arg = text.partition(" ")
            arg = arg.strip().lower()
            modes = [
                ("default", "Standard prompt-on-unmatched mode"),
                ("dont_ask", "Auto-allow all except deny rules"),
                ("bypass", "Allow all actions without prompting"),
            ]
            for m, desc in modes:
                if not arg or m.startswith(arg):
                    yield Completion(
                        m,
                        start_position=-len(text[6:]),
                        display=m,
                        display_meta=desc,
                    )
            return

        # 6. Dynamic MCP server suggestions for /mcps and /mcp
        if text.startswith(("/mcps ", "/mcp ")):
            command, _, rest = text.partition(" ")
            rest_parts = rest.split()
            # If user hasn't typed a subcommand or is typing "retry"
            if not rest_parts or (len(rest_parts) == 1 and not rest.endswith(" ")):
                prefix = rest_parts[0].lower() if rest_parts else ""
                if "retry".startswith(prefix):
                    yield Completion(
                        "retry",
                        start_position=-len(prefix),
                        display="retry",
                        display_meta="Retry failed MCP servers",
                    )
                return

            if rest_parts[0].lower() == "retry":
                server_prefix = rest_parts[1].lower() if len(rest_parts) > 1 else ""
                if len(rest_parts) == 1 and not rest.endswith(" "):
                    return
                try:
                    mcps = self.get_mcps_fn() if self.get_mcps_fn else []
                    for m in mcps:
                        sid = m.get("id", "")
                        state = (m.get("state") or "unknown").upper()
                        if not server_prefix or sid.lower().startswith(server_prefix):
                            yield Completion(
                                sid,
                                start_position=-len(server_prefix),
                                display=sid,
                                display_meta=f"[{state}]",
                            )
                except Exception:
                    pass
                return

        # 7. Suggestions for /memory
        if text.startswith("/memory "):
            _, _, arg = text.partition(" ")
            arg = arg.strip().lower()
            options = [
                ("list", "List all long-term memories"),
                ("user", "Filter user-layer memories"),
                ("project", "Filter project-layer memories"),
            ]
            for opt, desc in options:
                if not arg or opt.startswith(arg):
                    yield Completion(
                        opt,
                        start_position=-len(text[8:]),
                        display=opt,
                        display_meta=desc,
                    )
            return
