"""Main application: clean & minimal interactive terminal CLI for AgentDesk."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding

import cli.ui as ui
from cli.client import AgentApiClient, AgentApiError
from cli.commands import COMMAND_MAP
from cli.completer import SlashCompleter
from cli.streamer import MarkdownStreamer
from cli.theme import (
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

CLI_STYLE = Style.from_dict({
    "prompt": f"bold {CYAN}",
    "prompt-session": f"{CYAN_BRIGHT}",
    "prompt-bracket": f"{DIM}",
    "completion-menu": f"bg:{BG_CARD} {FG_MUTED}",
    "completion-menu.completion": f"{FG}",
    "completion-menu.completion.current": f"bg:{BG_SELECTION} bold {WHITE}",
    "completion-menu.meta": f"bg:{BG_CARD} {DIM}",
    "completion-menu.meta.current": f"bg:{BG_SELECTION} {CYAN}",
})


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


class AnnaCliApp:
    """AgentDesk interactive terminal CLI application."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url
            or os.environ.get("AGENT_API_URL")
            or os.environ.get("AGENTDESK_API_URL")
            or "http://127.0.0.1:8000/api"
        )
        self.client = AgentApiClient(base_url=self.base_url)
        self.current_session: dict[str, Any] | None = None
        self._pending_title: str | None = None
        self._pending_tools: list[str] | None = None
        self.running = True
        self.model_name = ui.get_current_model()
        self._health: dict[str, Any] | None = None

        self._last_interrupt_time: float = 0.0

        # Keybindings: standard Ctrl+C & Esc line clearing
        kb = KeyBindings()

        @kb.add("escape")
        def _on_escape(event: Any) -> None:
            buf = event.current_buffer
            if buf.text:
                buf.text = ""
                buf.cursor_position = 0

        @kb.add("escape", "enter")
        def _on_alt_enter(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _on_ctrl_c(event: Any) -> None:
            buf = event.current_buffer
            if buf.text:
                buf.text = ""
                buf.cursor_position = 0
            else:
                event.app.exit(exception=KeyboardInterrupt())

        history_path = os.path.expanduser("~/.anna_cli_history")
        self.session_prompt = PromptSession(
            history=FileHistory(history_path),
            completer=SlashCompleter(
                get_sessions_fn=self._safe_list_sessions,
            ),
            key_bindings=kb,
            style=CLI_STYLE,
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
        )

    def _safe_list_sessions(self) -> list[dict[str, Any]]:
        """Safely fetch sessions list."""
        try:
            return self.client.list_sessions()
        except Exception:
            return []

    def _safe_health(self) -> dict[str, Any] | None:
        """Probe backend health on startup."""
        try:
            return self.client.health()
        except Exception:
            return None

    def print_header(self) -> None:
        """Print startup information banner."""
        title = self.current_session.get("title") if self.current_session else (self._pending_title or None)
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
        self.current_session = None
        self._pending_title = pending_title
        self._pending_tools = pending_tools

    def auto_init_session(self) -> None:
        """Start with a clean initial page; session is created on the first question."""
        self.reset_to_new_session()

    def open_session(self, session: dict[str, Any] | None, is_new: bool = False) -> None:
        """Switch to session and display message history."""
        if not session or not session.get("id"):
            ui.print_error_message("无法打开会话：会话不存在或未成功创建。")
            return

        self.current_session = session
        self._pending_title = None
        self._pending_tools = None
        title = session.get("title") or "Untitled"
        sid = (session.get("id") or "")[:8]
        if is_new:
            ui.console.print(f"[green]Created session:[/green] [white]{title}[/white] [dim]({sid})[/dim]\n")
        else:
            ui.console.print(f"[green]Switched to session:[/green] [white]{title}[/white] [dim]({sid})[/dim]\n")
            try:
                messages = self.client.list_messages(session["id"])
                ui.print_history_list(messages, limit=10)
            except Exception as exc:
                ui.print_error_message(f"Failed to load message history: {exc}")

    def rewind_and_resume(self) -> None:
        """Select a past turn in the current session and fork a new session."""
        if not self.current_session:
            ui.console.print("[dim]No active session.[/dim]\n")
            return

        try:
            messages = self.client.list_messages(self.current_session["id"])
        except Exception as exc:
            ui.print_error_message(f"Failed to read session messages: {exc}")
            return

        user_asks = [
            m for m in messages
            if m.get("role") == "human" and (m.get("metadata") or {}).get("source") != "agent"
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
            with ui.console.status(f"[cyan]Forking new session from step {seq}...[/cyan]", spinner="dots"):
                new_session = self.client.resume_session(
                    session_id=self.current_session["id"],
                    title=new_title,
                    parent_last_seq=seq,
                )
            self.open_session(new_session)
            ui.console.print(f"[green]Forked new session from step {seq}:[/green] [white]{new_title}[/white]\n")
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

        # If no active session yet, create one lazily on first user question
        if not self.current_session:
            candidate_title = self._pending_title or make_title_from_question(clean_q)
            try:
                unique_title = self._get_unique_title(candidate_title)
                self.current_session = self.client.create_session(
                    title=unique_title,
                    allowed_tools=self._pending_tools,
                )
                self._pending_title = None
                self._pending_tools = None
            except Exception as exc:
                ui.print_error_message(f"Failed to create session: {exc}")
                return

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
                    msg = "Upstream model provider timed out (streaming silence cap exceeded). Please retry."
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

    def _safe_list_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Safely fetch message list for a session."""
        try:
            return self.client.list_messages(session_id)
        except Exception:
            return []

    def _wait_for_turn_completion(self, session_id: str, timeout: float = 120.0) -> None:
        """Poll session until active turn finishes and display the assistant reply."""
        start_time = time.time()
        initial_messages = self._safe_list_messages(session_id)
        initial_count = len(initial_messages)

        status = ui.console.status("[cyan]Waiting for active turn to finish...[/cyan]", spinner="dots")
        status.start()
        try:
            while time.time() - start_time < timeout:
                time.sleep(1.5)
                current_messages = self._safe_list_messages(session_id)
                if len(current_messages) > initial_count:
                    new_msgs = current_messages[initial_count:]
                    assistants = [m for m in new_msgs if m.get("role") == "assistant"]
                    if assistants:
                        status.stop()
                        ui.console.print("\n[green]✓ Active turn completed:[/green]")
                        for a in assistants:
                            ui.print_message(a)
                        ui.console.print()
                        return
            status.stop()
            ui.console.print(f"[{DIM}]Active turn still running in background. Type /history to view progress.[/{DIM}]\n")
        except KeyboardInterrupt:
            status.stop()
            ui.console.print("\n[yellow]> Stopped waiting. Turn continues in background.[/yellow]\n")
        finally:
            if status:
                try:
                    status.stop()
                except Exception:
                    pass

    def _prompt_ask_user(self, payload: dict[str, Any]) -> str | None:
        """Prompt user for multiple-choice selection cleanly without redundant headers.

        Returns:
            The chosen option string, or None if skipped/cancelled.
        """
        options = payload.get("options") or []
        recommended = payload.get("recommended")

        if not options or not isinstance(options, list):
            return None

        # Launch inline choice picker directly without redundant clarification panel
        from cli.picker import AskUserPicker

        picker = AskUserPicker(options=options, recommended=recommended)
        chosen = picker.run()

        # Handle cancellation / skip
        if not chosen:
            ui.console.print(f"[{DIM}](Clarification skipped)[/{DIM}]\n")
            return None

        return chosen

    def _ask_agent_stream(self, question: str) -> dict[str, Any] | None:
        """Stream response tokens formatted as Markdown and handle tool calls and queued events.

        Returns ask_user payload dictionary if an ask_user tool call was intercepted, else None.
        """
        console = ui.console
        status: Any = None
        streamer = MarkdownStreamer(console)
        full_text = ""
        streamed_in_turn = False
        ask_user_payload: dict[str, Any] | None = None

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
            stop_status()
            status = console.status(f"[cyan]{msg}[/cyan]", spinner="dots")
            status.start()

        start_status("Thinking...")

        try:
            for event in self.client.ask_stream(self.current_session["id"], question):
                name = event.get("event")
                data = event.get("data") or {}

                if name == "tool_call":
                    stop_status()
                    if full_text:
                        streamer.finish(full_text)
                        full_text = ""
                        streamer = MarkdownStreamer(console)
                    t_name = data.get("name") or "tool"
                    raw_args = data.get("arguments") or {}
                    if t_name == "ask_user":
                        if isinstance(raw_args, str):
                            try:
                                ask_user_payload = json.loads(raw_args)
                            except Exception:
                                ask_user_payload = {"question": raw_args, "options": []}
                        elif isinstance(raw_args, dict):
                            ask_user_payload = dict(raw_args)
                    else:
                        ui.print_tool_call_start(t_name, raw_args)
                        start_status(f"Running {t_name}...")

                elif name == "tool_result":
                    stop_status()
                    t_name = data.get("name") or ""
                    raw_content = str(data.get("content", ""))
                    if t_name == "ask_user" or (ask_user_payload and not t_name):
                        if isinstance(raw_content, str) and raw_content.startswith("{"):
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
                        ui.print_tool_result_end(raw_content)
                        start_status("Thinking...")

                elif name == "text_delta":
                    delta = data.get("content", "")
                    if delta:
                        stop_status()
                        if not full_text:
                            console.print()
                        full_text += delta
                        streamed_in_turn = True
                        streamer.update(full_text)

                elif name == "message_end":
                    stop_status()
                    msg = data.get("message") or {}
                    content = (msg.get("content") or "").strip()
                    if full_text:
                        streamer.finish(full_text)
                        full_text = ""
                    elif content and not streamed_in_turn:
                        streamer.finish(content)
                    usage = (msg.get("metadata") or {}).get("usage") or {}
                    console.print()
                    if usage:
                        console.print(
                            f"[{DIM}]tokens: {usage.get('input_tokens', 0):,} in · "
                            f"{usage.get('output_tokens', 0):,} out · "
                            f"{usage.get('total_tokens', 0):,} total[/{DIM}]\n"
                        )

                elif name == "queued":
                    stop_status()
                    if full_text:
                        streamer.finish(full_text)
                        full_text = ""
                    console.print(
                        f"\n[cyan]● Message queued[/cyan] "
                        f"[{DIM}]— session currently has an active turn running; message saved to conversation.[/{DIM}]"
                    )
                    console.print(f"[{DIM}]  Waiting for active turn to finish (Ctrl+C to skip wait)...[/{DIM}]\n")
                    self._wait_for_turn_completion(self.current_session["id"])

                elif name == "error":
                    stop_status()
                    if full_text:
                        streamer.finish(full_text)
                        full_text = ""
                    raw_msg = data.get("message", "Execution failed")
                    if "'Queued' object has no attribute 'message'" in raw_msg:
                        console.print(
                            f"\n[cyan]● Message queued[/cyan] "
                            f"[{DIM}]— session currently has an active turn running; message saved to conversation.[/{DIM}]"
                        )
                        console.print(f"[{DIM}]  Waiting for active turn to finish (Ctrl+C to skip wait)...[/{DIM}]\n")
                        self._wait_for_turn_completion(self.current_session["id"])
                        return None
                    if "No streaming chunk received for" in raw_msg:
                        raw_msg = "Upstream model provider timed out (streaming silence cap exceeded). Please retry."
                    raise AgentApiError(raw_msg, status_code=502, details=data)

            return ask_user_payload

        except KeyboardInterrupt:
            stop_status()
            if full_text:
                streamer.finish(full_text)
                full_text = ""
            self.client.abort_stream()
            now = time.time()
            if now - self._last_interrupt_time < 1.2:
                self.running = False
                return None
            self._last_interrupt_time = now
            console.print("\n[yellow]> Request interrupted[/yellow]\n")
            return None
        finally:
            stop_status()
            if full_text:
                streamer.finish(full_text)

    def get_prompt_text(self) -> HTML:
        """Simple prompt display text."""
        return HTML("<prompt>&gt; </prompt>")

    def run(self) -> None:
        """Start the primary interactive loop."""
        self._health = self._safe_health()
        self.auto_init_session()
        self.print_header()

        while self.running:
            try:
                user_input = self.session_prompt.prompt(self.get_prompt_text()).strip()
            except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
                break

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=1)
                cmd_name = parts[0].lower()
                cmd_arg = parts[1] if len(parts) > 1 else ""

                if cmd_name in ("new", "resume", "subagents", "subagent", "sub", "rewind", "fork", "clear"):
                    sys.stdout.write("\033[1A\033[2K\r")
                    sys.stdout.flush()

                handler = COMMAND_MAP.get(cmd_name)
                if handler:
                    try:
                        handler(self, cmd_arg)
                    except AgentApiError as exc:
                        ui.print_error_message(f"Request failed: {exc}")
                    except Exception as exc:
                        ui.print_error_message(f"Command execution failed: {exc}")
                else:
                    ui.print_error_message(f"Unknown command: /{cmd_name}. Type /help for commands.")
                continue

            # Standard user chat: prompt_toolkit already rendered user input once on screen
            self.ask_agent(user_input, print_prompt=False)

        self.running = False
        try:
            self.client.abort_stream()
            self.client.close()
        except Exception:
            pass

    def run_repl(self) -> None:
        """Compatibility entry point."""
        self.run()
