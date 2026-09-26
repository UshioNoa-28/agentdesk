"""Interactive session, subagent & message history pickers."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Callable

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style

from cli.theme import (
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
from cli.ui import format_relative_time, is_summary_text, message_metadata

PICKER_STYLE = Style.from_dict(
    {
        "title": f"bold {CYAN}",
        "hint": f"{DIM}",
        "filter.label": f"{DIM}",
        "filter.query": f"bold {CYAN_BRIGHT}",
        "filter.placeholder": f"{DIM}",
        "item": f"{FG}",
        "item.selected": f"bold {WHITE} bg:{BG_SELECTION}",
        "item.id": f"{DIM}",
        "item.id.selected": f"{CYAN_BRIGHT} bg:{BG_SELECTION}",
        "item.time": f"{DIM}",
        "item.time.selected": f"{DIM} bg:{BG_SELECTION}",
        "pointer": f"bold {CYAN}",
        "pointer.selected": f"bold {CYAN_BRIGHT} bg:{BG_SELECTION}",
        "footer": f"{DIM}",
        "ask.selected": f"bold {WHITE} bg:{BG_SELECTION}",
        "ask.unselected": f"{FG}",
        "assistant.text": f"{FG_MUTED}",
        "assistant.selected": f"italic {WHITE} bg:{BG_SELECTION}",
        "choice.pointer": f"{CYAN}",
        "choice.num": f"{DIM}",
        "choice.num.selected": f"{CYAN}",
        "choice.text": f"{FG_MUTED}",
        "choice.text.selected": f"{WHITE}",
        "choice.recommended": f"{GREEN}",
        "choice.recommended.selected": f"{GREEN}",
        "choice.custom": f"{DIM}",
        "choice.custom.selected": f"{WHITE}",
        "status.connected": f"bold {GREEN}",
        "status.connected.selected": f"bold {GREEN} bg:{BG_SELECTION}",
        "status.failed": f"bold {RED}",
        "status.failed.selected": f"bold {RED} bg:{BG_SELECTION}",
        "status.retrying": f"bold {CYAN_BRIGHT}",
        "status.retrying.selected": f"bold {CYAN_BRIGHT} bg:{BG_SELECTION}",
        "status.disabled": f"{DIM}",
        "status.disabled.selected": f"{DIM} bg:{BG_SELECTION}",
    }
)


def _safe_run_app(app: Application[Any]) -> None:
    """Safely execute prompt_toolkit application."""
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass

    try:
        import asyncio

        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        app.run(in_thread=True)
    else:
        app.run()


class SessionPicker:
    """Interactive inline session picker."""

    def __init__(
        self, sessions: list[dict[str, Any]], current_session_id: str | None = None
    ) -> None:
        self.sessions = [s for s in sessions if not s.get("main_session_id")] or sessions
        self.current_session_id = current_session_id
        self.search_query: str = ""
        self.selected_index = 0
        self.page_size = 6
        self.result: dict[str, Any] | None = None

        if current_session_id:
            for idx, s in enumerate(self.sessions):
                if s.get("id") == current_session_id:
                    self.selected_index = idx
                    break

    def _get_filtered(self) -> list[dict[str, Any]]:
        if not self.search_query.strip():
            return self.sessions
        q = self.search_query.strip().lower()
        res = []
        for idx, s in enumerate(self.sessions, 1):
            title = (s.get("title") or "").lower()
            sid = (s.get("id") or "").lower()
            if q in title or q in sid or q == str(idx):
                res.append(s)
        return res

    def _get_formatted_text(self) -> FormattedText:
        lines: list[tuple[str, str]] = []
        filtered = self._get_filtered()
        total = len(filtered)

        lines.append(("class:title", "Sessions "))
        lines.append(("class:filter.label", "[filter: "))
        if self.search_query:
            lines.append(("class:filter.query", self.search_query))
        else:
            lines.append(("class:filter.placeholder", "type to search..."))
        lines.append(("class:filter.label", "]\n"))

        if not filtered:
            lines.append(("class:hint", "  (no matching sessions)\n\n"))
            lines.append(("class:footer", "  [0/0]\n"))
            return FormattedText(lines)

        if self.selected_index >= total:
            self.selected_index = max(0, total - 1)

        start = max(0, min(self.selected_index - self.page_size // 2, total - self.page_size))
        end = min(total, start + self.page_size)

        for idx in range(start, end):
            s = filtered[idx]
            is_sel = idx == self.selected_index
            is_cur = s.get("id") == self.current_session_id

            time_str = format_relative_time(s.get("updated_at") or s.get("created_at"))
            time_padded = f"{time_str:<10}"

            title = s.get("title") or "Untitled"
            if len(title) > 42:
                title = title[:40] + "…"

            cur_mark = " (current)" if is_cur else ""
            sid_short = (s.get("id") or "")[:8]

            prefix = " > " if is_sel else "   "
            line_style = "class:item.selected" if is_sel else "class:item"
            ptr_style = "class:pointer.selected" if is_sel else "class:pointer"
            time_style = "class:item.time.selected" if is_sel else "class:item.time"
            id_style = "class:item.id.selected" if is_sel else "class:item.id"

            lines.append((ptr_style, prefix))
            lines.append((time_style, time_padded))
            lines.append((line_style, f" {title}{cur_mark}"))
            lines.append((id_style, f"  ({sid_short})\n"))

        cur_pos = f"[{self.selected_index + 1}/{total}]"
        lines.append(
            (
                "class:footer",
                f"  {cur_pos}   [↑/↓: navigate · Enter: select · Esc: cancel · Type: filter]\n",
            )
        )
        return FormattedText(lines)

    def run(self) -> dict[str, Any] | None:
        if not self.sessions:
            return None

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("c-p")
        def _up(event: Any) -> None:
            if self.selected_index > 0:
                self.selected_index -= 1

        @kb.add("down")
        @kb.add("c-n")
        def _down(event: Any) -> None:
            filtered = self._get_filtered()
            if self.selected_index < len(filtered) - 1:
                self.selected_index += 1

        @kb.add("pageup")
        def _pageup(event: Any) -> None:
            self.selected_index = max(0, self.selected_index - self.page_size)

        @kb.add("pagedown")
        def _pagedown(event: Any) -> None:
            filtered = self._get_filtered()
            self.selected_index = min(
                max(0, len(filtered) - 1), self.selected_index + self.page_size
            )

        @kb.add("home")
        def _home(event: Any) -> None:
            self.selected_index = 0

        @kb.add("end")
        def _end(event: Any) -> None:
            filtered = self._get_filtered()
            self.selected_index = max(0, len(filtered) - 1)

        @kb.add("backspace")
        def _backspace(event: Any) -> None:
            if self.search_query:
                self.search_query = self.search_query[:-1]
                self.selected_index = 0

        @kb.add("c-u")
        def _clear_query(event: Any) -> None:
            self.search_query = ""
            self.selected_index = 0

        @kb.add("enter")
        def _enter(event: Any) -> None:
            filtered = self._get_filtered()
            if filtered and 0 <= self.selected_index < len(filtered):
                self.result = filtered[self.selected_index]
            else:
                self.result = None
            event.app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _exit(event: Any) -> None:
            self.result = None
            event.app.exit()

        @kb.add(Keys.Any)
        def _type_char(event: Any) -> None:
            char = event.key_sequence[0].data
            if char and char.isprintable():
                self.search_query += char
                self.selected_index = 0

        content_control = FormattedTextControl(self._get_formatted_text)
        window = Window(
            content=content_control,
            height=Dimension(min=6, max=10, preferred=8),
            dont_extend_height=True,
        )
        layout = Layout(HSplit([window]))

        app: Application[None] = Application(
            layout=layout,
            key_bindings=kb,
            style=PICKER_STYLE,
            full_screen=True,
        )
        _safe_run_app(app)
        return self.result


class SubagentPicker:
    """Interactive inline subagent picker."""

    def __init__(
        self,
        main_session: dict[str, Any],
        subagents: list[dict[str, Any]],
        current_session_id: str | None = None,
    ) -> None:
        self.main_session = main_session
        self.subagents = subagents
        self.current_session_id = current_session_id or main_session.get("id")
        self.search_query: str = ""

        self.items: list[dict[str, Any]] = [
            {
                "id": main_session.get("id", ""),
                "title": f"Main [{main_session.get('title') or 'default'}]",
                "role": "main",
                "is_main": True,
                "raw": main_session,
            }
        ]
        for sub in subagents:
            raw_title = sub.get("title") or "Untitled Subagent"
            sub_name = (
                raw_title.rsplit("_subagent_", 1)[-1] if "_subagent_" in raw_title else raw_title
            )
            self.items.append(
                {
                    "id": sub.get("id", ""),
                    "title": f"Subagent [{sub_name}]",
                    "role": "subagent",
                    "is_main": False,
                    "raw": sub,
                }
            )

        self.selected_index = 0
        for idx, item in enumerate(self.items):
            if item["id"] == self.current_session_id:
                self.selected_index = idx
                break

        self.result: dict[str, Any] | None = None
        self.page_size = 6

    def _get_filtered(self) -> list[dict[str, Any]]:
        if not self.search_query.strip():
            return self.items
        q = self.search_query.strip().lower()
        res = []
        for idx, item in enumerate(self.items, 1):
            title = (item.get("title") or "").lower()
            sid = (item.get("id") or "").lower()
            role = (item.get("role") or "").lower()
            if q in title or q in sid or q in role or q == str(idx):
                res.append(item)
        return res

    def _get_formatted_text(self) -> FormattedText:
        lines: list[tuple[str, str]] = []
        filtered = self._get_filtered()
        total = len(filtered)

        lines.append(("class:title", "Subagents "))
        lines.append(("class:filter.label", "[filter: "))
        if self.search_query:
            lines.append(("class:filter.query", self.search_query))
        else:
            lines.append(("class:filter.placeholder", "type to search..."))
        lines.append(("class:filter.label", "]\n"))

        if not filtered:
            lines.append(("class:hint", "  (no matching subagents)\n\n"))
            lines.append(("class:footer", "  [0/0]\n"))
            return FormattedText(lines)

        if self.selected_index >= total:
            self.selected_index = max(0, total - 1)

        start = max(0, min(self.selected_index - self.page_size // 2, total - self.page_size))
        end = min(total, start + self.page_size)

        for idx in range(start, end):
            item = filtered[idx]
            is_sel = idx == self.selected_index
            is_cur = item["id"] == self.current_session_id

            prefix = " > " if is_sel else "   "
            num_str = f"[{idx + 1}] "
            title_str = item["title"]
            if is_cur:
                title_str = f"{title_str} (current)"
            title_padded = f"{title_str:<32}"
            sid_str = (item["id"] or "")[:8]

            line_style = "class:item.selected" if is_sel else "class:item"
            ptr_style = "class:pointer.selected" if is_sel else "class:pointer"
            id_style = "class:item.id.selected" if is_sel else "class:item.id"

            lines.append((ptr_style, prefix))
            lines.append((line_style, f"{num_str}{title_padded}"))
            lines.append((id_style, f"  ({sid_str})\n"))

        cur_pos = f"[{self.selected_index + 1}/{total}]"
        lines.append(
            (
                "class:footer",
                f"  {cur_pos}   [↑/↓: navigate · Enter: select · Esc: cancel · Type: filter]\n",
            )
        )
        return FormattedText(lines)

    def run(self) -> dict[str, Any] | None:
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("c-p")
        def _up(event: Any) -> None:
            if self.selected_index > 0:
                self.selected_index -= 1

        @kb.add("down")
        @kb.add("c-n")
        def _down(event: Any) -> None:
            filtered = self._get_filtered()
            if self.selected_index < len(filtered) - 1:
                self.selected_index += 1

        @kb.add("backspace")
        def _backspace(event: Any) -> None:
            if self.search_query:
                self.search_query = self.search_query[:-1]
                self.selected_index = 0

        @kb.add("c-u")
        def _clear_query(event: Any) -> None:
            self.search_query = ""
            self.selected_index = 0

        @kb.add("enter")
        def _enter(event: Any) -> None:
            filtered = self._get_filtered()
            if filtered and 0 <= self.selected_index < len(filtered):
                self.result = filtered[self.selected_index]["raw"]
            else:
                self.result = None
            event.app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _exit(event: Any) -> None:
            self.result = None
            event.app.exit()

        @kb.add(Keys.Any)
        def _type_char(event: Any) -> None:
            char = event.key_sequence[0].data
            if char and char.isprintable():
                self.search_query += char
                self.selected_index = 0

        content_control = FormattedTextControl(self._get_formatted_text)
        window = Window(
            content=content_control,
            height=Dimension(min=6, max=10, preferred=8),
            dont_extend_height=True,
        )
        layout = Layout(HSplit([window]))

        app: Application[None] = Application(
            layout=layout,
            key_bindings=kb,
            style=PICKER_STYLE,
            full_screen=True,
        )
        _safe_run_app(app)
        return self.result


class HistoryMessageResumer:
    """Interactive inline history message turn picker for rewind and fork."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.visible_messages: list[dict[str, Any]] = [
            m
            for m in messages
            if m.get("role") != "system"
            and message_metadata(m).get("kind") not in ("summary", "maintenance", "system")
            and not is_summary_text(str(m.get("content") or ""))
        ]

        self.turns: list[dict[str, Any]] = []
        current_turn: dict[str, Any] | None = None

        for msg in self.visible_messages:
            if msg.get("role") == "human" and message_metadata(msg).get("source") != "agent":
                current_turn = {
                    "ask": msg,
                    "responses": [],
                }
                self.turns.append(current_turn)
            else:
                if current_turn is not None:
                    current_turn["responses"].append(msg)

        self.selected_turn_idx = max(0, len(self.turns) - 1)
        self.search_query: str = ""
        self.result: dict[str, Any] | None = None
        self.page_size = 3

    def _get_filtered(self) -> list[dict[str, Any]]:
        if not self.search_query.strip():
            return self.turns
        q = self.search_query.strip().lower()
        res = []
        for turn in self.turns:
            ask_text = (turn["ask"].get("content") or "").lower()
            resp_texts = " ".join((r.get("content") or "").lower() for r in turn["responses"])
            if q in ask_text or q in resp_texts:
                res.append(turn)
        return res

    def _get_formatted_text(self) -> FormattedText:
        lines: list[tuple[str, str]] = []
        filtered = self._get_filtered()
        total = len(filtered)

        lines.append(("class:title", "Rewind & Fork "))
        lines.append(("class:filter.label", "[filter: "))
        if self.search_query:
            lines.append(("class:filter.query", self.search_query))
        else:
            lines.append(("class:filter.placeholder", "type to search..."))
        lines.append(("class:filter.label", "]\n"))

        if not filtered:
            lines.append(("class:hint", "  (no matching conversation turns)\n\n"))
            lines.append(("class:footer", "  [0/0]\n"))
            return FormattedText(lines)

        if self.selected_turn_idx >= total:
            self.selected_turn_idx = max(0, total - 1)

        start = max(0, min(self.selected_turn_idx - self.page_size // 2, total - self.page_size))
        end = min(total, start + self.page_size)

        for idx in range(start, end):
            turn = filtered[idx]
            is_sel = idx == self.selected_turn_idx

            ask_msg = turn["ask"]
            seq = ask_msg.get("seq", idx + 1)
            content = (ask_msg.get("content") or "").strip().replace("\n", " ")
            if len(content) > 60:
                content = content[:58] + "…"

            prefix = " > " if is_sel else "   "
            turn_tag = f"[Turn {idx + 1}|seq {seq}] "
            ask_line = f"{turn_tag}{content}"

            line_style = "class:ask.selected" if is_sel else "class:ask.unselected"
            ptr_style = "class:pointer.selected" if is_sel else "class:pointer"

            lines.append((ptr_style, prefix))
            lines.append((line_style, f"{ask_line}\n"))

            first_resp = turn["responses"][0] if turn["responses"] else None
            if first_resp:
                resp_text = (first_resp.get("content") or "").strip().replace("\n", " ")
                if len(resp_text) > 64:
                    resp_text = resp_text[:62] + "…"
                resp_style = "class:assistant.selected" if is_sel else "class:assistant.text"
                lines.append(("class:hint", "     ↳ "))
                lines.append((resp_style, f"{resp_text}\n"))

        cur_pos = f"[{self.selected_turn_idx + 1}/{total}]"
        lines.append(
            (
                "class:footer",
                f"  {cur_pos}   [↑/↓: navigate · Enter: fork from here · "
                "Esc: cancel · Type: filter]\n",
            )
        )
        return FormattedText(lines)

    def run(self) -> dict[str, Any] | None:
        if not self.turns:
            return None

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("c-p")
        def _up(event: Any) -> None:
            if self.selected_turn_idx > 0:
                self.selected_turn_idx -= 1

        @kb.add("down")
        @kb.add("c-n")
        def _down(event: Any) -> None:
            filtered = self._get_filtered()
            if self.selected_turn_idx < len(filtered) - 1:
                self.selected_turn_idx += 1

        @kb.add("backspace")
        def _backspace(event: Any) -> None:
            if self.search_query:
                self.search_query = self.search_query[:-1]
                self.selected_turn_idx = 0

        @kb.add("c-u")
        def _clear_query(event: Any) -> None:
            self.search_query = ""
            self.selected_turn_idx = 0

        @kb.add("enter")
        def _enter(event: Any) -> None:
            filtered = self._get_filtered()
            if filtered and 0 <= self.selected_turn_idx < len(filtered):
                selected_turn = filtered[self.selected_turn_idx]
                target_msg = selected_turn["ask"]
                self.result = {
                    "message": target_msg,
                    "seq": target_msg.get("seq", 1),
                    "content": target_msg.get("content", ""),
                }
            else:
                self.result = None
            event.app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _exit(event: Any) -> None:
            self.result = None
            event.app.exit()

        @kb.add(Keys.Any)
        def _type_char(event: Any) -> None:
            char = event.key_sequence[0].data
            if char and char.isprintable():
                self.search_query += char
                self.selected_turn_idx = 0

        content_control = FormattedTextControl(self._get_formatted_text)
        window = Window(
            content=content_control,
            height=Dimension(min=6, max=10, preferred=8),
            dont_extend_height=True,
        )
        layout = Layout(HSplit([window]))

        app: Application[None] = Application(
            layout=layout,
            key_bindings=kb,
            style=PICKER_STYLE,
            full_screen=True,
        )
        _safe_run_app(app)
        return self.result


class AskUserPicker:
    """Interactive choice picker for agent clarification questions (ask_user)."""

    def __init__(
        self,
        options: list[str],
        recommended: str | None = None,
        allow_custom: bool = True,
    ) -> None:
        self.options = [opt.strip() for opt in options if opt and opt.strip()]
        self.recommended = recommended.strip() if recommended and recommended.strip() else None
        self.allow_custom = allow_custom
        self.result: str | None = None
        self.custom_text: str = ""
        self.custom_cursor_idx: int = 0

        self.items: list[dict[str, Any]] = []
        for opt in self.options:
            self.items.append(
                {
                    "type": "option",
                    "text": opt,
                    "is_recommended": (self.recommended is not None and opt == self.recommended),
                }
            )

        if self.allow_custom:
            self.items.append(
                {
                    "type": "custom",
                    "text": "Type your own answer...",
                    "is_recommended": False,
                }
            )
            self.custom_index = len(self.items) - 1
        else:
            self.custom_index = -1

        self.selected_index = 0
        if self.recommended:
            for idx, item in enumerate(self.items):
                if item["is_recommended"]:
                    self.selected_index = idx
                    break

    def _get_formatted_text(self) -> FormattedText:
        lines: list[tuple[str, str]] = []

        for idx, item in enumerate(self.items):
            is_sel = idx == self.selected_index
            is_custom = item["type"] == "custom"
            is_rec = item["is_recommended"]
            num_str = f"[{idx + 1}] "

            if is_sel:
                prefix = " ❯ "
                ptr_style = "class:choice.pointer"
                num_style = "class:choice.num.selected"
                text_style = "class:choice.text.selected"
                rec_style = "class:choice.recommended.selected"
            else:
                prefix = "   "
                ptr_style = "class:choice.pointer"
                num_style = "class:choice.num"
                text_style = "class:choice.text"
                rec_style = "class:choice.recommended"

            lines.append((ptr_style, prefix))
            lines.append((num_style, num_str))

            if is_custom:
                if is_sel:
                    if self.custom_text:
                        cur = max(0, min(self.custom_cursor_idx, len(self.custom_text)))
                        before = self.custom_text[:cur]
                        if cur < len(self.custom_text):
                            at = self.custom_text[cur]
                            after = self.custom_text[cur + 1 :]
                            if before:
                                lines.append(("class:choice.custom.selected", before))
                            lines.append(("[SetCursorPosition] class:choice.custom.selected", at))
                            if after:
                                lines.append(("class:choice.custom.selected", after))
                        else:
                            if before:
                                lines.append(("class:choice.custom.selected", before))
                            lines.append(("[SetCursorPosition]", " "))
                    else:
                        placeholder = item["text"]
                        lines.append(("[SetCursorPosition] class:choice.custom", placeholder[:1]))
                        if len(placeholder) > 1:
                            lines.append(("class:choice.custom", placeholder[1:]))
                else:
                    lines.append(("class:choice.custom", item["text"]))
            else:
                lines.append((text_style, item["text"]))

            if is_rec:
                lines.append((rec_style, " (Recommended)"))
            if idx < len(self.items) - 1:
                lines.append(("", "\n"))

        return FormattedText(lines)

    def _confirm_index(self, idx: int, app: Application[Any]) -> None:
        if 0 <= idx < len(self.items):
            item = self.items[idx]
            if item["type"] == "custom":
                self.result = self.custom_text.strip() or None
            else:
                self.result = item["text"]
        else:
            self.result = None
        app.exit()

    def run(self, input: Any = None, output: Any = None) -> str | None:
        if not self.items:
            return None

        if not sys.stdin.isatty() and input is None:
            if self.recommended:
                return self.recommended
            return self.options[0] if self.options else None

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("c-p")
        def _up(event: Any) -> None:
            if self.selected_index > 0:
                self.selected_index -= 1
                if self.selected_index == self.custom_index:
                    self.custom_cursor_idx = len(self.custom_text)

        @kb.add("down")
        @kb.add("c-n")
        def _down(event: Any) -> None:
            if self.selected_index < len(self.items) - 1:
                self.selected_index += 1
                if self.selected_index == self.custom_index:
                    self.custom_cursor_idx = len(self.custom_text)

        @kb.add("left")
        def _left(event: Any) -> None:
            if self.selected_index == self.custom_index and self.custom_cursor_idx > 0:
                self.custom_cursor_idx -= 1

        @kb.add("right")
        def _right(event: Any) -> None:
            if self.selected_index == self.custom_index and self.custom_cursor_idx < len(
                self.custom_text
            ):
                self.custom_cursor_idx += 1

        @kb.add("home")
        @kb.add("c-a")
        def _home(event: Any) -> None:
            if self.selected_index == self.custom_index:
                self.custom_cursor_idx = 0

        @kb.add("end")
        @kb.add("c-e")
        def _end(event: Any) -> None:
            if self.selected_index == self.custom_index:
                self.custom_cursor_idx = len(self.custom_text)

        @kb.add("backspace")
        @kb.add("c-h")
        def _backspace(event: Any) -> None:
            if self.selected_index == self.custom_index and self.custom_cursor_idx > 0:
                self.custom_text = (
                    self.custom_text[: self.custom_cursor_idx - 1]
                    + self.custom_text[self.custom_cursor_idx :]
                )
                self.custom_cursor_idx -= 1

        @kb.add("delete")
        @kb.add("c-d")
        def _delete(event: Any) -> None:
            if self.selected_index == self.custom_index and self.custom_cursor_idx < len(
                self.custom_text
            ):
                self.custom_text = (
                    self.custom_text[: self.custom_cursor_idx]
                    + self.custom_text[self.custom_cursor_idx + 1 :]
                )

        @kb.add("c-u")
        def _clear(event: Any) -> None:
            if self.selected_index == self.custom_index:
                self.custom_text = ""
                self.custom_cursor_idx = 0

        @kb.add("enter")
        def _enter(event: Any) -> None:
            if self.selected_index != self.custom_index:
                self.result = self.items[self.selected_index]["text"]
                event.app.exit()
            else:
                text = self.custom_text.strip()
                if text:
                    self.result = text
                    event.app.exit()

        @kb.add("escape")
        def _escape(event: Any) -> None:
            if self.selected_index == self.custom_index and self.custom_text:
                self.custom_text = ""
                self.custom_cursor_idx = 0
            else:
                self.result = None
                event.app.exit()

        @kb.add("c-c")
        def _exit(event: Any) -> None:
            self.result = None
            event.app.exit()

        if self.allow_custom:

            @kb.add(Keys.Any)
            def _on_key(event: Any) -> None:
                char = event.data
                if not char or not char.isprintable():
                    return
                self.selected_index = self.custom_index
                self.custom_text = (
                    self.custom_text[: self.custom_cursor_idx]
                    + char
                    + self.custom_text[self.custom_cursor_idx :]
                )
                self.custom_cursor_idx += len(char)

        content_control = FormattedTextControl(self._get_formatted_text)
        total_lines = len(self.items)
        window = Window(
            content=content_control,
            height=Dimension.exact(total_lines),
            dont_extend_height=True,
            always_hide_cursor=Condition(lambda: self.selected_index != self.custom_index),
        )
        layout = Layout(HSplit([window]))

        app: Application[None] = Application(
            layout=layout,
            key_bindings=kb,
            style=PICKER_STYLE,
            full_screen=False,
            erase_when_done=True,
            input=input,
            output=output,
        )
        _safe_run_app(app)
        return self.result


class ModelPicker:
    """Interactive inline model profile picker."""

    def __init__(
        self, models: list[dict[str, Any]], current_model_id: str | None = None
    ) -> None:
        self.models = models
        self.current_model_id = current_model_id
        self.search_query: str = ""
        self.selected_index = 0
        self.page_size = 6
        self.result: dict[str, Any] | None = None

        if current_model_id:
            for idx, m in enumerate(self.models):
                mid = m.get("id") or m.get("model_id") or ""
                if mid == current_model_id:
                    self.selected_index = idx
                    break

    def _get_filtered(self) -> list[dict[str, Any]]:
        if not self.search_query.strip():
            return self.models
        q = self.search_query.strip().lower()
        res = []
        for idx, m in enumerate(self.models, 1):
            mid = (m.get("id") or m.get("model_id") or "").lower()
            name = (m.get("name") or "").lower()
            if q in mid or q in name or q == str(idx):
                res.append(m)
        return res

    def _get_formatted_text(self) -> FormattedText:
        lines: list[tuple[str, str]] = []
        filtered = self._get_filtered()
        total = len(filtered)

        lines.append(("class:title", "Models "))
        lines.append(("class:filter.label", "[filter: "))
        if self.search_query:
            lines.append(("class:filter.query", self.search_query))
        else:
            lines.append(("class:filter.placeholder", "type to search..."))
        lines.append(("class:filter.label", "]\n"))

        if not filtered:
            lines.append(("class:hint", "  (no matching models)\n\n"))
            lines.append(("class:footer", "  [0/0]\n"))
            return FormattedText(lines)

        if self.selected_index >= total:
            self.selected_index = max(0, total - 1)

        start = max(0, min(self.selected_index - self.page_size // 2, total - self.page_size))
        end = min(total, start + self.page_size)

        for idx in range(start, end):
            m = filtered[idx]
            is_sel = idx == self.selected_index
            mid = m.get("id") or m.get("model_id") or ""
            is_cur = mid == self.current_model_id

            name = m.get("name") or mid
            if len(name) > 35:
                name = name[:33] + "…"

            cur_mark = " (current)" if is_cur else ""

            prefix = " > " if is_sel else "   "
            line_style = "class:item.selected" if is_sel else "class:item"
            ptr_style = "class:pointer.selected" if is_sel else "class:pointer"
            id_style = "class:item.id.selected" if is_sel else "class:item.id"

            lines.append((ptr_style, prefix))
            lines.append((line_style, f"{name:<35}{cur_mark}"))
            lines.append((id_style, f"  ({mid})\n"))

        cur_pos = f"[{self.selected_index + 1}/{total}]"
        lines.append(
            (
                "class:footer",
                f"  {cur_pos}   [↑/↓: navigate · Enter: select · Esc: cancel · Type: filter]\n",
            )
        )
        return FormattedText(lines)

    def run(self, input: Any = None, output: Any = None) -> dict[str, Any] | None:
        if not self.models:
            return None

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("c-p")
        def _up(event: Any) -> None:
            if self.selected_index > 0:
                self.selected_index -= 1

        @kb.add("down")
        @kb.add("c-n")
        def _down(event: Any) -> None:
            filtered = self._get_filtered()
            if self.selected_index < len(filtered) - 1:
                self.selected_index += 1

        @kb.add("pageup")
        def _pageup(event: Any) -> None:
            self.selected_index = max(0, self.selected_index - self.page_size)

        @kb.add("pagedown")
        def _pagedown(event: Any) -> None:
            filtered = self._get_filtered()
            self.selected_index = min(
                max(0, len(filtered) - 1), self.selected_index + self.page_size
            )

        @kb.add("home")
        def _home(event: Any) -> None:
            self.selected_index = 0

        @kb.add("end")
        def _end(event: Any) -> None:
            filtered = self._get_filtered()
            self.selected_index = max(0, len(filtered) - 1)

        @kb.add("enter")
        def _enter(event: Any) -> None:
            filtered = self._get_filtered()
            if filtered and 0 <= self.selected_index < len(filtered):
                self.result = filtered[self.selected_index]
            event.app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _cancel(event: Any) -> None:
            self.result = None
            event.app.exit()

        @kb.add("backspace")
        def _backspace(event: Any) -> None:
            if self.search_query:
                self.search_query = self.search_query[:-1]
                self.selected_index = 0

        @kb.add(Keys.Any)
        def _char(event: Any) -> None:
            c = event.data
            if c and c.isprintable():
                self.search_query += c
                self.selected_index = 0

        content_control = FormattedTextControl(self._get_formatted_text)
        window = Window(
            content=content_control,
            height=Dimension(min=3, max=self.page_size + 3),
            dont_extend_height=True,
        )
        layout = Layout(HSplit([window]))
        app: Application[None] = Application(
            layout=layout,
            key_bindings=kb,
            style=PICKER_STYLE,
            full_screen=False,
            erase_when_done=True,
            input=input,
            output=output,
        )
        _safe_run_app(app)
        return self.result


class McpPicker:
    """Interactive inline MCP server picker for viewing and reconnecting in-place."""

    def __init__(
        self,
        mcps: list[dict[str, Any]],
        retry_fn: Callable[[str], dict[str, Any]] | None = None,
        auto_retry_ids: list[str] | None = None,
    ) -> None:
        self.mcps = [dict(m) for m in mcps]
        self.retry_fn = retry_fn
        self.search_query: str = ""
        self.selected_index = 0
        for i, m in enumerate(self.mcps):
            if (m.get("state") or "").upper() == "FAILED":
                self.selected_index = i
                break
        if auto_retry_ids:
            for i, m in enumerate(self.mcps):
                if m.get("id") == auto_retry_ids[0]:
                    self.selected_index = i
                    break

        self.page_size = 6
        self.result: dict[str, Any] | None = None
        self.retried_results: dict[str, dict[str, Any]] = {}

        self._auto_retry_ids = list(auto_retry_ids or [])
        self._retrying_servers: set[str] = set()
        self._retry_messages: dict[str, str] = {}
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._ticker_running = False
        self._lock = threading.RLock()
        self._app: Application[Any] | None = None
        self._worker_threads: list[threading.Thread] = []

    def _invalidate_app(self) -> None:
        app = self._app
        if app and getattr(app, "is_running", False):
            try:
                app.invalidate()
            except Exception:
                pass

    def _start_ticker(self) -> None:
        with self._lock:
            if self._ticker_running:
                return
            self._ticker_running = True

        def _ticker() -> None:
            while True:
                time.sleep(0.08)
                with self._lock:
                    if not self._retrying_servers:
                        self._ticker_running = False
                        break
                    self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
                self._invalidate_app()

        t = threading.Thread(target=_ticker, daemon=True)
        t.start()

    def _trigger_retry(self, sid: str) -> None:
        if not self.retry_fn:
            return
        with self._lock:
            if sid in self._retrying_servers:
                return
            self._retrying_servers.add(sid)
            self._retry_messages[sid] = "Connecting..."
            self._start_ticker()

        self._invalidate_app()

        def _worker() -> None:
            try:
                assert self.retry_fn is not None
                res = self.retry_fn(sid)
                with self._lock:
                    for idx, m in enumerate(self.mcps):
                        if m.get("id") == sid:
                            self.mcps[idx] = dict(res)
                            break
                    self.retried_results[sid] = dict(res)
                    state = (res.get("state") or "").upper()
                    if state == "CONNECTED":
                        self._retry_messages[sid] = "✓ Reconnected"
                    else:
                        err = res.get("error") or "Failed"
                        first_err = err.split("\n")[0].strip()
                        if len(first_err) > 36:
                            first_err = first_err[:34] + "…"
                        self._retry_messages[sid] = f"Err: {first_err}"
            except Exception as exc:
                with self._lock:
                    err_msg = str(exc).split("\n")[0].strip()
                    if len(err_msg) > 36:
                        err_msg = err_msg[:34] + "…"
                    for m in self.mcps:
                        if m.get("id") == sid:
                            m["state"] = "failed"
                            m["error"] = str(exc)
                            break
                    self.retried_results[sid] = {
                        "id": sid,
                        "state": "failed",
                        "error": str(exc),
                        "tool_count": 0,
                    }
                    self._retry_messages[sid] = f"Err: {err_msg}"
            finally:
                with self._lock:
                    self._retrying_servers.discard(sid)
                self._invalidate_app()

        t = threading.Thread(target=_worker, daemon=True)
        self._worker_threads.append(t)
        t.start()

    def wait_retries(self, timeout: float = 5.0) -> None:
        """Wait for active retry workers to complete."""
        for t in list(self._worker_threads):
            t.join(timeout=timeout)

    def _get_filtered(self) -> list[dict[str, Any]]:
        if not self.search_query.strip():
            return self.mcps
        q = self.search_query.strip().lower()
        res = []
        for idx, m in enumerate(self.mcps, 1):
            sid = (m.get("id") or "").lower()
            desc = (m.get("description") or "").lower()
            state = (m.get("state") or "").lower()
            if q in sid or q in desc or q in state or q == str(idx):
                res.append(m)
        return res

    def _get_formatted_text(self) -> FormattedText:
        lines: list[tuple[str, str]] = []
        filtered = self._get_filtered()
        total = len(filtered)

        lines.append(("class:title", "MCP Servers "))
        lines.append(("class:filter.label", "[filter: "))
        if self.search_query:
            lines.append(("class:filter.query", self.search_query))
        else:
            lines.append(("class:filter.placeholder", "type to search..."))
        lines.append(("class:filter.label", "]\n"))

        if not filtered:
            lines.append(("class:hint", "  (no matching MCP servers)\n\n"))
            lines.append(("class:footer", "  [0/0]\n"))
            return FormattedText(lines)

        if self.selected_index >= total:
            self.selected_index = max(0, total - 1)

        start = max(0, min(self.selected_index - self.page_size // 2, total - self.page_size))
        end = min(total, start + self.page_size)

        for idx in range(start, end):
            m = filtered[idx]
            is_sel = idx == self.selected_index
            sid = m.get("id") or ""
            with self._lock:
                is_retrying = sid in self._retrying_servers
                msg = self._retry_messages.get(sid)
                spinner_char = self._spinner_frames[self._spinner_idx]

            raw_state = (m.get("state") or "unknown").upper()
            tool_count = m.get("tool_count", 0)

            prefix = " > " if is_sel else "   "
            line_style = "class:item.selected" if is_sel else "class:item"
            ptr_style = "class:pointer.selected" if is_sel else "class:pointer"
            id_style = "class:item.id.selected" if is_sel else "class:item.id"

            if is_retrying:
                status_tag = f"[{spinner_char} RETRYING]"
                status_style = (
                    "class:status.retrying.selected" if is_sel else "class:status.retrying"
                )
            elif raw_state == "CONNECTED":
                status_tag = f"[{raw_state}]"
                status_style = (
                    "class:status.connected.selected" if is_sel else "class:status.connected"
                )
            elif raw_state == "FAILED":
                status_tag = f"[{raw_state}]"
                status_style = (
                    "class:status.failed.selected" if is_sel else "class:status.failed"
                )
            elif raw_state == "DISABLED":
                status_tag = f"[{raw_state}]"
                status_style = (
                    "class:status.disabled.selected" if is_sel else "class:status.disabled"
                )
            else:
                status_tag = f"[{raw_state}]"
                status_style = "class:item.time.selected" if is_sel else "class:item.time"

            lines.append((ptr_style, prefix))
            lines.append((line_style, f"{sid:<16} "))
            lines.append((status_style, f"{status_tag:<14}"))
            lines.append((id_style, f"({tool_count} tools)  "))

            if is_retrying:
                retrying_style = (
                    "class:status.retrying.selected" if is_sel else "class:status.retrying"
                )
                lines.append((retrying_style, "Connecting…"))
            elif msg and msg.startswith("✓"):
                conn_style = (
                    "class:status.connected.selected" if is_sel else "class:status.connected"
                )
                lines.append((conn_style, msg))
            elif msg and msg.startswith("Err:"):
                fail_style = (
                    "class:status.failed.selected" if is_sel else "class:status.failed"
                )
                lines.append((fail_style, msg))
            elif is_sel:
                err = m.get("error")
                if err:
                    first_err = err.split("\n")[0].strip()
                    if len(first_err) > 36:
                        first_err = first_err[:34] + "…"
                    lines.append(("class:status.failed.selected", f"Err: {first_err}"))
                else:
                    desc = (m.get("description") or "").split("\n")[0].strip()
                    if desc:
                        if len(desc) > 36:
                            desc = desc[:34] + "…"
                        desc_style = (
                            "class:item.time.selected" if is_sel else "class:item.time"
                        )
                        lines.append((desc_style, desc))
            lines.append(("", "\n"))

        cur_pos = f"[{self.selected_index + 1}/{total}]"
        lines.append(
            (
                "class:footer",
                f"  {cur_pos}   [↑/↓: navigate · Enter: reconnect · Esc: exit · Type: filter]\n",
            )
        )
        return FormattedText(lines)

    def run(self, input: Any = None, output: Any = None) -> dict[str, Any] | None:
        if not self.mcps:
            return None

        for sid in self._auto_retry_ids:
            self._trigger_retry(sid)
        self._auto_retry_ids.clear()

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("c-p")
        def _up(event: Any) -> None:
            if self.selected_index > 0:
                self.selected_index -= 1

        @kb.add("down")
        @kb.add("c-n")
        def _down(event: Any) -> None:
            filtered = self._get_filtered()
            if self.selected_index < len(filtered) - 1:
                self.selected_index += 1

        @kb.add("pageup")
        def _pageup(event: Any) -> None:
            self.selected_index = max(0, self.selected_index - self.page_size)

        @kb.add("pagedown")
        def _pagedown(event: Any) -> None:
            filtered = self._get_filtered()
            self.selected_index = min(
                max(0, len(filtered) - 1), self.selected_index + self.page_size
            )

        @kb.add("home")
        def _home(event: Any) -> None:
            self.selected_index = 0

        @kb.add("end")
        def _end(event: Any) -> None:
            filtered = self._get_filtered()
            self.selected_index = max(0, len(filtered) - 1)

        @kb.add("enter")
        def _enter(event: Any) -> None:
            filtered = self._get_filtered()
            if filtered and 0 <= self.selected_index < len(filtered):
                target = filtered[self.selected_index]
                self.result = target
                sid = target.get("id")
                if self.retry_fn and sid:
                    self._trigger_retry(sid)
                else:
                    event.app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _cancel(event: Any) -> None:
            event.app.exit()

        @kb.add("q")
        def _q(event: Any) -> None:
            if not self.search_query:
                event.app.exit()
            else:
                self.search_query += "q"
                self.selected_index = 0

        @kb.add("backspace")
        def _backspace(event: Any) -> None:
            if self.search_query:
                self.search_query = self.search_query[:-1]
                self.selected_index = 0

        @kb.add(Keys.Any)
        def _char(event: Any) -> None:
            c = event.data
            if c and c.isprintable():
                self.search_query += c
                self.selected_index = 0

        content_control = FormattedTextControl(self._get_formatted_text, show_cursor=False)
        window = Window(
            content=content_control,
            height=Dimension(min=3, max=self.page_size + 3),
            dont_extend_height=True,
        )
        layout = Layout(HSplit([window]))
        app: Application[None] = Application(
            layout=layout,
            key_bindings=kb,
            style=PICKER_STYLE,
            full_screen=False,
            erase_when_done=True,
            input=input,
            output=output,
        )
        self._app = app
        try:
            _safe_run_app(app)
        finally:
            self._app = None
        return self.result
