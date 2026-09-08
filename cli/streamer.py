"""Progressive sliding-window Markdown streamer for clean, non-duplicating terminal output."""

from __future__ import annotations

import io
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text


from cli.ui import cli_theme


class MarkdownStreamer:
    """Streams markdown incrementally using a sliding window to prevent duplicate lines.

    When rendering long markdown responses with rich.live.Live, setting vertical_overflow='visible'
    on the entire document causes lines that scroll above the terminal viewport to be reprinted
    on every tick (because the terminal clamps cursor movement at row 0).

    This streamer solves it by:
    1. Rendering the accumulated markdown with proper styling and right margin padding.
    2. Splitting output lines into 'stable' lines (older than live_window) and 'active' lines.
    3. Emitting stable lines directly to console.print() above the Live window (permanent scrollback).
    4. Only rendering the active tail (last live_window lines) inside the Live window.
    5. On finish, flushing all remaining lines and stopping Live cleanly.
    """

    def __init__(
        self,
        console: Console,
        live_window: int = 5,
        min_delay: float = 0.04,
    ) -> None:
        self.console = console
        self.live_window = live_window
        self.min_delay = min_delay
        self.printed_lines: list[str] = []
        self.last_update_time: float = 0.0
        self.live: Live | None = None
        self._started = False
        self.right_pad = max(6, int(console.width * 0.12))

    def _render_to_lines(self, text: str) -> list[str]:
        buf = io.StringIO()
        render_console = Console(
            file=buf,
            force_terminal=True,
            width=self.console.width,
            color_system=self.console.color_system,
            theme=cli_theme,
        )
        md = Padding(Markdown(text, code_theme="one-dark"), (0, self.right_pad, 0, 0))
        render_console.print(md)
        return buf.getvalue().splitlines(keepends=True)

    def start(self) -> None:
        if not self._started:
            self.live = Live(
                Text(""),
                console=self.console,
                refresh_per_second=15,
                vertical_overflow="crop",
            )
            self.live.start()
            self._started = True

    def update(self, text: str, final: bool = False) -> None:
        if not text:
            return

        if not self._started:
            self.start()

        now = time.time()
        if not final and now - self.last_update_time < self.min_delay:
            return
        self.last_update_time = now

        lines = self._render_to_lines(text)
        num_lines = len(lines)
        stable_count = num_lines if final else max(0, num_lines - self.live_window)

        if stable_count > len(self.printed_lines):
            new_stable = lines[len(self.printed_lines):stable_count]
            out = Text.from_ansi("".join(new_stable))
            if self.live:
                self.live.console.print(out)
            else:
                self.console.print(out)
            self.printed_lines = lines[:stable_count]

        if final:
            if self.live:
                try:
                    self.live.update(Text(""))
                    self.live.stop()
                except Exception:
                    pass
                self.live = None
            self._started = False
            return

        if self.live:
            tail = lines[stable_count:]
            self.live.update(Text.from_ansi("".join(tail)))

    def finish(self, full_text: str | None = None) -> None:
        if full_text:
            self.update(full_text, final=True)
        elif self.live:
            try:
                self.live.update(Text(""))
                self.live.stop()
            except Exception:
                pass
            self.live = None
            self._started = False
