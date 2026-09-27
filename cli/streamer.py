"""Progressive sliding-window Markdown streamer for clean, non-duplicating terminal output."""

from __future__ import annotations

import io
import time

from rich.console import Console
from rich.errors import LiveError
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text

from cli.ui import LIVE_RENDER_LOCK, cli_theme


class MarkdownStreamer:
    """Streams markdown incrementally using a sliding window to prevent duplicate lines.

    When rendering long markdown responses with rich.live.Live, setting vertical_overflow='visible'
    on the entire document causes lines that scroll above the terminal viewport to be reprinted
    on every tick (because the terminal clamps cursor movement at row 0).

    This streamer solves it by:
    1. Rendering the accumulated markdown with proper styling and right margin padding.
    2. Splitting output lines into 'stable' lines (older than live_window) and
       'active' lines.
    3. Emitting stable lines directly to console.print() above the Live window
       (permanent scrollback).
    4. Only rendering the active tail (last live_window lines) inside the Live window.
    5. On finish, flushing all remaining lines and stopping Live cleanly.
    """

    def __init__(
        self,
        console: Console,
        live_window: int = 5,
        min_delay: float = 0.04,
        live_enabled: bool = True,
    ) -> None:
        self.console = console
        self.live_window = live_window
        self.min_delay = min_delay
        self.live_enabled = live_enabled
        self.printed_lines: list[str] = []
        self.last_update_time: float = 0.0
        self.live: Live | None = None
        self._started = False
        self._lease_acquired = False
        self.right_pad = 2
        self._last_text = ""

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

    @staticmethod
    def _find_open_code_fence(text: str) -> tuple[str, int, int] | None:
        """Return (fence_str, fence_start_idx, content_start_idx) if inside an unclosed block."""
        lines = text.split("\n")
        pos = 0
        in_code = False
        fence_char = ""
        fence_len = 0
        fence_start = 0
        content_start = 0
        for line in lines:
            stripped = line.lstrip()
            if not in_code:
                if stripped.startswith("```") or stripped.startswith("~~~"):
                    fence_char = stripped[0]
                    fence_len = len(stripped) - len(stripped.lstrip(fence_char))
                    in_code = True
                    fence_start = pos
                    content_start = pos + len(line) + 1
            else:
                if stripped.startswith(fence_char * fence_len):
                    in_code = False
            pos += len(line) + 1
        if in_code:
            return (fence_char * fence_len, fence_start, min(content_start, len(text)))
        return None

    @classmethod
    def _get_renderable_markdown(cls, text: str, final: bool = False) -> tuple[str, bool]:
        """Prepare markdown for incremental streaming.

        Returns (render_text, is_open_code_block).
        Incomplete code blocks, in-progress tables, and incomplete headings are held back
        so that prematurely emitted lines with artificial syntax padding or shifting
        borders/columns do not cause permanent truncation in the terminal.
        """
        if final or not text:
            return text, False

        open_fence = cls._find_open_code_fence(text)
        if open_fence:
            fence_str, fence_start, content_start = open_fence
            code_after_fence = text[content_start:]
            last_nl_in_code = code_after_fence.rfind("\n")
            if last_nl_in_code >= 0:
                completed_code = code_after_fence[: last_nl_in_code + 1]
                render_text = text[:content_start] + completed_code + fence_str
            else:
                render_text = text[:fence_start]
            return render_text, True

        # Check if inside an in-progress table (last non-empty line starts with |)
        if not text.endswith("\n\n"):
            raw_lines = text.split("\n")
            non_empty = [item for item in raw_lines if item.strip()]
            if non_empty and non_empty[-1].lstrip().startswith("|"):
                i = len(raw_lines) - 1
                while i >= 0 and (
                    raw_lines[i].lstrip().startswith("|") or not raw_lines[i].strip()
                ):
                    i -= 1
                table_start_pos = sum(len(raw_lines[j]) + 1 for j in range(i + 1))
                render_text = text[:table_start_pos]
                return render_text, False

        # Check if last line is an incomplete heading
        last_nl = text.rfind("\n")
        last_line = text[last_nl + 1 :] if last_nl >= 0 else text
        stripped = last_line.lstrip()
        if stripped.startswith("#"):
            render_text = text[: last_nl + 1] if last_nl >= 0 else ""
            return render_text, False

        return text, False

    def start(self) -> None:
        if self._started:
            return

        if not self.live_enabled:
            self._started = True
            return

        if not LIVE_RENDER_LOCK.acquire(blocking=False):
            # Another worker owns the animated output. Keep this response in
            # buffered mode and flush it through Console.print at boundaries.
            self._started = True
            return

        self._lease_acquired = True
        try:
            self.live = Live(
                Text(""),
                console=self.console,
                refresh_per_second=15,
                vertical_overflow="crop",
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self.live.start()
        except LiveError:
            # An uncoordinated Rich Live may already own this Console. Falling
            # back must also release the process-wide animation lease.
            self._close_live()
            self._started = True
        except BaseException:
            self._close_live()
            raise
        else:
            self._started = True

    def _owns_live(self) -> bool:
        """Return whether this streamer still owns the Console live slot."""
        return self.live is not None and getattr(self.console, "_live", None) is self.live

    def _release_lease(self) -> None:
        if self._lease_acquired:
            self._lease_acquired = False
            LIVE_RENDER_LOCK.release()

    def _close_live(self) -> None:
        """Stop only our Live display and always release its render lease."""
        try:
            if self._owns_live():
                try:
                    self.live.stop()
                except BaseException:
                    pass
        finally:
            self.live = None
            self._release_lease()

    def _reset(self) -> None:
        self._close_live()
        self._started = False
        self.printed_lines.clear()
        self.last_update_time = 0.0
        self._last_text = ""

    def update(self, text: str, final: bool = False) -> None:
        if not text:
            return

        if not self._started:
            self.start()

        now = time.time()
        if not final and now - self.last_update_time < self.min_delay:
            return
        self.last_update_time = now
        self._last_text = text

        try:
            if final:
                render_text, is_open = text, False
            else:
                render_text, is_open = self._get_renderable_markdown(text, final=False)

            lines: list[str] = []
            stable_count = 0
            if render_text:
                lines = self._render_to_lines(render_text)
                num_lines = len(lines)
                if final:
                    stable_count = num_lines
                else:
                    effective_window = max(1, self.live_window) if is_open else self.live_window
                    stable_count = max(0, num_lines - effective_window)

                if stable_count > len(self.printed_lines):
                    new_stable = lines[len(self.printed_lines) : stable_count]
                    out = Text.from_ansi("".join(new_stable))
                    if self._owns_live():
                        try:
                            self.live.console.print(out)
                        except LiveError:
                            self._close_live()
                            self.console.print(out)
                    else:
                        if self.live is not None:
                            # A Live outside this coordinator replaced our handle;
                            # do not stop that display when this response finishes.
                            self._close_live()
                        self.console.print(out)
                    self.printed_lines = lines[:stable_count]

            if final:
                if self._owns_live():
                    try:
                        self.live.update(Text(""))
                    except Exception:
                        pass
                self._reset()
                return

            if self._owns_live():
                if render_text:
                    tail = lines[stable_count:]
                    try:
                        self.live.update(Text.from_ansi("".join(tail)))
                    except LiveError:
                        self._close_live()
            elif self.live is not None:
                self._close_live()
        except BaseException:
            self._reset()
            raise

    def finish(self, full_text: str | None = None) -> None:
        target = full_text if full_text is not None else self._last_text
        if target:
            self.update(target, final=True)
            return
        if self._owns_live():
            try:
                self.live.update(Text(""))
            except Exception:
                pass
        self._reset()
