"""``read_file`` workspace Meta Tool."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
from agent.infrastructure.metatools.workspace._base import (
    DEFAULT_MAX_FILE_BYTES,
    absolute_user_path,
    clamp_limit,
    positive_limit,
)

READ_FILE_DEFINITION = ToolDefinition(
    name="read_file",
    description=(
        "Read a UTF-8 text file by absolute path. Every returned line is "
        "prefixed with its true file line number in cat -n style; the prefix is "
        "display-only, strip it before feeding content back to edit_file. "
        "Results are byte-limited and may be marked truncated."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Absolute file path.",
            },
            "max_bytes": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional maximum number of file bytes to return.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional one-based first line to return.",
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional inclusive one-based last line to return.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    keywords=("file", "read", "source", "text"),
    use_cases=("Inspect source files", "Read configuration or documentation"),
)


class ReadFileTool(AgentTool):
    """Read bounded text from a file addressed by absolute path."""

    def __init__(
        self,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self._max_file_bytes = positive_limit(max_file_bytes, DEFAULT_MAX_FILE_BYTES)

    @property
    def definition(self) -> ToolDefinition:
        return READ_FILE_DEFINITION

    def permission_targets(self, arguments: Mapping[str, object]) -> tuple[str, ...]:
        """权限层按归一化后的绝对路径匹配；path 非法直接抛参数错误，不询问。"""

        return (absolute_user_path(arguments.get("path")).as_posix(),)

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        try:
            path = absolute_user_path(arguments.get("path"))
            max_bytes = clamp_limit(arguments, "max_bytes", self._max_file_bytes)
            start_line: int | None = arguments.get("start_line")
            end_line: int | None = arguments.get("end_line")
            if start_line is not None and end_line is not None and end_line < start_line:
                raise ValueError("end_line must be greater than or equal to start_line")
            payload = await asyncio.to_thread(
                self._read,
                path,
                max_bytes,
                start_line,
                end_line,
            )
            return ok_result(payload)
        except (OSError, UnicodeError, ValueError) as exc:
            return error_result("read_file_failed", str(exc))

    def _read(
        self,
        path: Path,
        max_bytes: int,
        start_line: int | None,
        end_line: int | None,
    ) -> dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path.as_posix()}")
        if not path.is_file():
            raise IsADirectoryError(f"path is not a regular file: {path.as_posix()}")
        size = path.stat().st_size
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
        truncated = len(raw) > max_bytes
        text = raw[:max_bytes].decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        first = (start_line or 1) - 1
        last = end_line if end_line is not None else len(lines)
        numbered = "".join(
            f"{number:6d}\t{line}" for number, line in enumerate(lines[first:last], start=first + 1)
        )
        return {
            "path": path.as_posix(),
            "content": numbered,
            "size_bytes": size,
            "truncated": truncated,
            "start_line": start_line,
            "end_line": end_line,
        }


__all__ = ["READ_FILE_DEFINITION", "ReadFileTool"]
