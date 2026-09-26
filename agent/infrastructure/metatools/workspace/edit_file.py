"""``edit_file`` workspace Meta Tool."""

from __future__ import annotations

import asyncio
import os
import stat
import tempfile
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
    positive_limit,
)

EDIT_FILE_DEFINITION = ToolDefinition(
    name="edit_file",
    description=(
        "Replace one exact snippet in an existing file by absolute path: "
        "old_text must occur exactly once and is swapped for new_text. The write "
        "uses a same-directory temporary file and atomic replacement. An empty "
        "new_text deletes the matched snippet."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Absolute file path.",
            },
            "old_text": {
                "type": "string",
                "minLength": 1,
                "description": "Exact existing text to replace; it must occur exactly once.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text; an empty string deletes old_text.",
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
    keywords=("file", "edit", "replace"),
    use_cases=("Apply a reviewed file change",),
)


class EditFileTool(AgentTool):
    """Atomically swap one unique snippet inside an existing file."""

    def __init__(
        self,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self._max_file_bytes = positive_limit(max_file_bytes, DEFAULT_MAX_FILE_BYTES)

    @property
    def definition(self) -> ToolDefinition:
        return EDIT_FILE_DEFINITION

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
            old_text: str = arguments["old_text"]
            new_text: str = arguments["new_text"]
            if len(new_text.encode("utf-8")) > self._max_file_bytes:
                raise ValueError(f"new_text exceeds the {self._max_file_bytes}-byte file limit")
            payload = await asyncio.to_thread(self._edit, path, old_text, new_text)
            return ok_result(payload)
        except (OSError, UnicodeError, ValueError) as exc:
            return error_result("edit_file_failed", str(exc))

    def _edit(self, path: Path, old_text: str, new_text: str) -> dict[str, object]:
        if not path.is_file():
            raise FileNotFoundError(f"not a readable regular file: {path.as_posix()}")

        with path.open("rb") as handle:
            existing_raw = handle.read(self._max_file_bytes + 1)
        if len(existing_raw) > self._max_file_bytes:
            raise ValueError(f"existing file exceeds the {self._max_file_bytes}-byte edit limit")
        try:
            existing = existing_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("target file is not valid UTF-8 text") from exc

        occurrences = existing.count(old_text)
        if occurrences != 1:
            raise ValueError(f"old_text must occur exactly once; found {occurrences} occurrences")
        encoded = existing.replace(old_text, new_text, 1).encode("utf-8")
        if len(encoded) > self._max_file_bytes:
            raise ValueError(f"result exceeds the {self._max_file_bytes}-byte file limit")

        try:
            previous_mode: int | None = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            previous_mode = None

        temporary_name: str | None = None
        try:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".agentdesk-tmp", dir=str(path.parent)
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if previous_mode is not None:
                os.chmod(temporary_name, previous_mode)
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

        return {
            "path": path.as_posix(),
            "bytes_written": len(encoded),
        }


__all__ = ["EDIT_FILE_DEFINITION", "EditFileTool"]
