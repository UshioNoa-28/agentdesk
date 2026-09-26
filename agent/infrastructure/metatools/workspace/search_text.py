"""``search_text`` workspace Meta Tool."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator, Mapping
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
    DEFAULT_MAX_SEARCH_RESULTS,
    absolute_user_path,
    clamp_limit,
    positive_limit,
)

SEARCH_TEXT_DEFINITION = ToolDefinition(
    name="search_text",
    description=(
        "Search for a literal text string in files under an absolute path (a "
        "file, or a directory searched recursively; symlinked directories are "
        "not followed). Results include file and line numbers and are bounded by "
        "a result-count limit."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Literal text to find (not a regular expression).",
            },
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Absolute file or directory path to search.",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether matching is case-sensitive (default true).",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional maximum number of matching lines.",
            },
        },
        "required": ["query", "path"],
        "additionalProperties": False,
    },
    keywords=("search", "grep", "find", "text"),
    use_cases=("Find symbol references", "Locate configuration values"),
)


class SearchTextTool(AgentTool):
    """Search literal text in bounded files addressed by absolute path."""

    def __init__(
        self,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self._max_file_bytes = positive_limit(max_file_bytes, DEFAULT_MAX_FILE_BYTES)

    @property
    def definition(self) -> ToolDefinition:
        return SEARCH_TEXT_DEFINITION

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
            query = arguments["query"]
            path = absolute_user_path(arguments.get("path"))
            case_sensitive = arguments.get("case_sensitive", True)
            max_results = clamp_limit(arguments, "max_results", DEFAULT_MAX_SEARCH_RESULTS)
            payload = await asyncio.to_thread(
                self._search,
                path,
                query,
                case_sensitive,
                max_results,
            )
            return ok_result(payload)
        except (OSError, UnicodeError, ValueError) as exc:
            return error_result("search_text_failed", str(exc))

    def _search(
        self,
        path: Path,
        query: str,
        case_sensitive: bool,
        max_results: int,
    ) -> dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(f"path not found: {path.as_posix()}")
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, object]] = []
        files_scanned = 0
        truncated = False
        for file_path in _iter_files(path):
            if len(matches) >= max_results:
                truncated = True
                break
            try:
                files_scanned += 1
                with file_path.open("rb") as handle:
                    raw = handle.read(self._max_file_bytes + 1)
                if len(raw) > self._max_file_bytes:
                    truncated = True
                text = raw[: self._max_file_bytes].decode("utf-8", errors="replace")
                if "\x00" in text:
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    haystack = line if case_sensitive else line.casefold()
                    if needle in haystack:
                        matches.append(
                            {
                                "path": file_path.as_posix(),
                                "line": line_number,
                                "text": line[:4096],
                            }
                        )
                        if len(matches) >= max_results:
                            truncated = True
                            break
            except (OSError, UnicodeError):
                continue
        return {
            "query": query,
            "matches": matches,
            "match_count": len(matches),
            "files_scanned": files_scanned,
            "truncated": truncated,
        }


def _iter_files(path: Path) -> Iterator[Path]:
    """Yield regular files without following symlinked directories."""

    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                children = sorted(iterator, key=lambda item: item.name, reverse=True)
        except OSError:
            continue
        for item in children:
            child = Path(item.path)
            try:
                if item.is_symlink():
                    continue
                if item.is_dir(follow_symlinks=False):
                    stack.append(child)
                elif item.is_file(follow_symlinks=False):
                    yield child
            except OSError:
                continue


__all__ = ["SEARCH_TEXT_DEFINITION", "SearchTextTool"]
