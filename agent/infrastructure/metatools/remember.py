"""固定 ``remember`` Meta Tool：把一条长期记忆写进指定层的记忆目录。

每层（用户级 ``~/.agent-desk/memory/``、项目级 ``<repo>/.agent-desk/memory/``）
的形态一致：

- ``MEMORY.md``：纯索引，一行一条 ``- [标题](标题.md) — 一行描述``，
  正文不在索引里。
- ``<标题>.md``：记忆本体，文件名就是 title（中文合法），frontmatter 只记
  ``written``，其后是正文。``written`` 由本工具在每次写入时盖时间戳
  （同 title 覆盖也刷新）。

写入是本协议唯一的窄口：笔记文件与索引行的同步是最容易悄悄失散的
invariant，不让模型直接 edit 这两个文件。读取不经过这里——system prompt
只给目录路径清单，索引和细节由模型用通用 Read/Grep 在记忆目录里自行
检索；当 grep 撞出多条互相矛盾的旧记忆时，``written`` 较新的一条为准。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from agent.domain.memory import MemoryLayer
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
from agent.infrastructure.memory import (
    MEMORY_INDEX_FILE_NAME,
    format_index_line,
    parse_index_line,
    read_index,
)
from agent.infrastructure.project_settings import atomic_write

_TITLE_MAX_CHARS = 64


REMEMBER_DEFINITION = ToolDefinition(
    name="remember",
    description=(
        "Persist one long-term memory note into a memory directory and keep its "
        "index line in MEMORY.md in sync. Use layer \"user\" for facts about the "
        "person and their preferences that outlive this project, and \"project\" "
        "for facts tied to this repository. The title doubles as the note file "
        "name: re-using an existing title overwrites that note instead of "
        "creating a duplicate. Before writing, search the memory directory for "
        "related terms: if contradicting notes exist (e.g. \"likes A\" vs "
        "\"likes B\"), the note with the newer frontmatter \"written\" timestamp "
        "is current - consolidate or delete the stale one."
    ),
    parameters={
        "type": "object",
        "properties": {
            "layer": {
                "type": "string",
                "enum": [layer.value for layer in MemoryLayer],
                "description": "Which memory layer to write to.",
            },
            "title": {
                "type": "string",
                "description": (
                    "Short noun phrase naming this memory, used verbatim as the "
                    "note file name (e.g. \"游戏偏好\"); any language is fine. "
                    "Must not contain '/', '\\', '[', ']', '(' or ')', start "
                    "with '.', or exceed 64 characters."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "One-line summary of the note, appended to its MEMORY.md "
                    "index entry; this is what future sessions skim to decide "
                    "whether to open the note. Single line, no square brackets."
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "The memory content; may be as short as one sentence "
                    "(\"user likes A\") or full markdown."
                ),
            },
        },
        "required": ["layer", "title", "description", "body"],
        "additionalProperties": False,
    },
)


class RememberTool(AgentTool):
    """按层写入记忆笔记，并原子维护该层的 MEMORY.md 索引。"""

    def __init__(self, *, user_dir: Path, project_dir: Path) -> None:
        self._dirs = {
            MemoryLayer.USER: user_dir,
            MemoryLayer.PROJECT: project_dir,
        }

    @property
    def definition(self) -> ToolDefinition:
        return REMEMBER_DEFINITION

    def permission_targets(self, arguments: Mapping[str, object]) -> tuple[str, ...]:
        """目标就是笔记文件的绝对路径；layer/title 非法直接抛参数错误，不询问。"""

        directory, title = self._resolve(arguments)
        return (self._note_path(directory, title).resolve(strict=False).as_posix(),)

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        try:
            layer = MemoryLayer(str(arguments["layer"]))
            description = str(arguments["description"])
            body = str(arguments["body"])
            directory, title = self._resolve(arguments)
            _validate_description(description)
            if not body.strip():
                raise ValueError("Memory body must not be empty")

            note_path = self._note_path(directory, title)
            note_path.parent.mkdir(parents=True, exist_ok=True)
            written = datetime.now().astimezone().isoformat(timespec="seconds")
            note = "\n".join(
                [
                    "---",
                    f"written: {written}",
                    "---",
                    "",
                    body.rstrip(),
                    "",
                ]
            )
            atomic_write(note_path, note)
            index_line = format_index_line(title, description)
            atomic_write(
                directory / MEMORY_INDEX_FILE_NAME,
                _upsert_index(
                    read_index(directory / MEMORY_INDEX_FILE_NAME),
                    title,
                    index_line,
                ),
            )
        except ValueError as exc:
            return error_result("invalid_memory", str(exc))

        return ok_result(
            {
                "remembered": {
                    "layer": layer.value,
                    "title": title,
                    "description": description,
                    "written": written,
                    "path": note_path.as_posix(),
                    "index_line": index_line,
                },
            }
        )

    def _resolve(self, arguments: Mapping[str, object]) -> tuple[Path, str]:
        layer = MemoryLayer(str(arguments["layer"]))
        title = str(arguments["title"])
        _validate_title(title)
        return self._dirs[layer], title

    @staticmethod
    def _note_path(directory: Path, title: str) -> Path:
        return directory / f"{title}.md"


def _validate_title(title: str) -> None:
    if not title or not title.strip():
        raise ValueError("Memory title must not be empty")
    if title != title.strip():
        raise ValueError("Memory title must not have leading or trailing whitespace")
    if len(title) > _TITLE_MAX_CHARS:
        raise ValueError(f"Memory title must be at most {_TITLE_MAX_CHARS} characters")
    if title.startswith("."):
        raise ValueError("Memory title must not start with '.'")
    if any(ord(char) < 32 or ord(char) == 127 for char in title):
        raise ValueError("Memory title must not contain control characters")
    for char in ("/", "\\", "[", "]", "(", ")"):
        if char in title:
            raise ValueError(f"Memory title must not contain '{char}'")


def _validate_description(description: str) -> None:
    if not description.strip():
        raise ValueError("Memory description must not be empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in description):
        raise ValueError("Memory description must be a single line")
    if "[" in description or "]" in description:
        raise ValueError("Memory description must not contain square brackets")


def _upsert_index(index_text: str, title: str, line: str) -> str:
    lines = index_text.splitlines()
    for i, existing in enumerate(lines):
        parsed = parse_index_line(existing)
        if parsed is not None and parsed[0] == title:
            lines[i] = line
            break
    else:
        lines.append(line)
    return "\n".join(lines) + "\n"


__all__ = ["REMEMBER_DEFINITION", "RememberTool"]
