"""长期记忆的路径口径、索引 wire format 与只读目录服务。

两层目录（用户级 ``~/.agent-desk/memory/``、项目级 ``<workspace>/.agent-desk/memory/``）
的形态协议住在这里：``MEMORY.md`` 纯索引（一行
``- [标题](标题.md) — 描述``），每条记忆一个 ``<标题>.md`` 笔记文件。
写入端是 ``remember`` Meta Tool（见 ``agent.infrastructure.metatools.remember``），
与它共用 ``format_index_line`` / ``parse_index_line`` 一套口径；读取端是
``MemoryCatalog`` 的两个接口——system prompt 的目录路径清单（``load_prompt``）
和给 CLI 展示用的结构化 list（``list_entries``）。本模块只读不写、不持有状态。
"""

from __future__ import annotations

import re
from pathlib import Path

from agent.domain.memory import MemoryEntry, MemoryLayer
from agent.infrastructure.project_settings import AGENTDESK_DIR_NAME

MEMORY_DIR_NAME = "memory"
MEMORY_INDEX_FILE_NAME = "MEMORY.md"
MEMORY_INDEX_SEPARATOR = " — "

_INDEX_LINE_RE = re.compile(
    r"^- \[(?P<title>[^\]]+)\]\([^\)\[\]]+\.md\)"
    + re.escape(MEMORY_INDEX_SEPARATOR)
    + r"(?P<desc>.*)$"
)


def memory_layer_dirs(workspace: Path) -> dict[MemoryLayer, Path]:
    """两层记忆目录的唯一路径口径：remember 工具与 prompt 注入共用。"""

    return {
        MemoryLayer.USER: Path.home() / AGENTDESK_DIR_NAME / MEMORY_DIR_NAME,
        MemoryLayer.PROJECT: workspace / AGENTDESK_DIR_NAME / MEMORY_DIR_NAME,
    }


def read_index(path: Path) -> str:
    """读取一个 MEMORY.md 索引原文；文件不存在是正常状态，等价于空。"""

    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def format_index_line(title: str, description: str) -> str:
    """索引行的唯一写出形态；与 parse_index_line 严格互逆。"""

    return f"- [{title}]({title}.md){MEMORY_INDEX_SEPARATOR}{description}"


def parse_index_line(line: str) -> tuple[str, str] | None:
    """解析一条索引行，返回 (title, description)；不符合协议返回 None。"""

    match = _INDEX_LINE_RE.match(line)
    if match is None:
        return None
    return match.group("title"), match.group("desc")


def load_memory_prompt(user_dir: Path, project_dir: Path) -> str:
    """渲染两层记忆目录的路径清单。

    只给路径、不注入索引全文：system prompt 要保持静态前缀（KV cache），
    remember 的写入不再使 prompt 变化；模型需要索引时自己现读 MEMORY.md。
    """

    return (
        f"- user layer: {user_dir.as_posix()}\n"
        f"- project layer: {project_dir.as_posix()}"
    )


class MemoryCatalog:
    """两层记忆的只读服务：load（prompt 用的路径清单）与 list（结构化索引）。

    list 每次调用都重新读盘——记忆会在会话中途被 remember 改写，CLI 要
    看到最新状态。list 止步于索引（title + description），正文由消费方用
    通用读文件能力自行获取；不符合协议的行直接跳过——展示路径宽容，
    严格校验由写入端的 remember 工具把关。
    """

    def __init__(self, *, user_dir: Path, project_dir: Path) -> None:
        self._user_dir = user_dir
        self._project_dir = project_dir

    def load_prompt(self) -> str:
        return load_memory_prompt(self._user_dir, self._project_dir)

    def list_entries(self) -> tuple[MemoryEntry, ...]:
        layers = (
            (MemoryLayer.USER, self._user_dir),
            (MemoryLayer.PROJECT, self._project_dir),
        )
        entries: list[MemoryEntry] = []
        for layer, directory in layers:
            index = read_index(directory / MEMORY_INDEX_FILE_NAME)
            for line in index.splitlines():
                parsed = parse_index_line(line)
                if parsed is None:
                    continue
                title, description = parsed
                entries.append(MemoryEntry(layer=layer, title=title, description=description))
        return tuple(entries)

    def read_note(self, layer: MemoryLayer | str, title: str) -> str | None:
        raw_layer = layer.value if hasattr(layer, "value") else str(layer)
        directory = self._user_dir if raw_layer == MemoryLayer.USER else self._project_dir
        note_file = directory / f"{title}.md"
        if not note_file.exists():
            return None
        return note_file.read_text(encoding="utf-8")


__all__ = [
    "MEMORY_DIR_NAME",
    "MEMORY_INDEX_FILE_NAME",
    "MEMORY_INDEX_SEPARATOR",
    "MemoryCatalog",
    "format_index_line",
    "load_memory_prompt",
    "memory_layer_dirs",
    "parse_index_line",
    "read_index",
]
