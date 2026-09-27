"""文件式长期记忆的存储端口。

写入端（``remember`` / ``forget``）只需要"追加一条待整理记录"；后台整理者
才需要读日志和改笔记。笔记这一面单独成端口：主 Agent 的工具永远拿不到它，
隔离靠"谁的构造参数里出现 ``MemoryNoteBookPort``"就能一眼看穿。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from agent.domain.memory import (
    MemoryLayer,
    MemoryNote,
    MemoryNoteEntry,
    MemoryOperation,
    PendingRecord,
)


class MemoryStorePort(Protocol):
    """两层记忆目录与 append-only 待整理日志的读写入口。"""

    @property
    def enabled(self) -> bool:
        """长期记忆是否开启；关闭时记录类操作一律拒绝。"""
        ...

    def layer_root(self, layer: MemoryLayer) -> Path:
        """某一层记忆的根目录（笔记与索引都在其下）。"""
        ...

    def inbox_path(self, layer: MemoryLayer, session_id: str) -> Path:
        """某层某个 session 的待整理日志路径；目录可能尚不存在。"""
        ...

    def pending_records(self, layer: MemoryLayer) -> tuple[PendingRecord, ...]:
        """某层全部尚未整理的记录（跨 session），按写入时间排序。"""
        ...

    def discard_pending(
        self,
        layer: MemoryLayer,
        records: Sequence[PendingRecord],
    ) -> None:
        """摘掉整理成功的记录；期间新追加的行必须留下。"""
        ...

    def record(
        self,
        operation: MemoryOperation,
        content: str,
        *,
        session_id: str,
        layer: MemoryLayer,
    ) -> PendingRecord:
        """校验并追加一条待整理记录；非法输入抛 ``MemoryWriteError``。"""
        ...


class MemoryNoteBookPort(Protocol):
    """笔记与 ``MEMORY.md`` 索引的唯一写入口。

    索引每行只存三样：标题、时间、描述。标题即身份——正文文件与索引同级，
    就叫 ``<标题>.md``，没有单独的 id。实现者负责索引行的序列化与落盘顺序；
    调用方只提供标题、描述和正文，拿不到"手写索引"这个面。找不到笔记抛
    ``MemoryNotFound``，输入不合规或落盘失败抛 ``MemoryWriteError``。
    """

    def list_entries(self, layer: MemoryLayer) -> tuple[MemoryNoteEntry, ...]:
        """某层的全部索引条目，顺序即索引里的书写顺序。"""
        ...

    def read_note(self, layer: MemoryLayer, title: str) -> MemoryNote:
        """按标题读一条笔记（条目 + 正文）。"""
        ...

    def write_note(
        self,
        layer: MemoryLayer,
        *,
        title: str,
        description: str,
        body: str,
    ) -> MemoryNote:
        """新建一条笔记并把它追加进索引；标题即文件名。"""
        ...

    def update_note(
        self,
        layer: MemoryLayer,
        title: str,
        *,
        description: str | None = None,
        body: str | None = None,
    ) -> MemoryNote:
        """按标题改一条笔记；未给出的字段保持不变，时间戳刷新。"""
        ...

    def delete_note(self, layer: MemoryLayer, title: str) -> None:
        """按标题从索引摘除并删掉正文文件。"""
        ...


class MemoryConsolidatorPort(Protocol):
    """后台消化器：把两层日志里的待整理记录折叠成笔记。"""

    async def consolidate(self, *, session_id: str) -> None:
        """消化当前积压；没有待整理记录时静默返回，失败只记日志。"""
        ...


__all__ = [
    "MemoryConsolidatorPort",
    "MemoryNoteBookPort",
    "MemoryStorePort",
]
