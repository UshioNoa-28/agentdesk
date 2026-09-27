"""两层长期记忆的落盘原语：层级目录与 append-only 待整理日志。

主 agent 只能往这里追加记录，永远不碰笔记和索引：一条记录就是
``<layer>/inbox/<session_id>.md`` 里的一行。谁追加、追加多少次都不会破坏结
构，结构由后台整理者负责（它读这些日志，改写入笔记与 ``MEMORY.md``，成功后
删除已消化的日志）。

每个 session 一个日志文件是刻意的：同一 workspace 里并存的会话共享同一个
inbox 目录，按 session 分文件后写入端连锁都不需要。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from agent.domain.memory import MemoryLayer, MemoryOperation, MemoryWriteError, PendingRecord
from agent.infrastructure.project_settings import AGENTDESK_DIR_NAME, atomic_write
from agent.ports.services import MemoryStorePort

MEMORY_DIR_NAME = ".memory"
INBOX_DIR_NAME = "inbox"
MAX_RECORD_CHARS = 2_000

# 一行就是一条待整理记录：``- <时间> <动作> <内容>``。人和整理者都读得懂，
# 内容里的换行在写入时已被折叠，所以按行切分不会错位。
_RECORD_LINE = re.compile(
    r"^- (?P<at>\S+) (?P<op>remember|forget) (?P<content>.*)$"
)

# session_id 直接成为文件名：uuid4 的字符集远小于此模式，放宽只为兼容历史
# id，同时仍挡掉分隔符与 "." 这类能改写路径形状的字符。
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


class MemoryStore(MemoryStorePort):
    """定位两层记忆目录，并把记录追加进对应 session 的待整理日志。"""

    def __init__(
        self,
        *,
        start: Path,
        home: Path | None = None,
        enabled: bool = False,
    ) -> None:
        workspace = start.expanduser().resolve(strict=False)
        home_dir = (Path.home() if home is None else home).expanduser().resolve(strict=False)
        self._enabled = enabled
        self._roots = {
            MemoryLayer.USER: home_dir / AGENTDESK_DIR_NAME / MEMORY_DIR_NAME,
            MemoryLayer.PROJECT: workspace / AGENTDESK_DIR_NAME / MEMORY_DIR_NAME,
        }

    @property
    def enabled(self) -> bool:
        """长期记忆是否开启；关闭时记录类操作一律拒绝。"""

        return self._enabled

    def layer_root(self, layer: MemoryLayer) -> Path:
        """某一层记忆的根目录（笔记与索引都在其下）。"""

        return self._roots[layer]

    def inbox_path(self, layer: MemoryLayer, session_id: str) -> Path:
        """某层某个 session 的待整理日志路径；目录可能尚不存在。"""

        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise MemoryWriteError(f"session id cannot be used as a file name: {session_id!r}")
        return self.layer_root(layer) / INBOX_DIR_NAME / f"{session_id}.md"

    def record(
        self,
        operation: MemoryOperation,
        content: str,
        *,
        session_id: str,
        layer: MemoryLayer,
    ) -> PendingRecord:
        """校验并追加一条待整理记录；返回落盘后的记录本身。"""

        text = " ".join(content.split())
        if not text:
            raise MemoryWriteError("content must not be empty")
        if len(text) > MAX_RECORD_CHARS:
            raise MemoryWriteError(
                f"content exceeds the {MAX_RECORD_CHARS}-character per-record limit; "
                "split it into separate records"
            )
        pending = PendingRecord(
            operation=operation,
            content=text,
            session_id=session_id,
            created_at=datetime.now(UTC),
        )
        path = self.inbox_path(layer, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{_line(pending)}\n")
        return pending

    def discard_pending(
        self,
        layer: MemoryLayer,
        records: Sequence[PendingRecord],
    ) -> None:
        """把整理成功的记录从日志里摘掉。

        不能整文件删：整理者跑这一趟期间，主 agent 完全可能又 ``remember`` 了
        新行。日志是 append-only 且时间戳到微秒，所以"行文本相同"就是"同一条
        记录"，按行剔除即可，既不丢新行也不需要锁。
        """

        by_session: dict[str, set[str]] = {}
        for pending in records:
            by_session.setdefault(pending.session_id, set()).add(_line(pending))
        for session_id, digested in by_session.items():
            path = self.inbox_path(layer, session_id)
            if not path.is_file():
                continue
            kept = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and line not in digested
            ]
            if not kept:
                path.unlink(missing_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(path, "".join(f"{line}\n" for line in kept))

    def pending_records(self, layer: MemoryLayer) -> tuple[PendingRecord, ...]:
        """读出某层全部待整理记录（跨 session），按写入时间排序。"""

        records: list[PendingRecord] = []
        inbox = self.layer_root(layer) / INBOX_DIR_NAME
        for path in sorted(inbox.glob("*.md")) if inbox.is_dir() else []:
            stamp = path.stem
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                match = _RECORD_LINE.fullmatch(line)
                if match is None:
                    raise MemoryWriteError(
                        f"pending record line does not match the entry format: {line!r}"
                    )
                if not _SESSION_ID_PATTERN.fullmatch(stamp):
                    raise MemoryWriteError(f"pending log file name is not a session id: {path}")
                records.append(
                    PendingRecord(
                        operation=MemoryOperation(match["op"]),
                        content=match["content"],
                        session_id=stamp,
                        created_at=datetime.fromisoformat(match["at"]),
                    )
                )
        return tuple(sorted(records, key=lambda item: item.created_at))


def _line(pending: PendingRecord) -> str:
    """一条记录在日志里的那一行（不含换行）；写与摘共用，格式才不会分叉。"""

    return f"- {pending.created_at.isoformat()} {pending.operation.value} {pending.content}"


__all__ = ["INBOX_DIR_NAME", "MAX_RECORD_CHARS", "MEMORY_DIR_NAME", "MemoryStore"]
