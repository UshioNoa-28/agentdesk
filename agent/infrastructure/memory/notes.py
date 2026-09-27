"""笔记与 ``MEMORY.md`` 索引：结构由代码维护，模型只提供内容。

物理布局就是一层目录摊平：

- ``MEMORY.md`` 每行三列（制表符分隔）：``标题	时间	描述``；
- 正文另起一个文件 ``<标题>.md``，与索引同级。

标题即身份：它同时是索引第一列和正文文件名，没有另外一套 id，换标题等于换
一条笔记。整理者拿不到"手写索引"这个面——它只能通过 ``NoteBook`` 的增删改，
索引行由代码序列化后再落盘。

写入顺序按"索引是可见性闸门"来定：新建与修改先写正文再改索引，删除先改索引
再删文件。中途崩溃的后果都是一个文件多出来而索引里没有它，也就是一条看不到
的记忆，不会产生半条错误事实。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent.domain.memory import (
    MemoryLayer,
    MemoryNote,
    MemoryNoteEntry,
    MemoryNotFound,
    MemoryWriteError,
)
from agent.infrastructure.project_settings import atomic_write
from agent.ports.services import MemoryNoteBookPort, MemoryStorePort

MEMORY_INDEX_FILE_NAME = "MEMORY.md"
ENTRY_SEPARATOR = "\t"
NOTE_SUFFIX = ".md"

MAX_TITLE_CHARS = 64
MAX_DESCRIPTION_CHARS = 500
MAX_BODY_CHARS = 20_000


class NoteBook(MemoryNoteBookPort):
    """一层记忆的读写：索引行与正文文件同步变化，索引是唯一目录。"""

    def __init__(self, store: MemoryStorePort) -> None:
        self._store = store

    def index_path(self, layer: MemoryLayer) -> Path:
        """某层的 ``MEMORY.md``；文件不存在是正常状态，等价于零条记忆。"""

        return self._store.layer_root(layer) / MEMORY_INDEX_FILE_NAME

    def note_path(self, layer: MemoryLayer, title: str) -> Path:
        """某层某条笔记的正文文件路径；标题即文件名。"""

        return self._store.layer_root(layer) / f"{title}{NOTE_SUFFIX}"

    def read_index(self, layer: MemoryLayer) -> str:
        path = self.index_path(layer)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def list_entries(self, layer: MemoryLayer) -> tuple[MemoryNoteEntry, ...]:
        """解析索引全文；不合规的行直接抛错，宁可停下也不静默丢记忆。"""

        entries: list[MemoryNoteEntry] = []
        for line in self.read_index(layer).splitlines():
            if not line.strip():
                continue
            entries.append(_parse_entry(line))
        return tuple(entries)

    def read_note(self, layer: MemoryLayer, title: str) -> MemoryNote:
        name = _title_of(title)
        entry = self._entry(layer, name)
        path = self.note_path(layer, name)
        if not path.is_file():
            raise MemoryNotFound(f"note {name!r} is indexed but its file is missing: {path}")
        return MemoryNote(entry=entry, body=path.read_text(encoding="utf-8"))

    def write_note(
        self,
        layer: MemoryLayer,
        *,
        title: str,
        description: str,
        body: str,
    ) -> MemoryNote:
        name = _title_of(title)
        if any(item.title == name for item in self.list_entries(layer)):
            raise MemoryWriteError(f"a note titled {name!r} already exists; update it instead")
        entry = MemoryNoteEntry(
            title=name,
            description=_description_of(description),
            timestamp=datetime.now(UTC),
        )
        text = _normalized_body(body)
        self._write_body(layer, name, text)
        self._replace_entries(layer, [*self.list_entries(layer), entry])
        return MemoryNote(entry=entry, body=text)

    def update_note(
        self,
        layer: MemoryLayer,
        title: str,
        *,
        description: str | None = None,
        body: str | None = None,
    ) -> MemoryNote:
        name = _title_of(title)
        note = self.read_note(layer, name)
        updated = MemoryNoteEntry(
            title=name,
            description=note.entry.description
            if description is None
            else _description_of(description),
            timestamp=datetime.now(UTC),
        )
        text = note.body
        if body is not None:
            text = _normalized_body(body)
            self._write_body(layer, name, text)
        self._replace_entries(
            layer,
            [updated if item.title == name else item for item in self.list_entries(layer)],
        )
        return MemoryNote(entry=updated, body=text)

    def delete_note(self, layer: MemoryLayer, title: str) -> None:
        name = _title_of(title)
        self._entry(layer, name)
        self._replace_entries(
            layer,
            [item for item in self.list_entries(layer) if item.title != name],
        )
        self.note_path(layer, name).unlink(missing_ok=True)

    def _entry(self, layer: MemoryLayer, title: str) -> MemoryNoteEntry:
        for entry in self.list_entries(layer):
            if entry.title == title:
                return entry
        raise MemoryNotFound(f"no memory note titled {title!r} in layer {layer.value!r}.")

    def _write_body(self, layer: MemoryLayer, title: str, body: str) -> None:
        path = self.note_path(layer, title)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, _normalized_body(body))

    def _replace_entries(self, layer: MemoryLayer, entries: list[MemoryNoteEntry]) -> None:
        path = self.index_path(layer)
        if not entries:
            # 索引空了就删掉：一条笔记都没有的层不该留下一个空文件。
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, "".join(_serialize(entry) for entry in entries))


def _serialize(entry: MemoryNoteEntry) -> str:
    return (
        f"{entry.title}{ENTRY_SEPARATOR}"
        f"{entry.timestamp.isoformat()}{ENTRY_SEPARATOR}"
        f"{entry.description}\n"
    )


def _parse_entry(line: str) -> MemoryNoteEntry:
    parts = line.split(ENTRY_SEPARATOR)
    if len(parts) != 3:
        raise MemoryWriteError(f"memory index line is not three tab-separated columns: {line!r}")
    title, stamp, description = parts
    try:
        timestamp = datetime.fromisoformat(stamp)
    except ValueError as exc:
        raise MemoryWriteError(f"memory index line has an unparseable timestamp: {line!r}") from exc
    return MemoryNoteEntry(title=title, description=description, timestamp=timestamp)


def _title_of(title: str) -> str:
    """标题即文件名：只容得下小写 slug，超限或含分隔符直接报错。"""

    name = " ".join(title.split()).lower().replace(" ", "-")
    if not name:
        raise MemoryWriteError("title must not be empty")
    if len(name) > MAX_TITLE_CHARS:
        raise MemoryWriteError(f"title exceeds {MAX_TITLE_CHARS} characters; shorten it")
    if "/" in name or "\\" in name or name.startswith("."):
        raise MemoryWriteError("title cannot be used as a file name; use words and dashes")
    return name


def _description_of(description: str) -> str:
    collapsed = " ".join(description.replace("\t", " ").split())
    if not collapsed:
        raise MemoryWriteError("description must not be empty")
    if len(collapsed) > MAX_DESCRIPTION_CHARS:
        raise MemoryWriteError(
            f"description exceeds {MAX_DESCRIPTION_CHARS} characters; shorten it"
        )
    return collapsed


def _normalized_body(body: str) -> str:
    if "\x00" in body:
        raise MemoryWriteError("note body must not contain NUL bytes")
    text = body.replace("\r\n", "\n").strip()
    if not text:
        raise MemoryWriteError("note body must not be empty")
    if len(text) > MAX_BODY_CHARS:
        raise MemoryWriteError(f"note body exceeds {MAX_BODY_CHARS} characters; split it")
    return text + "\n"


__all__ = [
    "ENTRY_SEPARATOR",
    "MEMORY_INDEX_FILE_NAME",
    "MAX_BODY_CHARS",
    "MAX_DESCRIPTION_CHARS",
    "MAX_TITLE_CHARS",
    "NOTE_SUFFIX",
    "NoteBook",
]
