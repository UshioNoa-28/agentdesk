"""长期记忆的值对象。

记忆分两层，语义不同：``user`` 层属于个人（主目录），``project`` 层随仓库
走（可能被提交或分享）。任何一条记录先落进所在层的待整理日志，由后台整理
者改写成笔记和 ``MEMORY.md`` 索引；本模块只描述记录本身，不碰文件系统。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MemoryLayer(StrEnum):
    """记录归属的记忆层级。"""

    USER = "user"
    PROJECT = "project"


class MemoryOperation(StrEnum):
    """待整理记录的动作：追加事实或声明某条事实已失效。"""

    REMEMBER = "remember"
    FORGET = "forget"


@dataclass(frozen=True, slots=True)
class PendingRecord:
    """一条尚未整理的记忆记录。"""

    operation: MemoryOperation
    content: str
    session_id: str
    created_at: datetime


class MemoryWriteError(ValueError):
    """记录本身不合法：空白内容、超长，或 session id 不能作文件名。"""


class MemoryNotFound(ValueError):
    """被引用的笔记不存在（索引里没有这个标题）。"""


@dataclass(frozen=True, slots=True)
class MemoryNoteEntry:
    """索引里的一行：标题、一句话描述和最后一次改写的时间。

    标题就是这条记忆在本层的身份——正文文件与索引同级、就叫 ``<标题>.md``，所以
    不设单独的 id，换标题等于换一条笔记。
    """

    title: str
    description: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class MemoryNote:
    """一条完整笔记：索引行加正文。"""

    entry: MemoryNoteEntry
    body: str


__all__ = [
    "MemoryLayer",
    "MemoryNotFound",
    "MemoryNote",
    "MemoryNoteEntry",
    "MemoryOperation",
    "MemoryWriteError",
    "PendingRecord",
]
