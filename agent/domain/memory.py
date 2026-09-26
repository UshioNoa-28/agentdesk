"""供应商无关的长期记忆值对象。

目录发现与索引解析属于 infrastructure（``agent.infrastructure.memory``）；
消费方（system prompt 注入、CLI list）只接收这里的稳定值对象，因此以后
把本地目录换成远程记忆服务时，不需要修改上层。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MemoryLayer(StrEnum):
    """记忆的两层归属：跨项目的人 vs 当前仓库的事实。"""

    USER = "user"
    PROJECT = "project"


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """MEMORY.md 里一条已索引的记忆。

    list 只到索引为止：title 即笔记文件名，正文由消费方自行读取。
    """

    layer: MemoryLayer
    title: str
    description: str


__all__ = ["MemoryEntry", "MemoryLayer"]
