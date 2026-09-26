"""长期记忆只读目录的端口协议。"""

from __future__ import annotations

from typing import Protocol

from agent.domain.memory import MemoryEntry, MemoryLayer


class MemoryCatalogPort(Protocol):
    """system prompt 注入与 CLI list 共用的只读端口：load 与 list。"""

    def load_prompt(self) -> str: ...

    def list_entries(self) -> tuple[MemoryEntry, ...]: ...

    def read_note(self, layer: MemoryLayer | str, title: str) -> str | None: ...


__all__ = ["MemoryCatalogPort"]
