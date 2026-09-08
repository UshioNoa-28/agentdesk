"""长期记忆服务抽象端口协议。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class MemoryServicePort(Protocol):
    """跨会话长期记忆存储与语义检索端口。"""

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 3,
    ) -> list[str]:
        """按语义检索属于该用户的相关记忆事实。"""

        ...

    async def add(
        self,
        messages: Sequence[dict[str, str]] | str,
        *,
        user_id: str,
    ) -> None:
        """从增量对话或事实文本中提炼记忆，并自动进行消歧与持久化。"""

        ...

    async def get_all(
        self,
        *,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """获取属于该用户的所有长期记忆列表。"""

        ...

    async def delete(
        self,
        memory_id: str,
    ) -> None:
        """显式删除某一条记忆。"""

        ...


__all__ = ["MemoryServicePort"]
