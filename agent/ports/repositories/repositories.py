"""Agent Session 与 Message 仓储端口协议。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from agent.domain.entities import Session
from agent.domain.messages import Message
from agent.domain.model_messages import ModelMessage


class SessionRepository(Protocol):
    """长期会话元数据仓储。"""

    async def add(self, session: Session) -> None: ...

    async def get(self, session_id: str) -> Session | None: ...

    async def list_all(self) -> list[Session]: ...

    async def list_by_main_session(self, main_session_id: str) -> list[Session]: ...

    async def has_children(self, session_id: str) -> bool: ...

    async def save(self, session: Session) -> Session: ...

    async def delete(self, session_id: str) -> bool: ...


class MessageRepository(Protocol):
    """Session 消息历史仓储。"""

    async def add(self, message: Message) -> Message: ...

    async def add_with_next_seq(
        self,
        *,
        session_id: str,
        message: ModelMessage,
        metadata: Mapping[str, Any] | None = None,
    ) -> Message | None:
        """原子地按会话内 max(seq) 续号插入一条消息。

        seq 计算与插入在同一条语句内完成,并发追加互不撞号;会话行
        不存在时返回 ``None`` 且不写入。
        """
        ...

    async def list_for_session(
        self,
        session_id: str,
        *,
        max_seq: int | None = None,
    ) -> list[Message]: ...

    async def last_seq(self, session_id: str) -> int: ...

    async def inbound_last_seq(self, session_id: str) -> int:
        """返回session的最大seq的human message """
        ...

    async def delete_for_session(self, session_id: str) -> int: ...


__all__ = [
    "MessageRepository",
    "SessionRepository",
]
