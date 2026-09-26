"""Agent Message 持久化用例端口协议。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from agent.domain.messages import Message
from agent.domain.model_messages import ModelMessage


class MessageServicePort(Protocol):
    """封装 Message 仓储常用事务操作的应用服务端口。"""

    async def add(
        self,
        *,
        session_id: str,
        message: ModelMessage,
        metadata: Mapping[str, object] | None = None,
    ) -> Message:
        """分配 Session 内序号并提交一条模型消息。"""

        ...

    async def history(self, session_id: str) -> list[Message]:
        """读取当前 Session 的完整、可继承 Message 历史。"""

        ...
__all__ = ["MessageServicePort"]
