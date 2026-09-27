"""Agent Message 持久化应用服务。"""

from __future__ import annotations

from collections.abc import Mapping

from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage
from agent.exceptions import SessionNotFoundError
from agent.ports.repositories import MessageRepository, UnitOfWork
from agent.ports.services import MessageServicePort, SessionServicePort


def _metadata_with_default_kind(
    message: ModelMessage,
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    """未显式给 kind 时按消息角色推导默认领域分类。"""

    meta = dict(metadata or {})
    if "kind" not in meta:
        if message.role == "human":
            meta["kind"] = MessageKind.USER
        elif message.role == "assistant":
            meta["kind"] = (
                MessageKind.ASSISTANT_TOOL_CALL
                if message.tool_calls
                else MessageKind.ASSISTANT_ANSWER
            )
        elif message.role == "system":
            meta["kind"] = MessageKind.SYSTEM
        elif message.role == "tool":
            meta["kind"] = MessageKind.TOOL_RESULT
    return meta


class MessageService(MessageServicePort):
    """组合 Message Repository、Session history 和事务边界。"""

    def __init__(
        self,
        *,
        repository: MessageRepository,
        unit_of_work: UnitOfWork,
        session_service: SessionServicePort,
    ) -> None:
        """接收消息仓储、事务端口和 SessionService（history 委托）。"""

        self._repository = repository
        self._unit_of_work = unit_of_work
        self._session_service = session_service

    async def add(
        self,
        *,
        session_id: str,
        message: ModelMessage,
        metadata: Mapping[str, object] | None = None,
    ) -> Message:
        """在一个短事务中追加消息，seq 由仓储原子分配。

        业务规则：
        1. Session 不存在则抛出 SessionNotFoundError；
        2. seq 起点：已有消息取 max(seq)+1；空会话且为分叉子会话
           （parent_last_seq 存在）取 parent_last_seq+1；否则为 1。

        本方法只负责按接收顺序落库。异步到达的 agent 消息可能落在尚未闭合的
        assistant tool_calls 与 tool result 之间；该协议顺序由投影层
        （MessageContextProjector）统一修复，持久化层不再做补全。

        并发安全：seq 的读-改-写窗口由仓储的单条 INSERT..SELECT 在 SQLite
        语句级写锁内关闭；应用层无锁、无重试，多进程下同样成立。
        """

        async with self._unit_of_work:
            stored = await self._repository.add_with_next_seq(
                session_id=session_id,
                message=message,
                metadata=_metadata_with_default_kind(message, metadata),
            )
            if stored is None:
                raise SessionNotFoundError(session_id)
            await self._unit_of_work.commit()
            return stored

    async def history(self, session_id: str) -> list[Message]:
        """委托 SessionService 读取完整的可继承历史。"""

        return await self._session_service.history(session_id)


__all__ = ["MessageService"]
