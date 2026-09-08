"""Agent Message 持久化应用服务。"""

from __future__ import annotations

from collections.abc import Mapping

from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage
from agent.exceptions import SessionNotFoundError
from agent.ports.repositories import MessageRepository, SessionRepository, UnitOfWork
from agent.ports.services import MessageServicePort, SessionServicePort


class MessageService(MessageServicePort):
    """组合 Message Repository、Session history 和事务边界。"""

    def __init__(
        self,
        *,
        repository: MessageRepository,
        session_repository: SessionRepository,
        unit_of_work: UnitOfWork,
        session_service: SessionServicePort,
    ) -> None:
        """接收消息仓储、会话仓储、SessionService 和当前请求的事务端口。"""

        self._repository = repository
        self._session_repository = session_repository
        self._unit_of_work = unit_of_work
        self._session_service = session_service

    async def add(
        self,
        *,
        session_id: str,
        message: ModelMessage,
        metadata: Mapping[str, object] | None = None,
    ) -> Message:
        """在一个短事务中追加消息并自动分配单调递增的 seq 序号。

        业务规则：
        1. 检查 Session 是否存在（不存在则抛出 SessionNotFoundError）；
        2. 如果会话已有本地消息，seq = last_seq + 1；
        3. 如果会话尚无本地消息，但为分叉子会话（parent_last_seq 存在），seq = parent_last_seq + 1；
        4. 否则（根会话初始消息），seq = 1。

        本方法只负责按接收顺序落库。异步到达的 agent 消息可能落在尚未闭合的
        assistant tool_calls 与 tool result 之间；该协议顺序由投影层
        （MessageContextProjector）统一修复，持久化层不再做补全。

        并发安全：``seq`` 的分配通过锁定对应 Session 行（``SELECT ... FOR UPDATE``）
        串行化同一会话的并发写入，避免 ``max(seq) + 1`` 的非原子读改写。
        """

        async with self._unit_of_work:
            session = await self._session_repository.get_for_update(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)

            current_max = await self._repository.last_seq(session_id)
            if current_max > 0:
                next_seq = current_max + 1
            elif session.parent_last_seq is not None:
                next_seq = session.parent_last_seq + 1
            else:
                next_seq = 1

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

            entity = Message.create(
                session_id=session_id,
                seq=next_seq,
                message=message,
                metadata=meta,
            )
            stored = await self._repository.add(entity)
            await self._unit_of_work.commit()
            return stored

    async def history(self, session_id: str) -> list[Message]:
        """委托 SessionService 读取完整的可继承历史。"""

        return await self._session_service.history(session_id)


__all__ = ["MessageService"]
