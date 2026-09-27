"""Agent Session、Message 的 SQLite Repository 实现。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, bindparam, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent.domain.entities import Session
from agent.domain.messages import Message, message_kind_from_value
from agent.domain.model_messages import ModelMessage, ToolCall
from agent.exceptions import SessionNotFoundError
from agent.infrastructure.db.models import MessageTable, SessionTable, utc_now
from agent.ports.repositories import MessageRepository, SessionRepository


def _session_from_table(row: SessionTable) -> Session:
    """把 Session ORM 行恢复成领域实体。"""

    return Session.rehydrate(
        id=row.id,
        title=row.title,
        user_id=row.user_id,
        parent_session_id=row.parent_session_id,
        main_session_id=row.main_session_id,
        allowed_tools=tuple(row.allowed_tools or ()),
        parent_last_seq=row.parent_last_seq,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _message_from_table(row: MessageTable) -> Message:
    """把消息 ORM 行恢复成中立 Message。"""

    calls = tuple(
        ToolCall(
            id=str(item.get("id", "")),
            name=str(item.get("name", "")),
            arguments=dict(item.get("arguments", {})),
        )
        for item in (row.tool_calls or [])
        if isinstance(item, dict)
    )
    return Message(
        id=row.id,
        session_id=row.session_id,
        seq=row.seq,
        role=row.role,
        content=row.content,
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        tool_calls=calls,
        metadata=dict(row.message_metadata or {}),
        created_at=row.created_at,
    )


def _message_table_from_entity(message: Message) -> MessageTable:
    """把领域 Message 映射成 ORM 行。"""

    return MessageTable(
        id=message.id,
        session_id=message.session_id,
        seq=message.seq,
        role=message.role,
        content=message.content,
        tool_call_id=message.tool_call_id,
        tool_name=message.tool_name,
        tool_calls=[
            {
                "id": call.id,
                "name": call.name,
                "arguments": dict(call.arguments),
            }
            for call in message.tool_calls
        ],
        message_metadata=dict(message.metadata),
        created_at=message.created_at,
    )


class SqlAlchemySessionRepository(SessionRepository):
    """Session 领域仓储的 SQLAlchemy 实现。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, session: Session) -> None:
        self._session.add(
            SessionTable(
                id=session.id,
                title=session.title,
                user_id=session.user_id,
                parent_session_id=session.parent_session_id,
                main_session_id=session.main_session_id,
                allowed_tools=list(session.allowed_tools),
                parent_last_seq=session.parent_last_seq,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
        )
        await self._session.flush()

    async def get(self, session_id: str) -> Session | None:
        stmt = select(SessionTable).where(SessionTable.id == session_id)
        result = await self._session.execute(stmt)
        row = result.scalars().first()
        return _session_from_table(row) if row is not None else None

    async def list_all(self) -> list[Session]:
        stmt = select(SessionTable).order_by(
            SessionTable.updated_at.desc(),
            SessionTable.created_at.desc(),
        )
        result = await self._session.execute(stmt)
        return [_session_from_table(row) for row in result.scalars().all()]

    async def list_by_main_session(self, main_session_id: str) -> list[Session]:
        stmt = select(SessionTable).where(
            SessionTable.main_session_id == main_session_id
        ).order_by(
            SessionTable.created_at.asc(),
        )
        result = await self._session.execute(stmt)
        return [_session_from_table(row) for row in result.scalars().all()]

    async def has_children(self, session_id: str) -> bool:
        stmt = select(SessionTable.id).where(SessionTable.parent_session_id == session_id).limit(1)
        result = await self._session.execute(stmt)
        return result.scalars().first() is not None

    async def save(self, session: Session) -> Session:
        stmt = select(SessionTable).where(SessionTable.id == session.id)
        result = await self._session.execute(stmt)
        row = result.scalars().first()
        if row is None:
            # 并发删除（rename 持有的实体在提交前被删）必须落到 404，
            # 不能让裸 KeyError 落进兜底 500。
            raise SessionNotFoundError(session.id)
        row.title = session.title
        row.updated_at = session.updated_at
        await self._session.flush()
        return session

    async def delete(self, session_id: str) -> bool:
        stmt = select(SessionTable).where(SessionTable.id == session_id)
        result = await self._session.execute(stmt)
        row = result.scalars().first()
        if row is None:
            return False
        await self._session.delete(row)
        await self._session.flush()
        return True


class SqlAlchemyMessageRepository(MessageRepository):
    """Message 领域仓储的 SQLAlchemy 实现。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, message: Message) -> Message:
        row = _message_table_from_entity(message)
        self._session.add(row)

        now = utc_now()
        await self._session.execute(
            update(SessionTable)
            .where(SessionTable.id == message.session_id)
            .values(updated_at=now)
        )

        await self._session.flush()
        return message

    async def add_with_next_seq(
        self,
        *,
        session_id: str,
        message: ModelMessage,
        metadata: Mapping[str, Any] | None = None,
    ) -> Message | None:
        """一条 INSERT..SELECT 原子续号追加;seq 计算与插入不可分割。

        seq 种子在同一语句内完成:会话内 max(seq) → 空会话取分叉点
        ``parent_last_seq`` → 否则 0,+1 后写入。外层 ``FROM sessions
        WHERE id=:sid`` 兼作存在性检查:会话不存在时插入 0 行。
        并发互斥由 SQLite 的语句级写锁保证,应用层无锁、无重试。
        """

        message_id = str(uuid4())
        created_at = utc_now()
        values = dict(metadata or {})
        values["kind"] = message_kind_from_value(values.get("kind")).value
        tool_calls = [
            {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
            for call in message.tool_calls
        ]

        next_seq = (
            select(
                func.coalesce(
                    select(func.max(MessageTable.seq))
                    .where(MessageTable.session_id == session_id)
                    .scalar_subquery(),
                    select(SessionTable.parent_last_seq)
                    .where(SessionTable.id == session_id)
                    .scalar_subquery(),
                    0,
                )
                + 1
            )
            .scalar_subquery()
        )

        stmt = insert(MessageTable).from_select(
            [
                MessageTable.id,
                MessageTable.session_id,
                MessageTable.seq,
                MessageTable.role,
                MessageTable.content,
                MessageTable.tool_call_id,
                MessageTable.tool_name,
                MessageTable.tool_calls,
                MessageTable.message_metadata,
                MessageTable.created_at,
            ],
            select(
                bindparam("id", message_id),
                bindparam("sid", session_id),
                next_seq,
                bindparam("role", message.role),
                bindparam("content", message.content),
                bindparam("tool_call_id", message.tool_call_id),
                bindparam("tool_name", message.tool_name),
                bindparam("tool_calls", tool_calls, type_=JSON()),
                bindparam("metadata", values, type_=JSON()),
                bindparam("created_at", created_at, type_=DateTime(timezone=True)),
            ).where(SessionTable.id == session_id),
        )

        result = await self._session.execute(stmt)
        if result.rowcount == 0:
            return None

        seq = (
            await self._session.execute(
                select(MessageTable.seq).where(MessageTable.id == message_id)
            )
        ).scalar_one()

        await self._session.execute(
            update(SessionTable)
            .where(SessionTable.id == session_id)
            .values(updated_at=utc_now())
        )
        await self._session.flush()

        return Message(
            id=message_id,
            session_id=session_id,
            seq=seq,
            role=message.role,
            content=message.content,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            tool_calls=message.tool_calls,
            metadata=values,
            created_at=created_at,
        )

    async def list_for_session(
        self,
        session_id: str,
        *,
        max_seq: int | None = None,
    ) -> list[Message]:
        stmt = select(MessageTable).where(MessageTable.session_id == session_id)
        if max_seq is not None:
            stmt = stmt.where(MessageTable.seq <= max_seq)
        stmt = stmt.order_by(MessageTable.seq.asc())
        result = await self._session.execute(stmt)
        return [_message_from_table(row) for row in result.scalars().all()]

    async def last_seq(self, session_id: str) -> int:
        stmt = (
            select(func.coalesce(func.max(MessageTable.seq), 0))
            .where(MessageTable.session_id == session_id)
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def inbound_last_seq(self, session_id: str) -> int:
        # 只数入站(human):本轮自己落库的 assistant 回答/tool 结果/压缩摘要
        # 不得移动水位,否则 finish 比较永远不等。
        stmt = (
            select(func.coalesce(func.max(MessageTable.seq), 0))
            .where(MessageTable.session_id == session_id, MessageTable.role == "human")
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def delete_for_session(self, session_id: str) -> int:
        stmt = select(MessageTable).where(MessageTable.session_id == session_id)
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        for row in rows:
            await self._session.delete(row)
        await self._session.flush()
        return len(rows)


__all__ = [
    "SqlAlchemyMessageRepository",
    "SqlAlchemySessionRepository",
]
