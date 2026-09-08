"""Agent PostgreSQL ORM 表模型。"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """为 ORM 默认值生成当前 UTC 时间。"""

    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Agent ORM metadata 基类。"""


class SessionTable(Base):
    """sessions 表：保存会话元数据与父子关联。"""

    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint("title", name="uq_sessions_title"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    parent_session_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("sessions.id"),
        nullable=True,
        index=True,
    )
    main_session_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="default_user",
        index=True,
    )
    allowed_tools: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    parent_last_seq: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    parent: Mapped[SessionTable | None] = relationship(
        "SessionTable",
        remote_side="SessionTable.id",
        foreign_keys=[parent_session_id],
        back_populates="children",
        lazy="selectin",
    )
    children: Mapped[list[SessionTable]] = relationship(
        "SessionTable",
        foreign_keys=[parent_session_id],
        back_populates="parent",
        lazy="selectin",
    )
    messages: Mapped[list[MessageTable]] = relationship(
        "MessageTable",
        back_populates="session",
        lazy="selectin",
        cascade="all, delete-orphan",
    )


class MessageTable(Base):
    """messages 表：Session 内按 seq 排序的中立对话记录。"""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_messages_session_seq"),
        CheckConstraint(
            "role <> 'tool' OR "
            "(tool_call_id IS NOT NULL AND tool_call_id <> '')",
            name="ck_messages_tool_requires_call_id",
        ),
        Index("ix_messages_session_seq", "session_id", "seq"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_call_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tool_calls: Mapped[list[dict[str, object]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    message_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    session: Mapped[SessionTable] = relationship(
        SessionTable,
        back_populates="messages",
        lazy="selectin",
    )


__all__ = [
    "Base",
    "MessageTable",
    "SessionTable",
    "utc_now",
]
