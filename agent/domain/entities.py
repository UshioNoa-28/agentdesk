"""Agent 服务自己的持久化领域实体。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from agent.domain.exceptions import DomainValidationError
from agent.domain.time import utc_now


@dataclass(slots=True)
class Session:
    """长期会话边界。

    对话消息不嵌套在 Session 实体中，而是通过 ``session_id`` 关联到 Message 表。
    会话可通过 ``parent_session_id`` 继承父会话历史快照，或通过 ``main_session_id``
    标记归属于某个主智能体会话。
    """

    id: str
    title: str
    parent_session_id: str | None
    created_at: datetime
    updated_at: datetime
    main_session_id: str | None = None
    allowed_tools: tuple[str, ...] = ()
    user_id: str = "default_user"
    parent_last_seq: int | None = None

    @classmethod
    def create(
        cls,
        *,
        title: str,
        user_id: str = "default_user",
        parent_session_id: str | None = None,
        main_session_id: str | None = None,
        allowed_tools: tuple[str, ...] = (),
        parent_last_seq: int | None = None,
    ) -> "Session":
        """创建一段长期会话。"""

        normalized_title = title.strip() if title is not None else ""
        if not normalized_title:
            raise DomainValidationError("Session title cannot be empty")
        normalized_user_id = (
            user_id.strip()
            if user_id is not None and user_id.strip()
            else "default_user"
        )
        now = utc_now()
        tools = tuple(allowed_tools)
        return cls(
            id=str(uuid4()),
            title=normalized_title,
            user_id=normalized_user_id,
            parent_session_id=parent_session_id,
            main_session_id=main_session_id,
            allowed_tools=tools,
            created_at=now,
            updated_at=now,
            parent_last_seq=parent_last_seq,
        )

    @classmethod
    def rehydrate(
        cls,
        *,
        id: str,
        title: str,
        user_id: str = "default_user",
        parent_session_id: str | None,
        main_session_id: str | None = None,
        allowed_tools: tuple[str, ...] = (),
        parent_last_seq: int | None = None,
        created_at: datetime,
        updated_at: datetime,
    ) -> "Session":
        """从数据库恢复 Session。"""

        tools = tuple(allowed_tools)
        return cls(
            id=id,
            title=title,
            user_id=user_id or "default_user",
            parent_session_id=parent_session_id,
            main_session_id=main_session_id,
            allowed_tools=tools,
            parent_last_seq=parent_last_seq,
            created_at=created_at,
            updated_at=updated_at,
        )

    def rename(self, title: str) -> None:
        """修改标题并刷新更新时间。"""

        normalized = title.strip()
        if not normalized:
            raise DomainValidationError("Session title cannot be empty")
        self.title = normalized
        self.touch()

    def touch(self) -> None:
        """刷新 Session 的更新时间。"""

        self.updated_at = utc_now()

    def resume(
        self,
        *,
        title: str | None = None,
        parent_last_seq: int | None = None,
    ) -> "Session":
        """从当前会话创建一个子会话。"""

        return Session.create(
            title=title or f"Resume: {self.title}",
            user_id=self.user_id,
            parent_session_id=self.id,
            main_session_id=self.main_session_id,
            allowed_tools=self.allowed_tools,
            parent_last_seq=parent_last_seq,
        )


__all__ = ["Session"]
