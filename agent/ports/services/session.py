"""Session 资源的应用用例端口协议。"""

from __future__ import annotations

from typing import Protocol

from agent.domain.entities import Session
from agent.domain.messages import Message


class SessionServicePort(Protocol):
    """Session 资源的应用用例端口。"""

    async def create(
        self,
        *,
        title: str,
        user_id: str = "default_user",
        parent_session_id: str | None = None,
        main_session_id: str | None = None,
        allowed_tools: tuple[str, ...] = (),
        custom_system_prompt: str | None = None,
    ) -> Session:
        """创建并持久化一个长期 Session。"""

        ...

    async def get(self, session_id: str) -> Session:
        """按 ID 查询 Session。"""

        ...

    async def list(self) -> list[Session]:
        """读取全部 Session。"""

        ...

    async def list_by_main_session(self, main_session_id: str) -> list[Session]:
        """读取属于指定主会话的全部子智能体 Session。"""

        ...

    async def inbound_last_seq(self, session_id: str) -> int:
        """该会话入站消息(role=human)的最大seq 主要用于判断是否有新的message在graph的run期间入库"""

        ...

    async def rename(self, *, session_id: str, title: str) -> Session:
        """修改 Session 标题并保存更新时间。"""

        ...

    async def resume(
        self,
        *,
        session_id: str,
        title: str | None = None,
        parent_last_seq: int | None = None,
    ) -> Session:
        """从指定 Session 分叉出一个带有 parent_session_id 的新 Session。

        ``parent_last_seq`` 指定父会话的分叉 seq；省略时使用父会话最新 seq。
        """

        ...

    async def delete(self, session_id: str) -> None:
        """删除一个没有子会话的 Session 聚合。"""

        ...

    async def history(self, session_id: str) -> list[Message]:
        """读取当前 Session 可继承的父快照和本地消息。"""

        ...


__all__ = ["SessionServicePort"]
