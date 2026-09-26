"""Agent 的 SQLAlchemy 事务控制实现。"""

from __future__ import annotations

from types import TracebackType

from sqlalchemy.ext.asyncio import AsyncSession

from agent.ports.repositories import UnitOfWork


class SqlAlchemyUnitOfWork(UnitOfWork):
    """以 SQLAlchemy AsyncSession 实现最小 UnitOfWork 端口。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> SqlAlchemyUnitOfWork:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.rollback()

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()


__all__ = ["SqlAlchemyUnitOfWork"]
