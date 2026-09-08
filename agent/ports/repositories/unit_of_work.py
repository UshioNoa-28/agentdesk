"""事务边界 UnitOfWork 端口协议。"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol


class UnitOfWork(Protocol):
    """一次短生命周期事务的端口，不绑定任何 Repository。"""

    async def __aenter__(self) -> UnitOfWork: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


__all__ = ["UnitOfWork"]
