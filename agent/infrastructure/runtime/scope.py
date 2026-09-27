"""Dishka 容器到 Runtime 作用域端口的适配器。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dishka import AsyncContainer


class DishkaRequestScopes:
    """每次调用进入独立的 Dishka 请求作用域。

    RoutedAgent 实例由 AutoGen Runtime 按 ``AgentId`` 缓存并跨请求复用，
    因此消息处理必须通过本适配器开辟全新作用域，保证 UnitOfWork 与
    AsyncSession 不会跨请求共享。
    """

    def __init__(self, container: AsyncContainer) -> None:
        self._container = container

    @asynccontextmanager
    async def scope(self) -> AsyncGenerator[AsyncContainer, None]:
        async with self._container() as request_container:
            yield request_container


__all__ = ["DishkaRequestScopes"]
