"""Multi-Agent 运行时测试共享替身。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any


class FakeScope:
    """按端口类型返回预注册实例的请求作用域替身。"""

    def __init__(self, services: dict[type, Any]) -> None:
        self._services = services

    async def get(self, port_type: type) -> Any:
        if port_type not in self._services:
            raise KeyError(f"No service registered for port '{port_type.__name__}'")
        return self._services[port_type]


class FakeRuntimeScopes:
    """为 RoutedAgent 提供独立请求作用域的测试替身。"""

    def __init__(self, services: dict[type, Any] | None = None, **kwargs: Any) -> None:
        self._services: dict[type, Any] = dict(services or {})
        self._services.update(kwargs)

    def register(self, port_type: type, instance: Any) -> None:
        """注册或替换端口实现。"""

        self._services[port_type] = instance

    def scope(self) -> Any:
        provider = self

        @asynccontextmanager
        async def _context() -> AsyncGenerator[FakeScope, None]:
            yield FakeScope(provider._services)

        return _context()


__all__ = ["FakeRuntimeScopes"]
