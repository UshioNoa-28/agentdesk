"""运行时作用域抽象端口协议。"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol, TypeVar

T = TypeVar("T")


class RuntimeScope(Protocol):
    """已进入的请求级作用域，可按端口类型解析依赖实现。"""

    async def get(self, port_type: type[T]) -> T:
        """解析当前作用域内的端口实现。"""

        ...


class RuntimeScopeProvider(Protocol):
    """提供进入独立请求作用域的上下文管理器。

    RoutedAgent 实例由运行时缓存并跨请求复用，因此每条消息处理都必须通过
    本协议开辟全新的请求作用域，保证 UnitOfWork 与 AsyncSession 的事务隔离。
    """

    def scope(self) -> AbstractAsyncContextManager[RuntimeScope]:
        """开启一个新的请求作用域。"""

        ...


__all__ = [
    "RuntimeScope",
    "RuntimeScopeProvider",
]
