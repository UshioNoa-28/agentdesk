"""Mem0 长期记忆 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide
from mem0 import AsyncMemory

from agent.infrastructure.memory import Mem0MemoryService, build_mem0_async_client
from agent.infrastructure.settings import MemorySettings
from agent.ports.memory import MemoryServicePort


class MemoryProvider(Provider):
    """提供 Mem0 客户端单例和记忆应用服务。"""

    @provide(scope=Scope.APP)
    def provide_async_memory(
        self,
        settings: MemorySettings,
    ) -> AsyncMemory | None:
        """由 Dishka 容器管理的 Mem0 AsyncMemory 客户端单例（禁用时返回 None）。"""

        return build_mem0_async_client(settings)

    @provide(scope=Scope.APP)
    def provide_memory_service(
        self,
        settings: MemorySettings,
        client: AsyncMemory | None,
    ) -> MemoryServicePort:
        """由 Dishka 容器注入 AsyncMemory 客户端单例并组装 MemoryServicePort。"""

        return Mem0MemoryService(settings=settings, client=client)


__all__ = ["MemoryProvider"]
