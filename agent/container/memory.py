"""长期记忆目录 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.domain.memory import MemoryLayer
from agent.infrastructure.memory import MemoryCatalog, memory_layer_dirs
from agent.infrastructure.settings import AgentRuntimeSettings
from agent.ports.tools import MemoryCatalogPort


class MemoryProvider(Provider):
    """提供进程级记忆目录；数据源是两层 MEMORY.md，每次调用现读。"""

    @provide(scope=Scope.APP)
    def provide_memory_catalog(self, settings: AgentRuntimeSettings) -> MemoryCatalogPort:
        dirs = memory_layer_dirs(settings.workspace_start)
        return MemoryCatalog(
            user_dir=dirs[MemoryLayer.USER],
            project_dir=dirs[MemoryLayer.PROJECT],
        )


__all__ = ["MemoryProvider"]
