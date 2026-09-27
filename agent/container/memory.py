"""文件式长期记忆的 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.application.memory import MemoryConsolidator
from agent.infrastructure.memory import MemoryStore, load_memory_enabled
from agent.infrastructure.memory.notes import NoteBook
from agent.infrastructure.prompt import PromptBuilder
from agent.infrastructure.settings import AgentRuntimeSettings
from agent.ports.model import AgentModelPort
from agent.ports.services import (
    MemoryConsolidatorPort,
    MemoryNoteBookPort,
    MemoryStorePort,
)
from agent.ports.tools import MetaToolRegistryPort


class MemoryProvider(Provider):
    """提供两层记忆目录与笔记读写入口；``memory.enabled`` 随进程启动解析一次。

    工具本身在 ``MetaToolProvider`` 里按普通 Meta Tool 注册，这里只给它们要用的
    存储：``MemoryStore`` 是 append-only 日志，``NoteBook`` 是笔记与索引。
    """

    @provide(scope=Scope.APP)
    def provide_memory_store(self, settings: AgentRuntimeSettings) -> MemoryStorePort:
        start = settings.workspace_start
        return MemoryStore(start=start, enabled=load_memory_enabled(start))

    @provide(scope=Scope.APP)
    def provide_note_book(self, store: MemoryStorePort) -> MemoryNoteBookPort:
        return NoteBook(store)

    # 消化要查注册表拿笔记工具，与 REQUEST 作用域的 MetaToolRegistry 同域；
    # max_turns 沿用运行时上限，不另设一个只有后台用的旋钮。
    @provide(scope=Scope.REQUEST)
    def provide_consolidator(
        self,
        store: MemoryStorePort,
        notebook: MemoryNoteBookPort,
        meta_tools: MetaToolRegistryPort,
        model: AgentModelPort,
        prompt_builder: PromptBuilder,
        settings: AgentRuntimeSettings,
    ) -> MemoryConsolidatorPort:
        return MemoryConsolidator(
            store=store,
            notebook=notebook,
            meta_tools=meta_tools,
            model=model,
            prompt_builder=prompt_builder,
            max_turns=settings.max_turns,
        )


__all__ = ["MemoryProvider"]
