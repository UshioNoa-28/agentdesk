"""Agent 微服务的 Dishka 组合根。

按职责把 Provider 拆到同目录的多个模块，这里只负责聚合。
"""

from __future__ import annotations

from dishka import Provider

from agent.container.application import ApplicationProvider
from agent.container.context import ContextProvider
from agent.container.mcp import McpProvider
from agent.container.memory import MemoryProvider
from agent.container.metatools import MetaToolProvider
from agent.container.model import AgentModelProvider
from agent.container.runtime import RuntimeProvider
from agent.container.settings import AgentSettingsProvider
from agent.container.skills import SkillProvider
from agent.container.tokenizer import TokenizerProvider
from agent.infrastructure.db import PostgresProvider


def providers() -> tuple[Provider, ...]:
    """返回 Agent 进程的全部基础设施和业务 Provider。"""

    return (
        PostgresProvider(),
        AgentSettingsProvider(),
        TokenizerProvider(),
        MemoryProvider(),
        McpProvider(),
        SkillProvider(),
        AgentModelProvider(),
        RuntimeProvider(),
        ApplicationProvider(),
        MetaToolProvider(),
        ContextProvider(),
    )


__all__ = ["providers"]
