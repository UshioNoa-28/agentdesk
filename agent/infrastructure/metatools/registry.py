"""Agent 进程内固定 Meta Tool Registry。"""

from __future__ import annotations

from collections.abc import Sequence

from agent.domain.tools import AgentTool
from agent.ports.tools import MetaToolRegistryPort


class MetaToolRegistry(MetaToolRegistryPort):
    """持有由组合根注入的 Meta Tool 集合，并按白名单过滤。"""

    def __init__(self, tools: Sequence[AgentTool]) -> None:
        self._tools = tuple(tools)

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        """返回所有已注册的 Meta Tools。"""

        return self._tools

    def get_tools(self, allowed_tools: tuple[str, ...] = ()) -> tuple[AgentTool, ...]:
        """按名称白名单过滤 Meta Tools。"""

        allowed_set = set(allowed_tools)
        return tuple(tool for tool in self._tools if tool.definition.name in allowed_set)


__all__ = ["MetaToolRegistry"]
