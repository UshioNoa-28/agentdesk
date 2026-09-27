"""Meta Tool 注册端口协议。"""

from __future__ import annotations

from typing import Protocol

from agent.domain.tools import AgentTool


class MetaToolRegistryPort(Protocol):
    """每次 Agent 请求都自动激活的本地 Meta Tool 集合端口。"""

    @property
    def tools(self) -> tuple[AgentTool, ...]: ...

    def get_tools(self, allowed_tools: tuple[str, ...] = ()) -> tuple[AgentTool, ...]: ...


__all__ = ["MetaToolRegistryPort"]
