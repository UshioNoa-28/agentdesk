"""智能体运行态状态的进程内注册表端口。

``AgentStatus``（idle/running/error/…）是 actor 实例上的**易失内存态**，随消息
处理而变、不落库：库里的 Session 行只描述"这个子代理存在及其配置"，不描述"它
这一会儿是否在跑"。本端口把这份内存态收敛为一个按 ``session_id`` 定位的可读视图，
供 ``list_subagents`` 这类观测工具查询，使主控无需向正在 RUNNING 的子代理投递
消息（那会被排队、拿不到实时状态）即可读到其当前状态。
"""

from __future__ import annotations

from typing import Protocol

from agent.domain.multi_agent import AgentStatus


class AgentStatusRegistryPort(Protocol):
    """按 session 记录并读取智能体运行态状态的进程内注册表。

    唯一实现：``agent.infrastructure.runtime.status_registry.InMemoryAgentStatusRegistry``
    （APP 作用域单例，与运行它的 AutoGen 单线程 Runtime 同生命周期）。
    """

    def set_status(self, session_id: str, status: AgentStatus) -> None:
        """记录 ``session_id`` 对应智能体的当前状态；实现不应阻塞。"""

        ...

    def get_status(self, session_id: str) -> AgentStatus | None:
        """读取当前状态；该 session 尚无在途实例（从未被实例化）时返回 ``None``。"""

        ...


__all__ = ["AgentStatusRegistryPort"]
