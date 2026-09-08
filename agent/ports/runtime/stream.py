"""Agent 流式事件的进程内传输端口。

流式输出是 ask RPC 的一条旁路：actor 侧通过 ``AgentStreamSink`` 发布事件，
HTTP 侧注册队列并消费，队列的摘除交给本轮 ``ask`` 任务。两条路径互不阻塞，
事件丢失（无注册队列）时调用方应自然退化为非流式行为。
"""

from __future__ import annotations

import asyncio
from typing import Protocol, TypeAlias

from agent.domain.streaming import AgentStreamEvent, StreamFrame

AgentStreamQueue: TypeAlias = asyncio.Queue[StreamFrame | None]


class AgentStreamSink(Protocol):
    """graph 与工具循环发布流式事件的最小出口。

    唯一实现：``agent.runtime.stream_hub.QueueStreamSink``（由
    ``StreamHub.sink_for`` 创建，包裹 HTTP 侧注册的 asyncio.Queue）。
    """

    async def publish(self, event: AgentStreamEvent) -> None:
        """发布一个流式事件；实现不应因发布失败中断业务流程。"""

        ...


class AgentStreamHubPort(Protocol):
    """按 session 管理流式队列的进程内注册表。

    队列元素是 ``(SSE 事件名, JSON 载荷)`` 帧，``None`` 是消费者约定的
    终止哨兵（由本轮 ``ask`` 任务的收尾回调入队）。队列随本轮图运行而生
    灭：注册存在即"这一轮还在跑"，与是否仍有人在读无关。
    """

    def register(self, session_id: str) -> AgentStreamQueue | None:
        """为 session 注册事件队列；已有在途轮次时返回 ``None``。"""

        ...

    def sink_for(self, session_id: str) -> AgentStreamSink | None:
        """返回 session 当前注册的发布端；未注册时返回 ``None``。"""

        ...

    def remove(self, session_id: str) -> None:
        """摘除 session 的队列注册；由本轮 ``ask`` 结束时调用，幂等。"""

        ...


__all__ = ["AgentStreamHubPort", "AgentStreamQueue", "AgentStreamSink"]
