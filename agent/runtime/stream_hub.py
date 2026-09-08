"""进程内流式事件注册表：session → 事件帧队列。

生命周期约定：``register`` 归 HTTP 侧（SSE endpoint）所有，``remove`` 归本轮
``ask`` 任务所有——队列跟着那一轮图生死，不跟着 HTTP 连接生死，所以"注册存在"
的含义是"这一轮还在跑"，与还有没有人在读无关。actor 侧只通过 ``sink_for``
读取；session 未注册时返回 ``None``，调用方自然退化为非流式行为。同一 session
同时只允许一条活跃流。

队列里流动的是 SSE 就绪帧 ``(事件名, JSON 载荷)``；领域事件在发布端经
``stream_frame`` 一次性映射，消费端无需感知领域类型。
"""

from __future__ import annotations

import asyncio
import logging

from agent.domain.streaming import AgentStreamEvent, stream_frame
from agent.ports.runtime.stream import AgentStreamQueue, AgentStreamSink

logger = logging.getLogger(__name__)


class QueueStreamSink(AgentStreamSink):
    """``AgentStreamSink`` 的唯一实现：把领域事件映射成帧写入 asyncio.Queue。

    无界队列 + put_nowait，发布端永不阻塞、永不因队列满而失败。
    """

    __slots__ = ("_queue",)

    def __init__(self, queue: AgentStreamQueue) -> None:
        self._queue = queue

    async def publish(self, event: AgentStreamEvent) -> None:
        """发布一个流式事件帧；put_nowait 在无界队列上不会失败。"""

        self._queue.put_nowait(stream_frame(event))


class StreamHub:
    """按 session_id 保存活跃流式队列；APP 级单例。"""

    def __init__(self) -> None:
        self._queues: dict[str, AgentStreamQueue] = {}

    def register(self, session_id: str) -> AgentStreamQueue | None:
        """为 session 注册事件队列；已有在途轮次时返回 ``None``。"""

        if session_id in self._queues:
            return None
        queue: AgentStreamQueue = asyncio.Queue()
        self._queues[session_id] = queue
        return queue

    def sink_for(self, session_id: str) -> AgentStreamSink | None:
        """返回 session 当前注册的发布端（``QueueStreamSink``）；未注册时 ``None``。"""

        queue = self._queues.get(session_id)
        return None if queue is None else QueueStreamSink(queue)

    def remove(self, session_id: str) -> None:
        """摘除 session 的队列注册；由本轮 ``ask`` 结束时调用，幂等。"""

        if self._queues.pop(session_id, None) is not None:
            logger.debug("Stream hub removed queue: session_id=%s", session_id)


__all__ = ["QueueStreamSink", "StreamHub"]
