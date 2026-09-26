"""AutoGen 事件发布器：把领域消息发布为运行时事件（fire-and-forget）。

只依赖 AutoGen Runtime 本身，不感知名字解析与边界计数，因此可以被
AgentFactory → RoutedAgent 注入用于崩溃告警，避免与管理器互相依赖。
"""

from __future__ import annotations

import logging

from autogen_core import SingleThreadedAgentRuntime, TopicId

logger = logging.getLogger(__name__)

MAIN_AGENT_TOPIC = "main_agent"
SUBAGENT_TOPIC = "subagent"


class RuntimeEventPublisher:
    """按 TopicId(type=类别, source=会话ID) 发布事件；source 即实例 key。"""

    def __init__(self, runtime: SingleThreadedAgentRuntime) -> None:
        self._runtime = runtime

    async def publish(self, message: object, *, topic_type: str, key: str) -> None:
        """向指定主题发布事件；投递即返回，不等待处理结果。"""

        topic = TopicId(type=topic_type, source=key)
        await self._runtime.publish_message(message=message, topic_id=topic)
        logger.debug(
            "[MultiAgent] Published event to topic '%s' (key: %s)", topic_type, key
        )


__all__ = ["MAIN_AGENT_TOPIC", "RuntimeEventPublisher", "SUBAGENT_TOPIC"]
