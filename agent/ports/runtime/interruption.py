"""人机中断的进程内传输端口。

由 :class:`~agent.infrastructure.interruption.InterruptionBroker` 实现：按
ID 保存挂起的 Future，把请求投递到活跃流，收到带外回复后唤醒等待方。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from agent.domain.interruption import (
    InterruptionReplyStatus,
    InterruptionRequest,
    InterruptionResult,
)


class InterruptionBrokerPort(Protocol):
    """一次性人机中断请求的内存 broker。"""

    async def request(self, request: InterruptionRequest) -> InterruptionResult:
        """发布请求并等待入口回复；进程内没有可用流时快速返回 UNAVAILABLE。

        ``request`` 由 caller 构造并携带自己那套 payload；broker 只搬运，
        不理解权限/澄清语义。
        """

        ...

    def reply(
        self,
        *,
        interruption_id: str,
        payload: Mapping[str, object],
    ) -> InterruptionReplyStatus:
        """按中断 ID 唤醒等待中的协程，投递入口回填的原始映射。"""

        ...

    def pending(self, interruption_id: str) -> InterruptionRequest | None:
        """返回在途中断的原始请求；ID 未知返回 None。

        供 caller 在唤醒前反查自己关心的语义（如权限规则落盘所需的工具名），
        broker 不解释请求内容。
        """

        ...

    def close(self) -> None:
        """关闭 broker，并以 CANCELLED 唤醒所有仍在等待的请求。"""

        ...


__all__ = ["InterruptionBrokerPort"]
