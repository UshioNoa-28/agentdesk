"""进程内的一次性人机中断 broker。

把 :class:`~agent.domain.interruption.InterruptionRequest` 投递到当前活跃的
流式队列，并用 Future 等待入口的一次性带外回复。权限审批与 ``ask_user`` 澄清
都经本 broker 完成"挂起—唤醒"，broker 本身不解释请求语义。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from agent.domain.interruption import (
    InterruptionReplyStatus,
    InterruptionRequest,
    InterruptionResult,
    InterruptionStatus,
    PendingInterruption,
)
from agent.ports.runtime.interruption import InterruptionBrokerPort
from agent.ports.runtime.stream import AgentStreamHubPort

logger = logging.getLogger(__name__)


class InterruptionBroker(InterruptionBrokerPort):
    """保存 ``interruption_id -> PendingInterruption``，按 ID 路由入口回复。"""

    def __init__(self, stream_hub: AgentStreamHubPort) -> None:
        self._stream_hub = stream_hub
        self._pending: dict[str, PendingInterruption] = {}
        self._closed = False

    @property
    def pending_count(self) -> int:
        """当前尚未收到回复的中断数；主要用于诊断和测试。"""

        return len(self._pending)

    async def request(self, request: InterruptionRequest) -> InterruptionResult:
        """发布中断请求并等待回复；没有可用流时快速返回 UNAVAILABLE。"""

        target_session_id = request.session_id.strip()
        interruption_id = request.interruption_id.strip()
        if self._closed or not target_session_id or not interruption_id:
            return InterruptionResult(status=InterruptionStatus.UNAVAILABLE)

        future: asyncio.Future[InterruptionResult] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[interruption_id] = PendingInterruption(
            future=future, request=request
        )
        try:
            published = False
            try:
                published = await self._stream_hub.publish(
                    target_session_id, request
                )
            except Exception:
                logger.exception(
                    "Failed to publish interruption request: id=%s", interruption_id
                )
            if not published:
                return InterruptionResult(status=InterruptionStatus.UNAVAILABLE)
            return await future
        finally:
            entry = self._pending.get(interruption_id)
            if entry is not None and entry.future is future:
                self._pending.pop(interruption_id, None)

    def pending(self, interruption_id: str) -> InterruptionRequest | None:
        """返回在途中断的原始请求；ID 未知返回 None。"""

        entry = self._pending.get(interruption_id.strip())
        return entry.request if entry is not None else None

    def reply(
        self,
        *,
        interruption_id: str,
        payload: Mapping[str, object],
    ) -> InterruptionReplyStatus:
        """唤醒对应的等待协程，投递入口回填的原始映射。"""

        return self._settle(
            interruption_id,
            InterruptionResult(status=InterruptionStatus.RESOLVED, payload=payload),
        )

    def _settle(
        self,
        interruption_id: str,
        result: InterruptionResult,
    ) -> InterruptionReplyStatus:
        """按 ID 定位在途中断并用 ``result`` 唤醒其 Future。"""

        normalized_id = interruption_id.strip()
        if not normalized_id:
            return InterruptionReplyStatus.NOT_FOUND
        entry = self._pending.get(normalized_id)
        if entry is None:
            return InterruptionReplyStatus.NOT_FOUND
        if entry.future.done():
            return InterruptionReplyStatus.ALREADY_RESOLVED
        entry.future.set_result(result)
        return InterruptionReplyStatus.ACCEPTED

    def close(self) -> None:
        """关闭 broker，并让所有等待中的中断以 ``cancelled`` 收尾。"""

        if self._closed:
            return
        self._closed = True
        pending = tuple(self._pending.values())
        self._pending.clear()
        for entry in pending:
            if not entry.future.done():
                entry.future.set_result(
                    InterruptionResult(status=InterruptionStatus.CANCELLED)
                )
        logger.debug("Interruption broker closed: pending=%s", len(pending))


__all__ = ["InterruptionBroker"]
