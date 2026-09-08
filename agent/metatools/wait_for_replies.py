"""固定 ``wait_for_replies`` Meta Tool（仅主控可用）。

控制流原语：轮询会话入站水位（role=human 的最大 seq,由 ``SessionService``
提供），新入站消息一落库就提前返回,不再睡满固定时长;返回后下一轮
``before_model`` 重读历史自然消化。超时保留为活性兜底（子代理猝死且连
告警都没发出时的唯一出口）,语义不变。

未注入 session_service 视为装配错误直接抛异常——本工具仅主控可用
（``MAIN_AGENT_ONLY_TOOL_NAMES``），任何调用点都该有真实的 SessionService。
默认与最大超时由 ``AgentRuntimeSettings`` 从环境变量
``WAIT_DEFAULT_TIMEOUT_SECONDS`` / ``WAIT_MAX_TIMEOUT_SECONDS`` 注入。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

from agent.domain.tools import AgentTool, ToolContext, ToolDefinition, ToolResult
from agent.ports.services import SessionServicePort

WAIT_FOR_REPLIES_DEFINITION = ToolDefinition(
    name="wait_for_replies",
    description=(
        "Wait for incoming subagent replies or user messages. Returns EARLY, as "
        "soon as any new inbound message has been persisted, so do not poll for "
        "replies with repeated short waits; one call is enough. On return your "
        "next turn already sees the new messages in context. If nothing arrives, "
        "it returns after a timeout as a liveness fallback; the default and "
        "maximum timeouts come from the deployment configuration."
    ),
    parameters={
        "type": "object",
        "properties": {
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Maximum seconds to wait if no new message arrives; defaults to "
                    "and is capped by the deployment configuration."
                ),
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)


class WaitForRepliesTool(AgentTool):
    """轮询入站水位的控制流工具：来消息即醒,无消息则超时兜底。"""

    def __init__(
        self,
        *,
        default_timeout: float,
        max_timeout: float,
        session_service: SessionServicePort,
    ) -> None:
        self._default_timeout = default_timeout
        self._max_timeout = max_timeout
        self._sessions = session_service

    @property
    def definition(self) -> ToolDefinition:
        return WAIT_FOR_REPLIES_DEFINITION

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        raw_timeout = arguments.get("timeout_seconds")
        try:
            timeout = min(max(float(raw_timeout), 1.0), self._max_timeout)
        except (TypeError, ValueError):
            timeout = self._default_timeout

        if self._sessions is None:
            raise RuntimeError("wait_for_replies requires a session_service")

        # 仅主控可用(MAIN_AGENT_ONLY):subagent 会话的 main_session_id 指向其
        # 归属主会话(主控为 None)。白名单理应挡住 subagent 拿到本工具,这里是
        # 纵深防御——拿到即越权,抛异常炸给模型看,绝不代睡。
        caller = await self._sessions.get(context.caller_session_id)
        if caller.main_session_id is not None:
            raise RuntimeError("wait_for_replies is main-agent-only; subagents must not call it")

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        # 基线=进入等待那一刻的水位。派发完到本行之间晚到的回报不触发提前醒,
        # 由 ``_route_after_model`` 的收尾水位比对兜底,不会冷落。
        baseline = await self._sessions.inbound_last_seq(context.caller_session_id)
        while True:
            remaining = started_at + timeout - loop.time()
            if remaining <= 0:
                return self._result(int(timeout), arrived=False)
            # 轮询粒度 1s:相对 60s 量级的超时完全无所谓。
            await asyncio.sleep(min(1, remaining))
            if await self._sessions.inbound_last_seq(context.caller_session_id) > baseline:
                return self._result(round(loop.time() - started_at), arrived=True)

    @staticmethod
    def _result(waited_seconds: int, *, arrived: bool) -> ToolResult:
        note = (
            "A new inbound message arrived; it is already in your context for the next turn."
            if arrived
            else "Timed out with no new message. Subagent replies, if any arrived "
            "meanwhile, are already in your context for the next turn."
        )
        return ToolResult(
            content=json.dumps(
                {"ok": True, "waited_seconds": waited_seconds, "note": note},
                ensure_ascii=False,
            )
        )


__all__ = ["WAIT_FOR_REPLIES_DEFINITION", "WaitForRepliesTool"]
