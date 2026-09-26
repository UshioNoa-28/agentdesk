"""Agent 消息编排用例端口协议。"""

from __future__ import annotations

from typing import Protocol

from agent.domain.multi_agent import AskOutcome


class AgentServicePort(Protocol):
    """Agent 消息编排用例端口。"""

    async def ask(
        self,
        *,
        session_id: str,
        question: str,
        sender: str,
    ) -> AskOutcome:
        """投递一条 UserMessage，返回本轮结果（回答，或已排队不再回复）。"""

        ...

    async def cancel(
        self,
        *,
        session_id: str,
        sender: str,
    ) -> None:
        """中断该会话主控在途的一轮并向其 RUNNING 子代理级联取消（fire-and-forget）。

        与 ``ask`` 对称：调用方正 ``await ask`` 阻塞时，须从另一个并发任务发起本方法；
        被打断的那次 ``ask`` 会以 ``asyncio.CancelledError`` 结束，由调用方接住。
        """

        ...


__all__ = ["AgentServicePort"]
