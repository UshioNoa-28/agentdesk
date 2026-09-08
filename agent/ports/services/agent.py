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


__all__ = ["AgentServicePort"]
