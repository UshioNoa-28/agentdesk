"""Agent 问答用例服务实现。"""

from __future__ import annotations

from agent.domain.exceptions import DomainValidationError
from agent.domain.multi_agent import MAIN_AGENT_NAME, AgentMessage, AskOutcome
from agent.ports.services import AgentServicePort
from agent.ports.tools import AgentRuntimePort


class AgentService(AgentServicePort):
    """把外部用户输入投递给 main_agent 并等待其图自然收尾。"""

    def __init__(
        self,
        *,
        runtime: AgentRuntimePort,
    ) -> None:
        self._runtime = runtime

    async def ask(
        self,
        *,
        session_id: str,
        question: str,
        sender: str,
    ) -> AskOutcome:
        """投递用户问题并返回本轮结果。

        send_message 是 RPC：主控的图在模型中不再产生工具调用时结束，此时结果是
        ``Answered``。若主控已在跑，问题照常入库，结果是 ``Queued`` —— 本轮不产出
        回复，调用方不应重发。
        """

        clean_question = question.strip()
        if not clean_question:
            raise DomainValidationError("Question cannot be empty")
        clean_sender = sender.strip()
        if not clean_sender:
            raise DomainValidationError("Sender cannot be empty")

        return await self._runtime.send_message(
            AgentMessage(
                content=clean_question,
                sender=clean_sender,
                recipient=MAIN_AGENT_NAME,
                session_id=session_id,
            )
        )


__all__ = ["AgentService"]
