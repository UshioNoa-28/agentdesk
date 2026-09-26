"""Agent 问答用例服务实现。"""

from __future__ import annotations

from agent.domain.multi_agent import AskOutcome, CancelMessage, MainAgentMessage
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
        回复，调用方不应重发。非空校验由 ``MainAgentMessage`` 充血模型承担。
        """

        return await self._runtime.send_message(
            MainAgentMessage(
                content=question,
                sender=sender,
                session_id=session_id,
            )
        )

    async def cancel(
        self,
        *,
        session_id: str,
        sender: str,
    ) -> None:
        """中断该会话主控在途的一轮（fire-and-forget），由其向下级联取消子代理。

        与 ``ask`` 对称：调用方正阻塞在 ``await ask`` 时，须从另一个并发任务发起本方法；
        被打断的那次 ``ask`` 以 ``asyncio.CancelledError`` 结束。非空校验由
        ``CancelMessage`` 充血模型承担。
        """

        await self._runtime.cancel(
            CancelMessage(
                session_id=session_id,
                sender=sender,
            )
        )


__all__ = ["AgentService"]
