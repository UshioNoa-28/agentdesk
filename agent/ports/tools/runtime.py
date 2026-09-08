"""Multi-Agent 运行时总线抽象端口。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent.domain.multi_agent import AgentMessage, AskOutcome


@runtime_checkable
class AgentRuntimePort(Protocol):
    """把 AgentMessage 路由到目标智能体的总线端口。

    两种传输语义按流量类型固定映射（星形拓扑）：

    - ``send_message``（RPC）：仅 main_agent 的外部同步入口，等待本轮结果；
    - ``dispatch``（事件）：agent 间异步投递，发布即返回。
    """

    async def send_message(self, message: AgentMessage) -> AskOutcome:
        """RPC 驱动 main_agent 并返回本轮结果。

        收件人不是 ``main_agent`` 时实现方必须拒绝。会话已有 turn 在跑时消息
        照常入库，返回 ``Queued`` —— 那是被接受的结果，不是失败。
        """

        ...

    async def dispatch(self, message: AgentMessage) -> None:
        """非阻塞投递事件给收件人；不等待处理结果。"""

        ...

    @property
    def is_running(self) -> bool:
        """后台消息循环是否已启动。"""

        ...

    async def stop(self) -> None:
        """等待在途消息处理完毕后停止后台消息循环。"""

        ...


__all__ = ["AgentRuntimePort"]
