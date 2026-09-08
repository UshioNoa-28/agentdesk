"""RoutedAgent 工厂：向 AutoGen Runtime 注册按 AgentId 惰性实例化的工厂。

``type`` 表示智能体类别（决定工厂），全部子代理共享同一个 ``subagent``
类别；个体身份由 ``AgentId.key``（子会话 ID）与 Session 持久化配置承载，
因此运行时注册表面恒定为 ``main_agent`` 与 ``subagent`` 两类，不受 LLM
命名的影响。
"""

from __future__ import annotations

import logging

from autogen_core import (
    AgentInstantiationContext,
    SingleThreadedAgentRuntime,
)

from agent.domain.multi_agent import MAIN_AGENT_NAME
from agent.ports.runtime.routed_agent import RuntimeScopeProvider
from agent.runtime.routed_agent import RoutedAgent

logger = logging.getLogger(__name__)

SUBAGENT_TYPE = "subagent"


class AgentFactory:
    """把智能体类型工厂注册到 AutoGen 运行时（幂等）。

    工厂闭包在运行时首次投递消息时被调用，从 ``AgentInstantiationContext``
    读取当前 ``AgentId``：``key`` 即目标会话 ID，从而实现按会话的惰性实例化
    与缓存（Lazy Rehydration）。
    """

    def __init__(self, *, scopes: RuntimeScopeProvider) -> None:
        self._scopes = scopes
        self._installed = False

    def _create_from_instantiation_context(self) -> RoutedAgent:
        agent_id = AgentInstantiationContext.current_agent_id()
        return RoutedAgent(
            name=agent_id.type,
            session_id=agent_id.key,
            scopes=self._scopes,
        )

    async def install(self, runtime: SingleThreadedAgentRuntime) -> None:
        """注册全部固定类型；应在首次派发前调用一次。"""

        if self._installed:
            return
        for agent_type in (MAIN_AGENT_NAME, SUBAGENT_TYPE):
            await runtime.register_factory(
                agent_type,
                self._create_from_instantiation_context,
            )
            logger.info("[MultiAgent] Registered RoutedAgent factory: %s", agent_type)
        self._installed = True


__all__ = ["SUBAGENT_TYPE", "AgentFactory"]
