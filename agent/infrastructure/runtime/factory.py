from __future__ import annotations

import logging

from autogen_core import (
    AgentInstantiationContext,
    SingleThreadedAgentRuntime,
)

from agent.domain.multi_agent import MAIN_AGENT_NAME
from agent.infrastructure.runtime.routed_agent import RoutedAgent
from agent.ports.runtime.routed_agent import RuntimeScopeProvider
from agent.ports.runtime.status import AgentStatusRegistryPort

logger = logging.getLogger(__name__)

SUBAGENT_TYPE = "subagent"


class AgentFactory:
    """把智能体类型工厂注册到 AutoGen 运行时（幂等）。

    工厂闭包在运行时首次投递消息时被调用，从 ``AgentInstantiationContext``
    读取当前 ``AgentId``：``key`` 即目标会话 ID，从而实现按会话的惰性实例化
    与缓存（Lazy Rehydration）。
    """

    def __init__(
        self,
        *,
        scopes: RuntimeScopeProvider,
        status_registry: AgentStatusRegistryPort,
    ) -> None:
        self._scopes = scopes
        self._status_registry = status_registry
        self._installed = False

    def _create_from_instantiation_context(self) -> RoutedAgent:
        agent_id = AgentInstantiationContext.current_agent_id()
        # 当未创建实例的routed agent 应该被创建的时候 无论是 send type+key 还是 publish topic
        # 都会调用改函数创建
        return RoutedAgent(
            session_id=agent_id.key,
            scopes=self._scopes,
            status_registry=self._status_registry,
        )

    async def install(self, runtime: SingleThreadedAgentRuntime) -> None:
        """注册全部固定类型；应在首次派发前调用一次。"""

        if self._installed:
            return
        for agent_type in (MAIN_AGENT_NAME, SUBAGENT_TYPE):
            await runtime.register_factory(
                agent_type,
                self._create_from_instantiation_context,
            ) # 注册对应的type以及工厂函数 
            logger.info("[MultiAgent] Registered RoutedAgent factory: %s", agent_type)
        self._installed = True


__all__ = ["SUBAGENT_TYPE", "AgentFactory"]
