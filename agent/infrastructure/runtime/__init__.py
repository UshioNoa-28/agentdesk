"""Multi-Agent 运行时与生命周期模块。"""

from agent.infrastructure.runtime.factory import AgentFactory
from agent.infrastructure.runtime.manager import AgentRuntimeManager
from agent.infrastructure.runtime.routed_agent import RoutedAgent
from agent.infrastructure.runtime.scope import DishkaRequestScopes
from agent.infrastructure.runtime.status_registry import InMemoryAgentStatusRegistry

__all__ = [
    "AgentFactory",
    "AgentRuntimeManager",
    "DishkaRequestScopes",
    "InMemoryAgentStatusRegistry",
    "RoutedAgent",
]
