"""AutoGen 多智能体运行时 Provider。"""

from __future__ import annotations

from autogen_core import SingleThreadedAgentRuntime
from dishka import AsyncContainer, Provider, Scope, provide

from agent.infrastructure.interruption import InterruptionBroker
from agent.infrastructure.permission import load_permission_manager
from agent.infrastructure.runtime.event_publisher import RuntimeEventPublisher
from agent.infrastructure.runtime.factory import AgentFactory
from agent.infrastructure.runtime.manager import AgentRuntimeManager
from agent.infrastructure.runtime.scope import DishkaRequestScopes
from agent.infrastructure.runtime.status_registry import InMemoryAgentStatusRegistry
from agent.infrastructure.runtime.stream_hub import StreamHub
from agent.infrastructure.settings import AgentRuntimeSettings
from agent.ports.runtime.interruption import InterruptionBrokerPort
from agent.ports.runtime.status import AgentStatusRegistryPort
from agent.ports.runtime.stream import AgentStreamHubPort
from agent.ports.services import PermissionManagerPort
from agent.ports.tools import AgentRuntimePort


class RuntimeProvider(Provider):
    """提供 AutoGen Runtime、Actor 工厂和消息路由管理器。"""

    @provide(scope=Scope.APP)
    def provide_permission_manager(
        self,
        settings: AgentRuntimeSettings,
    ) -> PermissionManagerPort:
        return load_permission_manager(settings.workspace_start)

    @provide(scope=Scope.APP)
    def provide_autogen_runtime(self) -> SingleThreadedAgentRuntime:
        """提供进程级 AutoGen 单线程事件驱动运行时（Actor 注册表与调度器）。"""

        return SingleThreadedAgentRuntime()

    @provide(scope=Scope.APP)
    def provide_runtime_scopes(self, container: AsyncContainer) -> DishkaRequestScopes:
        """提供进入独立请求作用域的适配器，保证 Actor 消息间事务隔离。"""

        return DishkaRequestScopes(container)

    @provide(scope=Scope.APP)
    def provide_agent_status_registry(self) -> AgentStatusRegistryPort:
        """提供进程内智能体运行态状态注册表（actor 写、观测工具读）。"""

        return InMemoryAgentStatusRegistry()

    @provide(scope=Scope.APP)
    def provide_agent_factory(
        self,
        scopes: DishkaRequestScopes,
        status_registry: AgentStatusRegistryPort,
    ) -> AgentFactory:
        """提供按 AgentId 惰性实例化 RoutedAgent 的工厂。"""

        return AgentFactory(scopes=scopes, status_registry=status_registry)

    @provide(scope=Scope.APP)
    def provide_event_publisher(
        self,
        autogen_runtime: SingleThreadedAgentRuntime,
    ) -> RuntimeEventPublisher:
        """提供 AutoGen 事件发布器（fire-and-forget 投递）。"""

        return RuntimeEventPublisher(autogen_runtime)

    @provide(scope=Scope.APP)
    def provide_stream_hub(self) -> AgentStreamHubPort:
        """提供进程内流式事件注册表（session → 事件队列）。"""

        return StreamHub()

    @provide(scope=Scope.APP)
    def provide_interruption_broker(
        self,
        stream_hub: AgentStreamHubPort,
    ) -> InterruptionBrokerPort:
        """提供进程内一次性中断 broker；pending 请求不落库。"""

        return InterruptionBroker(stream_hub)

    @provide(scope=Scope.APP)
    def provide_agent_runtime(
        self,
        autogen_runtime: SingleThreadedAgentRuntime,
        factory: AgentFactory,
        scopes: DishkaRequestScopes,
        publisher: RuntimeEventPublisher,
    ) -> AgentRuntimePort:
        """提供基于 AutoGen Runtime 的多智能体消息路由管理器。"""

        return AgentRuntimeManager(
            runtime=autogen_runtime,
            factory=factory,
            scopes=scopes,
            publisher=publisher,
        )


__all__ = ["RuntimeProvider"]
