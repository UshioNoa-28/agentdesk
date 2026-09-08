"""AutoGen 多智能体运行时 Provider。"""

from __future__ import annotations

from autogen_core import SingleThreadedAgentRuntime
from dishka import AsyncContainer, Provider, Scope, provide

from agent.ports.runtime.stream import AgentStreamHubPort
from agent.ports.tools import AgentRuntimePort
from agent.runtime.event_publisher import RuntimeEventPublisher
from agent.runtime.factory import AgentFactory
from agent.runtime.manager import AgentRuntimeManager
from agent.runtime.scope import DishkaRequestScopes
from agent.runtime.stream_hub import StreamHub


class RuntimeProvider(Provider):
    """提供 AutoGen Runtime、Actor 工厂和消息路由管理器。"""

    @provide(scope=Scope.APP)
    def provide_autogen_runtime(self) -> SingleThreadedAgentRuntime:
        """提供进程级 AutoGen 单线程事件驱动运行时（Actor 注册表与调度器）。"""

        return SingleThreadedAgentRuntime()

    @provide(scope=Scope.APP)
    def provide_runtime_scopes(self, container: AsyncContainer) -> DishkaRequestScopes:
        """提供进入独立请求作用域的适配器，保证 Actor 消息间事务隔离。"""

        return DishkaRequestScopes(container)

    @provide(scope=Scope.APP)
    def provide_agent_factory(
        self,
        scopes: DishkaRequestScopes,
    ) -> AgentFactory:
        """提供按 AgentId 惰性实例化 RoutedAgent 的工厂。"""

        return AgentFactory(scopes=scopes)

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
