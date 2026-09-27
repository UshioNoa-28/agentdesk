"""应用服务与仓储 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio import AsyncSession

from agent.application.services.agent_service import AgentService
from agent.application.services.message_service import MessageService
from agent.application.services.permission_service import PermissionService
from agent.application.services.session_service import SessionService
from agent.infrastructure.db import (
    SqlAlchemyMessageRepository,
    SqlAlchemySessionRepository,
)
from agent.ports.repositories import (
    MessageRepository,
    SessionRepository,
    UnitOfWork,
)
from agent.ports.runtime.interruption import InterruptionBrokerPort
from agent.ports.services import (
    AgentServicePort,
    MessageServicePort,
    PermissionManagerPort,
    PermissionServicePort,
    SessionServicePort,
)
from agent.ports.tools import (
    AgentRuntimePort,
)


class ApplicationProvider(Provider):
    """提供 Agent 请求作用域的仓储与应用服务。"""

    @provide(scope=Scope.REQUEST)
    def provide_session_repository(self, session: AsyncSession) -> SessionRepository:
        """绑定 Agent 自己拥有的 Session 仓储。"""

        return SqlAlchemySessionRepository(session)

    @provide(scope=Scope.REQUEST)
    def provide_message_repository(self, session: AsyncSession) -> MessageRepository:
        """绑定 Agent 自己拥有的 Message 仓储。"""

        return SqlAlchemyMessageRepository(session)

    @provide(scope=Scope.REQUEST)
    def provide_message_service(
        self,
        repository: MessageRepository,
        unit_of_work: UnitOfWork,
        session_service: SessionServicePort,
    ) -> MessageServicePort:
        """把 Message 仓储和事务边界组装成消息应用服务。"""

        return MessageService(
            repository=repository,
            unit_of_work=unit_of_work,
            session_service=session_service,
        )

    @provide(scope=Scope.REQUEST)
    def provide_session_service(
        self,
        repository: SessionRepository,
        messages: MessageRepository,
        unit_of_work: UnitOfWork,
    ) -> SessionServicePort:
        """组装 Session CRUD、自定义 System Prompt 持久化、resume 和聚合删除用例。

        默认 system prompt 不落库：每轮由 ContextManager 用活的
        MCP/Skill/工具授权/记忆目录现拼。
        """

        return SessionService(
            repository=repository,
            messages=messages,
            unit_of_work=unit_of_work,
        )

    @provide(scope=Scope.REQUEST)
    def provide_agent_service(
        self,
        runtime: AgentRuntimePort,
    ) -> AgentServicePort:
        """组装 Agent 消息问答用例。"""

        return AgentService(runtime=runtime)

    @provide(scope=Scope.REQUEST)
    def provide_permission_service(
        self,
        broker: InterruptionBrokerPort,
        manager: PermissionManagerPort,
        session_service: SessionServicePort,
    ) -> PermissionServicePort:
        return PermissionService(
            broker=broker,
            manager=manager,
            session_service=session_service,
        )


__all__ = ["ApplicationProvider"]
