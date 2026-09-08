"""Agent PostgreSQL 连接池和 UnitOfWork 的 Dishka Provider。"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agent.infrastructure.db.models import Base, MessageTable, SessionTable
from agent.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from agent.infrastructure.settings import DatabaseSettings
from agent.ports.repositories import UnitOfWork


class PostgresProvider(Provider):
    """提供连接池、Session 工厂和 PostgreSQL 事务端口。"""

    @provide(scope=Scope.APP)
    async def provide_engine(
        self,
        settings: DatabaseSettings,
    ) -> AsyncGenerator[AsyncEngine, None]:
        """创建当前微服务自己的异步连接池。"""

        engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout_seconds,
        )
        yield engine
        await engine.dispose()

    @provide(scope=Scope.APP)
    def provide_session_factory(
        self,
        engine: AsyncEngine,
    ) -> async_sessionmaker[AsyncSession]:
        """从当前服务的连接池创建 Session 工厂。"""

        return async_sessionmaker(engine, expire_on_commit=False)

    @provide(scope=Scope.REQUEST)
    async def provide_db_session(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> AsyncGenerator[AsyncSession, None]:
        """为一个 HTTP 请求或后台处理作用域提供 Session。"""

        async with session_factory() as session:
            yield session

    @provide(scope=Scope.REQUEST)
    def provide_unit_of_work(self, session: AsyncSession) -> UnitOfWork:
        """把当前 request-scoped Session 绑定为事务端口。"""

        return SqlAlchemyUnitOfWork(session)


async def initialize_agent_database(engine: AsyncEngine) -> None:
    """初始化 Agent 拥有的 Session、Message 表。"""

    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=(
                SessionTable.__table__,
                MessageTable.__table__,
            ),
        )


__all__ = [
    "PostgresProvider",
    "initialize_agent_database",
]
