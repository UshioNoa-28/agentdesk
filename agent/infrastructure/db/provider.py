"""Agent SQLite 引擎和 UnitOfWork 的 Dishka Provider。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

from dishka import Provider, Scope, provide
from sqlalchemy import event, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agent.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from agent.infrastructure.settings import DatabaseSettings
from agent.ports.repositories import UnitOfWork

_SQLITE_BUSY_TIMEOUT_SECONDS = 30.0


def _apply_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    """每个 SQLite 连接建立时启用外键、WAL 和忙等超时。

    外键约束 SQLite 默认关闭；WAL 让读不阻塞写；busy_timeout 把偶发的
    跨进程写冲突（如命令行工具手滑打开同一文件）变成短暂等待而非报错。
    """

    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    timeout_ms = int(_SQLITE_BUSY_TIMEOUT_SECONDS * 1000)
    cursor.execute(f"PRAGMA busy_timeout={timeout_ms}")
    cursor.close()


def build_database_engine(settings: DatabaseSettings) -> AsyncEngine:
    """创建 SQLite 异步引擎；文件型数据库的父目录在此确保存在。"""

    url = settings.database_url
    database = make_url(url).database
    if database and database != ":memory:":
        Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(
        url,
        connect_args={"check_same_thread": False},
    )
    event.listens_for(engine.sync_engine, "connect")(_apply_sqlite_pragmas)
    return engine


class DatabaseProvider(Provider):
    """提供 SQLite 引擎、Session 工厂和事务端口。"""

    @provide(scope=Scope.APP)
    async def provide_engine(
        self,
        settings: DatabaseSettings,
    ) -> AsyncGenerator[AsyncEngine, None]:
        """创建进程级共享的异步引擎（连接池随引擎生命周期）。"""

        engine = build_database_engine(settings)
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


__all__ = [
    "DatabaseProvider",
]
