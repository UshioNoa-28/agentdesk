"""Alembic 异步迁移环境(PostgreSQL + asyncpg)。

被两种入口执行:CLI(``alembic upgrade head``)与应用启动时的
``command.upgrade``(见 agent/infrastructure/db/migrations.py)。
两种路径都经由本脚本末尾的在线/离线分支;``asyncio.run`` 要求执行线程
没有正在运行的事件循环,应用侧因此必须经 ``asyncio.to_thread`` 调用。
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import make_url, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from agent.infrastructure.db.models import Base
from agent.infrastructure.settings import DatabaseSettings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """连接串与运行时同源:DatabaseSettings(进程 env 优先于默认值)。"""

    return DatabaseSettings().database_url


def run_migrations_offline() -> None:
    """离线模式:不连数据库,把迁移 SQL 打印到 stdout。

    渲染 SQL 只需方言元数据,同步入口不接受 async 驱动,
    所以去掉 ``+asyncpg`` / ``+aiosqlite`` 驱动后缀。
    """

    url = make_url(_database_url())
    context.configure(
        url=url.set(drivername=url.drivername.split("+", 1)[0]),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在线模式核心:借同步 Connection 驱动 context(alembic 为同步 API)。"""

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
