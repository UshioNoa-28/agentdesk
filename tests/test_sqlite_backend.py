"""真 SQLite 后端的集成回归。

覆盖 mock 仓储测不到的两点:``FOR UPDATE`` 在 SQLite 下被编译为空操作
后 seq 仍必须串行,以及连接 PRAGMA(foreign_keys/WAL)确实在每个连接生效。
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, mock

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from agent.application.services.message_service import MessageService
from agent.domain.entities import Session
from agent.domain.model_messages import ModelMessage
from agent.exceptions import SessionNotFoundError
from agent.infrastructure.db import (
    SqlAlchemyMessageRepository,
    SqlAlchemySessionRepository,
    SqlAlchemyUnitOfWork,
    apply_migrations,
    build_database_engine,
)
from agent.infrastructure.settings import DatabaseSettings


class SqliteBackendIntegrationTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        url = f"sqlite+aiosqlite:///{Path(directory.name) / 'agentdesk.db'}"
        # env.py 经 DatabaseSettings 读 DATABASE_URL(进程 env 最高优先级)。
        env_patch = mock.patch.dict(os.environ, {"DATABASE_URL": url})
        env_patch.start()
        self.addCleanup(env_patch.stop)

        await asyncio.to_thread(apply_migrations)
        self.engine = build_database_engine(DatabaseSettings())
        self._factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()

    async def _make_session(self) -> Session:
        session = Session.create(title="race")
        async with self._factory() as db:
            uow = SqlAlchemyUnitOfWork(db)
            async with uow:
                await SqlAlchemySessionRepository(db).add(session)
                await uow.commit()
        return session

    async def _add(self, session_id: str, index: int):
        async with self._factory() as db:
            service = MessageService(
                repository=SqlAlchemyMessageRepository(db),
                unit_of_work=SqlAlchemyUnitOfWork(db),
                session_service=None,  # add 路径不触达 history 委托
            )
            return await service.add(
                session_id=session_id,
                message=ModelMessage.human(f"concurrent {index}"),
            )

    async def test_concurrent_add_to_same_session_allocates_unique_sequential_seq(
        self,
    ) -> None:
        session = await self._make_session()

        stored = await asyncio.gather(
            *(self._add(session.id, index) for index in range(12))
        )

        seqs = sorted(message.seq for message in stored)
        self.assertEqual(list(range(1, 13)), seqs)

    async def test_fork_seed_starts_from_parent_last_seq(self) -> None:
        session = Session.create(title="forked", parent_last_seq=5)
        async with self._factory() as db:
            uow = SqlAlchemyUnitOfWork(db)
            async with uow:
                await SqlAlchemySessionRepository(db).add(session)
                await uow.commit()

        msg = await self._add(session.id, 0)
        self.assertEqual(6, msg.seq)

    async def test_add_to_missing_session_raises_not_found(self) -> None:
        with self.assertRaises(SessionNotFoundError):
            await self._add("00000000-0000-0000-0000-000000000000", 0)

    async def test_connection_pragmas_enabled(self) -> None:
        async with self.engine.connect() as conn:
            foreign_keys = (await conn.execute(text("PRAGMA foreign_keys"))).scalar()
            journal_mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar()

        self.assertEqual(1, int(foreign_keys))
        self.assertEqual("wal", str(journal_mode).lower())
