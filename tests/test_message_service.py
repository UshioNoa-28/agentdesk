from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase

from agent.application.services.message_service import MessageService
from agent.domain.entities import Session
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage
from agent.exceptions import SessionNotFoundError


class _UnitOfWork:
    def __init__(self) -> None:
        self.commits = 0

    async def __aenter__(self) -> "_UnitOfWork":
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1


class _MessageRepository:
    """模拟真实端口:seq 在"语句内"计算并落库,会话不存在返回 None。

    计算与 append 之间不得有 await——这镜像 SQLite 单条 INSERT..SELECT
    不可被打断的语义;若未来把两步拆散,本文件的并发测试会先撞号。
    """

    def __init__(self) -> None:
        self.items: dict[str, list[Message]] = {}
        self.sessions: dict[str, Session] = {}

    async def add(self, message: Message) -> Message:
        self.items.setdefault(message.session_id, []).append(message)
        return message

    async def add_with_next_seq(self, *, session_id, message, metadata=None):
        session = self.sessions.get(session_id)
        if session is None:
            return None
        current = max((m.seq for m in self.items.get(session_id, ())), default=0)
        if current > 0:
            seq = current + 1
        elif session.parent_last_seq is not None:
            seq = session.parent_last_seq + 1
        else:
            seq = 1
        entity = Message.create(
            session_id=session_id,
            seq=seq,
            message=message,
            metadata=dict(metadata or {}),
        )
        self.items.setdefault(session_id, []).append(entity)
        return entity

    async def list_for_session(
        self, session_id: str, *, max_seq: int | None = None
    ) -> list[Message]:
        msgs = self.items.get(session_id, [])
        return [m for m in msgs if max_seq is None or m.seq <= max_seq]

    async def last_seq(self, session_id: str) -> int:
        return max((m.seq for m in self.items.get(session_id, ())), default=0)

    async def delete_for_session(self, session_id: str) -> int:
        return len(self.items.pop(session_id, []))


class _FakeSessionService:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def history(self, _session_id: str) -> list[Message]:
        return []


def _service(
    repository: _MessageRepository,
) -> tuple[MessageService, _UnitOfWork]:
    uow = _UnitOfWork()
    service = MessageService(
        repository=repository,
        unit_of_work=uow,
        session_service=_FakeSessionService(),
    )
    return service, uow


class MessageServiceTests(IsolatedAsyncioTestCase):
    def test_removed_legacy_mcp_result_kind_fails_loudly(self) -> None:
        """破坏性迁移后旧 kind 必须直接暴露为错误。"""

        with self.assertRaisesRegex(ValueError, "unsupported message kind"):
            Message.create(
                session_id="session-1",
                seq=1,
                message=ModelMessage.tool(
                    name="execute_mcp",
                    tool_call_id="call-1",
                    content="legacy result",
                ),
                metadata={"kind": "mcp_tool_result"},
            )

    async def test_add_owns_commit_and_allocates_sequential_seq(self) -> None:
        session = Session.create(title="test")
        repository = _MessageRepository()
        repository.sessions[session.id] = session
        service, uow = _service(repository)

        msg1 = await service.add(
            session_id=session.id,
            message=ModelMessage.human("hello 1"),
            metadata={"source": "test", "kind": MessageKind.USER},
        )
        msg2 = await service.add(
            session_id=session.id,
            message=ModelMessage.assistant(content="reply 1"),
            metadata={"source": "test", "kind": MessageKind.ASSISTANT_ANSWER},
        )

        self.assertEqual("hello 1", msg1.content)
        self.assertEqual(1, msg1.seq)
        self.assertEqual("reply 1", msg2.content)
        self.assertEqual(2, msg2.seq)
        self.assertEqual(2, uow.commits)

    async def test_add_allocates_seq_starting_from_parent_last_seq_for_resumed_session(
        self,
    ) -> None:
        resumed_session = Session.create(title="resumed", parent_last_seq=5)
        repository = _MessageRepository()
        repository.sessions[resumed_session.id] = resumed_session
        service, _ = _service(repository)

        msg1 = await service.add(
            session_id=resumed_session.id,
            message=ModelMessage.human("resumed first msg"),
        )
        msg2 = await service.add(
            session_id=resumed_session.id,
            message=ModelMessage.assistant(content="resumed second msg"),
        )

        self.assertEqual(6, msg1.seq)
        self.assertEqual(7, msg2.seq)

    async def test_concurrent_adds_never_share_a_seq(self) -> None:
        """并发 gather 下,原子端口方法不允许出现重复/空洞 seq。"""

        session = Session.create(title="race")
        repository = _MessageRepository()
        repository.sessions[session.id] = session
        service, _ = _service(repository)

        results = await asyncio.gather(
            *(
                service.add(
                    session_id=session.id,
                    message=ModelMessage.human(f"concurrent {index}"),
                )
                for index in range(8)
            )
        )

        seqs = sorted(message.seq for message in results)
        self.assertEqual(list(range(1, 9)), seqs)

    async def test_add_translates_missing_session(self) -> None:
        service, _ = _service(_MessageRepository())

        with self.assertRaises(SessionNotFoundError):
            await service.add(
                session_id="non-existent-session",
                message=ModelMessage.human("hello"),
            )
