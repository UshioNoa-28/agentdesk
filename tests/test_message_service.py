from __future__ import annotations

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
    def __init__(self) -> None:
        self.items: dict[str, list[Message]] = {}

    async def add(self, message: Message) -> Message:
        self.items.setdefault(message.session_id, []).append(message)
        return message

    async def list_for_session(
        self, session_id: str, *, max_seq: int | None = None
    ) -> list[Message]:
        msgs = self.items.get(session_id, [])
        return [m for m in msgs if max_seq is None or m.seq <= max_seq]

    async def last_seq(self, session_id: str) -> int:
        return max((m.seq for m in self.items.get(session_id, ())), default=0)

    async def delete_for_session(self, session_id: str) -> int:
        return len(self.items.pop(session_id, []))


class _FakeSessionRepository:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def get_for_update(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)


class _FakeSessionService:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def history(self, _session_id: str) -> list[Message]:
        return []


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
        session_service = _FakeSessionService()
        session_service.sessions[session.id] = session
        session_repository = _FakeSessionRepository()
        session_repository.sessions[session.id] = session
        repository = _MessageRepository()
        unit_of_work = _UnitOfWork()
        service = MessageService(
            repository=repository,
            session_repository=session_repository,
            unit_of_work=unit_of_work,
            session_service=session_service,
        )

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
        self.assertEqual(2, unit_of_work.commits)

    async def test_add_allocates_seq_starting_from_parent_last_seq_for_resumed_session(
        self,
    ) -> None:
        resumed_session = Session.create(title="resumed", parent_last_seq=5)
        session_service = _FakeSessionService()
        session_service.sessions[resumed_session.id] = resumed_session
        session_repository = _FakeSessionRepository()
        session_repository.sessions[resumed_session.id] = resumed_session
        repository = _MessageRepository()
        unit_of_work = _UnitOfWork()
        service = MessageService(
            repository=repository,
            session_repository=session_repository,
            unit_of_work=unit_of_work,
            session_service=session_service,
        )

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

    async def test_add_translates_missing_session(self) -> None:
        repository = _MessageRepository()
        service = MessageService(
            repository=repository,
            session_repository=_FakeSessionRepository(),
            unit_of_work=_UnitOfWork(),
            session_service=_FakeSessionService(),
        )

        with self.assertRaises(SessionNotFoundError):
            await service.add(
                session_id="non-existent-session",
                message=ModelMessage.human("hello"),
            )
