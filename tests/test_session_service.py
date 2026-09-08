from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from agent.application.services.message_service import MessageService
from agent.application.services.session_service import SessionService
from agent.domain.entities import Session
from agent.domain.exceptions import DomainValidationError
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage
from agent.domain.tools import MAIN_AGENT_ONLY_TOOL_NAMES
from agent.exceptions import DomainStateError, SessionDeletionConflictError, SessionNotFoundError


class _FakeUnitOfWork:
    def __init__(self) -> None:
        self.commits = 0
        self.rolled_back = False

    async def __aenter__(self) -> "_FakeUnitOfWork":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.rolled_back = True
        return None

    async def commit(self) -> None:
        self.commits += 1


class _FakeSessionRepository:
    def __init__(self) -> None:
        self.items: dict[str, Session] = {}

    async def add(self, session: Session) -> None:
        self.items[session.id] = session

    async def get(self, session_id: str) -> Session | None:
        return self.items.get(session_id)

    async def get_for_update(self, session_id: str) -> Session | None:
        return self.items.get(session_id)

    async def list_all(self) -> list[Session]:
        return sorted(self.items.values(), key=lambda s: (s.updated_at, s.created_at), reverse=True)

    async def list_by_main_session(self, main_session_id: str) -> list[Session]:
        return [s for s in self.items.values() if s.main_session_id == main_session_id]

    async def has_children(self, session_id: str) -> bool:
        return any(item.parent_session_id == session_id for item in self.items.values())

    async def save(self, session: Session) -> Session:
        self.items[session.id] = session
        return session

    async def delete(self, session_id: str) -> bool:
        return self.items.pop(session_id, None) is not None


class _FakeMessageRepository:
    def __init__(self, sessions: _FakeSessionRepository | None = None) -> None:
        self.sessions = sessions
        self.items: dict[str, list[Message]] = {}
        self.fail_on_add = False

    async def add(self, message: Message) -> Message:
        if self.fail_on_add:
            raise RuntimeError("Database error during message add")
        self.items.setdefault(message.session_id, []).append(message)
        if self.sessions is not None:
            sess = await self.sessions.get(message.session_id)
            if sess is not None:
                sess.updated_at = message.created_at
        return message

    async def seed(self, message: Message) -> None:
        self.items.setdefault(message.session_id, []).append(message)

    async def list_for_session(
        self, session_id: str, *, max_seq: int | None = None
    ) -> list[Message]:
        messages = sorted(self.items.get(session_id, ()), key=lambda message: message.seq)
        return [message for message in messages if max_seq is None or message.seq <= max_seq]

    async def last_seq(self, session_id: str) -> int:
        return max((message.seq for message in self.items.get(session_id, ())), default=0)

    async def inbound_last_seq(self, session_id: str) -> int:
        return max(
            (message.seq for message in self.items.get(session_id, ()) if message.role == "human"),
            default=0,
        )

    async def delete_for_session(self, session_id: str) -> int:
        return len(self.items.pop(session_id, []))


class _FakeMcpRegistry:
    def server_descriptions(self) -> tuple[tuple[str, str], ...]:
        return ()


class _FakeSkillCatalog:
    def metadata(self) -> tuple:
        return ()


class _FakeMetaTools:
    def get_tools(self, allowed_tools: tuple[str, ...] = ()) -> tuple:
        return ()


class SessionServiceTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.uow = _FakeUnitOfWork()
        self.sessions = _FakeSessionRepository()
        self.messages = _FakeMessageRepository(self.sessions)
        self.mcp_registry = _FakeMcpRegistry()
        self.skill_catalog = _FakeSkillCatalog()
        self.service = SessionService(
            repository=self.sessions,
            messages=self.messages,
            unit_of_work=self.uow,
            mcp_registry=self.mcp_registry,
            skill_catalog=self.skill_catalog,
        )
        self.message_service = MessageService(
            repository=self.messages,
            session_repository=self.sessions,
            unit_of_work=self.uow,
            session_service=self.service,
        )

    async def _append(self, session_id: str, content: str) -> Message:
        return await self.message_service.add(
            session_id=session_id,
            message=ModelMessage.human(content),
            metadata={"kind": MessageKind.USER},
        )

    async def test_create_session_persists_default_system_prompt(self) -> None:
        session = await self.service.create(title="Test Session")
        messages = await self.service.history(session.id)

        self.assertEqual(1, len(messages))
        system_msg = messages[0]
        self.assertEqual(1, system_msg.seq)
        self.assertEqual("system", system_msg.role)
        self.assertEqual(MessageKind.SYSTEM, system_msg.metadata.get("kind"))
        self.assertIn("You are AgentDesk", system_msg.content)

    async def test_create_session_with_custom_system_prompt(self) -> None:
        session = await self.service.create(
            title="Subagent Session",
            custom_system_prompt="You are a subagent worker.",
        )
        messages = await self.service.history(session.id)

        self.assertEqual(1, len(messages))
        system_msg = messages[0]
        self.assertEqual(1, system_msg.seq)
        self.assertEqual("system", system_msg.role)
        self.assertEqual(MessageKind.SYSTEM, system_msg.metadata.get("kind"))
        self.assertEqual("You are a subagent worker.", system_msg.content)

    async def test_create_session_resumed_cannot_override_system_prompt(self) -> None:
        parent = await self.service.create(title="Parent")
        with self.assertRaises(DomainValidationError) as ctx:
            await self.service.create(
                title="Resumed Session",
                parent_session_id=parent.id,
                custom_system_prompt="Should fail",
            )
        self.assertIn("A resumed session cannot define a new system prompt.", str(ctx.exception))

    async def test_create_session_empty_custom_prompt_rejected(self) -> None:
        with self.assertRaises(DomainValidationError) as ctx:
            await self.service.create(title="Session 1", custom_system_prompt="   ")
        self.assertIn("Custom system prompt cannot be empty.", str(ctx.exception))

    async def test_create_session_empty_title_rejected(self) -> None:
        with self.assertRaises(DomainValidationError) as ctx:
            await self.service.create(title="   ")
        self.assertIn("Session title cannot be empty", str(ctx.exception))

    async def test_resumed_session_inherits_single_parent_system_prompt(self) -> None:
        parent = await self.service.create(title="Parent")
        await self._append(parent.id, "first user query")
        child = await self.service.resume(session_id=parent.id)
        await self._append(child.id, "child query")

        history = await self.service.history(child.id)
        system_msgs = [m for m in history if m.role == "system"]
        self.assertEqual(1, len(system_msgs))
        self.assertEqual(0, history.index(system_msgs[0]))
        self.assertEqual(3, len(history))

    async def test_session_creation_rollback_on_failure(self) -> None:
        self.messages.fail_on_add = True
        with self.assertRaises(RuntimeError):
            await self.service.create(title="Failing Session")
        self.assertTrue(self.uow.rolled_back)

    async def test_resume_creates_child_session(self) -> None:
        parent = await self.service.create(title="原会话")  # seq 1 = system
        await self._append(parent.id, "old")           # seq 2 = user
        await self._append(parent.id, "answer")        # seq 3 = user

        resumed = await self.service.resume(session_id=parent.id)

        self.assertNotEqual(parent.id, resumed.id)
        self.assertEqual(parent.id, resumed.parent_session_id)
        self.assertEqual("Resume: 原会话", resumed.title)

    async def test_resume_multiple_times_auto_deduplicates_titles_and_validates_custom_titles(
        self,
    ) -> None:
        """验证多次 resume 会话自动生成不重名的序号标题，且对显式重复标题抛出校验异常。"""
        parent = await self.service.create(title="原会话")

        # 第 1 次自动命名: "Resume: 原会话"
        resumed1 = await self.service.resume(session_id=parent.id)
        self.assertEqual("Resume: 原会话", resumed1.title)

        # 第 2 次自动命名: "Resume: 原会话 (2)"
        resumed2 = await self.service.resume(session_id=parent.id)
        self.assertEqual("Resume: 原会话 (2)", resumed2.title)

        # 第 3 次自动命名: "Resume: 原会话 (3)"
        resumed3 = await self.service.resume(session_id=parent.id)
        self.assertEqual("Resume: 原会话 (3)", resumed3.title)

        # 显式指定重复名称 -> 抛出 DomainValidationError
        with self.assertRaises(DomainValidationError) as ctx1:
            await self.service.resume(session_id=parent.id, title="Resume: 原会话")
        self.assertIn("already exists", str(ctx1.exception))

        # 显式指定空名称 -> 抛出 DomainValidationError
        with self.assertRaises(DomainValidationError) as ctx2:
            await self.service.resume(session_id=parent.id, title="   ")
        self.assertIn("cannot be empty", str(ctx2.exception))

    async def test_history_uses_parent_and_child_messages(self) -> None:
        parent = await self.service.create(title="原会话")
        first = await self._append(parent.id, "old")
        second = await self._append(parent.id, "answer")
        child = await self.service.resume(session_id=parent.id)
        child_message = await self._append(child.id, "new")

        history = await self.service.history(child.id)
        self.assertEqual(4, len(history))
        self.assertEqual("system", history[0].role)
        self.assertEqual(first.id, history[1].id)
        self.assertEqual(second.id, history[2].id)
        self.assertEqual(child_message.id, history[3].id)

    async def test_delete_requires_leaf_and_deletes_messages(self) -> None:
        parent = await self.service.create(title="父会话")
        child = await self.service.resume(session_id=parent.id)
        await self._append(child.id, "child")

        with self.assertRaises(SessionDeletionConflictError):
            await self.service.delete(parent.id)

        await self.service.delete(child.id)
        self.assertIsNone(await self.sessions.get(child.id))
        self.assertEqual([], self.messages.items.get(child.id, []))

    async def test_resumed_session_seq_continues_monotonically_from_parent(self) -> None:
        parent = await self.service.create(title="Parent")  # seq 1 = system
        await self._append(parent.id, "first user query")  # seq 2
        await self._append(parent.id, "second user query") # seq 3
        child = await self.service.resume(session_id=parent.id)
        child_msg1 = await self._append(child.id, "child query 1")
        child_msg2 = await self._append(child.id, "child query 2")

        self.assertEqual(4, child_msg1.seq)
        self.assertEqual(5, child_msg2.seq)

        history = await self.service.history(child.id)
        self.assertEqual([1, 2, 3, 4, 5], [m.seq for m in history])

    async def test_resumed_session_history_is_isolated_from_subsequent_parent_messages(
        self,
    ) -> None:
        """验证 Resume 分支快照语义：子会话历史严格截断，不受父会话后续追加消息的影响。"""
        # 1. 祖父 G：创建 (seq 1=system) + 2 条消息 (seq 2, 3)
        grandparent = await self.service.create(title="祖父 G")
        await self._append(grandparent.id, "G1")
        await self._append(grandparent.id, "G2")

        # 2. 父 P：从 G 分叉，此时 G 的 last_seq = 3
        parent = await self.service.resume(session_id=grandparent.id, title="父 P")
        self.assertEqual(3, parent.parent_last_seq)

        # 3. G 后续又聊了 2 句 (seq 4, 5) —— 不应被 P 或 C 看到
        await self._append(grandparent.id, "G3-后续消息")
        await self._append(grandparent.id, "G4-后续消息")

        # 4. P 发送 1 句本地消息 (seq 4)
        await self._append(parent.id, "P1")

        # 5. 子 C：从 P 分叉，此时 P 的 last_seq = 4
        child = await self.service.resume(session_id=parent.id, title="子 C")
        self.assertEqual(4, child.parent_last_seq)

        # 6. P 后续又发了 1 句 (seq 5) —— 不应被 C 看到
        await self._append(parent.id, "P2-后续消息")

        # 7. C 发送 1 句本地消息 (seq 5)
        await self._append(child.id, "C1")

        # 8. 验证 C 的历史消息：严格包含 [G_system, G1, G2, P1, C1]，不含 G3, G4, P2
        c_history = await self.service.history(child.id)
        c_contents = [m.content for m in c_history]
        self.assertIn("G1", c_contents)
        self.assertIn("G2", c_contents)
        self.assertIn("P1", c_contents)
        self.assertIn("C1", c_contents)
        self.assertNotIn("G3-后续消息", c_contents)
        self.assertNotIn("G4-后续消息", c_contents)
        self.assertNotIn("P2-后续消息", c_contents)
        self.assertEqual(5, len(c_history))

        # 9. 验证 P 的历史消息：严格包含 [G_system, G1, G2, P1, P2]，不含 G3, G4
        p_history = await self.service.history(parent.id)
        p_contents = [m.content for m in p_history]
        self.assertIn("G1", p_contents)
        self.assertIn("G2", p_contents)
        self.assertIn("P1", p_contents)
        self.assertIn("P2-后续消息", p_contents)
        self.assertNotIn("G3-后续消息", p_contents)
        self.assertNotIn("G4-后续消息", p_contents)
        self.assertEqual(5, len(p_history))

    async def test_create_session_with_main_session_id(self) -> None:
        main_sess = await self.service.create(title="Main Agent Session")
        sub_sess = await self.service.create(
            title="subagent_coder",
            main_session_id=main_sess.id,
            custom_system_prompt="You are a coder subagent.",
        )

        self.assertEqual(main_sess.id, sub_sess.main_session_id)
        self.assertIsNone(sub_sess.parent_session_id)
        self.assertEqual("subagent_coder", sub_sess.title)

    async def test_create_session_with_nonexistent_main_session_id_raises_not_found(self) -> None:
        with self.assertRaises(SessionNotFoundError):
            await self.service.create(
                title="Orphan subagent",
                main_session_id="non-existent-id",
            )

    async def test_create_subagent_session_rejects_main_agent_only_tools(self) -> None:
        """C1 兜底：子代理会话（main_session_id 非空）禁止授予主控专用工具。"""
        main_sess = await self.service.create(title="Main Session")
        for tool_name in MAIN_AGENT_ONLY_TOOL_NAMES:
            with self.assertRaises(DomainValidationError):
                await self.service.create(
                    title=f"subagent_bad_{tool_name}",
                    main_session_id=main_sess.id,
                    custom_system_prompt="prompt",
                    allowed_tools=(tool_name,),
                )

    async def test_create_subagent_session_allows_worker_tools(self) -> None:
        """子代理会话授予 worker 工具不受主控专用工具约束影响。"""
        main_sess = await self.service.create(title="Main Session")
        sub_sess = await self.service.create(
            title="subagent_worker",
            main_session_id=main_sess.id,
            custom_system_prompt="prompt",
            allowed_tools=("send_message", "execute_python"),
        )
        self.assertEqual(("send_message", "execute_python"), sub_sess.allowed_tools)

    async def test_rename_subagent_session_is_rejected(self) -> None:
        """子代理标题即路由（{主会话id}_subagent_{name}），禁止 rename。"""
        main_sess = await self.service.create(title="Main Session")
        sub_sess = await self.service.create(
            title="subagent_coder",
            main_session_id=main_sess.id,
            custom_system_prompt="prompt",
        )
        with self.assertRaises(DomainStateError):
            await self.service.rename(session_id=sub_sess.id, title="new-name")

    async def test_resumed_session_preserves_main_session_id(self) -> None:
        main_sess = await self.service.create(title="Main Session")
        sub_sess = await self.service.create(
            title="subagent_researcher",
            main_session_id=main_sess.id,
            custom_system_prompt="Research role.",
        )
        resumed_sub = await self.service.resume(session_id=sub_sess.id)

        self.assertEqual(sub_sess.id, resumed_sub.parent_session_id)
        self.assertEqual(main_sess.id, resumed_sub.main_session_id)

    async def test_create_session_with_allowed_tools_persists_and_resumes(self) -> None:
        sess = await self.service.create(
            title="Scoped Worker",
            allowed_tools=("execute_python", "send_message"),
        )
        self.assertEqual(("execute_python", "send_message"), sess.allowed_tools)

        fetched = await self.service.get(sess.id)
        self.assertEqual(("execute_python", "send_message"), fetched.allowed_tools)

        resumed = await self.service.resume(session_id=sess.id)
        self.assertEqual(("execute_python", "send_message"), resumed.allowed_tools)

    async def test_create_session_with_parent_session_id_captures_parent_last_seq(self) -> None:
        """验证直接通过 create(parent_session_id=...) 创建子会话也能正确捕获父分叉点。"""
        parent = await self.service.create(title="Parent Directly")
        await self._append(parent.id, "query 1")
        await self._append(parent.id, "query 2")

        child = await self.service.create(
            title="Child Directly",
            parent_session_id=parent.id,
        )
        self.assertEqual(3, child.parent_last_seq)

    def test_rehydrate_session_with_empty_allowed_tools_preserves_empty_tuple(self) -> None:
        """验证 allowed_tools 为空元组/列表时，rehydrate 不会回退到全部工具。"""
        from agent.domain.time import utc_now
        now = utc_now()
        session = Session.rehydrate(
            id="test-empty-tools",
            title="No Tools Session",
            parent_session_id=None,
            allowed_tools=(),
            created_at=now,
            updated_at=now,
        )
        self.assertEqual((), session.allowed_tools)

    def test_sqlalchemy_session_and_message_models_configuration(self) -> None:
        """验证 SQLAlchemy ORM 表模型和多外键关系配置无歧义。"""
        from sqlalchemy.orm import configure_mappers

        from agent.infrastructure.db.models import MessageTable, SessionTable

        # 触发全量 ORM 关系配置与外键校验
        configure_mappers()
        self.assertIsNotNone(SessionTable.__table__)
        self.assertIsNotNone(MessageTable.__table__)

    async def test_sqlalchemy_message_repository_add_creates_valid_entity(self) -> None:
        """验证 SqlAlchemyMessageRepository.add 能正确将 ModelMessage 转为 Message 实体并入库。"""
        from unittest.mock import AsyncMock, MagicMock

        from agent.infrastructure.db.repositories import SqlAlchemyMessageRepository

        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        repo = SqlAlchemyMessageRepository(session=mock_session)
        entity = Message.create(
            session_id="session-1",
            seq=1,
            message=ModelMessage.human("hello"),
            metadata={"kind": MessageKind.USER},
        )
        msg = await repo.add(entity)
        self.assertEqual("session-1", msg.session_id)
        self.assertEqual(1, msg.seq)
        self.assertEqual("human", msg.role)
        self.assertEqual("hello", msg.content)
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    async def test_create_and_rename_duplicate_title_raises_validation_error(self) -> None:
        """验证创建或重命名重复标题的 Session 时会抛出 DomainValidationError。"""
        service = SessionService(
            repository=_FakeSessionRepository(),
            messages=_FakeMessageRepository(),
            unit_of_work=_FakeUnitOfWork(),
            mcp_registry=_FakeMcpRegistry(),
            skill_catalog=_FakeSkillCatalog(),
        )

        sess = await service.create(title="test")
        self.assertEqual("test", sess.title)

        with self.assertRaises(DomainValidationError):
            await service.create(title="test")

        with self.assertRaises(DomainValidationError):
            await service.create(title="  test  ")

        sess2 = await service.create(title="other")
        with self.assertRaises(DomainValidationError):
            await service.rename(session_id=sess2.id, title="test")

    async def test_list_sessions_orders_by_recently_active_updated_at(self) -> None:
        """验证会话列表按最近活跃更新时间排序：旧会话发新消息后排到最前面。"""
        import asyncio

        # 1. 创建会话 A，并稍作等待
        session_a = await self.service.create(title="会话 A")
        await asyncio.sleep(0.01)

        # 2. 创建会话 B
        session_b = await self.service.create(title="会话 B")
        await asyncio.sleep(0.01)

        # 此时 B 比 A 新，B 应排在前面
        sessions = await self.service.list()
        self.assertEqual([session_b.id, session_a.id], [s.id for s in sessions])

        # 3. 往旧会话 A 中发送一条新消息
        await self._append(session_a.id, "会话 A 的最新消息")

        # 此时 A 的 updated_at 被刷新，A 应重新排在最前面
        sessions_after = await self.service.list()
        self.assertEqual([session_a.id, session_b.id], [s.id for s in sessions_after])

    async def test_list_by_main_session_retrieves_subagent_sessions(self) -> None:
        """验证读取主会话关联的全部子智能体会话。"""
        main_sess = await self.service.create(title="Main Session")
        sub1 = await self.service.create(title="Subagent 1", main_session_id=main_sess.id)
        sub2 = await self.service.create(title="Subagent 2", main_session_id=main_sess.id)

        sub_res = await self.service.list_by_main_session(main_sess.id)
        self.assertEqual({sub1.id, sub2.id}, {s.id for s in sub_res})

    async def test_resume_with_custom_parent_last_seq(self) -> None:
        """验证指定合法 parent_last_seq 进行会话分叉与历史截断。"""
        parent = await self.service.create(title="Parent Session")
        # seq 1 is system prompt
        await self._append(parent.id, "Message 1")  # seq 2
        await self._append(parent.id, "Message 2")  # seq 3
        await self._append(parent.id, "Message 3")  # seq 4

        # 分叉到 seq 2（只保留到 Message 1）
        child = await self.service.resume(
            session_id=parent.id,
            title="Child at seq 2",
            parent_last_seq=2,
        )
        self.assertEqual(2, child.parent_last_seq)
        history = await self.service.history(child.id)
        self.assertEqual(2, len(history))  # system prompt + Message 1

    async def test_resume_with_invalid_parent_last_seq_raises_error(self) -> None:
        """验证非法或超界的 parent_last_seq 抛出 DomainValidationError。"""
        parent = await self.service.create(title="Parent")
        await self._append(parent.id, "Message 1")  # max seq is 2

        with self.assertRaises(DomainValidationError):
            await self.service.resume(session_id=parent.id, parent_last_seq=0)

        with self.assertRaises(DomainValidationError):
            await self.service.resume(session_id=parent.id, parent_last_seq=99)





