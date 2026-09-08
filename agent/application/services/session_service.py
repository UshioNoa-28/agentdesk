from __future__ import annotations

from agent.domain.entities import Session
from agent.domain.exceptions import DomainValidationError
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage
from agent.domain.tools import MAIN_AGENT_ONLY_TOOL_NAMES
from agent.exceptions import DomainStateError, SessionDeletionConflictError, SessionNotFoundError
from agent.ports.repositories import (
    MessageRepository,
    SessionRepository,
    UnitOfWork,
)
from agent.ports.services import SessionServicePort
from agent.ports.tools import McpRegistryPort, SkillCatalogPort
from agent.prompt.system import build_system_prompt


class SessionService(SessionServicePort):
    """负责 Session 元数据、初始系统提示词持久化、resume 和删除。"""

    def __init__(
        self,
        *,
        repository: SessionRepository,
        messages: MessageRepository,
        unit_of_work: UnitOfWork,
        mcp_registry: McpRegistryPort,
        skill_catalog: SkillCatalogPort,
    ) -> None:
        self._repository = repository
        self._messages = messages
        self._unit_of_work = unit_of_work
        self._mcp_registry = mcp_registry
        self._skill_catalog = skill_catalog

    async def create(
        self,
        *,
        title: str,
        user_id: str = "default_user",
        parent_session_id: str | None = None,
        main_session_id: str | None = None,
        allowed_tools: tuple[str, ...] = (),
        custom_system_prompt: str | None = None,
    ) -> Session:
        """创建 Session；根会话原子持久化初始 System Prompt，Resume 会话继承父 Prompt。"""

        clean_title = title.strip()
        if not clean_title:
            raise DomainValidationError("Session title cannot be empty.")

        if custom_system_prompt is not None and not custom_system_prompt.strip():
            raise DomainValidationError("Custom system prompt cannot be empty.")

        if parent_session_id is not None and custom_system_prompt is not None:
            raise DomainValidationError("A resumed session cannot define a new system prompt.")

        async with self._unit_of_work:
            all_sessions = await self._repository.list_all()
            if any(s.title == clean_title for s in all_sessions):
                raise DomainValidationError(f"Session with title '{clean_title}' already exists.")
            parent_last_seq: int | None = None
            if parent_session_id is not None:
                parent = await self._repository.get(parent_session_id)
                if parent is None:
                    raise SessionNotFoundError(parent_session_id)
                parent_last_seq = await self._messages.last_seq(parent_session_id)
            if main_session_id is not None:
                main_sess = await self._repository.get(main_session_id)
                if main_sess is None:
                    raise SessionNotFoundError(main_session_id)
                # 星形拓扑硬约束：子代理会话永远拿不到主控专用工具。
                # define_subagent 已在工具入口过滤，这里兜底所有其它创建路径。
                forbidden = [
                    name for name in allowed_tools if name in MAIN_AGENT_ONLY_TOOL_NAMES
                ]
                if forbidden:
                    raise DomainValidationError(
                        "Subagent sessions cannot be granted main-agent-only tools: "
                        + ", ".join(forbidden)
                        + ". Only the main agent may use them."
                    )

            session = Session.create(
                title=clean_title,
                user_id=user_id,
                parent_session_id=parent_session_id,
                main_session_id=main_session_id,
                allowed_tools=allowed_tools,
                parent_last_seq=parent_last_seq,
            )
            await self._repository.add(session)

            if parent_session_id is None:
                if custom_system_prompt is not None:
                    system_content = custom_system_prompt.strip()
                else:
                    skills = self._skill_catalog.metadata()
                    tools = self._mcp_registry.server_descriptions()
                    system_content = build_system_prompt(
                        server_descriptions=tools,
                        skill_descriptions=skills,
                        meta_tools=allowed_tools,
                    )

                system_message = Message.create(
                    session_id=session.id,
                    seq=1,
                    message=ModelMessage.system(system_content),
                    metadata={"kind": MessageKind.SYSTEM},
                )
                await self._messages.add(system_message)

            await self._unit_of_work.commit()
            return session

    async def get(self, session_id: str) -> Session:
        async with self._unit_of_work:
            session = await self._repository.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            return session

    async def list(self) -> list[Session]:
        async with self._unit_of_work:
            return await self._repository.list_all()

    async def list_by_main_session(self, main_session_id: str) -> list[Session]:
        async with self._unit_of_work:
            return await self._repository.list_by_main_session(main_session_id)

    async def inbound_last_seq(self, session_id: str) -> int:
        async with self._unit_of_work:
            return await self._messages.inbound_last_seq(session_id)

    async def rename(self, *, session_id: str, title: str) -> Session:
        clean_title = title.strip()
        if not clean_title:
            raise DomainValidationError("Session title cannot be empty.")

        async with self._unit_of_work:
            all_sessions = await self._repository.list_all()
            if any(s.title == clean_title and s.id != session_id for s in all_sessions):
                raise DomainValidationError(f"Session with title '{clean_title}' already exists.")

            session = await self._repository.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            # 子代理标题是路由机制本身（{主会话id}_subagent_{name}），改名即断路由。
            if session.main_session_id is not None:
                raise DomainStateError("Subagent sessions cannot be renamed.")
            session.rename(clean_title)
            saved = await self._repository.save(session)
            await self._unit_of_work.commit()
            return saved

    async def resume(
        self,
        *,
        session_id: str,
        title: str | None = None,
        parent_last_seq: int | None = None,
    ) -> Session:
        async with self._unit_of_work:
            parent = await self._repository.get(session_id)
            if parent is None:
                raise SessionNotFoundError(session_id)

            all_sessions = await self._repository.list_all()
            existing_titles = {s.title for s in all_sessions}

            if title is not None:
                clean_title = title.strip()
                if not clean_title:
                    raise DomainValidationError("Session title cannot be empty.")
                if clean_title in existing_titles:
                    raise DomainValidationError(
                        f"Session with title '{clean_title}' already exists."
                    )
                target_title = clean_title
            else:
                base_title = f"Resume: {parent.title}"
                target_title = base_title
                counter = 2
                while target_title in existing_titles:
                    target_title = f"{base_title} ({counter})"
                    counter += 1

            if parent_last_seq is None:
                parent_last_seq = await self._messages.last_seq(session_id)
            elif parent_last_seq < 1:
                raise DomainValidationError("parent_last_seq must be positive.")
            else:
                current_max = await self._messages.last_seq(session_id)
                if parent_last_seq > current_max:
                    raise DomainValidationError(
                        f"parent_last_seq {parent_last_seq} exceeds parent session "
                        f"last seq {current_max}."
                    )

            child = parent.resume(title=target_title, parent_last_seq=parent_last_seq)
            await self._repository.add(child)
            await self._unit_of_work.commit()
            return child

    async def delete(self, session_id: str) -> None:
        async with self._unit_of_work:
            has_children = await self._repository.has_children(session_id)
            if has_children:
                raise SessionDeletionConflictError(session_id=session_id, reason="children")
            await self._messages.delete_for_session(session_id)
            deleted = await self._repository.delete(session_id)
            if not deleted:
                raise SessionNotFoundError(session_id)
            await self._unit_of_work.commit()

    async def history(self, session_id: str) -> list[Message]:
        """读取当前 Session 可继承的祖先消息和本地消息（严格按分叉点截断）。"""

        async with self._unit_of_work:
            session = await self._repository.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            chain = await self._session_chain(session)
            history: list[Message] = []
            for i in range(len(chain) - 1):
                ancestor = chain[i]
                child = chain[i + 1]
                history.extend(
                    await self._messages.list_for_session(
                        ancestor.id,
                        max_seq=child.parent_last_seq,
                    )
                )
            history.extend(await self._messages.list_for_session(session.id))
            return history

    async def _session_chain(self, session: Session) -> list[Session]:
        chain = [session]
        parent_id = session.parent_session_id
        while parent_id is not None:
            parent = await self._repository.get(parent_id)
            if parent is None:
                raise SessionNotFoundError(parent_id)
            chain.append(parent)
            parent_id = parent.parent_session_id
        chain.reverse()
        return chain


__all__ = ["SessionService"]
