"""基于 AutoGen RoutedAgent 的智能体实现。

实例由 AutoGen Runtime 按 ``AgentId(type, key=session_id)`` 惰性创建并缓存，
因此构造函数只持有 APP 级依赖（作用域提供者）；业务服务、工作流和模型全部
在消息处理时从独立的请求作用域解析，保证事务与 AsyncSession 隔离。

两种传输入口共用同一个消费路径，只有失败处理按角色区分：

- 主控（``main_session_id is None``）：失败向调用方抛出（ask 的 RPC 在等）；
- 子代理：失败以普通语义文本通知主控（崩溃的正是它自己）。

主控的事件只落库绝不驱动——主控唯一的图驱动入口是 ask 的 RPC，
保证一个 request = 一次 graph.invoke。
"""

from __future__ import annotations

import asyncio
import logging
import time

from autogen_core import (
    MessageContext,
    message_handler,
)
from autogen_core import (
    RoutedAgent as AutoGenRoutedAgent,
)

from agent.domain.exceptions import DomainValidationError
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage
from agent.domain.multi_agent import (
    MAIN_AGENT_NAME,
    AgentMessage,
    AgentStatus,
    Answered,
    AskOutcome,
    Queued,
)
from agent.exceptions import AgentExecutionError, AgentInternalError
from agent.graph.agent_graph import AgentGraphExecutionError
from agent.infrastructure.model import ModelProviderError
from agent.ports.memory import MemoryServicePort
from agent.ports.model import AgentModelPort, AgentWorkflow
from agent.ports.runtime.routed_agent import RuntimeScopeProvider
from agent.ports.runtime.stream import AgentStreamHubPort
from agent.ports.services import MessageServicePort, SessionServicePort
from agent.ports.tools import MetaToolRegistryPort
from agent.runtime.event_publisher import MAIN_AGENT_TOPIC, RuntimeEventPublisher

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task[None]] = set()


def _involves_remote_service(failure: BaseException) -> bool:
    """沿异常因果链判断是否为模型 provider 等上游依赖失败。"""

    current: BaseException | None = failure
    while current is not None:
        if isinstance(current, ModelProviderError):
            return True
        current = current.__cause__
    return False


class RoutedAgent(AutoGenRoutedAgent):
    """具备状态机、自动持久化与独立请求作用域的 AutoGen Actor。"""

    def __init__(
        self,
        *,
        name: str,
        session_id: str,
        scopes: RuntimeScopeProvider,
        description: str = "",
    ) -> None:
        super().__init__(description)
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("RoutedAgent name cannot be empty.")
        if not session_id.strip():
            raise ValueError("RoutedAgent session_id cannot be empty.")
        self._name = clean_name
        self._session_id = session_id
        self._scopes = scopes
        # 子代理的逻辑显示名（如 "coder"）从 Session 标题解析后按实例缓存。
        self._display_name: str | None = None

        self._status = AgentStatus.IDLE
        self._status_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def status(self) -> AgentStatus:
        return self._status

    async def set_status(self, status: AgentStatus) -> None:
        """安全更新生命周期状态。"""

        async with self._status_lock:
            self._status = status

    async def try_acquire_running(self) -> bool:
        """原子检查并尝试进入 RUNNING；已在 RUNNING 时返回 False。"""

        async with self._status_lock:
            if self._status == AgentStatus.RUNNING:
                return False
            self._status = AgentStatus.RUNNING
            return True

    @message_handler
    async def on_message(
        self,
        message: AgentMessage,
        ctx: MessageContext,
    ) -> AskOutcome | None:
        """唯一入口：ask 的 RPC 与全部事件投递都走到这里。"""

        return await self._consume(message, is_rpc=ctx.is_rpc)

    async def _consume(self, message: AgentMessage, *, is_rpc: bool) -> AskOutcome | None:
        """统一消费路径：持久化入站消息、按角色与传输方式决定是否驱动工作流。

        - 主控的 ask(RPC) 与子代理的任务(事件) → 落库 + 驱动；
        - 主控的事件（回报/告警等）→ 只落库，绝不驱动（主控唯一驱动入口是
          ask 的 RPC，保证一个 request 对应一次 graph.invoke）。

        RPC 的两个出口是 ``Answered`` 与 ``Queued``；事件投递返回 ``None``
        （publish 方忽略返回值）。
        """

        content = message.content.strip()
        if not content:
            raise DomainValidationError("Question cannot be empty")

        started_at = time.perf_counter()
        sender_name = message.sender.strip() if message.sender else None

        # 每条消息开辟全新请求作用域：服务与工作流绝不跨消息复用事务。
        async with self._scopes.scope() as scope:
            message_service: MessageServicePort = await scope.get(MessageServicePort)
            session_service: SessionServicePort = await scope.get(SessionServicePort)
            meta_tools: MetaToolRegistryPort = await scope.get(MetaToolRegistryPort)
            graph: AgentWorkflow = await scope.get(AgentWorkflow)
            model: AgentModelPort = await scope.get(AgentModelPort)
            memory_service: MemoryServicePort = await scope.get(MemoryServicePort)

            # 1. 入站消息立即持久化；其 seq 作为本轮请求的基线（wait 工具据此判新）
            await message_service.add(
                session_id=self._session_id,
                message=ModelMessage.human(content, name=sender_name),
                metadata={
                    "source": "agent" if sender_name and sender_name != "user" else "user",
                    "name": sender_name,
                    "kind": MessageKind.USER,
                },
            )
            session = await session_service.get(self._session_id)
            display_name = self._resolve_display_name(session)

            # 2.5 主控的事件只落库，绝不驱动：主控的驱动入口只有 ask 的 RPC，
            # 保证一个 request = 一次 graph.invoke（晚到的回报不会触发多余图）。
            if session.main_session_id is None and not is_rpc:
                logger.info(
                    "Main agent '%s' received an event; persisted only: session_id=%s",
                    display_name,
                    self._session_id,
                )
                return None

            # 2. 状态锁保护：RUNNING 期间新消息只落库，运行中的工作流下一轮自会感知
            if not await self.try_acquire_running():
                logger.info(
                    "Agent '%s' is already RUNNING; message persisted for later consumption: "
                    "session_id=%s",
                    display_name,
                    self._session_id,
                )
                return Queued(session_id=self._session_id) if is_rpc else None

            try:
                # 3. 按 Session 白名单组装工具并驱动工作流；调用者会话 ID 是
                # 工具上下文的唯一会话来源，身份归属由工具按需从持久层推导。
                # 流式是 ask RPC 的旁路：HTTP 侧注册了队列时才发布事件，
                # 否则 sink 为 None，工作流走非流式路径。
                stream_hub: AgentStreamHubPort = await scope.get(AgentStreamHubPort)
                sink = stream_hub.sink_for(self._session_id)
                tools = meta_tools.get_tools(session.allowed_tools)
                if sink is None:
                    result = await graph.ainvoke(
                        messages=[],
                        model=model,
                        tools=tools,
                        caller_session_id=self._session_id,
                    )
                else:
                    result = await graph.astream(
                        messages=[],
                        model=model,
                        tools=tools,
                        caller_session_id=self._session_id,
                        events=sink,
                    )
            except asyncio.CancelledError:
                await self.set_status(AgentStatus.IDLE)
                logger.warning(
                    "Agent '%s' invocation cancelled: session_id=%s",
                    display_name,
                    self._session_id,
                )
                raise
            except AgentGraphExecutionError as exc:
                return await self._handle_failure(
                    publisher=await scope.get(RuntimeEventPublisher),
                    session=session,
                    failure=exc,
                )
            except Exception as exc:
                return await self._handle_failure(
                    publisher=await scope.get(RuntimeEventPublisher),
                    session=session,
                    failure=exc,
                )

            await self.set_status(AgentStatus.IDLE)

            latest_persisted = result.get("persisted_assistant")
            if isinstance(latest_persisted, Message):
                final_message = latest_persisted
            else:
                history = await session_service.history(self._session_id)
                assistants = [item for item in history if item.role == "assistant"]
                if not assistants:
                    raise AgentInternalError(
                        f"Agent '{display_name}' completed without persisting "
                        "an assistant message.",
                        details={"session_id": self._session_id},
                    )
                final_message = assistants[-1]

            logger.info(
                "Agent '%s' completed: session_id=%s seq=%s turns=%s elapsed_ms=%.1f",
                display_name,
                self._session_id,
                final_message.seq,
                result.get("turns", 0),
                (time.perf_counter() - started_at) * 1000,
            )

            # 4. 主会话异步提炼长期记忆
            if session.main_session_id is None:
                task = asyncio.create_task(
                    memory_service.add(
                        messages=[
                            {"role": "user", "content": content},
                            {"role": "assistant", "content": final_message.content},
                        ],
                        user_id=session.user_id,
                    )
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

            # 如果 subagent 没给 main agent send message，就默认提取 final message 发给主控。
            if (
                session.main_session_id is not None
                and not result.get("sent_to_main", False)
                and final_message.content.strip()
            ):
                try:
                    publisher = await scope.get(RuntimeEventPublisher)
                    await publisher.publish(
                        AgentMessage(
                            content=final_message.content,
                            sender=display_name,
                            recipient=MAIN_AGENT_NAME,
                            session_id=session.main_session_id,
                        ),
                        topic_type=MAIN_AGENT_TOPIC,
                        key=session.main_session_id,
                    )
                except Exception:
                    logger.exception(
                        "Subagent '%s' failed to auto-deliver its result to main_agent",
                        display_name,
                    )

            return Answered(message=final_message)

    async def _handle_failure(
        self,
        *,
        publisher: RuntimeEventPublisher,
        session: object,
        failure: Exception,
    ) -> Message | None:
        """按角色处理工作流失败。

        主控失败向调用方抛出；子代理失败以普通语义文本通知主控（崩溃告警）。
        """

        await self.set_status(AgentStatus.ERROR)
        owner_session_id = getattr(session, "main_session_id", None)
        if owner_session_id is None:
            # 主控：错误属于本轮 ask，向上抛；按异常链区分上游依赖失败（502）
            # 与图内部逻辑失败（500），让调用方和告警能区分"该重试"和"该修 bug"。
            logger.exception(
                "Agent '%s' invocation failed: session_id=%s",
                self._display_name or self._name,
                self._session_id,
            )
            error_class = (
                AgentExecutionError
                if _involves_remote_service(failure)
                else AgentInternalError
            )
            raise error_class(
                str(failure), details={"session_id": self._session_id}
            ) from failure
        # 子代理：让主控从消息语义里自己判断怎么处理。
        logger.exception(
            "Subagent '%s' invocation failed: session_id=%s",
            self._display_name or self._name,
            self._session_id,
        )
        try:
            await publisher.publish(
                AgentMessage(
                    content=(
                        f"subagent '{self._display_name or self._name}' failed: "
                        f"{type(failure).__name__}: {failure}"
                    ),
                    sender=self._display_name or self._name,
                    recipient=MAIN_AGENT_NAME,
                    session_id=owner_session_id,
                ),
                topic_type=MAIN_AGENT_TOPIC,
                key=owner_session_id,
            )
        except Exception:
            logger.exception(
                "Failed to publish failure notice for '%s'",
                self._display_name or self._name,
            )
        return None

    def _resolve_display_name(self, session: object) -> str:
        """解析用于日志与工具上下文的逻辑名：主控固定，子代理取自标题后缀。"""

        if self._display_name is not None:
            return self._display_name
        main_session_id = getattr(session, "main_session_id", None)
        if main_session_id is None:
            self._display_name = MAIN_AGENT_NAME
            return self._display_name
        title = getattr(session, "title", "")
        marker = "_subagent_"
        self._display_name = title.rsplit(marker, 1)[-1] if marker in title else self._name
        return self._display_name


__all__ = ["RoutedAgent"]