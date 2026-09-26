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
from typing import Any

from autogen_core import (
    MessageContext,
    message_handler,
)
from autogen_core import (
    RoutedAgent as AutoGenRoutedAgent,
)

from agent.domain.entities import Session
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage
from agent.domain.multi_agent import (
    MAIN_AGENT_NAME,
    AgentStatus,
    Answered,
    AskOutcome,
    BusMessage,
    CancelMessage,
    MainAgentMessage,
    Queued,
    SubAgentMessage,
)
from agent.exceptions import AgentExecutionError, AgentInternalError
from agent.infrastructure.model import ModelProviderError
from agent.infrastructure.runtime.event_publisher import (
    MAIN_AGENT_TOPIC,
    SUBAGENT_TOPIC,
    RuntimeEventPublisher,
)
from agent.ports.model import AgentModelPort, AgentWorkflow
from agent.ports.runtime.routed_agent import RuntimeScope, RuntimeScopeProvider
from agent.ports.runtime.status import AgentStatusRegistryPort
from agent.ports.runtime.stream import AgentStreamHubPort
from agent.ports.services import MessageServicePort, SessionServicePort
from agent.ports.tools import MetaToolRegistryPort

logger = logging.getLogger(__name__)


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
        session_id: str,  # 作为 key
        scopes: RuntimeScopeProvider,
        status_registry: AgentStatusRegistryPort,
        description: str = "",
    ) -> None:
        super().__init__(description)
        if not session_id.strip():
            raise ValueError("RoutedAgent session_id cannot be empty.")
        self._session_id = session_id
        self._scopes = scopes
        # 进程内状态注册表：本实例的易失内存态镜像于此，供 list_subagents 观测。
        self._status_registry = status_registry
        # 逻辑显示名 用于main agent到subagent的通信
        self._display_name: str | None = None

        self._status = AgentStatus.IDLE
        self._status_lock = asyncio.Lock()
        # 本轮工作流的专属子 task（非 AutoGen activation task）。取消只作用于它，
        # 使 activation 本体不被 cancel，从而 DB 收尾能在健康上下文里正常关闭。
        self._drive_task: asyncio.Task[Any] | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def status(self) -> AgentStatus:
        return self._status

    def _mirror_status(self) -> None:
        """把当前状态镜像到进程内注册表（须在持有 _status_lock 时调用）。"""

        self._status_registry.set_status(self._session_id, self._status)

    async def set_status(self, status: AgentStatus) -> None:
        """安全更新生命周期状态。"""

        async with self._status_lock:
            self._status = status
            self._mirror_status()

    async def try_acquire_running(self) -> bool:
        """原子检查并尝试进入 RUNNING；已在 RUNNING 时返回 False。"""

        async with self._status_lock:
            if self._status == AgentStatus.RUNNING:
                return False
            self._status = AgentStatus.RUNNING
            self._mirror_status()
            return True

    @message_handler
    async def on_main_message(
        self,
        message: MainAgentMessage,
        ctx: MessageContext,
    ) -> AskOutcome | None:
        """主控入口：ask 的 RPC 驱动本轮，事件投递只落库不驱动。

        角色由消息类型（``MainAgentMessage``）本身确定；主控专属的两件事
        （非 RPC 事件只落库、跑完提炼长期记忆）就地表达，不再是通用路径里的分支。
        """

        async with self._scopes.scope() as scope:
            session = await self._persist_inbound(scope, message)
            display_name = self._resolve_display_name(session)

            # 主控的事件只落库，绝不驱动：主控唯一驱动入口是 ask 的 RPC，
            # 保证一个 request = 一次 graph.invoke（晚到的回报不触发多余图）。
            if not ctx.is_rpc:
                logger.info(
                    "Main agent '%s' received an event; persisted only: session_id=%s",
                    display_name,
                    self._session_id,
                )
                return None

            if not await self._acquire_running(display_name):# 正在running 直接return
                return Queued(session_id=self._session_id)

            started_at = time.perf_counter()
            self._drive_task = asyncio.create_task(self._drive(scope, session))
            try:
                result = await self._drive_task
            except asyncio.CancelledError:
                logger.warning(
                    "Agent '%s' invocation cancelled: session_id=%s",
                    display_name,
                    self._session_id,
                )
                raise  # 状态归位/级联由 on_cancel_message 承担；此处仅回抛给 ask RPC
            except Exception as exc:
                await self._fail_main(display_name, exc)
                raise  # _fail_main 始终抛错；此行仅为满足类型/控制流契约
            finally:
                self._drive_task = None

            await self.set_status(AgentStatus.IDLE)
            final_message = await self._finalize(scope, display_name, result, started_at)

            return Answered(message=final_message)

    @message_handler
    async def on_sub_message(
        self,
        message: SubAgentMessage,
        ctx: MessageContext,
    ) -> None:
        """子代理入口：收到的必然是任务事件，驱动本轮；fire-and-forget 无返回。

        子代理专属的收尾（失败告警、未回报时兜底投递）就地表达。
        """

        async with self._scopes.scope() as scope:
            session = await self._persist_inbound(scope, message)
            display_name = self._resolve_display_name(session)

            if not await self._acquire_running(display_name):
                return None

            started_at = time.perf_counter()
            self._drive_task = asyncio.create_task(self._drive(scope, session))
            try:
                result = await self._drive_task
            except asyncio.CancelledError:
                # 子代理经 publish 进入：CancelledError 若逃逸出 handler，会被
                # _process_publish 存为 _background_exception 并立即关停整个 runtime。
                # 故这里吞掉——fire-and-forget 无人等待其"答案"，取消即静默收场。
                logger.warning(
                    "Subagent '%s' invocation cancelled: session_id=%s",
                    display_name,
                    self._session_id,
                )
                return None
            except Exception as exc:
                await self._fail_sub(scope, session, display_name, exc)
                return None
            finally:
                self._drive_task = None

            await self.set_status(AgentStatus.IDLE)
            final_message = await self._finalize(scope, display_name, result, started_at)

            # 子代理没主动 send_message 给主控时，兜底把最终文本投递回去。
            if not result.get("sent_to_main", False) and final_message.content.strip():
                await self._deliver_to_main(scope, session, display_name, final_message.content)
            return None

    @message_handler
    async def on_cancel_message(self, message: CancelMessage, ctx: MessageContext) -> None:
        """取消本轮（控制面，fire-and-forget）。

        本 handler 运行在**独立的健康 activation** 上，与被中断的那一轮不是同一个
        task，因此其中的 await 不受取消影响；那一轮自身只把子 drive task 的取消回抛
        （主控）或吞掉（子代理），状态归位与向下级联全部集中于此。
        """

        self._cancel_running_turn()
        await self._cascade_cancel_to_running_children()
        await self.set_status(AgentStatus.IDLE)

    def _cancel_running_turn(self) -> None:
        """中断在途工作流：只 cancel 本轮专属子 task，不动 activation 本体。"""

        task = self._drive_task
        if task is not None and not task.done():
            task.cancel()

    async def _cascade_cancel_to_running_children(self) -> None:
        """向当前 RUNNING 的子代理级联取消（星形拓扑：主控→子，逐级一层）。

        子代理 ``list_by_main_session`` 恒为空，故本方法对子代理是自然 no-op。
        取一次 RUNNING 快照后投递：快照之后新起的子代理属于取消后另起的一轮，不在
        本次级联范围内。
        """

        async with self._scopes.scope() as scope:
            session_service: SessionServicePort = await scope.get(SessionServicePort)
            children = await session_service.list_by_main_session(self._session_id)
            running = [
                child
                for child in children
                if self._status_registry.get_status(child.id) == AgentStatus.RUNNING
            ]
            if not running:
                return
            publisher: RuntimeEventPublisher = await scope.get(RuntimeEventPublisher)
            for child in running:
                await publisher.publish(
                    CancelMessage(session_id=child.id, sender=MAIN_AGENT_NAME),
                    topic_type=SUBAGENT_TOPIC,
                    key=child.id,
                )

    async def _persist_inbound(self, scope: RuntimeScope, message: BusMessage) -> Session:
        """入站消息立即持久化；其 seq 作为本轮请求的基线（wait 工具据此判新）。"""

        message_service: MessageServicePort = await scope.get(MessageServicePort)
        session_service: SessionServicePort = await scope.get(SessionServicePort)
        await message_service.add(
            session_id=self._session_id,
            message=ModelMessage.human(message.content, name=message.sender),
            metadata={
                "source": "agent" if message.sender and message.sender != "user" else "user",
                "name": message.sender,
                "kind": MessageKind.USER,
            },
        )
        return await session_service.get(self._session_id)

    async def _acquire_running(self, display_name: str) -> bool:
        """状态锁保护：RUNNING 期间新消息只落库，运行中的工作流下一轮自会感知。"""

        if await self.try_acquire_running():
            return True
        logger.info(
            "Agent '%s' is already RUNNING; message persisted for later consumption: "
            "session_id=%s",
            display_name,
            self._session_id,
        )
        return False

    async def _drive(self, scope: RuntimeScope, session: Session) -> dict:
        """按 Session 白名单组装工具并驱动工作流。

        流式是 ask RPC 的旁路：入口注册了队列时才发布事件，否则 sink 为 None，
        工作流走非流式路径。
        """

        meta_tools: MetaToolRegistryPort = await scope.get(MetaToolRegistryPort)
        graph: AgentWorkflow = await scope.get(AgentWorkflow)
        model: AgentModelPort = await scope.get(AgentModelPort)
        stream_hub: AgentStreamHubPort = await scope.get(AgentStreamHubPort)
        sink = stream_hub.sink_for(self._session_id)
        tools = meta_tools.get_tools(session.allowed_tools)
        if sink is None:
            return await graph.ainvoke(
                messages=[],
                model=model,
                tools=tools,
                caller_session_id=self._session_id,
            )
        return await graph.astream(
            messages=[],
            model=model,
            tools=tools,
            caller_session_id=self._session_id,
            events=sink,
        )

    async def _finalize(
        self,
        scope: RuntimeScope,
        display_name: str,
        result: dict,
        started_at: float,
    ) -> Message:
        """取回本轮最终 assistant 消息并记录完成日志。"""

        latest_persisted = result.get("persisted_assistant")
        if isinstance(latest_persisted, Message):
            final_message = latest_persisted
        else:
            session_service: SessionServicePort = await scope.get(SessionServicePort)
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
        return final_message

    async def _fail_main(self, display_name: str, failure: Exception) -> None:
        """主控失败：错误属于本轮 ask，标记 ERROR 后向上抛。

        按异常链区分上游依赖失败（502）与图内部逻辑失败（500），让调用方和告警
        能区分"该重试"和"该修 bug"。
        """

        await self.set_status(AgentStatus.ERROR)
        logger.exception(
            "Agent '%s' invocation failed: session_id=%s",
            display_name,
            self._session_id,
        )
        error_class = (
            AgentExecutionError if _involves_remote_service(failure) else AgentInternalError
        )
        raise error_class(str(failure), details={"session_id": self._session_id}) from failure

    async def _fail_sub(
        self,
        scope: RuntimeScope,
        session: Session,
        display_name: str,
        failure: Exception,
    ) -> None:
        """子代理失败：标记 ERROR 后以普通语义文本通知主控（崩溃告警）。"""

        await self.set_status(AgentStatus.ERROR)
        logger.exception(
            "Subagent '%s' invocation failed: session_id=%s",
            display_name,
            self._session_id,
        )
        await self._deliver_to_main(
            scope,
            session,
            display_name,
            f"subagent '{display_name}' failed: {type(failure).__name__}: {failure}",
        )

    async def _deliver_to_main(
        self,
        scope: RuntimeScope,
        session: Session,
        display_name: str,
        content: str,
    ) -> None:
        """把子代理的文本（回报或失败告警）投递给归属主会话。"""

        owner_session_id = session.main_session_id
        if not owner_session_id:
            return
        try:
            publisher: RuntimeEventPublisher = await scope.get(RuntimeEventPublisher)
            await publisher.publish(
                MainAgentMessage(
                    content=content,
                    sender=display_name,
                    session_id=owner_session_id,
                ),
                topic_type=MAIN_AGENT_TOPIC,
                key=owner_session_id,
            )
        except Exception:
            logger.exception(
                "Subagent '%s' failed to deliver to main_agent", display_name
            )

    def _resolve_display_name(self, session: Session) -> str:
        """解析用于日志与工具上下文的逻辑名：主控固定，子代理取自标题后缀。"""

        if self._display_name is not None:  # 缓存
            return self._display_name
        if session.main_session_id is None:
            self._display_name = MAIN_AGENT_NAME
            return self._display_name
        marker = "_subagent_"
        # 标题里带 "{主会话id}_subagent_{名字}" 时取真名；否则退化为类别名。
        self._display_name = (
            session.title.rsplit(marker, 1)[-1] if marker in session.title else "subagent"
        )
        return self._display_name


__all__ = ["RoutedAgent"]