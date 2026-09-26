"""Multi-Agent 运行时管理器：名字路由、事件派发与心跳守护。

AutoGen Runtime 是唯一的智能体注册表（按 ``AgentId(type, key=session_id)``
惰性实例化并缓存）。本管理器负责：

- ``send_message``（RPC）：外部 ask 前门专用，首轮驱动主控；
- ``dispatch``（事件）：agent 间异步投递，发布即返回，不做任何计数与判断；
- 心跳发送器：按退避节拍向进行中的请求主控发送时间流逝通知，
  是否继续等待/催促/放弃完全由主控 LLM 决定。
"""

from __future__ import annotations

import asyncio
import logging

from autogen_core import AgentId, SingleThreadedAgentRuntime, TypeSubscription

from agent.domain.multi_agent import (
    MAIN_AGENT_NAME,
    Answered,
    AskOutcome,
    BusMessage,
    CancelMessage,
    MainAgentMessage,
    Queued,
)
from agent.exceptions import AgentInternalError
from agent.infrastructure.runtime.event_publisher import (
    MAIN_AGENT_TOPIC,
    SUBAGENT_TOPIC,
    RuntimeEventPublisher,
)
from agent.infrastructure.runtime.factory import SUBAGENT_TYPE, AgentFactory
from agent.ports.runtime.routed_agent import RuntimeScopeProvider
from agent.ports.services import SessionServicePort

logger = logging.getLogger(__name__)


class AgentRuntimeManager:
    """把 MainAgentMessage / SubAgentMessage 路由到目标 RoutedAgent。

    同时管理运行时与心跳循环的生命周期。
    """

    def __init__(
        self,
        *,
        runtime: SingleThreadedAgentRuntime,
        factory: AgentFactory,
        scopes: RuntimeScopeProvider,
        publisher: RuntimeEventPublisher,
    ) -> None:
        self._runtime = runtime
        self._factory = factory
        self._scopes = scopes
        self._publisher = publisher
        self._started = False
        self._start_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """后台消息循环是否已启动。"""

        return self._started

    async def _ensure_started(self) -> None:
        # 快速路径：已启动则直接返回，避免每次都抢锁。
        if self._started:
            return

        # 双检锁：并发首次 ask/dispatch 只会有一个真正执行 install/start，
        # 其余等待锁后发现 _started 已置位，直接返回，避免重复
        # register_factory / add_subscription / runtime.start()。
        async with self._start_lock:
            if self._started:
                return
            await self._factory.install(self._runtime)
            for topic_type, agent_type in (
                (MAIN_AGENT_TOPIC, MAIN_AGENT_NAME),
                (SUBAGENT_TOPIC, SUBAGENT_TYPE),
            ):
                await self._runtime.add_subscription(
                    TypeSubscription(topic_type=topic_type, agent_type=agent_type)
                ) # topic 用于 publish agent_type 用于 标识一个类型的 agent key 用于type内标记 
                # 这里仅提供两种topic用于fire and forget
            self._runtime.start()
            self._started = True

    async def stop(self) -> None:
        """等待在途消息处理完毕后停止后台消息循环。"""

        if not self._started:
            return
        await self._runtime.stop_when_idle()
        self._started = False

    async def send_message(self, message: MainAgentMessage) -> AskOutcome:
        """外部 ask 前门：RPC 驱动 main_agent 并等待本轮结果。

        RPC 语义只属于主控（星形拓扑：子代理通信一律走 dispatch 事件），
        因此不做收件人分派，直接以消息所属会话定位 main_agent 实例。
        类型即契约：``MainAgentMessage`` 已保证收件人是主控、会话非空。
        """

        target = AgentId(type=MAIN_AGENT_NAME, key=message.session_id)
        await self._ensure_started()

        logger.info(
            "[MultiAgent] Dispatching RPC message from '%s' to '%s' (session: %s)",
            message.sender,
            MAIN_AGENT_NAME,
            target.key,
        )
        reply = await self._runtime.send_message(message, target)

        if not isinstance(reply, (Answered, Queued)):
            # RPC 的正常出口只有 Answered（本轮产出最终消息）与 Queued（已入库、
            # 本轮不回复）；图失败走异常。AutoGen 的返回值无类型，这里守住契约。
            raise AgentInternalError(
                f"Agent '{MAIN_AGENT_NAME}' did not return a turn outcome "
                f"(received: {type(reply).__name__}).",
                details={"session_id": message.session_id},
            )
        return reply

    async def dispatch(self, message: BusMessage) -> None:
        """非阻塞投递：按消息类型发布事件给目标智能体，投递即返回。

        角色由消息类型表达：``MainAgentMessage`` → 主控通知；``SubAgentMessage``
        → 子代理任务。不再从 recipient 字符串反推。
        """

        if isinstance(message, MainAgentMessage):
            await self._ensure_started()
            await self._publisher.publish(
                message,
                topic_type=MAIN_AGENT_TOPIC,
                key=message.session_id,
            )
            return

        subagent_session_id = await self._resolve_subagent_session(
            name=message.recipient,
            owner_session_id=message.session_id,
        )
        await self._ensure_started()
        logger.info(
            "[MultiAgent] Published task from '%s' to '%s' (subagent session: %s)",
            message.sender,
            message.recipient,
            subagent_session_id,
        )
        await self._publisher.publish(
            message,
            topic_type=SUBAGENT_TOPIC,
            key=subagent_session_id,
        )

    async def cancel(self, message: CancelMessage) -> None:
        """外部取消前门：向主控发布控制面事件（fire-and-forget），不等待结果。

        只定位主控实例（星形拓扑下取消自上而下发起）；子代理由主控自身在同一
        机制下级联。控制面事件不入库、不驱动工作流。
        """

        await self._ensure_started()
        logger.info(
            "[MultiAgent] Cancel requested by '%s' for main agent (session: %s)",
            message.sender,
            message.session_id,
        )
        await self._publisher.publish(
            message,
            topic_type=MAIN_AGENT_TOPIC,
            key=message.session_id,
        )

    async def _resolve_subagent_session(self, *, name: str, owner_session_id: str) -> str:
        """按裸名解析子智能体会话 ID；子会话标题格式为 ``{主会话id}_subagent_{name}``。"""

        async with self._scopes.scope() as scope:
            session_service: SessionServicePort = await scope.get(SessionServicePort)
            sub_sessions = await session_service.list_by_main_session(owner_session_id)

        expected_title = f"{owner_session_id}_subagent_{name}"
        for sub in sub_sessions:
            if sub.title == expected_title:
                return sub.id
        available = [
            sub.title.removeprefix(f"{owner_session_id}_subagent_") for sub in sub_sessions
        ]
        raise ValueError(
            f"Subagent '{name}' is not defined for session '{owner_session_id}'. "
            f"Currently available: {available}"
        )

__all__ = ["AgentRuntimeManager"]
