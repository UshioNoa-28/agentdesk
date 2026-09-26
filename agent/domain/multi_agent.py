"""Multi-Agent 领域模型与通信载荷。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from agent.domain.exceptions import DomainValidationError, MessageRoutingError
from agent.domain.messages import Message
from agent.domain.time import utc_now


class AgentStatus(StrEnum):
    """RoutedAgent 智能体生命周期状态机。"""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


MAIN_AGENT_NAME = "main_agent"

# 保留名：子代理一律不许占用。
# - main_agent：可伪造主控 sender 身份，绕过星形拓扑约束向其它子代理发消息；
# - user：会污染消息来源判定（source == "user"），带偏压缩锚点解析。
RESERVED_AGENT_NAMES = frozenset({MAIN_AGENT_NAME, "user"})


@dataclass(frozen=True, slots=True)
class MainAgentMessage:
    """投递给主控的载荷。

    两种传输共用同一类型，靠 ``MessageContext.is_rpc`` 区分对待：

    - RPC（用户 ask）：驱动本轮工作流并返回 ``AskOutcome``；
    - 事件（子代理回报、心跳）：只入库，绝不驱动（主控唯一图驱动入口是 RPC）。

    收件人恒为主控，因此不设 ``recipient`` 字段——目标角色由消息类型本身表达。
    """

    content: str
    sender: str
    session_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        content = self.content.strip()
        sender = self.sender.strip()
        session_id = self.session_id.strip()
        if not content:
            raise DomainValidationError("Message content is required and cannot be empty.")
        if not sender:
            raise DomainValidationError("Sender is required and cannot be empty.")
        if not session_id:
            raise DomainValidationError("Session id is required and cannot be empty.")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "sender", sender)
        object.__setattr__(self, "session_id", session_id)

        # 自环约束下沉到领域：主控不会给自己投递事件（那只会落库、不驱动，
        # 对调用方是误导），主控发起交互一律走新的 RPC。
        if sender == MAIN_AGENT_NAME:
            raise MessageRoutingError(
                "self_message_not_allowed",
                f"Cannot send a message to yourself ('{MAIN_AGENT_NAME}'). "
                "The main agent should simply continue its turn.",
            )

    @property
    def recipient(self) -> str:
        """兼容只读 ``recipient`` 的调用方；主控消息的收件人恒为主控名。"""

        return MAIN_AGENT_NAME


@dataclass(frozen=True, slots=True)
class SubAgentMessage:
    """投递给某个子代理的任务。星形拓扑下只由主控发起。

    ``session_id`` 是路由锚点（主会话 ID），子会话由运行时按 ``recipient``
    裸名解析。
    """

    content: str
    sender: str
    recipient: str
    session_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        content = self.content.strip()
        sender = self.sender.strip()
        recipient = self.recipient.strip()
        session_id = self.session_id.strip()
        if not content:
            raise DomainValidationError("Message content is required and cannot be empty.")
        if not sender:
            raise DomainValidationError("Sender is required and cannot be empty.")
        if not recipient:
            raise DomainValidationError("Recipient is required and cannot be empty.")
        if not session_id:
            raise DomainValidationError("Session id is required and cannot be empty.")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "sender", sender)
        object.__setattr__(self, "recipient", recipient)
        object.__setattr__(self, "session_id", session_id)

        # 星形拓扑与自环约束下沉到领域：子任务只能由主控发起、只能发给子代理，
        # 且不能发给发起者自己。
        if sender != MAIN_AGENT_NAME:
            raise MessageRoutingError(
                "star_topology_violation",
                f"Subagents may only send messages to '{MAIN_AGENT_NAME}' "
                "(star topology); a subagent cannot dispatch tasks to peers.",
            )
        if recipient == MAIN_AGENT_NAME:
            raise MessageRoutingError(
                "invalid_recipient",
                f"Messages to '{MAIN_AGENT_NAME}' must be MainAgentMessage, not SubAgentMessage.",
            )
        if recipient == sender:
            raise MessageRoutingError(
                "self_message_not_allowed",
                f"Cannot dispatch a task to the sender itself ('{recipient}').",
            )


# 投递进运行时总线的全部消息类型；端口与调度按此联合收窄。
BusMessage = MainAgentMessage | SubAgentMessage


@dataclass(frozen=True, slots=True)
class CancelMessage:
    """取消目标 actor 当前轮次的控制面事件（fire-and-forget，不入库、不驱动工作流）。

    刻意**不并入** :data:`BusMessage`：BusMessage 是"要落库、可能驱动一轮"的数据面
    载荷，而 cancel 是纯控制面信号，只中断在途的 drive task 并级联给自己的 RUNNING
    子代理。``session_id`` 即路由锚点（目标 actor 的会话 ID，也是 publish 的 key）。
    """

    session_id: str
    sender: str
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        sender = self.sender.strip()
        session_id = self.session_id.strip()
        if not sender:
            raise DomainValidationError("Sender is required and cannot be empty.")
        if not session_id:
            raise DomainValidationError("Session id is required and cannot be empty.")
        object.__setattr__(self, "sender", sender)
        object.__setattr__(self, "session_id", session_id)



@dataclass(frozen=True, slots=True)
class Answered:
    """本轮工作流跑完，产出最终 AssistantMessage。"""

    message: Message


@dataclass(frozen=True, slots=True)
class Queued:
    """消息已入库，本轮不产出回复。

    会话已有 turn 在跑，因此不驱动新的工作流；运行中的 turn 会重读完整历史，
    这条消息是否被它当作待办指令并不保证。调用方不应重发同一问题。
    """

    session_id: str


AskOutcome = Answered | Queued


__all__ = [
    "MAIN_AGENT_NAME",
    "RESERVED_AGENT_NAMES",
    "AskOutcome",
    "Answered",
    "BusMessage",
    "CancelMessage",
    "MainAgentMessage",
    "Queued",
    "SubAgentMessage",
    "AgentStatus",
]
