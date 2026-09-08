"""Multi-Agent 领域模型与通信载荷。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

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
class AgentMessage:
    """Actor 间通信与任务流转的统一消息信封。

    所有语义字段一律显式传递，不提供策略性默认值。
    """

    content: str
    sender: str
    recipient: str
    session_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


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
    "AgentMessage",
    "AgentStatus",
    "AskOutcome",
    "Answered",
    "Queued",
]
