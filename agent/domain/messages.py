"""Agent Session 中持久化的消息实体。

``ModelMessage`` 描述一次模型请求所需的临时视图，位于
``agent.domain.model_messages``；这里的 ``Message`` 额外保存 Session、顺序
和 metadata，因此只属于 Agent 的持久化边界。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from agent.domain.model_messages import ModelMessage, ToolCall
from agent.domain.time import utc_now


class MessageKind(StrEnum):
    """Agent 消息在持久化和上下文投影中的领域分类"""

    # 系统提示词（每个会话 seq=1 的那条），投影时单独放到最前面。
    SYSTEM = "system"
    # 真正的用户问题。注意：子代理回报也是 human/USER，但靠 metadata["source"]
    # 区分（"user" vs "agent"），需要判断"当前用户问题"时要用 _latest_user。
    USER = "user"
    # assistant 返回了 tool_calls 的那一轮（需要紧跟对应 tool 结果）。
    ASSISTANT_TOOL_CALL = "assistant_tool_call"
    # assistant 的最终文本回答（没有 tool_calls）。
    ASSISTANT_ANSWER = "assistant_answer"
    # 普通工具执行结果（execute_mcp / execute_python / wait_for_replies 等）。
    # 唯一会被内容上限截断的 kind。
    TOOL_RESULT = "tool_result"
    # load_skill 返回的本地 Skill 正文，低优先级指导，不截断。
    SKILL_RESULT = "skill_result"
    # search_mcp 返回的远程工具 schema/定义，发现上下文，不截断。
    MCP_TOOL_DEFINITION = "mcp_tool_definition"
    # 上下文压缩生成的对话摘要，投影时包 <conversation_summary> 边界。
    SUMMARY = "summary"
    # 压缩检查点等维护性消息（cold compact），模型视图中直接跳过。
    MAINTENANCE = "maintenance"


def _enum_value(value: object) -> object:
    """把领域枚举转换成 JSONB 可稳定保存的字符串。"""

    return value.value if isinstance(value, StrEnum) else value


def message_kind_from_value(value: object) -> MessageKind:
    """将 JSONB 或应用边界的值解析为 MessageKind。"""

    if isinstance(value, MessageKind):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return MessageKind(value.strip())
        except ValueError as exc:
            raise ValueError(f"unsupported message kind: {value!r}") from exc
    raise ValueError(f"message metadata must include a MessageKind, got {value!r}")


@dataclass(frozen=True, slots=True)
class Message:
    """Session 内持久化的一条中立消息。

    ``ModelMessage`` 只描述一次模型请求需要的内容；本类型额外保存
    Session、消息 ID、顺序和运行时 metadata，因此不依赖 LangChain 或某个
    provider 的消息类。``metadata`` 主要承载 usage、合成中断结果等持久化
    信息；Provider Adapter 不会把它整体透传到外部协议。

    ``id`` 是数据库消息行的唯一标识，``seq`` 只负责 Session 内排序。
    ``tool_calls`` 保存 Assistant 一次返回的调用列表；``tool_call_id``
    保存 ToolMessage 对应的 provider 调用 ID，只用于恢复 Assistant/Tool
    协议配对，不作为 Agent 内部消息查询或 Compact 计划的标识。
    """

    id: str
    session_id: str
    seq: int
    role: str
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """校验消息身份和顺序，并复制 metadata 防止外部原地修改。"""

        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if self.seq <= 0:
            raise ValueError("message seq must be positive")
        if self.role not in {"system", "human", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role}")
        calls = tuple(self.tool_calls)
        if self.role == "tool":
            if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
                raise ValueError("tool messages require a non-empty tool_call_id")
        call_ids = [call.id for call in calls]
        if calls and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool calls")
        if any(not isinstance(call_id, str) or not call_id.strip() for call_id in call_ids):
            raise ValueError("assistant tool calls require non-empty IDs")
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("assistant tool call IDs must be unique within a message")
        metadata = {key: _enum_value(value) for key, value in self.metadata.items()}
        metadata["kind"] = message_kind_from_value(
            metadata.get("kind")
        ).value
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "tool_calls", calls)

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        seq: int,
        message: ModelMessage,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Message":
        """把一次模型上下文消息包装成可持久化记录。"""

        values = dict(metadata or {})
        values["kind"] = message_kind_from_value(
            values.get("kind")
        ).value
        return cls(
            id=str(uuid4()),
            session_id=session_id,
            seq=seq,
            role=message.role,
            content=message.content,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            tool_calls=message.tool_calls,
            metadata=values,
        )

    def to_model(self) -> ModelMessage:
        """转换为下一次模型请求使用的供应商无关消息。"""

        name = self.metadata.get("name")
        return ModelMessage(
            role=self.role,
            content=self.content,
            name=str(name) if name else None,
            tool_call_id=self.tool_call_id,
            tool_name=self.tool_name,
            tool_calls=self.tool_calls,
        )


__all__ = [
    "Message",
    "MessageKind",
    "message_kind_from_value",
]
