"""Agent 使用的供应商无关模型消息值对象。

持久化消息 ``Message`` 属于 Agent 的 Session 存储边界，位于
``agent.domain.messages``。这里仅保留投给模型的上下文消息、工具调用和
provider usage 类型。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型要求运行一次工具的结构化指令。"""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """投给模型的一条供应商无关消息。"""
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @classmethod
    def system(cls, content: str) -> ModelMessage:
        """创建 Agent 行为策略消息。

        Args:
            content (str): system policy 文本。

        Returns:
            ModelMessage: role 为 ``system`` 的消息。
        """

        return cls(role="system", content=content)

    @classmethod
    def human(cls, content: str, *, name: str | None = None) -> ModelMessage:
        """创建用户或对端 Agent 消息。

        Args:
            content (str): 用户问题或对端 Agent 消息正文。
            name (str | None): 可选的发言人/Agent 标识名。

        Returns:
            ModelMessage: role 为 ``human`` 的消息。
        """

        return cls(role="human", content=content, name=name)

    @classmethod
    def assistant(
        cls,
        *,
        content: str,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> ModelMessage:
        """创建助手消息，可选地携带工具调用。

        Args:
            content (str): 助手文本内容。
            tool_calls (tuple[ToolCall, ...]): 本次 assistant turn 发起的工具调用。

        Returns:
            ModelMessage: role 为 ``assistant`` 的消息。
        """

        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, *, name: str, tool_call_id: str, content: str) -> ModelMessage:
        """创建工具返回消息，并保留对应的调用 ID。

        Args:
            name (str): 返回结果对应的工具名。
            tool_call_id (str): assistant 发起调用时的稳定 ID。
            content (str): 工具结果文本。

        Returns:
            ModelMessage: role 为 ``tool``、可与调用配对的消息。
        """

        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ValueError("tool messages require a non-empty tool_call_id")
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            tool_name=name,
        )


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """一次模型请求由 provider 返回的 token 计数。"""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        """拒绝 provider 或 wire 层传来的负数计数。"""

        if self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")
        if self.output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")
        if self.total_tokens < 0:
            raise ValueError("total_tokens cannot be negative")


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """一次模型 turn 的统一结果。"""

    message: ModelMessage
    tool_calls: tuple[ToolCall, ...]
    usage: ModelUsage | None = None

    @property
    def is_tool_call(self) -> bool:
        """当前 turn 是否要求 Agent 继续执行工具。"""

        return bool(self.tool_calls)


__all__ = ["ModelMessage", "ModelTurn", "ModelUsage", "ToolCall"]
