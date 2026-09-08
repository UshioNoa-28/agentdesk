"""Agent 流式输出的事件值对象与 wire 词汇表。

流式事件是运行期的临时产物：由工具循环产生、模型增量经 LangGraph 的
messages 模式捕获，经进程内 StreamHub 直达 SSE 客户端，不落库、不参与
持久化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, Union

from agent.domain.model_messages import ToolCall


@dataclass(frozen=True, slots=True)
class TextDelta:
    """模型增量输出的一段可见文本。"""

    content: str


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """一次工具调用执行完成；``call`` 保留发起时的 ID、名称和参数。"""

    call: ToolCall
    content: str


# Agent 流（跨 actor 边界到达 SSE）的事件集合；assistant 发起工具调用时
# 直接复用领域 ToolCall 值对象。
AgentStreamEvent = Union[TextDelta, ToolCall, ToolResultEvent]

# SSE 事件名：队列里流动的帧就是 (事件名, JSON 载荷)，名称与 CLI 渲染
# 分发一一对应，改任何一边都要同步另一边。
STREAM_EVENT_TEXT_DELTA = "text_delta"
STREAM_EVENT_TOOL_CALL = "tool_call"
STREAM_EVENT_TOOL_RESULT = "tool_result"
STREAM_EVENT_MESSAGE_END = "message_end"
STREAM_EVENT_ERROR = "error"
# 消息已入库、本轮不产出回复（会话已有工作流在跑）。与 error 区分：这是被接受的结果。
STREAM_EVENT_QUEUED = "queued"

StreamFrame: TypeAlias = tuple[str, dict[str, object]]

def stream_frame(event: AgentStreamEvent) -> StreamFrame:
    """把领域流式事件转换成 (SSE 事件名, JSON 载荷) 帧。"""

    if isinstance(event, TextDelta):
        return STREAM_EVENT_TEXT_DELTA, {"content": event.content}
    if isinstance(event, ToolResultEvent):
        return STREAM_EVENT_TOOL_RESULT, {
            "tool_call_id": event.call.id,
            "name": event.call.name,
            "arguments": event.call.arguments,
            "content": event.content,
        }
    return STREAM_EVENT_TOOL_CALL, {
        "id": event.id,
        "name": event.name,
        "arguments": event.arguments,
    }


__all__ = [
    "STREAM_EVENT_ERROR",
    "STREAM_EVENT_MESSAGE_END",
    "STREAM_EVENT_QUEUED",
    "STREAM_EVENT_TEXT_DELTA",
    "STREAM_EVENT_TOOL_CALL",
    "STREAM_EVENT_TOOL_RESULT",
    "AgentStreamEvent",
    "StreamFrame",
    "TextDelta",
    "ToolResultEvent",
    "stream_frame",
]
