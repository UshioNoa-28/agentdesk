"""Message 资源的 HTTP 响应映射。"""

from __future__ import annotations

from agent.domain.messages import Message
from agent.dto.agent import MessageResponse, ToolCallResponse


def message_response(message: Message) -> MessageResponse:
    """把持久化 Message 映射成稳定的 HTTP wire DTO。

    Message 既会由 Agent 问答接口返回，也会由 Session 历史接口返回；
    映射逻辑放在 Controller 之外，避免两个资源 Controller 互相依赖。
    """

    return MessageResponse(
        id=message.id,
        session_id=message.session_id,
        seq=message.seq,
        role=message.role,
        content=message.content,
        tool_call_id=message.tool_call_id,
        tool_name=message.tool_name,
        tool_calls=[
            ToolCallResponse(
                id=call.id,
                name=call.name,
                arguments=call.arguments,
            )
            for call in message.tool_calls
        ],
        metadata=message.metadata,
        created_at=message.created_at,
    )


__all__ = ["message_response"]
