from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from agent.application.persistence.manager import PersistenceManager
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage, ModelTurn, ModelUsage, ToolCall
from agent.domain.tools import ToolResult


class _Messages:
    def __init__(self) -> None:
        self.items: list[Message] = []

    async def add(self, *, session_id, message, metadata=None) -> Message:
        stored = Message.create(
            session_id=session_id,
            seq=len(self.items) + 1,
            message=message,
            metadata=metadata,
        )
        self.items.append(stored)
        return stored


class _Projector:
    def project_tool_result(self, message: Message) -> ModelMessage:
        return ModelMessage.tool(
            name=message.tool_name or "tool",
            tool_call_id=message.tool_call_id or "unknown",
            content=f"projected:{message.content}",
        )


class PersistenceManagerTests(IsolatedAsyncioTestCase):
    async def test_persist_assistant_keeps_kind_and_usage_metadata(self) -> None:
        messages = _Messages()
        manager = PersistenceManager(message_service=messages, projector=_Projector())
        call = ToolCall(id="call-1", name="execute_mcp", arguments={})
        turn = ModelTurn(
            message=ModelMessage.assistant(content="", tool_calls=(call,)),
            tool_calls=(call,),
            usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
        )

        stored = await manager.persist_assistant(session_id="session-1", turn=turn)

        self.assertEqual(MessageKind.ASSISTANT_TOOL_CALL.value, stored.metadata["kind"])
        self.assertEqual(
            {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            stored.metadata["usage"],
        )

    async def test_persist_tool_stores_raw_message_and_returns_projection(self) -> None:
        messages = _Messages()
        manager = PersistenceManager(message_service=messages, projector=_Projector())
        message = ModelMessage.tool(
            name="execute_mcp",
            tool_call_id="call-1",
            content="untrusted result",
        )
        result = ToolResult(content="untrusted result")

        projected = await manager.persist_tool(
            session_id="session-1",
            result=result,
            message=message,
        )

        self.assertEqual("untrusted result", messages.items[0].content)
        self.assertEqual(MessageKind.TOOL_RESULT.value, messages.items[0].metadata["kind"])
        self.assertEqual("projected:untrusted result", projected.content)
