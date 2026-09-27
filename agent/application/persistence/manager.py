"""Persistence orchestration for one Agent turn."""

from __future__ import annotations

from collections.abc import Mapping

from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage, ModelTurn
from agent.domain.tools import ToolResult
from agent.exceptions import AgentInternalError
from agent.ports.context import ContextProjectorPort, PersistenceManagerPort
from agent.ports.services import MessageServicePort


class PersistenceManager(PersistenceManagerPort):
    """Persist Agent messages and prepare the model-visible tool projection.

    The manager owns the ordering and metadata rules for Assistant/Tool
    messages. The graph only supplies the explicit session, turn and result
    values; it does not know how Message rows are created or projected.
    """

    def __init__(
        self,
        *,
        message_service: MessageServicePort,
        projector: ContextProjectorPort,
    ) -> None:
        self._messages = message_service
        self._projector = projector

    async def persist_assistant(
        self,
        *,
        session_id: str,
        turn: ModelTurn,
    ) -> Message:
        """Persist one Assistant turn with its stable metadata."""

        if not isinstance(turn, ModelTurn):
            raise AgentInternalError("Agent model turn was not available to persistence.")
        return await self._messages.add(
            session_id=session_id,
            message=turn.message,
            metadata=_assistant_metadata(turn),
        )

    async def persist_tool(
        self,
        *,
        session_id: str,
        result: ToolResult,
        message: ModelMessage,
    ) -> ModelMessage:
        """Persist the raw Tool Message and return its model projection."""

        stored = await self._messages.add(
            session_id=session_id,
            message=message,
            metadata={"kind": result.kind},
        )
        return self._projector.project_tool_result(stored)


def _assistant_metadata(result: ModelTurn) -> Mapping[str, object]:
    metadata: dict[str, object] = {
        "kind": (
            MessageKind.ASSISTANT_TOOL_CALL
            if result.tool_calls
            else MessageKind.ASSISTANT_ANSWER
        )
    }
    if result.usage is not None:
        metadata["usage"] = {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        }
    return metadata


__all__ = ["PersistenceManager"]
