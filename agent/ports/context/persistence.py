"""Agent 消息持久化编排端口协议。"""

from __future__ import annotations

from typing import Protocol

from agent.domain.messages import Message
from agent.domain.model_messages import ModelMessage, ModelTurn
from agent.domain.tools import ToolResult


class PersistenceManagerPort(Protocol):
    """持久化 Agent 轮次消息并生成模型可见工具投影的端口。"""

    async def persist_assistant(
        self,
        *,
        session_id: str,
        turn: ModelTurn,
    ) -> Message: ...

    async def persist_tool(
        self,
        *,
        session_id: str,
        result: ToolResult,
        message: ModelMessage,
    ) -> ModelMessage: ...


__all__ = ["PersistenceManagerPort"]
