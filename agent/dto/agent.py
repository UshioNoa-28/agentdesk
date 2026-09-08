from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Agent 问答请求。"""

    question: str = Field(min_length=1)


class ToolCallResponse(BaseModel):
    """AssistantMessage 中的一次结构化工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]


class MessageResponse(BaseModel):
    """Session 中一条可持久化的供应商无关消息。"""

    id: str
    session_id: str
    seq: int
    role: str
    content: str
    tool_call_id: str | None
    tool_name: str | None
    tool_calls: list[ToolCallResponse]
    metadata: dict[str, Any]
    created_at: datetime


class TurnQueuedResponse(BaseModel):
    """消息已入库、本轮不产出回复（会话已有工作流在跑）。

    这是一次被接受的结果，不是失败：运行中的工作流会重读完整历史。是否把这条
    消息当作待办指令并不保证，因此调用方不应重发同一问题。
    """

    status: Literal["queued"] = "queued"
    session_id: str


__all__ = [
    "AskRequest",
    "MessageResponse",
    "ToolCallResponse",
    "TurnQueuedResponse",
]
