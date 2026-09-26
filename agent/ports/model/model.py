"""Agent 与模型通信的端口协议。"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from agent.domain.model_messages import ModelMessage, ModelTurn
from agent.domain.tools import ToolDefinition


class ModelCallPurpose(StrEnum):
    """本次模型调用的意图：正文生成或历史总结。"""

    CHAT = "chat"
    SUMMARY = "summary"


class AgentModelPort(Protocol):
    """Agent 调用模型的本地端口；``purpose`` 供 langgraph 流式筛选。"""

    async def ainvoke(
        self,
        *,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition] = (),
        tool_choice: str | None = None,
        max_output_tokens: int | None = None,
        purpose: ModelCallPurpose = ModelCallPurpose.CHAT,
    ) -> ModelTurn: ...


__all__ = ["AgentModelPort", "ModelCallPurpose"]
