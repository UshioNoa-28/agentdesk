"""Agent 与模型通信的端口协议。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent.domain.model_messages import ModelMessage, ModelTurn
from agent.domain.tools import ToolDefinition


class AgentModelPort(Protocol):
    """Agent 调用模型的本地端口。

    流式输出不在此端口上：LangGraph 的 messages 模式会在底层模型调用时
    通过 callbacks 捕获增量，``ainvoke`` 保持一次性语义即可。
    """

    async def ainvoke(
        self,
        *,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition] = (),
        tool_choice: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelTurn: ...


__all__ = ["AgentModelPort"]
