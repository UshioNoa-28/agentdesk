"""Agent 工作流执行端口协议。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from agent.domain.model_messages import ModelMessage
from agent.domain.tools import AgentTool
from agent.ports.model.model import AgentModelPort
from agent.ports.runtime.stream import AgentStreamSink


class AgentWorkflow(Protocol):
    """Agent 工作流端口，不暴露 LangGraph 内部细节。"""

    async def ainvoke(
        self,
        *,
        messages: list[ModelMessage],
        model: AgentModelPort,
        tools: Sequence[AgentTool],
        session_id: str | None = None,
        runtime: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """非流式运行工作流；流式请使用 ``astream``。"""

        ...

    async def astream(
        self,
        *,
        messages: list[ModelMessage],
        model: AgentModelPort,
        tools: Sequence[AgentTool],
        session_id: str | None = None,
        runtime: Mapping[str, object] | None = None,
        events: AgentStreamSink,
    ) -> dict[str, object]:
        """流式运行工作流；文本增量发布到 ``events``，返回 ainvoke 同形结果。

        ``events`` 必填；非流式调用应使用 ``ainvoke``。
        """

        ...


__all__ = ["AgentWorkflow"]
