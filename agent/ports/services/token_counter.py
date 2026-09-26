"""Application-facing token counting service port protocol."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from agent.domain.model_messages import ModelMessage
from agent.domain.tools import ToolDefinition


class TokenCounterServicePort(Protocol):
    """Agent 上下文预算所需的高层 token 能力。

    原语方法（estimate_* / truncate_text）服务单条消息处理策略；语义化
    测量方法（measure_*）定义"一次请求的完整体积"，公式唯一归属本服务，
    调用方不得自行拼装加法。测量只认调用方构造好的完整请求，不持有
    任何请求构成（prompt、指令等）的知识。
    """

    def estimate_text(self, content: str) -> int:
        """估算一段文本的 token 数。"""

        ...

    def estimate_message(self, message: ModelMessage) -> int:
        """估算一条模型消息及其工具调用参数的 token 数。"""

        ...

    def estimate_messages(self, messages: Iterable[ModelMessage]) -> int:
        """估算一组模型消息的 token 数。"""

        ...

    def estimate_tool_definitions(
        self,
        definitions: Sequence[ToolDefinition],
    ) -> int:
        """估算一组工具 schema 的 token 数。"""

        ...

    def truncate_text(self, content: str, max_tokens: int) -> str:
        """按 token 上限截断文本，并保留截断标记。"""

        ...

    def measure_model_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> int:
        """测量一次模型请求的完整输入体积（消息 + 工具 schema）。"""

        ...

    def measure_request(self, messages: Sequence[ModelMessage]) -> int:
        """测量一次已构造请求的完整输入体积（如摘要请求）。"""

        ...


__all__ = ["TokenCounterServicePort"]
