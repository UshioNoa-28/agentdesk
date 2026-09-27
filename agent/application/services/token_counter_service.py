"""TokenCounter 的应用层测量服务。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from agent.domain.model_messages import ModelMessage
from agent.domain.tools import ToolDefinition
from agent.ports.model import TokenCounter
from agent.ports.services import TokenCounterServicePort

DEFAULT_TRUNCATION_MARKER = "\n...[truncated; original result is retained]...\n"


class TokenCounterService(TokenCounterServicePort):
    """把底层计数器适配成 Agent 消息预算与请求测量服务。

    ``TokenCounter`` 只知道如何数一段文本；这里集中两件事：

    - 原语适配：provider 无关消息、工具 schema 和截断标记，避免各处
      重复实现；
    - 请求测量：一次请求的体积公式唯一归属这里。测量只认调用方构造
      好的完整请求（含 prompt、指令等全部开销），不持有任何请求构成
      的知识——构造属于调用方的策略职责。
    """

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def estimate_text(self, content: str) -> int:
        """估算一段文本的 token 数。"""

        return self._counter.count_text(content)

    def estimate_message(self, message: ModelMessage) -> int:
        """估算一条模型消息，包括工具调用参数。"""

        total = self.estimate_text(message.content) + 4
        for call in message.tool_calls:
            total += self.estimate_text(
                json.dumps(
                    {"id": call.id, "name": call.name, "arguments": call.arguments},
                    ensure_ascii=False,
                    default=str,
                )
            )
        return total

    def estimate_messages(self, messages: Iterable[ModelMessage]) -> int:
        """估算一组模型消息的 token 数。"""

        return sum(self.estimate_message(message) for message in messages)

    def estimate_tool_definitions(
        self,
        definitions: Sequence[ToolDefinition],
    ) -> int:
        """估算工具 schema 占用的 token 数。"""

        return sum(
            self.estimate_text(
                json.dumps(
                    {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": dict(definition.parameters),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
            + 4
            for definition in definitions
        )

    def truncate_text(self, content: str, max_tokens: int) -> str:
        """按 token 上限截断文本，并让截断标记计入同一预算。"""

        return self._counter.truncate_text(
            content,
            max_tokens,
            marker=DEFAULT_TRUNCATION_MARKER,
        )

    def measure_model_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> int:
        """测量一次模型请求的完整输入体积（消息 + 工具 schema）。"""

        return self.estimate_messages(messages) + self.estimate_tool_definitions(tools)

    def measure_request(self, messages: Sequence[ModelMessage]) -> int:
        """测量一次已构造请求的完整输入体积（如摘要请求）。"""

        return self.estimate_messages(messages)


__all__ = ["DEFAULT_TRUNCATION_MARKER", "TokenCounterService"]
