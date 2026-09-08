"""Agent 上下文预算使用的 token 计数端口协议。"""

from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    """Agent 在发送上下文前需要的 token 计数和截断能力。"""

    def count_text(self, content: str) -> int:
        """返回文本的 token 计数。"""

        ...

    def truncate_text(self, content: str, max_tokens: int, *, marker: str) -> str:
        """按 token 上限保留文本头尾，并插入截断标记。"""

        ...


__all__ = ["TokenCounter"]
