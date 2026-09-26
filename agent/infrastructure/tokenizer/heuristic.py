"""无依赖、无网络的 token 估算。

估算器只服务于一件事：上下文预算水位。它按字符类别计数——非 ASCII（CJK、
全角、emoji）1 字符记 1 token，ASCII 字母数字与空白 4 字符记 1 token，
ASCII 标点记 2 字符 1 token——相对真实 BPE 的结果系统性偏大，因此宁可提前
触发压缩，也不让请求在 provider 侧被拒。
"""

from __future__ import annotations

import math

ASCII_CHARS_PER_TOKEN = 4
PUNCTUATION_CHARS_PER_TOKEN = 2
TRUNCATION_HEAD_RATIO = 0.75
MIN_RETAINED_TOKENS = 1


class HeuristicTokenCounter:
    """``TokenCounter`` 端口的默认实现：纯函数计数与 token 预算截断。"""

    def count_text(self, content: str) -> int:
        """返回文本的估算 token 数；空文本返回 0。"""

        wide = plain = dense = 0
        for char in content:
            if not char.isascii():
                wide += 1
            elif char.isalnum() or char.isspace():
                plain += 1
            else:
                dense += 1
        return (
            wide
            + math.ceil(plain / ASCII_CHARS_PER_TOKEN)
            + math.ceil(dense / PUNCTUATION_CHARS_PER_TOKEN)
        )

    def truncate_text(self, content: str, max_tokens: int, *, marker: str) -> str:
        """按 token 预算保留文本头尾 (75% 头 + 25% 尾)，中间插入 marker。"""

        if max_tokens <= 0:
            return ""
        if self.count_text(content) <= max_tokens:
            return content

        marker_tokens = self.count_text(marker)
        if marker_tokens >= max_tokens:
            return self._fit(marker, max_tokens, from_end=False)

        available = max_tokens - marker_tokens
        head_tokens = max(MIN_RETAINED_TOKENS, int(available * TRUNCATION_HEAD_RATIO))
        head = self._fit(content, head_tokens, from_end=False)
        # tail 用剩余预算而不是名义份额：各段独立 ceil 会让拼接后的总数超出 max_tokens。
        tail_budget = max_tokens - self.count_text(head) - marker_tokens
        tail = self._fit(content, tail_budget, from_end=True)
        return f"{head}{marker}{tail}"

    def _fit(self, content: str, max_tokens: int, *, from_end: bool) -> str:
        """二分出不超过预算的最长前缀（``from_end`` 时为后缀）。"""

        low, high = 0, len(content)
        while low < high:
            mid = (low + high + 1) // 2
            piece = content[len(content) - mid :] if from_end else content[:mid]
            if self.count_text(piece) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        return content[len(content) - low :] if from_end else content[:low]


__all__ = ["HeuristicTokenCounter"]
