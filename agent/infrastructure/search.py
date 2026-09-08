"""Skill 与 MCP 搜索共用的 BM25 前置分词。

BM25 库（rank-bm25）只吃预分词的 token 序列；这里统一切分规则：
ASCII 词/下划线为词元，CJK 逐字成元（unigram，无分词器时的标准降级）。
"""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """切分并小写归一一段文本。"""

    return _TOKEN_PATTERN.findall(text.casefold())


__all__ = ["tokenize"]
