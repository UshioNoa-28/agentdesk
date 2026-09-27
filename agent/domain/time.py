"""Agent 领域实体使用的 UTC 时间工具。"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """生成带时区的 UTC 时间。

    Returns:
        datetime: ``tzinfo`` 为 UTC 的当前时间。
    """

    return datetime.now(timezone.utc)


__all__ = ["utc_now"]
