"""Shared implementation details for workspace Meta Tools."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

DEFAULT_MAX_FILE_BYTES = 256 * 1024
DEFAULT_MAX_SEARCH_RESULTS = 200


class UserPathError(ValueError):
    """A user-supplied path is not a usable absolute path."""


def absolute_user_path(raw: object) -> Path:
    """Coerce a tool argument into a normalized absolute path."""

    if not isinstance(raw, str) or not raw.strip():
        raise UserPathError("path must be a non-empty absolute path string")

    value = raw.strip()
    if "\x00" in value:
        raise UserPathError("path must not contain NUL bytes")

    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        raise UserPathError("absolute paths are required")
    # normpath 只折叠 "."、".." 与冗余分隔符，不跟随 symlink：
    # 权限 scope 匹配的始终是用户书写的逻辑路径，而不是磁盘物理布局。
    return Path(os.path.normpath(str(expanded)))


def positive_limit(value: object, fallback: int) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return max(number, 1)


def clamp_limit(arguments: Mapping[str, object], key: str, cap: int) -> int:
    """缺省取实例上限，给定值只向下夹；类型与下界由 schema 保证。"""

    return min(int(arguments.get(key, cap)), cap)  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_SEARCH_RESULTS",
    "UserPathError",
    "absolute_user_path",
    "clamp_limit",
    "positive_limit",
]
