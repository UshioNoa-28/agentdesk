"""``.agent-desk/settings.json`` 里 ``memory`` 段的加载。

发现链与 ``permissions`` 同规：从主目录向 workspace 根逐层扫描，标量无法
合并，最近写了 ``enabled`` 的层级整体覆盖外层值，但每一层都会被校验——拼写
错误即使被覆盖也照样启动失败。整段缺失是正常状态，等价于该层没有贡献，
最终吃 ``False``：长期记忆默认关闭。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.infrastructure.project_settings import (
    config_search_roots,
    project_settings_path,
    read_settings_object,
)

# 记忆关闭时工具回给模型的话：明确「没写进去」与「别再试」，否则模型会
# 把失败当成一次普通报错反复重试。
MEMORY_DISABLED_MESSAGE = (
    "Long-term memory is disabled by settings, so nothing was recorded. "
    "Do not retry; answer without memory."
)


def load_memory_enabled(start: Path, *, home: Path | None = None) -> bool:
    """逐层校验 ``memory`` 段，返回最近写了 ``enabled`` 的层级的值。"""

    enabled: bool | None = None
    roots = reversed(config_search_roots(start.expanduser().resolve(strict=False), home=home))
    for root in roots:
        path = project_settings_path(root)
        if not path.exists():
            continue
        section = _memory_section(read_settings_object(path), path)
        if "enabled" not in section:
            continue
        value = section["enabled"]
        if not isinstance(value, bool):
            raise ValueError(f"settings 'memory.enabled' must be a boolean: {path}")
        enabled = value
    return bool(enabled)


def _memory_section(data: dict[str, Any], path: Path) -> dict[str, Any]:
    """取出顶层 ``memory`` 对象；类型非法即抛 ``ValueError``。"""

    section = data.get("memory", {})
    if not isinstance(section, dict):
        raise ValueError(f"settings 'memory' must be an object: {path}")
    return section


__all__ = ["load_memory_enabled"]
