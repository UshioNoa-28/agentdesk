"""``.agent-desk/settings.json`` 里 ``permissions`` 段的加载与写入。

发现链复用 ``agent.infrastructure.project_settings`` 的通用原语
（``config_search_roots`` 等），本模块只负责 permissions 的语义与校验：

- ``permissions.allow`` / ``ask`` / ``deny``：跨层取并集并按 (tool, scope)
  去重。列表只做加法：内层无法把外层的条目从某个列表里删掉，要取消外层
  的允许，写进更严的列表即可，生效优先级是 deny > ask > allow（由决策方
  消费，见 ``PermissionManager.in_*``）。
- ``permissions.mode``：标量无法合并，最近的写了它的层级整体覆盖外层值；
  但每一层都会被校验——拼写错误即使被覆盖也照样启动失败。

未列出的 Meta Tool 一律需要审批（默认 ask）。文件缺失是正常状态，等价于
该层没有贡献；结构非法、坏 JSON 或未知工具名则启动失败——拼写错误必须
立刻可见。

写入只发生在最近的层级（workspace 根）：``append_permission_rule``
read-merge-write 并原子替换，未知键原样保留，为其他配置段预留共存空间。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agent.domain.permissions import PermissionAction, PermissionMode, ToolRule
from agent.domain.tools import ALL_META_TOOL_NAMES
from agent.infrastructure.project_settings import (
    atomic_write,
    config_search_roots,
    project_settings_path,
    read_settings_object,
)

logger = logging.getLogger(__name__)

_META_TOOL_NAMES = frozenset(ALL_META_TOOL_NAMES)


def load_permission_rule(
    start: Path,
    *,
    home: Path | None = None,
) -> dict[str, list[ToolRule]]:
    """跨层并集合并 ``permissions`` 的 allow/ask/deny，按 (tool, scope) 去重。"""

    merged: dict[str, list[ToolRule]] = {}
    roots = reversed(
        config_search_roots(start.expanduser().resolve(strict=False), home=home)
    )
    # 从 home 到 project
    for root in roots:
        path = project_settings_path(root)
        if not path.exists():
            continue
        layer = _load_permission_layer(read_settings_object(path), path)
        for action, rules in layer.items():
            bucket = merged.setdefault(action, [])
            for rule in rules:
                if rule not in bucket:
                    bucket.append(rule)
    return merged


def load_permission_mode(
    start: Path,
    *,
    home: Path | None = None,
) -> str:
    """从主目录向 workspace 根扫描，最近写了 ``mode`` 的层级获胜；每层都校验。"""

    permission_mode: str | None = None
    roots = reversed(
        config_search_roots(start.expanduser().resolve(strict=False), home=home)
    )
    for root in roots:
        path = project_settings_path(root)
        if not path.exists():
            continue
        data = read_settings_object(path)
        permissions = data.get("permissions", {})
        if not isinstance(permissions, dict):
            raise ValueError(f"settings 'permissions' must be an object: {path}")
        mode = permissions.get("mode")
        if mode is None:
            continue
        if not isinstance(mode, str) or not mode:
            raise ValueError(
                f"settings 'permissions.mode' must be a non-empty string: {path}"
            )
        permission_mode = mode
    return permission_mode or "default"


def append_permission_rule(
    start: Path,
    permission_action: str,
    tool_name: str,
    scope: str,
) -> None:
    """把一条工具规则追加进最近层级的 settings.json，原子写入。"""

    action = PermissionAction(permission_action)
    path = project_settings_path(start.expanduser().resolve(strict=False))
    if path.exists():
        data = read_settings_object(path)
    else:
        data = {}
    permissions = data.get("permissions")
    if permissions is None:
        permissions = {}
    elif not isinstance(permissions, dict):
        raise ValueError(f"settings 'permissions' must be an object: {path}")
    entries = permissions.get(action, [])
    if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
        raise ValueError(f"settings 'permissions.{action}' must be a string list: {path}")
    # 应该不需要考虑去重的问题；读入时跨层合并会去重
    rule = f"{tool_name}({scope})" if scope else tool_name
    permissions[action] = [*entries, rule]
    data["permissions"] = permissions

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    logger.info("Persisted permission rule %r under %r: %s", rule, action, path)


def set_permission_mode(start: Path, mode: str) -> None:
    """把 ``permissions.mode`` 写入项目级 settings 文件，不存在则创建。"""

    try:
        permission_mode = str(PermissionMode(mode))
    except ValueError as exc:
        raise ValueError(f"Unknown permission mode: {mode!r}") from exc
    path = project_settings_path(start.expanduser().resolve(strict=False))
    if path.exists():
        data = read_settings_object(path)
    else:
        data = {}
    permissions = data.get("permissions")
    if permissions is None:
        permissions = {}
    elif not isinstance(permissions, dict):
        raise ValueError(f"settings 'permissions' must be an object: {path}")
    permissions["mode"] = permission_mode
    data["permissions"] = permissions

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    logger.info("Persisted permission mode %r: %s", permission_mode, path)


def _load_permission_layer(
    data: dict[str, Any], path: Path
) -> dict[str, list[ToolRule]]:
    """从单个层级的顶层对象里校验并解析出 dict[action, list[ToolRule]]。"""

    permissions = data.get("permissions", {})
    if not isinstance(permissions, dict):
        raise ValueError(f"settings 'permissions' must be an object: {path}")
    res: dict[str, list[ToolRule]] = {}
    for action in PermissionAction:
        items = permissions.get(action, [])
        if not isinstance(items, list):
            raise ValueError(
                f"settings 'permissions.{action}' must be a string list: {path}"
            )
        rules: list[ToolRule] = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"settings 'permissions.{action}' requires non-empty strings: {path}"
                )
            tool_name, scope = _parse_permission_rule(item)
            if tool_name not in _META_TOOL_NAMES:
                raise ValueError(f"Unknown Meta Tool in {path}: {tool_name!r}")
            rules.append(ToolRule(tool=tool_name, scope=scope))
        res[action.value] = rules
    return res


def _parse_permission_rule(rule: str) -> tuple[str, str]:
    rule = rule.strip()
    if not rule:
        raise ValueError("rule can't be empty")
    if "(" not in rule:
        return rule, ""
    if not rule.endswith(")"):
        raise ValueError(f"Invalid permission rule: {rule!r}")
    tool_name, scope = rule[:-1].split("(", 1)
    if not tool_name or not scope:
        raise ValueError(f"Invalid permission rule: {rule!r}")
    return tool_name, scope


__all__ = [
    "append_permission_rule",
    "load_permission_mode",
    "load_permission_rule",
    "set_permission_mode",
]
