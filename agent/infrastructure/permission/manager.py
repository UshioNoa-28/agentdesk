from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

from agent.domain.permissions import (
    PermissionAction,
    PermissionMode,
    ToolRule,
)
from agent.domain.tools import ALL_META_TOOL_NAMES
from agent.infrastructure.permission.settings import (
    append_permission_rule,
    load_permission_mode,
    load_permission_rule,
    set_permission_mode,
)
from agent.ports.services.permission import PermissionManagerPort

logger = logging.getLogger(__name__)

_META_TOOL_NAMES = frozenset(ALL_META_TOOL_NAMES)

# scope 会被序列化成 "tool(scope)" 字符串写进 settings；单行、可打印、限长。
_MAX_SCOPE_LENGTH = 256


class PermissionManager(PermissionManagerPort):
    def __init__(
        self,
        *,
        start: Path,
        permission_rule: dict[str, list[ToolRule]] | None = None,
        mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> None:
        self._start = start
        self._permission_rule: dict[str, list[ToolRule]] = (
            {} if permission_rule is None else permission_rule
        )
        self._mode = mode

    @property
    def mode(self) -> PermissionMode:
        return self._mode

    @property
    def allowed(self) -> list[ToolRule]:
        return self._permission_rule.get(PermissionAction.ALLOW, [])

    @property
    def denied(self) -> list[ToolRule]:
        return self._permission_rule.get(PermissionAction.DENY, [])

    @property
    def asked(self) -> list[ToolRule]:
        return self._permission_rule.get(PermissionAction.ASK, [])

    def in_allow(self, tool_name: str, target: str = "") -> bool:
        return self._matches(PermissionAction.ALLOW, tool_name, target)

    def in_deny(self, tool_name: str, target: str = "") -> bool:
        return self._matches(PermissionAction.DENY, tool_name, target)

    def in_ask(self, tool_name: str, target: str = "") -> bool:
        return self._matches(PermissionAction.ASK, tool_name, target)

    def _matches(
        self,
        action: PermissionAction,
        tool_name: str,
        target: str,
    ) -> bool:
        return any(
            rule.tool == tool_name and _scope_matches(rule.scope, target)
            for rule in self._permission_rule.get(action, [])
        )

    def persist_rule(
        self,
        tool_name: str,
        permission_action: PermissionAction,
        scope: str = "",
    ) -> None:
        """落一条 (tool, scope) 规则：先校验形状，内存生效，磁盘尽力而为。

        - 未知 tool 名直接拒绝：读取侧对整个 settings 校验未知工具名，
          脏写会让下次启动直接失败，必须在写入端拦下；
        - 规则按 tool 键控，跨工具的 scope 只会匹配不上任何东西（惰性），
          不构成越权，因此不做语义校验。
        """

        if tool_name not in _META_TOOL_NAMES:
            raise ValueError(f"Unknown Meta Tool in permission rule: {tool_name!r}")
        if len(scope) > _MAX_SCOPE_LENGTH or any(
            ord(char) < 0x20 or char == "\x7f" for char in scope
        ):
            raise ValueError(f"Invalid permission scope: {scope!r}")
        rule = ToolRule(tool=tool_name, scope=scope)
        self._permission_rule.setdefault(permission_action, []).append(rule)
        try:
            append_permission_rule(
                self._start, permission_action, rule.tool, rule.scope
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "Permission grant for %r is active this process but was NOT "
                "persisted to %s: %s",
                rule.tool,
                self._start,
                exc,
            )

    def change_mode(self, mode: PermissionMode) -> None:
        """切换决策 mode：内存立即生效，并持久化到最近层级 settings。"""

        self._mode = mode
        try:
            set_permission_mode(self._start, str(mode))
        except (OSError, ValueError) as exc:
            logger.warning(
                "Permission mode %s is active this process but was NOT "
                "persisted to %s: %s",
                mode,
                self._start,
                exc,
            )


def _scope_matches(scope: str, target: str) -> bool:
    """scope 匹配：统一 fnmatch glob，一切从简。

    - 空 scope 匹配一切(整工具规则)；
    - target 为空时，带 scope 的规则一律不匹配(宁缺勿滥，fail-closed)；
    - 工具名本身就是语义域(read_file 的规则只会拿路径来匹配，bash 的只会
      拿命令行来匹配)，不需要额外的 style；
    - 路径统一归一 ``./`` 前缀；``dir/*`` 靠 ``*`` 跨目录命中深层文件；
    - 尾部 `` *`` 是可选尾巴：``git commit *`` 同时命中裸命令与带参数命令。
    """

    if not scope:
        return True
    if not target:
        return False
    name = target.strip().removeprefix("./")
    pattern = scope.strip().removeprefix("./")
    if pattern.endswith(" *"): # 针对command的做法 git commit *
        return fnmatch.fnmatchcase(name, pattern) or name == pattern[:-2]
    return fnmatch.fnmatchcase(name, pattern)


def load_permission_manager(
    start: Path,
    *,
    home: Path | None = None,
) -> PermissionManager:
    permission_rule = load_permission_rule(start, home=home)
    return PermissionManager(
        start=start,
        permission_rule=permission_rule,
        mode=PermissionMode(load_permission_mode(start, home=home)),
    )


__all__ = ["PermissionManager", "load_permission_manager"]
