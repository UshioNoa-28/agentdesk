from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"


class PermissionAuthorization(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"
    UNAVAILABLE = "unavailable"


class PermissionMode(StrEnum):
    DEFAULT = "default" # 按settings.json中的来 未匹配则 默认ask
    DONT_ASK = "dont_ask"# 允许所有除了deny的操作
    BYPASS="bypass"# 允许全部操作

class PermissionAction(StrEnum):
    ALLOW="allow"
    DENY="deny"
    ASK="ask"

@dataclass(slots=True,frozen=True)
class ToolRule:
    tool: str
    scope: str


__all__ = [
    "PermissionAction",
    "PermissionAuthorization",
    "PermissionDecision",
    "PermissionMode",
    "ToolRule",
]
