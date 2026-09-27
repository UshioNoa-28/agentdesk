"""供应商无关的 Skill 文档值对象。

Skill 的文件发现和解析属于 infrastructure；Agent 图只接收这里的稳定值对象，
因此以后把本地目录替换成对象存储或远程 Skill Registry 时，不需要修改编排层。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SkillState(StrEnum):
    """本地 Skill 配置项最近一次加载结果。"""

    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """一个 Skill 的轻量元数据。

    system prompt 只消费 name 和 description；path 供加载结果与状态快照使用。
    """

    name: str
    description: str
    path: str


@dataclass(frozen=True, slots=True)
class SkillDocument:
    """按名称加载到的完整 Markdown 指令。"""

    metadata: SkillMetadata
    instructions: str


@dataclass(frozen=True, slots=True)
class SkillStatus:
    """面向管理 API 的单个 Skill 状态快照。

    ``SkillStatus`` 描述配置中的每一项，而不是仅描述成功读取的文档；因此
    禁用项和单项读取失败也会出现在状态列表中。完整 Markdown 仍不会进入
    这个值对象。
    """

    name: str
    description: str
    path: str
    state: SkillState
    error: str | None = None


__all__ = ["SkillDocument", "SkillMetadata", "SkillState", "SkillStatus"]
