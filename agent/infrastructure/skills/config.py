"""``skills.yaml`` 配置读取与严格校验。

Skill 的名称、描述和文件路径由 YAML 显式登记；本模块只负责把这份清单
解析成已校验的 :class:`SkillConfiguration`，不触碰 Markdown 文件本身。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_VALUE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
DEFAULT_MAX_FILE_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class SkillConfig:
    """``skills.yaml`` 中登记的一个 Skill。

    配置元数据是权威来源；Markdown front matter 不会覆盖 name、description
    或 file_path。
    """

    name: str
    description: str
    file_path: str
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class SkillConfiguration:
    """Skill 清单和单个 Skill 的读取上限。"""

    skills: tuple[SkillConfig, ...]
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES


def load_skill_configuration(config_path: str | Path) -> SkillConfiguration:
    """读取并严格校验 skills.yaml；`${}` 占位符先展开，重复名/路径直接拒绝。"""

    path = Path(config_path)
    try:
        raw = yaml.safe_load(_expand_environment(path.read_text(encoding="utf-8")))
    except OSError as exc:
        raise RuntimeError(f"Could not read Skill config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid Skill config YAML: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Skill config must be an object")

    raw_skills = raw.get("skills")
    if not isinstance(raw_skills, list):
        raise ValueError("Skill config must contain a skills list")
    skills = tuple(_skill_config(item, index) for index, item in enumerate(raw_skills))
    names = [skill.name for skill in skills]
    if len(set(names)) != len(names):
        raise ValueError("Skill config contains duplicate names")
    paths = [skill.file_path for skill in skills]
    if len(set(paths)) != len(paths):
        raise ValueError("Skill config contains duplicate file paths")

    max_file_bytes = raw.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES)
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or not 1024 <= max_file_bytes <= 10 * 1024 * 1024
    ):
        raise ValueError("Skill config max_file_bytes must be an integer from 1024 to 10485760")
    return SkillConfiguration(skills=skills, max_file_bytes=max_file_bytes)


def _skill_config(raw: Any, index: int) -> SkillConfig:
    """校验并构造一个 Skill 配置项；index 用于错误定位。"""

    if not isinstance(raw, dict):
        raise ValueError(f"Skill at index {index} must be an object")
    name = _required_skill_string(raw, "name", index)
    if "/" in name or "\\" in name:
        raise ValueError("Skill name must not contain path separators")
    if len(name) > 128:
        raise ValueError("Skill name must not exceed 128 characters")
    description = _required_skill_string(raw, "description", index)
    if len(description) > 4000:
        raise ValueError("Skill description must not exceed 4000 characters")
    file_path = _required_skill_string(raw, "file_path", index)
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"Skill '{name}' enabled must be a boolean")
    return SkillConfig(
        name=name,
        description=description,
        file_path=file_path,
        enabled=enabled,
    )


def _required_skill_string(raw: dict[str, Any], key: str, index: int) -> str:
    """读取必须为非空字符串的字段，返回去除首尾空白后的值。"""

    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Skill at index {index} requires non-empty '{key}'")
    return value.strip()


def _expand_environment(text: str) -> str:
    """展开 ``${NAME}`` 与 ``${NAME:-default}``；无值且无默认值时报错。"""

    def replace(match: re.Match[str]) -> str:
        value = os.environ.get(match.group(1))
        if value is not None and value != "":
            return value
        default = match.group(2)
        if default is not None:
            return default
        raise ValueError(f"Skill config references unset environment variable: {match.group(1)}")

    return _ENV_VALUE.sub(replace, text)


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "SkillConfig",
    "SkillConfiguration",
    "load_skill_configuration",
]
