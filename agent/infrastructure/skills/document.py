"""单个 Skill 的 SKILL.md 读取与校验：front matter 是元数据的唯一来源。

约定：每个 Skill 是一个目录，内含 ``SKILL.md``；front matter 必须从文件
第一行开始，``name`` 与 ``description`` 必填，``name`` 必须等于父目录名
（与 Agent Skills 开放标准的目录约定对齐）。其余 front matter 字段一律忽
略，保证与 Claude Code 生态文件前向兼容。front matter 缺失或非法不在这
里降级处理，而是直接抛 ``ValueError``，由组合根在启动期失败。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from agent.domain.skills import SkillDocument, SkillMetadata

DEFAULT_MAX_FILE_BYTES = 256 * 1024
MAX_NAME_CHARS = 128
MAX_DESCRIPTION_CHARS = 4000

SKILL_FILE_NAME = "SKILL.md"

_NAME_PATTERN = re.compile(r"[a-z0-9](?:[a_-]?[a-z0-9])*")

_FRONT_MATTER = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)


def read_skill_document(
    skill_dir: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> SkillDocument:
    """读取 ``<skill_dir>/SKILL.md``，校验 front matter，返回完整文档。

    Raises:
        ValueError: 文件缺失、超过大小上限、front matter 无效，或
            name/description 缺失、非法、与目录名不一致。
    """

    path = skill_dir / SKILL_FILE_NAME
    if not path.is_file():
        raise ValueError(f"Skill directory is missing {SKILL_FILE_NAME}: {path}")
    size = path.stat().st_size
    if size > max_file_bytes:
        raise ValueError(f"Skill file is {size} bytes, above limit {max_file_bytes}: {path}")
    text = path.read_text(encoding="utf-8")

    match = _FRONT_MATTER.match(text)
    if match is None:
        raise ValueError(f"Skill must open with YAML front matter ('---' on line 1): {path}")
    metadata = _front_matter_fields(match.group(1), path)

    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Skill front matter requires a non-empty 'name': {path}")
    name = name.strip()
    if len(name) > MAX_NAME_CHARS:
        raise ValueError(f"Skill name must not exceed {MAX_NAME_CHARS} characters: {path}")
    if _NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"Skill name must match [a-z0-9_-] with no leading, trailing, or "
            f"consecutive separators: {name!r} ({path})"
        )
    if name != skill_dir.name:
        raise ValueError(
            f"Skill name {name!r} must match its directory name {skill_dir.name!r}: {path}"
        )

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Skill front matter requires a non-empty 'description': {path}")
    description = description.strip()
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(
            f"Skill description must not exceed {MAX_DESCRIPTION_CHARS} characters: {path}"
        )

    instructions = text[match.end() :].lstrip()
    if not instructions:
        raise ValueError(f"Skill instructions must not be empty: {path}")

    return SkillDocument(
        metadata=SkillMetadata(
            name=name,
            description=description,
            path=display_path(path),
        ),
        instructions=instructions,
    )


def _front_matter_fields(raw: str, path: Path) -> dict[str, object]:
    """解析 front matter YAML；只要求是映射，未知键由调用方忽略。"""

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid Skill front matter YAML: {path}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Skill front matter must be a mapping: {path}")
    return parsed


def display_path(path: Path) -> str:
    """优先返回相对项目根的路径，外部文件才返回绝对路径。"""

    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "MAX_DESCRIPTION_CHARS",
    "MAX_NAME_CHARS",
    "SKILL_FILE_NAME",
    "display_path",
    "read_skill_document",
]
