"""Skill Markdown 文件读取与展示路径工具。"""

from __future__ import annotations

import re
from pathlib import Path

from agent.domain.skills import SkillDocument, SkillMetadata
from agent.infrastructure.skills.config import SkillConfig

_FRONT_MATTER = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)


def read_skill_document(
    config: SkillConfig,
    path: Path,
    *,
    max_file_bytes: int,
) -> SkillDocument:
    """读取一个 Skill 文件，剥离可选 front matter，返回完整文档。"""

    if not path.is_file():
        raise ValueError("configured file_path is not a file")
    size = path.stat().st_size
    if size > max_file_bytes:
        raise ValueError(f"file is {size} bytes, above configured limit {max_file_bytes}")
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(text)
    instructions = text[match.end() :].lstrip() if match is not None else text.lstrip()
    if not instructions:
        raise ValueError("Skill instructions must not be empty")
    return SkillDocument(
        metadata=SkillMetadata(
            name=config.name,
            description=config.description,
            path=display_path(path),
        ),
        instructions=instructions,
    )


def display_path(path: Path) -> str:
    """优先返回相对项目根的路径，外部文件才返回绝对路径。"""

    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = ["display_path", "read_skill_document"]
