"""Skill 目录发现：从工作区根逐级向上扫描到用户主目录。

约定与 Claude Code 的目录式 Skill 对齐，根目录名换成本项目的
``.agent-desk``：每个可发现单元是 ``<ancestor>/.agent-desk/skills/<name>/``，
内含 ``SKILL.md``（解析见 ``document.py``）。优先级由近到远——工作区根最
高，逐层父目录次之，``~/.agent-desk/skills/`` 作为个人级兜底；若起点不在
主目录之下，主目录仍排在扫描链最后。同名 Skill 由近的层级遮蔽远的层级，
被遮蔽目录不读取内容，只记一条 warning。
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.infrastructure.project_settings import AGENTDESK_DIR_NAME, config_search_roots

logger = logging.getLogger(__name__)

SKILLS_DIR_NAME = "skills"


def discover_skill_dirs(start: Path, *, home: Path | None = None) -> tuple[Path, ...]:
    """按优先级收集有效的 Skill 目录；同名只保留最近的，遮蔽项跳过不读取。

    Raises:
        ValueError: 某个层级的 ``.agent-desk/skills`` 存在但不是目录。
    """

    roots = config_search_roots(start.expanduser().resolve(strict=False), home=home)
    active: list[Path] = []
    winners: dict[str, Path] = {}
    for root in roots:
        skills_root = root / AGENTDESK_DIR_NAME / SKILLS_DIR_NAME
        if not skills_root.exists():
            continue
        if not skills_root.is_dir():
            raise ValueError(f"Skills root is not a directory: {skills_root}")
        for entry in sorted(skills_root.iterdir(), key=lambda path: path.name):
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            if entry.name in winners:
                logger.warning(
                    "Skill '%s' shadowed by a nearer copy: %s (ignored: %s)",
                    entry.name,
                    winners[entry.name],
                    entry,
                )
                continue
            winners[entry.name] = entry
            active.append(entry)
    return tuple(active)


__all__ = [
    "AGENTDESK_DIR_NAME",
    "SKILLS_DIR_NAME",
    "discover_skill_dirs",
]
