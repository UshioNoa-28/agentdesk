"""进程内本地 Skill 目录：构造期加载一次、不可变。

目录发现见 ``discovery.py``，单文件解析见 ``document.py``。任何一个被发现
的 Skill 读取失败都会直接抛错——Skill 配置错误应当在启动期暴露，而不是
静默降级。目录在构造时加载一次；metadata 随之稳定，保证 system prompt 的
KV cache 命中。改了 Skill 文件就重启进程，不做热重载。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from agent.domain.skills import SkillDocument, SkillMetadata, SkillState, SkillStatus
from agent.infrastructure.skills.discovery import discover_skill_dirs
from agent.infrastructure.skills.document import (
    DEFAULT_MAX_FILE_BYTES,
    read_skill_document,
)
from agent.ports.tools import SkillCatalogPort


class SkillCatalog(SkillCatalogPort):
    """一个进程内、构造期加载的本地 Skill 目录。"""

    def __init__(
        self,
        skill_dirs: Sequence[Path],
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        """加载给定 Skill 目录；任一项校验或读取失败即抛错。"""

        if max_file_bytes < 1024:
            raise ValueError("max_file_bytes must be at least 1024")
        documents: dict[str, SkillDocument] = {}
        statuses: list[SkillStatus] = []
        for skill_dir in skill_dirs:
            document = read_skill_document(skill_dir, max_file_bytes=max_file_bytes)
            name = document.metadata.name
            if name in documents:
                raise ValueError(f"Duplicate Skill name: {name} ({skill_dir})")
            documents[name] = document
            statuses.append(
                SkillStatus(
                    name=name,
                    description=document.metadata.description,
                    path=document.metadata.path,
                    state=SkillState.LOADED,
                )
            )
        self._documents = dict(sorted(documents.items()))
        self._statuses = tuple(statuses)

    @classmethod
    def from_root(cls, start: Path, *, home: Path | None = None) -> SkillCatalog:
        """从工作区根出发发现 Skill 并构造目录。"""

        return cls(discover_skill_dirs(start, home=home))

    def list_statuses(self) -> tuple[SkillStatus, ...]:
        """按发现优先级（由近到远）返回每个 Skill 的加载状态。"""

        return self._statuses

    def metadata(self) -> tuple[SkillMetadata, ...]:
        """返回按名称排序的轻量元数据；system prompt 消费，不含正文。"""

        return tuple(document.metadata for document in self._documents.values())

    def get_document(self, name: str) -> SkillDocument | None:
        """按精确名称取完整文档；不存在返回 None。"""

        return self._documents.get(name)


__all__ = ["SkillCatalog"]
