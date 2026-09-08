"""进程内本地 Skill 目录：构造期加载一次、不可变。

配置解析见 ``config.py``，Markdown 读取见 ``document.py``。目录在构造时
加载一次；metadata 随之稳定，保证 system prompt 的 KV cache 命中。改了
Skill 文件就重启进程，不做热重载。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from agent.domain.skills import SkillDocument, SkillMetadata, SkillState, SkillStatus
from agent.infrastructure.skills.config import DEFAULT_MAX_FILE_BYTES, SkillConfig
from agent.infrastructure.skills.document import display_path, read_skill_document
from agent.ports.tools import SkillCatalogPort

logger = logging.getLogger(__name__)


class SkillCatalog(SkillCatalogPort):
    """一个进程内、构造期加载的本地 Skill 目录。"""

    def __init__(
        self,
        skills: Sequence[SkillConfig],
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        """加载启用的 Skill；单文件失败记为 failed，不影响其余条目。"""

        if max_file_bytes < 1024:
            raise ValueError("max_file_bytes must be at least 1024")
        self._skills = tuple(skills)
        self._max_file_bytes = max_file_bytes
        self._documents: dict[str, SkillDocument] = {}
        self._errors: tuple[str, ...] = ()
        self._statuses: tuple[SkillStatus, ...] = ()
        self._load()

    def list_statuses(self) -> tuple[SkillStatus, ...]:
        """按配置顺序返回每个 Skill 的加载状态（含 disabled / failed）。"""

        return self._statuses

    def metadata(self) -> tuple[SkillMetadata, ...]:
        """返回按名称排序的轻量元数据；system prompt 消费，不含正文。"""

        return tuple(document.metadata for document in self._documents.values())

    def get_document(self, name: str) -> SkillDocument | None:
        """按精确名称取完整文档；不存在返回 None。"""

        return self._documents.get(name)

    @property
    def errors(self) -> tuple[str, ...]:
        """返回构造期加载的可诊断错误。"""

        return self._errors

    def _load(self) -> None:
        """构造期一次性加载全部配置项，填充文档、状态与错误快照。"""

        documents: dict[str, SkillDocument] = {}
        errors: list[str] = []
        statuses: list[SkillStatus] = []
        for skill in self._skills:
            skill_path = Path(skill.file_path).expanduser().resolve()
            if not skill.enabled:
                statuses.append(
                    SkillStatus(
                        name=skill.name,
                        description=skill.description,
                        path=display_path(skill_path),
                        state=SkillState.DISABLED,
                    )
                )
                continue
            try:
                document = read_skill_document(
                    skill, skill_path, max_file_bytes=self._max_file_bytes
                )
            except (OSError, UnicodeError, ValueError) as exc:
                message = f"Could not load Skill {skill.name} ({skill_path}): {exc}"
                errors.append(message)
                logger.warning(message)
                statuses.append(
                    SkillStatus(
                        name=skill.name,
                        description=skill.description,
                        path=display_path(skill_path),
                        state=SkillState.FAILED,
                        error=message,
                    )
                )
                continue
            if document.metadata.name in documents:
                message = f"Duplicate Skill name ignored: {document.metadata.name} ({skill_path})"
                errors.append(message)
                logger.warning(message)
                statuses.append(
                    SkillStatus(
                        name=skill.name,
                        description=skill.description,
                        path=document.metadata.path,
                        state=SkillState.FAILED,
                        error=message,
                    )
                )
                continue
            documents[document.metadata.name] = document
            statuses.append(
                SkillStatus(
                    name=skill.name,
                    description=skill.description,
                    path=document.metadata.path,
                    state=SkillState.LOADED,
                )
            )
        self._documents = dict(sorted(documents.items()))
        self._errors = tuple(errors)
        self._statuses = tuple(statuses)


__all__ = ["SkillCatalog"]
