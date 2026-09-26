"""本地 Skill 知识库目录端口协议。"""

from __future__ import annotations

from typing import Protocol

from agent.domain.skills import SkillDocument, SkillMetadata, SkillStatus


class SkillCatalogPort(Protocol):
    """Agent 读取 Skill 元数据、状态和完整 Markdown 的端口。"""

    def list_statuses(self) -> tuple[SkillStatus, ...]: ...

    def metadata(self) -> tuple[SkillMetadata, ...]: ...

    def get_document(self, name: str) -> SkillDocument | None: ...


__all__ = ["SkillCatalogPort"]
