"""Skill 管理 API 的响应 DTO。"""

from __future__ import annotations

from pydantic import BaseModel

from agent.domain.skills import SkillState


class SkillStatusResponse(BaseModel):
    """返回给管理 API 的单个 Skill 状态快照。"""

    name: str
    description: str
    path: str
    state: SkillState
    error: str | None


__all__ = ["SkillStatusResponse"]
