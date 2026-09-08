from __future__ import annotations

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter

from agent.domain.skills import SkillStatus
from agent.dto.skill import SkillStatusResponse
from agent.ports.tools import SkillCatalogPort

router = APIRouter(tags=["skills"])


@router.get("/skills", response_model=list[SkillStatusResponse])
@inject
async def list_skills(catalog: FromDishka[SkillCatalogPort]) -> list[SkillStatusResponse]:
    """列出所有配置 Skill 的最近一次加载状态。

    Args:
        catalog (SkillCatalogPort): 由 Dishka 注入的进程级 Skill 目录。

    Returns:
        list[SkillStatusResponse]: 包含 loaded、failed 和 disabled 项的状态列表。
    """

    return [_response(status) for status in catalog.list_statuses()]


def _response(status: SkillStatus) -> SkillStatusResponse:
    """把领域状态快照转换为 HTTP DTO。

    Args:
        status (SkillStatus): 不包含 Markdown 正文的领域状态。

    Returns:
        SkillStatusResponse: 面向管理客户端的可序列化状态。
    """

    return SkillStatusResponse(
        name=status.name,
        description=status.description,
        path=status.path,
        state=status.state,
        error=status.error,
    )


__all__ = ["router"]
