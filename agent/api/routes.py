from fastapi import APIRouter

from agent.api.controllers import (
    agent_router,
    health_router,
    mcp_router,
    session_router,
    skill_router,
)

"""HTTP 路由总装配入口。

具体 Controller 按资源拆在 ``agent.api.controllers`` 中；这个文件只负责统一
挂载 `/api` 前缀，避免 FastAPI 应用入口需要知道每个 Controller 的细节。
"""


router = APIRouter(prefix="/api")
router.include_router(health_router)
router.include_router(session_router)
router.include_router(agent_router)
router.include_router(mcp_router)
router.include_router(skill_router)
