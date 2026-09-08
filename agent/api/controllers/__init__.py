"""Agent 资源 Controller。"""

from agent.api.controllers.agent_controller import router as agent_router
from agent.api.controllers.health_controller import router as health_router
from agent.api.controllers.mcp_controller import router as mcp_router
from agent.api.controllers.session_controller import router as session_router
from agent.api.controllers.skill_controller import router as skill_router

__all__ = [
    "agent_router",
    "health_router",
    "session_router",
    "mcp_router",
    "skill_router",
]
