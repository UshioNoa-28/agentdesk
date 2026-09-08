"""长期记忆检索 Meta Tool。"""

from __future__ import annotations

from collections.abc import Mapping

from agent.domain.messages import MessageKind
from agent.domain.tools import AgentTool, ToolContext, ToolDefinition, ToolResult
from agent.infrastructure.settings import MemorySettings
from agent.ports.memory import MemoryServicePort
from agent.ports.services import SessionServicePort

SEARCH_MEMORY_DEFINITION = ToolDefinition(
    name="search_memory",
    description=(
        "Search the user's long-term semantic memory for relevant facts, "
        "preferences, background information, or past project knowledge."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or keywords to look up in long-term memory.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 40,
                "description": "Maximum number of memory facts to return (default: 3, max: 40).",
                "default": 3,
            },
        },
        "required": ["query"],
    },
    keywords=("memory", "remember", "user preference", "past facts", "profile"),
    use_cases=("Retrieve user preferences", "Look up past discussion conclusions or facts"),
)


class SearchMemoryTool(AgentTool):
    """按语义检索长期记忆中存储的用户偏好、背景知识与历史事实。"""

    def __init__(
        self,
        *,
        memory_service: MemoryServicePort,
        settings: MemorySettings,
        session_service: SessionServicePort | None = None,
    ) -> None:
        self._memory_service = memory_service
        self._session_service = session_service
        self._max_limit = settings.memory_search_max_limit

    @property
    def definition(self) -> ToolDefinition:
        return SEARCH_MEMORY_DEFINITION

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        query_val = arguments.get("query")
        if not isinstance(query_val, str) or not query_val.strip():
            return ToolResult(
                content="Error: query parameter must be a non-empty string.",
                kind=MessageKind.TOOL_RESULT,
            )

        limit_val = arguments.get("limit", 3)
        try:
            limit = int(limit_val)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            limit = 3
        limit = min(max(limit, 1), self._max_limit)

        user_id = "default_user"
        try:
            session = await self._session_service.get(context.caller_session_id)
            user_id = session.user_id
        except Exception:
            pass

        memories = await self._memory_service.search(
            query=query_val.strip(),
            user_id=user_id,
            limit=limit,
        )

        if not memories:
            return ToolResult(
                content=f"No matching long-term memories found for query: {query_val.strip()!r}",
                kind=MessageKind.TOOL_RESULT,
            )

        formatted = "\n".join(f"- {m}" for m in memories)
        return ToolResult(
            content=f"Found {len(memories)} long-term memory facts:\n{formatted}",
            kind=MessageKind.TOOL_RESULT,
        )


__all__ = ["SEARCH_MEMORY_DEFINITION", "SearchMemoryTool"]
