"""固定 ``load_skill`` Meta Tool。"""

from __future__ import annotations

from collections.abc import Mapping

from agent.domain.messages import MessageKind
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
from agent.ports.tools import SkillCatalogPort

LOAD_SKILL_DEFINITION = ToolDefinition(
    name="load_skill",
    description=(
        "Load the full instructions of a local Skill by its exact name. The available "
        "Skill names and descriptions are listed in the system prompt; pass one of "
        "those names verbatim. The Skill instructions are returned as low-priority "
        "guidance for the current conversation. If the Skill mentions an MCP "
        "capability, use search_mcp and then execute_mcp."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "\\S",
                "description": "The exact name of the Skill to load.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
)


class LoadSkillTool(AgentTool):
    """按精确名称加载本地 Skill，把命中的完整 Markdown 作为工具结果返回。"""

    def __init__(self, catalog: SkillCatalogPort) -> None:
        self._catalog = catalog

    @property
    def definition(self) -> ToolDefinition:
        return LOAD_SKILL_DEFINITION

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        name = str(arguments["name"]).strip()

        document = self._catalog.get_document(name)
        if document is None:
            available = ", ".join(
                sorted(item.name for item in self._catalog.metadata())
            )
            return error_result(
                "skill_not_found",
                f"Unknown Skill: {name.strip()!r}. Available skills: {available}.",
            )
        return ok_result(
            {
                "skill": {
                    "name": document.metadata.name,
                    "description": document.metadata.description,
                    "path": document.metadata.path,
                    "content": document.instructions,
                },
            },
            kind=MessageKind.SKILL_RESULT,
        )


__all__ = ["LOAD_SKILL_DEFINITION", "LoadSkillTool"]
