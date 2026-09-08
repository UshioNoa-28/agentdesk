"""固定 ``load_skill`` Meta Tool。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from agent.domain.messages import MessageKind
from agent.domain.tools import AgentTool, ToolContext, ToolDefinition, ToolResult
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
                "minLength": 1,
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
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("invalid_arguments", "load_skill requires a non-empty name")

        document = self._catalog.get_document(name.strip())
        if document is None:
            available = ", ".join(
                sorted(item.name for item in self._catalog.metadata())
            )
            return _error(
                "skill_not_found",
                f"Unknown Skill: {name.strip()!r}. Available skills: {available}.",
            )
        return ToolResult(
            content=json.dumps(
                {
                    "ok": True,
                    "skill": {
                        "name": document.metadata.name,
                        "description": document.metadata.description,
                        "path": document.metadata.path,
                        "content": document.instructions,
                    },
                },
                ensure_ascii=False,
            ),
            kind=MessageKind.SKILL_RESULT,
        )


def _error(code: str, message: str) -> ToolResult:
    return ToolResult(
        content=json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        )
    )


__all__ = ["LOAD_SKILL_DEFINITION", "LoadSkillTool"]
