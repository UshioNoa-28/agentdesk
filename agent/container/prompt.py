"""prompt 策略基础设施 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.infrastructure.prompt import PromptBuilder
from agent.ports.tools import McpRegistryPort, SkillCatalogPort


class PromptProvider(Provider):
    """提供进程级 PromptBuilder；MCP/Skill 清单每次 build 时现读。"""

    @provide(scope=Scope.APP)
    def provide_prompt_builder(
        self,
        mcp_registry: McpRegistryPort,
        skill_catalog: SkillCatalogPort,
    ) -> PromptBuilder:
        return PromptBuilder(mcp_registry=mcp_registry, skill_catalog=skill_catalog)


__all__ = ["PromptProvider"]
