"""本地 Skill 目录 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.infrastructure.settings import AgentRuntimeSettings
from agent.infrastructure.skills import SkillCatalog
from agent.ports.tools import SkillCatalogPort


class SkillProvider(Provider):
    """提供进程级 Skill 目录；配置来源是 .agent-desk/skills 目录约定。"""

    @provide(scope=Scope.APP)
    def provide_skill_catalog(self, settings: AgentRuntimeSettings) -> SkillCatalogPort:
        """从 workspace 根（缺省为进程 cwd）逐级向上发现 Skill 并加载。"""

        start = settings.workspace_start
        return SkillCatalog.from_root(start)


__all__ = ["SkillProvider"]
