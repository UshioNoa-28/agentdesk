"""本地 Skill 目录 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.infrastructure.settings import SkillSettings
from agent.infrastructure.skills import (
    SkillCatalog,
    SkillConfiguration,
    load_skill_configuration,
)
from agent.ports.tools import SkillCatalogPort


class SkillProvider(Provider):
    """提供 Skill 配置读取和进程级 Skill 目录。"""

    @provide(scope=Scope.APP)
    def provide_skill_configuration(self, settings: SkillSettings) -> SkillConfiguration:
        """读取显式登记的 Skill 元数据；内容本身由 SkillCatalog 负责解析。"""

        return load_skill_configuration(settings.skill_config_path)

    @provide(scope=Scope.APP)
    def provide_skill_catalog(
        self,
        configuration: SkillConfiguration,
        settings: SkillSettings,
    ) -> SkillCatalogPort:
        """创建进程级 Skill 目录，并在启动时加载已启用条目。"""

        return SkillCatalog(
            configuration.skills,
            max_file_bytes=configuration.max_file_bytes,
        )


__all__ = ["SkillProvider"]
