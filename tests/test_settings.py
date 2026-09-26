from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from agent.infrastructure.settings import (
    AgentRuntimeSettings,
    DatabaseSettings,
)


class SettingsBoundaryTests(TestCase):
    """验证配置对象按职责拆分。"""

    def test_specialized_settings_keep_only_their_boundary_fields(self) -> None:
        self.assertEqual(
            {"database_url"},
            set(DatabaseSettings.model_fields),
        )

    def test_settings_are_kept_at_their_infrastructure_boundaries(self) -> None:
        # 模型配置已迁出 Settings 边界：改由 settings.json 的 model 段经
        # ModelCatalog 加载为扁平的 ModelProfile，不再有 ChatModelSettings。
        import agent.infrastructure.settings as settings_module

        self.assertFalse(hasattr(settings_module, "ChatModelSettings"))
        # 全局窗口与压缩水位已废除：预算随模型走（ModelProfile 的字段与
        # base_url 同级平铺在 model.<id> 条目里），不再有 ContextSettings 边界。
        self.assertFalse(hasattr(settings_module, "ContextSettings"))
        # 协议开关（responses api、超时、重试）不再是全局 env 边界：
        # 它们是同一条 profile 上的每模型字段。
        self.assertFalse(hasattr(settings_module, "ModelProtocolSettings"))
        # Mem0/Qdrant 长期记忆管线已拆除：不再有 MemorySettings 边界。
        self.assertFalse(hasattr(settings_module, "MemorySettings"))

    def test_root_dotenv_is_not_an_application_config_source(self) -> None:
        """根 .env 遗留内容不得重新混入分组配置。"""

        original_cwd = os.getcwd()
        try:
            with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                os.chdir(directory)
                Path(".env").write_text("MAX_TURNS=999\n", encoding="utf-8")
                (Path("config") / "env").mkdir(parents=True)
                (Path("config") / "env" / "agent.env").write_text(
                    "MAX_TURNS=12\n", encoding="utf-8"
                )

                settings = AgentRuntimeSettings()

            self.assertEqual(12, settings.max_turns)
        finally:
            os.chdir(original_cwd)
