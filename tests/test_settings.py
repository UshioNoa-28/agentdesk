from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from pydantic import ValidationError

from agent.infrastructure.settings import (
    ChatModelSettings,
    ContextSettings,
    DatabaseSettings,
    McpSettings,
    MemorySettings,
    SkillSettings,
)


class SettingsBoundaryTests(TestCase):
    """验证配置对象按职责拆分。"""

    def test_specialized_settings_keep_only_their_boundary_fields(self) -> None:
        self.assertEqual(
            {
                "database_url",
                "database_pool_size",
                "database_max_overflow",
                "database_pool_timeout_seconds",
            },
            set(DatabaseSettings.model_fields),
        )
        self.assertEqual(
            {"mcp_config_path"}, set(McpSettings.model_fields)
        )
        self.assertEqual({"skill_config_path"}, set(SkillSettings.model_fields))

    def test_settings_are_kept_at_their_infrastructure_boundaries(self) -> None:
        self.assertEqual(
            {
                "openai_api_key",
                "openai_base_url",
                "openai_model",
                "openai_max_output_tokens",
                "openai_use_responses_api",
                "openai_stream_chunk_timeout_seconds",
                "openai_request_timeout_seconds",
            },
            set(ChatModelSettings.model_fields),
        )
        # 摘要输出上限收敛进 ContextSettings：摘要模型与对话模型同源，
        # 不再有独立的 SummaryModelSettings 边界。
        self.assertNotIn(
            "summary_max_output_tokens",
            set(ContextSettings.model_fields),
        )
        self.assertIn("context_summary_max_tokens", set(ContextSettings.model_fields))

    def test_context_settings_read_their_scoped_dotenv_file(self) -> None:
        original_cwd = os.getcwd()
        try:
            with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                os.chdir(directory)
                (Path("config") / "env").mkdir(parents=True)
                (Path("config") / "env" / "context.env").write_text(
                    "CONTEXT_WINDOW_TOKENS=1234\n"
                    "CONTEXT_RESERVE_RATIO=0.1\n"
                    "CONTEXT_COMPACT_THRESHOLD_RATIO=0.7\n",
                    encoding="utf-8",
                )

                settings = ContextSettings()

            self.assertEqual(1234, settings.context_window_tokens)
            self.assertEqual(863, settings.compact_threshold_tokens)
        finally:
            os.chdir(original_cwd)

    def test_root_dotenv_is_not_an_application_config_source(self) -> None:
        """根 .env 遗留内容不得重新混入分组配置。"""

        original_cwd = os.getcwd()
        try:
            with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                os.chdir(directory)
                Path(".env").write_text("CONTEXT_WINDOW_TOKENS=999\n", encoding="utf-8")
                (Path("config") / "env").mkdir(parents=True)
                (Path("config") / "env" / "context.env").write_text(
                    "CONTEXT_WINDOW_TOKENS=1234\n",
                    encoding="utf-8",
                )

                settings = ContextSettings()

            self.assertEqual(1234, settings.context_window_tokens)
        finally:
            os.chdir(original_cwd)

    def test_memory_settings_read_their_scoped_dotenv_file(self) -> None:
        original_cwd = os.getcwd()
        try:
            with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                os.chdir(directory)
                (Path("config") / "env").mkdir(parents=True)
                (Path("config") / "env" / "memory.env").write_text(
                    "MEMORY_ENABLED=true\n"
                    "QDRANT_URL=http://test-qdrant:6333\n"
                    "MEMORY_LLM_MODEL=gpt-4o\n"
                    "MEMORY_LLM_BASE_URL=http://test-llm/v1\n"
                    "MEMORY_EMBEDDER_PROVIDER=ollama\n"
                    "MEMORY_EMBEDDER_DIMS=1024\n",
                    encoding="utf-8",
                )

                settings = MemorySettings()

            self.assertTrue(settings.memory_enabled)
            self.assertEqual("http://test-qdrant:6333", settings.qdrant_url)
            self.assertEqual("gpt-4o", settings.memory_llm_model)
        finally:
            os.chdir(original_cwd)

    def _isolated_memory_settings(self, **kwargs: object) -> MemorySettings:
        """在无 dotenv、无环境变量的沙箱里构造 MemorySettings。"""

        original_cwd = os.getcwd()
        try:
            with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                os.chdir(directory)
                return MemorySettings(**kwargs)  # type: ignore[arg-type]
        finally:
            os.chdir(original_cwd)

    def test_memory_validator_skips_all_checks_when_disabled(self) -> None:
        settings = self._isolated_memory_settings(memory_enabled=False)
        self.assertFalse(settings.memory_enabled)

    def test_memory_validator_requires_dims_when_enabled(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self._isolated_memory_settings(memory_enabled=True)
        self.assertIn("MEMORY_EMBEDDER_DIMS", str(ctx.exception))

    def test_memory_validator_requires_llm_key_or_base_url(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self._isolated_memory_settings(
                memory_enabled=True,
                memory_embedder_dims=1024,
                memory_embedder_provider="ollama",
            )
        self.assertIn("MEMORY_LLM", str(ctx.exception))

    def test_memory_validator_requires_embedder_key_or_base_url(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self._isolated_memory_settings(
                memory_enabled=True,
                memory_embedder_dims=1024,
                memory_llm_provider="ollama",
            )
        self.assertIn("MEMORY_EMBEDDER", str(ctx.exception))

    def test_memory_validator_accepts_base_url_only_without_key(self) -> None:
        """自定义 OpenAI 兼容端点免 key：有 base_url 即可通过。"""
        settings = self._isolated_memory_settings(
            memory_enabled=True,
            memory_embedder_dims=1024,
            memory_llm_base_url="https://example.com/v1",
            memory_embedder_provider="ollama",
        )
        self.assertIsNone(settings.memory_llm_api_key)

    def test_memory_validator_accepts_ollama_without_keys(self) -> None:
        settings = self._isolated_memory_settings(
            memory_enabled=True,
            memory_embedder_dims=1024,
            memory_llm_provider="ollama",
            memory_embedder_provider="ollama",
        )
        self.assertEqual(1024, settings.memory_embedder_dims)
