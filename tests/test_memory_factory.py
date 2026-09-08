"""Mem0 客户端工厂真实组装路径回归测试。

此前的 enabled 路径测试全部注入 mock client，组装字段缺失/错误测不出来
（2026-08-31 的 ollama 事故盲区）。这里 mock 掉 mem0 的 ``AsyncMemory``
本身，断言工厂产出的真实 config。
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

from agent.infrastructure.memory.factory import build_mem0_async_client
from agent.infrastructure.settings import MemorySettings


def _isolated_memory_settings(**kwargs: object) -> MemorySettings:
    """在无 dotenv、无环境变量的沙箱里构造 MemorySettings。"""

    original_cwd = os.getcwd()
    try:
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            os.chdir(directory)
            return MemorySettings(**kwargs)  # type: ignore[arg-type]
    finally:
        os.chdir(original_cwd)


class MemoryFactoryTests(TestCase):
    """build_mem0_async_client 组装行为。"""

    def test_disabled_memory_returns_none_without_building(self) -> None:
        settings = _isolated_memory_settings(memory_enabled=False)
        with patch("agent.infrastructure.memory.factory.AsyncMemory") as mock_async_memory:
            self.assertIsNone(build_mem0_async_client(settings))
        mock_async_memory.from_config.assert_not_called()

    def test_ollama_embedder_and_openai_llm_assembly(self) -> None:
        """LLM 走自定义 openai 端点免 key（占位值）；embedder 走 ollama 原生。"""
        with TemporaryDirectory() as temp_dir:
            settings = _isolated_memory_settings(
                memory_enabled=True,
                memory_embedder_dims=1024,
                memory_llm_api_key=None,
                memory_llm_base_url="https://tokenrhythm.example/v1",
                memory_embedder_provider="ollama",
                memory_embedder_model="mxbai-embed-large:latest",
                memory_embedder_base_url="http://host.docker.internal:11434",
                qdrant_url="http://qdrant:6333",
                memory_log_path=str(Path(temp_dir) / "memory.log"),
            )
            with patch("agent.infrastructure.memory.factory.AsyncMemory") as mock_async_memory:
                mock_async_memory.from_config.return_value = MagicMock()
                client = build_mem0_async_client(settings)

            self.assertIsNotNone(client)
            mock_async_memory.from_config.assert_called_once()
            config = mock_async_memory.from_config.call_args.args[0]
            self.assertEqual(
                {
                    "collection_name": "agent_memories",
                    "url": "http://qdrant:6333",
                    "embedding_model_dims": 1024,
                },
                config["vector_store"]["config"],
            )
            self.assertEqual("openai", config["llm"]["provider"])
            self.assertEqual("not-required", config["llm"]["config"]["api_key"])
            self.assertEqual(
                "https://tokenrhythm.example/v1",
                config["llm"]["config"]["openai_base_url"],
            )
            self.assertEqual("ollama", config["embedder"]["provider"])
            self.assertEqual(
                "mxbai-embed-large:latest", config["embedder"]["config"]["model"]
            )
            self.assertEqual(1024, config["embedder"]["config"]["embedding_dims"])
            self.assertEqual(
                "http://host.docker.internal:11434",
                config["embedder"]["config"]["ollama_base_url"],
            )
            self.assertTrue(config["custom_instructions"])

    def test_openai_embedder_api_key_passthrough(self) -> None:
        """显式提供 key 时原样透传；ollama LLM 分支不携带 api_key。"""
        with TemporaryDirectory() as temp_dir:
            settings = _isolated_memory_settings(
                memory_enabled=True,
                memory_embedder_dims=1536,
                memory_llm_provider="ollama",
                memory_llm_model="qwen2.5:7b",
                memory_embedder_api_key="sk-real-key",
                memory_log_path=str(Path(temp_dir) / "memory.log"),
            )
            with patch("agent.infrastructure.memory.factory.AsyncMemory") as mock_async_memory:
                mock_async_memory.from_config.return_value = MagicMock()
                build_mem0_async_client(settings)

            config = mock_async_memory.from_config.call_args.args[0]
            self.assertEqual("sk-real-key", config["embedder"]["config"]["api_key"])
            self.assertEqual("ollama", config["llm"]["provider"])
            self.assertNotIn("api_key", config["llm"]["config"])

    def test_init_failure_is_logged_then_reraised(self) -> None:
        """mem0 构建失败时写审计日志并原样抛出，不静默降级。"""
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "memory.log"
            settings = _isolated_memory_settings(
                memory_enabled=True,
                memory_embedder_dims=1024,
                memory_llm_provider="ollama",
                memory_embedder_provider="ollama",
                memory_log_path=str(log_path),
            )
            with patch("agent.infrastructure.memory.factory.AsyncMemory") as mock_async_memory:
                mock_async_memory.from_config.side_effect = RuntimeError("boom")
                with self.assertRaises(RuntimeError):
                    build_mem0_async_client(settings)

            self.assertTrue(log_path.exists())
            self.assertIn("MEMORY_CLIENT_INIT_FAILED", log_path.read_text(encoding="utf-8"))
