"""Mem0 AsyncMemory 客户端工厂。"""

from __future__ import annotations

from typing import Any

from mem0 import AsyncMemory

from agent.infrastructure.memory.logging import log_memory_event
from agent.infrastructure.settings import MemorySettings
from agent.prompt.memory import DEFAULT_MEMORY_EXTRACTION_POLICY


def _build_llm_config(settings: MemorySettings) -> dict[str, Any]:
    """按环境变量直接组装 LLM provider config。"""

    provider = settings.memory_llm_provider.lower()
    if provider == "ollama":
        config: dict[str, Any] = {"model": settings.memory_llm_model}
        if settings.memory_llm_base_url:
            config["ollama_base_url"] = settings.memory_llm_base_url
        return config

    config: dict[str, Any] = {
        "model": settings.memory_llm_model,
        "api_key": settings.memory_llm_api_key or "not-required",
    }
    if settings.memory_llm_base_url:
        config["openai_base_url"] = settings.memory_llm_base_url
    return config


def _build_embedder_config(settings: MemorySettings) -> dict[str, Any]:
    """按环境变量直接组装 Embedder provider config。"""

    provider = settings.memory_embedder_provider.lower()
    dims = settings.memory_embedder_dims
    if provider == "ollama":
        config = {
            "model": settings.memory_embedder_model,
            "embedding_dims": dims,
        }
        if settings.memory_embedder_base_url:
            config["ollama_base_url"] = settings.memory_embedder_base_url
        return config

    config: dict[str, Any] = {
        "model": settings.memory_embedder_model,
        "embedding_dims": dims,
        "api_key": settings.memory_embedder_api_key or "not-required",
    }
    if settings.memory_embedder_base_url:
        config["openai_base_url"] = settings.memory_embedder_base_url
    return config


def build_mem0_async_client(settings: MemorySettings) -> Any | None:
    """根据配置创建 Mem0 AsyncMemory 客户端；禁用时返回 None。

    开启 memory 时配置错误会直接抛异常（启动失败），不静默降级。
    """

    if not settings.memory_enabled:
        return None

    try:
        llm_config = _build_llm_config(settings)
        embedder_config = _build_embedder_config(settings)
        embedder_dims = embedder_config["embedding_dims"]

        vector_store_config: dict[str, Any] = {
            "collection_name": settings.qdrant_collection_name,
            "url": settings.qdrant_url,
            "embedding_model_dims": embedder_dims,
        }
        if settings.qdrant_api_key:
            vector_store_config["api_key"] = settings.qdrant_api_key

        client = AsyncMemory.from_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": vector_store_config,
                },
                "llm": {
                    "provider": settings.memory_llm_provider.lower(),
                    "config": llm_config,
                },
                "embedder": {
                    "provider": settings.memory_embedder_provider.lower(),
                    "config": embedder_config,
                },
                "custom_instructions": DEFAULT_MEMORY_EXTRACTION_POLICY,
            }
        )

        log_memory_event(
            settings.memory_log_path,
            "MEMORY_CLIENT_INIT_SUCCESS",
            details={
                "llm_provider": settings.memory_llm_provider.lower(),
                "llm_model": settings.memory_llm_model,
                "embedder_provider": settings.memory_embedder_provider.lower(),
                "embedder_model": settings.memory_embedder_model,
                "embedder_dims": embedder_dims,
                "qdrant_url": settings.qdrant_url,
                "collection_name": settings.qdrant_collection_name,
            },
        )
        return client
    except Exception as exc:
        log_memory_event(
            settings.memory_log_path,
            "MEMORY_CLIENT_INIT_FAILED",
            status="ERROR",
            error=str(exc),
            exc_info=True,
        )
        raise


__all__ = ["build_mem0_async_client"]
