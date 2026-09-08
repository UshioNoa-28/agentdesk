"""Mem0 长期记忆服务实现。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from mem0 import AsyncMemory

from agent.infrastructure.memory.logging import log_memory_event
from agent.infrastructure.settings import MemorySettings
from agent.ports.memory import MemoryServicePort
from agent.prompt.memory import DEFAULT_MEMORY_EXTRACTION_POLICY


class Mem0MemoryService(MemoryServicePort):
    """基于 Mem0 (AsyncMemory) 和 Qdrant 的长期记忆服务。"""

    def __init__(
        self,
        *,
        settings: MemorySettings,
        client: AsyncMemory | None = None,
    ) -> None:
        self._settings = settings
        self._client = client

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 3,
    ) -> list[str]:
        """按语义检索属于该用户的相关记忆事实。"""

        client = self._client
        if not self._settings.memory_enabled or client is None:
            return []

        start = time.perf_counter()
        try:
            results = await client.search(
                query=query,
                filters={"user_id": user_id},
                top_k=limit,
            )
            items = results.get("results", []) if isinstance(results, dict) else results
            memories: list[str] = []
            for item in items:
                if isinstance(item, dict) and "memory" in item:
                    memories.append(str(item["memory"]))
                elif isinstance(item, str):
                    memories.append(item)
            log_memory_event(
                self._settings.memory_log_path,
                "MEMORY_SEARCH",
                user_id=user_id,
                duration_s=time.perf_counter() - start,
                details={"matched_count": len(memories)},
            )
            return memories
        except Exception as exc:
            log_memory_event(
                self._settings.memory_log_path,
                "MEMORY_SEARCH_FAILED",
                status="ERROR",
                user_id=user_id,
                duration_s=time.perf_counter() - start,
                error=str(exc),
                exc_info=True,
            )
            return []

    async def add(
        self,
        messages: Sequence[dict[str, str]] | str,
        *,
        user_id: str,
    ) -> None:
        """从增量对话或事实文本中提炼记忆，并自动进行消歧与持久化。"""

        client = self._client
        if not self._settings.memory_enabled or client is None:
            return

        start = time.perf_counter()
        try:
            result = await client.add(
                messages,
                user_id=user_id,
                prompt=DEFAULT_MEMORY_EXTRACTION_POLICY,
            )
            items = result.get("results", []) if isinstance(result, dict) else result
            log_memory_event(
                self._settings.memory_log_path,
                "MEMORY_ADD",
                user_id=user_id,
                duration_s=time.perf_counter() - start,
                details={
                    "extracted_count": len(items) if isinstance(items, list) else 0,
                },
            )
        except Exception as exc:
            log_memory_event(
                self._settings.memory_log_path,
                "MEMORY_ADD_FAILED",
                status="ERROR",
                user_id=user_id,
                duration_s=time.perf_counter() - start,
                error=str(exc),
                exc_info=True,
            )

    async def get_all(
        self,
        *,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """获取属于该用户的所有长期记忆列表。"""

        client = self._client
        if not self._settings.memory_enabled or client is None:
            return []
        try:
            results = await client.get_all(filters={"user_id": user_id})
            items = results.get("results", []) if isinstance(results, dict) else results
            return list(items) if isinstance(items, list) else []
        except Exception as exc:
            log_memory_event(
                self._settings.memory_log_path,
                "MEMORY_GET_ALL_FAILED",
                status="ERROR",
                user_id=user_id,
                error=str(exc),
                exc_info=True,
            )
            return []

    async def delete(
        self,
        memory_id: str,
    ) -> None:
        """显式删除某一条记忆。"""

        client = self._client
        if not self._settings.memory_enabled or client is None:
            return
        try:
            await client.delete(memory_id)
        except Exception as exc:
            log_memory_event(
                self._settings.memory_log_path,
                "MEMORY_DELETE_FAILED",
                status="ERROR",
                details={"memory_id": memory_id},
                error=str(exc),
                exc_info=True,
            )


__all__ = ["Mem0MemoryService"]
