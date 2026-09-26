"""进程内智能体运行态状态注册表。"""

from __future__ import annotations

from agent.domain.multi_agent import AgentStatus
from agent.ports.runtime import AgentStatusRegistryPort


class InMemoryAgentStatusRegistry(AgentStatusRegistryPort):
    """用 dict 存储 session_id → AgentStatus 的内存实现。

    运行在 AutoGen SingleThreadedAgentRuntime（单 event loop、协作式调度）
    下，读写均为同步操作，不存在跨线程竞争；不同 actor 的 set_status 调用
    在时间上串行执行，因此无需加锁。
    """

    def __init__(self) -> None:
        self._store: dict[str, AgentStatus] = {}

    def set_status(self, session_id: str, status: AgentStatus) -> None:
        self._store[session_id] = status

    def get_status(self, session_id: str) -> AgentStatus | None:
        return self._store.get(session_id)


__all__ = ["InMemoryAgentStatusRegistry"]
