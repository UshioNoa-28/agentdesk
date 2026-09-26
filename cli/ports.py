"""CLI 传输端口:UI 层只依赖这里的协议,不绑定具体传输实现。

唯一实现是进程内 ``cli.local_client.LocalApiClient``;方法形状沿用当年
HTTP 契约(dict 载荷、``{"event","data"}`` 事件字典、``AgentApiError``),
``checkout/release`` 表达桌面多开的会话驾驶权。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable


class AgentApiError(Exception):
    """后端请求失败的统一异常;本地实现用 ``status_code=None`` 表达进程内错误。"""

    def __init__(self, message: str, status_code: int | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


@runtime_checkable
class AgentClientPort(Protocol):
    """终端 UI 消费的后端能力面。返回值均为 JSON 兼容的 dict/list。

    ``checkout``/``release`` 表达会话驾驶权:打开主会话前签出,切走/退出
    时释放;被其他窗口占用时抛 ``AgentApiError(status_code=409)``。
    实现方用 OS 级文件锁真正强制(见 local_client)。
    """

    def close(self) -> None: ...

    def checkout(self, session_id: str) -> None: ...

    def release(self, session_id: str) -> None: ...

    def health(self, *, timeout: float | None = None) -> dict[str, Any]: ...

    def list_sessions(self, *, timeout: float | None = None) -> list[dict[str, Any]]: ...

    def get_session(self, session_id: str) -> dict[str, Any]: ...

    def create_session(
        self,
        title: str | None = None,
        allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def rename_session(self, session_id: str, title: str) -> dict[str, Any]: ...

    def resume_session(
        self,
        session_id: str,
        title: str | None = None,
        parent_last_seq: int | None = None,
    ) -> dict[str, Any]: ...

    def delete_session(self, session_id: str) -> None: ...

    def compact_session(self, session_id: str) -> dict[str, Any]: ...

    def list_messages(
        self,
        session_id: str,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]: ...

    def ask(self, session_id: str, question: str) -> dict[str, Any]: ...

    def ask_stream(
        self,
        session_id: str,
        question: str,
        *,
        request_id: str | None = None,
    ) -> Iterator[dict[str, Any]]: ...

    def abort_stream(self, request_id: str | None = None) -> None: ...

    def cancel(self, session_id: str) -> None: ...

    def reply_permission(
        self,
        permission_id: str,
        decision: str,
        scope: Sequence[str] | None = None,
    ) -> dict[str, Any]: ...

    def reply_ask_user(
        self,
        interruption_id: str,
        answer: str,
    ) -> dict[str, Any]: ...

    def list_mcps(self) -> list[dict[str, Any]]: ...

    def retry_mcp(self, server_id: str) -> dict[str, Any]: ...

    def list_skills(self) -> list[dict[str, Any]]: ...

    def list_memories(self) -> list[dict[str, Any]]: ...

    def get_memory(self, layer: str, title: str) -> dict[str, Any] | None: ...

    def list_models(self) -> list[dict[str, Any]]: ...

    def get_current_model(self) -> dict[str, Any]: ...

    def select_model(self, model_id: str) -> dict[str, Any]: ...

    def get_permission_mode(self) -> str: ...

    def set_permission_mode(self, mode: str) -> str: ...


__all__ = ["AgentApiError", "AgentClientPort"]
