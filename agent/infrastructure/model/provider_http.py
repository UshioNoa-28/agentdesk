"""模型服务访问外部 OpenAI-compatible Provider 的异步 HTTP 日志客户端。"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)

_SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "api-key",
    "x-api-key",
}
_wire_log_locks: dict[Path, threading.Lock] = {}
_wire_log_locks_guard = threading.Lock()


class AsyncProviderHttpClient(httpx.AsyncClient):
    """记录外部 Provider 最终请求和响应的异步 httpx 客户端。

    模型服务只走 ``ChatOpenAI.ainvoke``，因此这里不再同时维护同步客户端。
    ``send`` 位于 SDK 的最终传输边界，能记录重试后真正发出的每个请求。
    """

    def __init__(
        self,
        *,
        wire_log_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化可选的 JSONL wire 日志。

        Args:
            wire_log_path (str | Path | None): 日志文件路径；为空时不记录 wire 内容。
            **kwargs (Any): 继续传给 ``httpx.AsyncClient`` 的连接配置。

        Note:
            日志会脱敏敏感 header，但可能包含 prompt、文档和模型输出；生产环境应
            仅在明确需要诊断时启用。
        """

        super().__init__(**kwargs)
        self._wire_log = (
            _ProviderWireLogWriter(wire_log_path) if wire_log_path is not None else None
        )

    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        """异步发送 Provider 请求并记录 wire 日志。"""

        started_at = time.perf_counter()
        request_body = _read_request_body(request)
        request_id = uuid4().hex
        try:
            response = await super().send(request, stream=stream, **kwargs)
        except Exception as exception:
            self._write(
                request_id=request_id,
                request=request,
                request_body=request_body,
                response=None,
                started_at=started_at,
                error={"type": type(exception).__name__, "message": str(exception)},
            )
            raise

        response_body: bytes | None = None
        if not stream:
            try:
                response_body = await response.aread()
            except Exception:
                # 记录失败不能改变 SDK 对响应的处理结果。
                response_body = None
        self._write(
            request_id=request_id,
            request=request,
            request_body=request_body,
            response=response,
            response_body=response_body,
            started_at=started_at,
        )
        return response

    def _write(
        self,
        *,
        request_id: str,
        request: httpx.Request,
        request_body: bytes | None,
        response: httpx.Response | None,
        started_at: float,
        response_body: bytes | None = None,
        error: dict[str, str] | None = None,
    ) -> None:
        """写入一次调用的 JSONL 记录；诊断失败不影响业务请求。"""

        if self._wire_log is None:
            return
        self._wire_log.write(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "provider": request.url.host,
                "method": request.method,
                "url": str(request.url),
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                "request": {
                    "headers": _headers_for_log(request.headers),
                    "body": _body_for_log(request_body),
                },
                "response": (
                    None
                    if response is None
                    else {
                        "status_code": response.status_code,
                        "headers": _headers_for_log(response.headers),
                        "body": _body_for_log(response_body),
                    }
                ),
                "error": error,
            }
        )


def build_provider_async_http_client(
    *,
    wire_log_path: str | Path | None,
    trust_env: bool = True,
) -> AsyncProviderHttpClient:
    """创建供 ``ChatOpenAI.ainvoke`` 使用的异步 Provider 客户端。"""

    return AsyncProviderHttpClient(
        wire_log_path=wire_log_path,
        trust_env=trust_env,
    )


def _read_request_body(request: httpx.Request) -> bytes | None:
    """读取已由 SDK 序列化的请求正文；读取失败不阻断业务调用。"""

    try:
        return request.read()
    except Exception:
        return None


def _headers_for_log(headers: httpx.Headers) -> dict[str, str]:
    """记录 header，但不把 API key 写进日志。"""

    return {
        key.lower(): "**REDACTED**" if key.lower() in _SENSITIVE_HEADERS else value
        for key, value in headers.items()
    }


def _body_for_log(body: bytes | None) -> str | dict[str, str] | None:
    """以完整 UTF-8 文本记录 JSON；非文本响应使用无损 Base64。"""

    if body is None:
        return None
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "encoding": "base64",
            "data": base64.b64encode(body).decode("ascii"),
        }


class _ProviderWireLogWriter:
    """线程安全地追加 Provider wire JSONL。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = _lock_for_wire_log(self._path)

    def write(self, record: dict[str, object]) -> None:
        """追加一行 JSON；日志文件异常只记录到 logger。"""

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.write("\n")
        except OSError:
            logger.exception("Could not write Provider wire log: %s", self._path)


def _lock_for_wire_log(path: Path) -> threading.Lock:
    """为同一进程内写入同一个 Provider 日志文件复用锁。"""

    normalized = path.resolve()
    with _wire_log_locks_guard:
        lock = _wire_log_locks.get(normalized)
        if lock is None:
            lock = threading.Lock()
            _wire_log_locks[normalized] = lock
        return lock


__all__ = ["AsyncProviderHttpClient", "build_provider_async_http_client"]
