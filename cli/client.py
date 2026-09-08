"""HTTP API Client for communication with the Agent FastAPI backend."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
import httpx


class AgentApiError(Exception):
    """API 请求异常。"""

    def __init__(self, message: str, status_code: int | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class AgentApiClient:
    """与 Agent 后端 REST API 交互的客户端。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/api", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        # trust_env=False prevents system HTTP_PROXY / ALL_PROXY from intercepting localhost API calls
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout=timeout, connect=5.0),
            trust_env=False,
        )
        self._active_response: httpx.Response | None = None

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise AgentApiError(f"请求执行超时 ({self.base_url})：任务执行时间过长，可能正在等待子代理响应或上游模型触发限流重试。") from exc
        except httpx.ConnectError as exc:
            raise AgentApiError(f"无法连接到 Agent 服务 ({self.base_url})：请确认后端服务是否正在运行（例如运行 docker compose up -d）。") from exc
        except httpx.RequestError as exc:
            raise AgentApiError(f"网络通信异常 ({self.base_url}): {exc}") from exc

        if response.status_code == 204:
            return None

        if response.status_code == 202:
            try:
                return response.json()
            except Exception:
                return {"status": "queued"}

        if not response.is_success:
            if not response.content:
                phrase = getattr(response, "reason_phrase", "") or "服务未正常响应"
                raise AgentApiError(
                    f"HTTP {response.status_code}: {phrase} ({self.base_url})",
                    status_code=response.status_code,
                )
            try:
                data = response.json()
            except Exception:
                raise AgentApiError(
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                )
            if response.status_code == 409:
                raise AgentApiError(
                    "会话当前正以另一种模式（同步/流式）运行，请等待当前轮次结束后再发送。",
                    status_code=409,
                    details=data,
                )
            err_detail = data.get("detail")
            if isinstance(data.get("error"), dict):
                err_detail = data["error"].get("message", err_detail)
            err_msg = err_detail or data.get("message") or str(data)
            raise AgentApiError(f"{err_msg}", status_code=response.status_code, details=data)

        if not response.content:
            return None

        try:
            return response.json()
        except Exception:
            raise AgentApiError(f"解析响应 JSON 失败 (HTTP {response.status_code})")

    def health(self) -> dict[str, Any]:
        """检查 Agent 服务 liveness。"""
        return self._request("GET", "/health")

    def list_sessions(self) -> list[dict[str, Any]]:
        """获取所有会话列表。"""
        return self._request("GET", "/sessions")

    def get_session(self, session_id: str) -> dict[str, Any]:
        """获取单个会话详情。"""
        return self._request("GET", f"/sessions/{session_id}")

    def create_session(self, title: str | None = None, allowed_tools: list[str] | None = None) -> dict[str, Any]:
        """创建新主会话（子代理会话仅由 Agent 内部 define_subagent 工具创建）。"""
        from datetime import datetime
        clean_title = title.strip() if title and title.strip() else f"Chat {datetime.now().strftime('%m%d-%H%M%S')}"
        payload: dict[str, Any] = {"title": clean_title}
        if allowed_tools is not None:
            payload["allowed_tools"] = allowed_tools
        return self._request("POST", "/sessions", json=payload)

    def rename_session(self, session_id: str, title: str) -> dict[str, Any]:
        """重命名会话。"""
        return self._request("PATCH", f"/sessions/{session_id}", json={"title": title})

    def resume_session(
        self,
        session_id: str,
        title: str | None = None,
        parent_last_seq: int | None = None,
    ) -> dict[str, Any]:
        """恢复/分叉会话（支持指定 parent_last_seq 截断）。"""
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if parent_last_seq is not None:
            payload["parent_last_seq"] = parent_last_seq
        return self._request("POST", f"/sessions/{session_id}/resume", json=payload if payload else None)

    def delete_session(self, session_id: str) -> None:
        """删除会话。"""
        self._request("DELETE", f"/sessions/{session_id}")

    def compact_session(self, session_id: str) -> dict[str, Any]:
        """手动触发会话上下文压缩。"""
        return self._request("POST", f"/sessions/{session_id}/compact")

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        """获取会话历史消息。"""
        return self._request("GET", f"/sessions/{session_id}/messages")

    def ask(self, session_id: str, question: str) -> dict[str, Any]:
        """向 Agent 发送提问并获取回复。"""
        return self._request("POST", f"/sessions/{session_id}/messages", json={"question": question})

    def abort_stream(self) -> None:
        """Abort any active streaming HTTP request."""
        resp = self._active_response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def ask_stream(self, session_id: str, question: str) -> Iterator[dict[str, Any]]:
        """流式提问：逐个产出 {"event": 事件名, "data": 载荷} 的 SSE 事件。"""
        try:
            with self._client.stream(
                "POST",
                f"/sessions/{session_id}/messages/stream",
                json={"question": question},
                headers={"Accept": "text/event-stream"},
            ) as response:
                self._active_response = response
                if response.status_code == 202:
                    body = response.read()
                    try:
                        data = json.loads(body)
                    except Exception:
                        data = {"session_id": session_id, "status": "queued"}
                    yield {"event": "queued", "data": data}
                    return
                elif response.status_code != 200:
                    body = response.read()
                    self._raise_for_error(response.status_code, body)
                event: str | None = None
                for line in response.iter_lines():
                    if not line:
                        event = None
                        continue
                    if line.startswith("event:"):
                        event = line[len("event:"):].strip()
                    elif line.startswith("data:") and event is not None:
                        raw = line[len("data:"):].strip()
                        try:
                            data: Any = json.loads(raw) if raw else {}
                        except json.JSONDecodeError:
                            data = {"raw": raw}
                        yield {"event": event, "data": data}
                        event = None
        except httpx.TimeoutException as exc:
            raise AgentApiError(f"请求执行超时 ({self.base_url})：任务执行时间过长，可能正在等待子代理响应或上游模型触发限流重试。") from exc
        except httpx.ConnectError as exc:
            raise AgentApiError(f"无法连接到 Agent 服务 ({self.base_url})：请确认后端服务是否正在运行。") from exc
        except httpx.RequestError as exc:
            raise AgentApiError(f"网络通信异常 ({self.base_url}): {exc}") from exc
        finally:
            self._active_response = None

    def _raise_for_error(self, status_code: int, body: bytes) -> None:
        """把非 2xx SSE 响应转成统一的 AgentApiError。"""
        try:
            data = json.loads(body)
        except Exception:
            raise AgentApiError(f"HTTP {status_code}: {body[:200].decode('utf-8', 'replace')}", status_code=status_code)
        if status_code == 409:
            raise AgentApiError(
                "会话当前正以另一种模式（同步/流式）运行，请等待当前轮次结束后再试。",
                status_code=409,
                details=data,
            )
        err = data.get("error")
        message = err.get("message") if isinstance(err, dict) else None
        message = message or data.get("detail") or data.get("message") or str(data)
        raise AgentApiError(f"{message}", status_code=status_code, details=data)

    def list_mcps(self) -> list[dict[str, Any]]:
        """获取所有 MCP 服务状态。"""
        return self._request("GET", "/mcps")

    def retry_mcp(self, server_id: str) -> dict[str, Any]:
        """重试指定 MCP 服务。"""
        return self._request("POST", f"/mcps/{server_id}/retry", json={})

    def list_skills(self) -> list[dict[str, Any]]:
        """获取所有 Skill 列表。"""
        return self._request("GET", "/skills")
