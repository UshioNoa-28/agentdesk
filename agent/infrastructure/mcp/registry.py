"""Application-scoped MCP registry backed directly by the official MCP SDK."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import Implementation
from rank_bm25 import BM25Plus

from agent.domain.tools import ToolDefinition, ToolResult
from agent.infrastructure.mcp.config import McpServerConfig
from agent.infrastructure.search import tokenize
from agent.ports.tools import McpRegistryPort

logger = logging.getLogger(__name__)


class McpServerState(StrEnum):
    """MCP Server 的生命周期状态。"""

    CONFIGURED = "configured"
    CONNECTED = "connected"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class McpServerStatus:
    """面向状态 API 的 MCP Server 快照。"""

    id: str
    description: str
    state: McpServerState
    error: str | None
    tool_count: int


@dataclass(frozen=True, slots=True)
class _McpToolSpec:
    """从 ``tools/list`` 响应解析出的远程工具定义。"""

    name: str
    description: str
    input_schema: Mapping[str, object]


class _McpConnection:
    """在一个 owner task 内持有官方 transport 和 ClientSession。

    官方 SDK 的 AnyIO cancel scope 要求进入和退出发生在同一个 task；这里的
    owner task 只负责持有上下文，实际工具调用仍直接发生在应用 event loop。
    """

    def __init__(self, config: McpServerConfig, server: Server | None = None) -> None:
        self._config = config
        self._server = server
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Event | None = None
        self._close_requested: asyncio.Event | None = None
        self.session: ClientSession | None = None
        self.tool_specs: tuple[_McpToolSpec, ...] = ()
        self._startup_error: Exception | None = None
        self._terminal_error: Exception | None = None

    async def start(self) -> None:
        """启动 owner task，并等待握手和工具发现完成。"""

        self._ready = asyncio.Event()
        self._close_requested = asyncio.Event()
        self._startup_error = None
        self._terminal_error = None
        self.tool_specs = ()
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self._config.id}")
        try:
            await asyncio.wait_for(self._ready.wait(), self._config.timeout_seconds)
        except TimeoutError as exc:
            await self.close()
            raise RuntimeError(
                f"MCP server '{self._config.id}' handshake timed out after "
                f"{self._config.timeout_seconds:g}s"
            ) from exc
        if self._startup_error is not None:
            error = self._startup_error
            await self.close()
            raise RuntimeError(
                f"Could not connect MCP server '{self._config.id}': {error}"
            ) from error
        if self._task.done() or self.session is None:
            await self.close()
            raise RuntimeError(f"MCP server '{self._config.id}' stopped during startup")

    async def close(self) -> None:
        """请求 owner task 退出，让官方上下文在原 task 中清理。"""

        task = self._task
        if task is None or task is asyncio.current_task():
            return
        if self._close_requested is not None:
            self._close_requested.set()
        if self._ready is not None and not self._ready.is_set() and not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self.session = None

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, object],
    ) -> object:
        """在当前应用 event loop 中调用已打开的官方 Session。"""

        if self._task is None or self._task.done() or self.session is None:
            reason = f": {self._terminal_error}" if self._terminal_error else ""
            raise RuntimeError(f"MCP server '{self._config.id}' is not connected{reason}")
        if self._close_requested is not None and self._close_requested.is_set():
            raise RuntimeError(f"MCP server '{self._config.id}' is closing")
        return await self.session.call_tool(
            name,
            dict(arguments),
            read_timeout_seconds=timedelta(seconds=self._config.timeout_seconds),
        )

    async def _run(self) -> None:
        try:
            if self._config.transport == "in-memory":
                if self._server is None:
                    raise RuntimeError(f"In-memory MCP server '{self._config.id}' was not provided")
                async with create_connected_server_and_client_session(self._server) as session:
                    self.session = session
                    self.tool_specs = await _discover_tools(session, self._config.id)
                    assert self._ready is not None and self._close_requested is not None
                    self._ready.set()
                    await self._close_requested.wait()
            else:
                async with _transport_context(self._config) as streams:
                    read_stream, write_stream = streams[:2]
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=self._config.timeout_seconds),
                        client_info=Implementation(name="agentdesk", version="0.1.0"),
                    ) as session:
                        self.session = session
                        await session.initialize()
                        self.tool_specs = await _discover_tools(session, self._config.id)
                        assert self._ready is not None and self._close_requested is not None
                        self._ready.set()
                        await self._close_requested.wait()
        except asyncio.CancelledError:
            if self._ready is not None and not self._ready.is_set():
                self._startup_error = RuntimeError("MCP connection was cancelled during startup")
            raise
        except Exception as exc:
            self._terminal_error = exc
            if self._ready is not None and not self._ready.is_set():
                self._startup_error = exc
        finally:
            if self._ready is not None:
                self._ready.set()
            self.session = None


class McpRegistry(McpRegistryPort):
    """应用级 MCP 连接、工具 schema 和状态注册表。

    Registry 不管理业务 Session ID。HTTP MCP 所需的协议 session 由官方
    ``streamable_http_client`` 在 transport 内部维护。
    """

    def __init__(
        self,
        servers: Sequence[McpServerConfig],
        *,
        in_memory_servers: Mapping[str, Server] | None = None,
    ) -> None:
        ids = [server.id for server in servers]
        if len(set(ids)) != len(ids):
            raise ValueError("MCP registry contains duplicate server ids")
        self._servers = {server.id: server for server in servers}
        self._in_memory_servers = dict(in_memory_servers or {})
        self._connections: dict[str, _McpConnection] = {}
        self._tool_specs: dict[str, tuple[_McpToolSpec, ...]] = {}
        self._statuses = {
            server.id: McpServerStatus(
                server.id,
                server.description,
                McpServerState.CONFIGURED if server.enabled else McpServerState.DISABLED,
                None,
                0,
            )
            for server in servers
        }
        self._server_locks = {server.id: asyncio.Lock() for server in servers}

    def register_in_memory_server(
        self,
        server_id: str,
        server: Server,
    ) -> None:
        """为已配置的 in-memory MCP Server 绑定进程内 Server 实例。"""
        self._in_memory_servers[server_id] = server

    async def start_all_async(self) -> None:
        """按配置启动全部启用的 MCP Server。"""

        for server_id, config in self._servers.items():
            if config.enabled:
                await self._start_server(server_id, config)

    async def close_async(self) -> None:
        """关闭所有官方 transport，并清空连接和工具缓存。"""

        for server_id in self._servers:
            async with self._server_locks[server_id]:
                connection = self._connections.pop(server_id, None)
                self._tool_specs.pop(server_id, None)
                if connection is not None:
                    await connection.close()

    def server_descriptions(self) -> tuple[tuple[str, str], ...]:
        """返回启用 Server 的稳定路由描述。"""

        return tuple(
            (server.id, server.description)
            for server in self._servers.values()
            if server.enabled
        )

    def list_statuses(self) -> tuple[McpServerStatus, ...]:
        """返回所有配置项的状态快照。"""

        return tuple(self._statuses[server_id] for server_id in self._servers)

    async def retry_async(self, server_id: str) -> McpServerStatus:
        """关闭旧连接并重试指定 Server。"""

        config = self._servers.get(server_id)
        if config is None:
            raise ValueError(f"Unknown MCP server: {server_id}")
        if not config.enabled:
            return self._disabled_status(config)
        return await self._start_server(server_id, config)

    async def _start_server(self, server_id: str, config: McpServerConfig) -> McpServerStatus:
        async with self._server_locks[server_id]:
            old = self._connections.pop(server_id, None)
            self._tool_specs.pop(server_id, None)
            if old is not None:
                await old.close()
            if not config.enabled:
                self._set_disabled(server_id, config)
                return self._statuses[server_id]
            try:
                server = self._in_memory_servers.get(server_id)
                connection = await _open_connection(config, server=server)
            except Exception as exc:
                self._mark_failed(server_id, str(exc))
            else:
                self._connections[server_id] = connection
                self._tool_specs[server_id] = connection.tool_specs
                self._statuses[server_id] = self._connected_status(
                    config, connection.tool_specs
                )
            return self._statuses[server_id]

    def search_tools(self, *, mcp: str, query: str, limit: int) -> tuple[ToolDefinition, ...]:
        """在已连接 Server 的缓存 schema 中搜索工具。"""

        if mcp not in self._servers:
            raise ValueError(f"Unknown MCP server: {mcp}")
        status = self._statuses[mcp]
        if status.state != McpServerState.CONNECTED:
            reason = "disabled" if status.state == McpServerState.DISABLED else status.error
            raise ValueError(f"MCP server '{mcp}' is unavailable: {reason or 'not connected'}")
        return tuple(
            ToolDefinition(spec.name, spec.description, spec.input_schema)
            for spec in _match_tools(self._tool_specs.get(mcp, ()), query, limit)
        )

    async def execute_tool(
        self,
        *,
        mcp: str,
        tool: str,
        arguments: Mapping[str, object],
    ) -> ToolResult:
        """校验 MCP/工具身份后调用官方 ``ClientSession``。"""

        if mcp not in self._servers:
            return _tool_error("mcp_server_unavailable", f"Unknown MCP server '{mcp}'")
        async with self._server_locks[mcp]:
            connection = self._connections.get(mcp)
            status = self._statuses[mcp]
            known = any(spec.name == tool for spec in self._tool_specs.get(mcp, ()))
            if connection is None:
                reason = "disabled" if status.state == McpServerState.DISABLED else status.error
                return _tool_error(
                    "mcp_server_unavailable",
                    f"MCP server '{mcp}' is unavailable: {reason or 'not connected'}",
                )
            if not known:
                return _tool_error(
                    "mcp_tool_not_found",
                    f"MCP server '{mcp}' has no tool named '{tool}'",
                )
            try:
                raw_result = await connection.call_tool(tool, arguments)
            except McpError as exc:
                return _tool_error("mcp_rpc_error", str(exc))
            except Exception as exc:
                logger.exception("MCP tool call failed: server=%s tool=%s", mcp, tool)
                await connection.close()
                self._mark_failed(mcp, str(exc))
                return _tool_error("mcp_tool_call_failed", str(exc))
            content = _content_to_text(
                raw_result.content,
                getattr(raw_result, "structuredContent", None),
            )
            if getattr(raw_result, "isError", False):
                return _tool_error("mcp_tool_error", content)
            return ToolResult(content=content)

    @staticmethod
    def _connected_status(
        config: McpServerConfig,
        specs: Sequence[_McpToolSpec],
    ) -> McpServerStatus:
        return McpServerStatus(
            config.id,
            config.description,
            McpServerState.CONNECTED,
            None,
            len(specs),
        )

    @staticmethod
    def _disabled_status(config: McpServerConfig) -> McpServerStatus:
        return McpServerStatus(config.id, config.description, McpServerState.DISABLED, None, 0)

    def _set_disabled(self, server_id: str, config: McpServerConfig) -> None:
        self._connections.pop(server_id, None)
        self._tool_specs.pop(server_id, None)
        self._statuses[server_id] = self._disabled_status(config)

    def _mark_failed(self, server_id: str, error: str) -> None:
        config = self._servers[server_id]
        self._connections.pop(server_id, None)
        self._tool_specs.pop(server_id, None)
        self._statuses[server_id] = McpServerStatus(
            config.id,
            config.description,
            McpServerState.FAILED,
            error[:1000],
            0,
        )


async def _open_connection(
    config: McpServerConfig,
    server: Server | None = None,
) -> _McpConnection:
    """启动一个由 owner task 持有官方上下文的连接。"""

    connection = _McpConnection(config, server=server)
    await connection.start()
    return connection


@asynccontextmanager
async def _transport_context(config: McpServerConfig):
    """创建官方 stdio 或 streamable HTTP transport。"""

    if config.transport == "http":
        if not config.url:
            raise RuntimeError(f"MCP HTTP server '{config.id}' has no URL")
        async with httpx.AsyncClient(
            headers=dict(config.headers),
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=True,
            trust_env=_trust_environment_for_url(config.url),
        ) as client:
            async with streamable_http_client(config.url, http_client=client) as streams:
                yield streams
        return
    parameters = StdioServerParameters(
        command=config.command,
        args=list(config.args),
        env=dict(os.environ),
        cwd=os.getcwd(),
    )
    async with stdio_client(parameters) as streams:
        yield streams


async def _discover_tools(
    session: ClientSession,
    server_id: str,
) -> tuple[_McpToolSpec, ...]:
    """通过官方分页 API 发现并校验所有工具定义。"""

    raw_tools: list[dict[str, object]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        listed = await session.list_tools(cursor=cursor)
        raw_tools.extend(
            tool.model_dump(by_alias=True, mode="python", exclude_none=True)
            for tool in listed.tools
        )
        next_cursor = listed.nextCursor
        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            raise RuntimeError(f"MCP server '{server_id}' returned a repeated tools/list cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return _parse_tool_specs(raw_tools, server_id)


def _trust_environment_for_url(url: str) -> bool:
    """只为公网 MCP URL 信任代理环境变量。"""

    host = urlsplit(url).hostname
    if not host:
        return False
    normalized = host.casefold()
    if normalized in {"localhost", "host.docker.internal"} or "." not in normalized:
        return False
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return True
    return not (address.is_private or address.is_loopback or address.is_link_local)


def _match_tools(
    specs: Sequence[_McpToolSpec],
    query: str,
    limit: int,
) -> tuple[_McpToolSpec, ...]:
    terms = tokenize(query)
    if not terms:
        return tuple(specs[:limit])
    if not specs:
        return ()
    corpus = [tokenize(f"{spec.name} {spec.description}") for spec in specs]
    scores = BM25Plus(corpus).get_scores(terms)
    ranked = [
        (score, index, spec)
        for score, (index, spec) in zip(scores, enumerate(specs), strict=True)
        if score > 0
    ]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in ranked[:limit])


def _parse_tool_specs(raw_tools: object, server_id: str) -> tuple[_McpToolSpec, ...]:
    if not isinstance(raw_tools, list):
        raise RuntimeError("MCP tools/list result does not contain a tools list")
    specs: list[_McpToolSpec] = []
    names: set[str] = set()
    for item in raw_tools:
        if not isinstance(item, dict):
            raise RuntimeError(f"MCP server '{server_id}' returned an invalid tool definition")
        name = item.get("name")
        schema = item.get("inputSchema")
        if not isinstance(name, str) or not name.strip() or not isinstance(schema, dict):
            raise RuntimeError(f"MCP server '{server_id}' returned an invalid tool definition")
        if name in names:
            raise RuntimeError(f"MCP server '{server_id}' returned duplicate tool: {name}")
        names.add(name)
        specs.append(_McpToolSpec(name, str(item.get("description") or ""), dict(schema)))
    return tuple(specs)


def _content_to_text(
    content: object,
    structured_content: object = None,
) -> str:
    blocks: list[Mapping[str, object]] = []
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        for block in content:
            if hasattr(block, "model_dump"):
                value = block.model_dump(by_alias=True, mode="json", exclude_none=True)
            elif isinstance(block, Mapping):
                value = dict(block)
            else:
                value = {"type": "text", "text": str(block)}
            if isinstance(value, Mapping):
                blocks.append(dict(value))
    texts = [item.get("text", "") for item in blocks if item.get("type") == "text"]
    if texts:
        return "\n".join(str(text) for text in texts)
    if isinstance(structured_content, Mapping):
        return json.dumps(structured_content, ensure_ascii=False, default=str)
    return json.dumps(blocks, default=str)


def _tool_error(code: str, message: str) -> ToolResult:
    return ToolResult(
        content=json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        )
    )


__all__ = ["McpRegistry", "McpServerState", "McpServerStatus"]
