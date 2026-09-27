"""配置驱动的 MCP Server 注册信息。

配置文件为 ``.agent-desk/mcp.json``（条目格式沿用 Claude Code 的
``.mcp.json`` 映射），按 Skill 相同的层级规则发现：从 workspace 根逐级
向上直到用户主目录，越近优先；不同名
Server 跨层合并，同名整条由近的层级遮蔽远的层级（记 warning）。任何
层级都没有该文件即视为「未配置 MCP」，按零 Server 正常启动。

格式对齐 Claude Code 的 ``mcpServers`` 映射（key 即 Server 名），但保留
本项目的设计：``description`` 必填（供 ``search_mcp`` 路由），``enabled``
可选；超时字段用 ``timeout_seconds``（秒）。文件按严格 JSON 解析，顶层与
条目级未识别的键均视为非法。任何一层文件读取、解析或校验失败都不抛错：
该层记 warning 后整体跳过（宁缺勿炸），其余层照常合并。配置只选择
内置的 transport/Server 命令，不允许指定任意 Python import path。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.infrastructure.environment import expand_environment_values, load_dotenv_environment
from agent.infrastructure.project_settings import AGENTDESK_DIR_NAME, config_search_roots

logger = logging.getLogger(__name__)

MCP_FILE_NAME = "mcp.json"
MAX_DESCRIPTION_CHARS = 4000

_ENTRY_FIELDS = frozenset(
    {
        "type",
        "command",
        "args",
        "url",
        "headers",
        "timeout_seconds",
        "description",
        "enabled",
    }
)

@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """一个 MCP Server 的静态连接配置。

    ``stdio`` 使用 command/args 启动子进程，``http`` 使用 url 建立 HTTP
    请求；``enabled`` 只控制是否启动和是否出现在模型路由描述中。
    """

    id: str
    description: str
    command: str
    args: tuple[str, ...]
    timeout_seconds: float = 30.0
    transport: str = "stdio"
    url: str | None = None
    # 禁用项仍出现在状态 API 中，但不会建立连接或展示给模型；字段放在最后，
    # 以保持既有集成和测试使用的位置参数构造兼容性。
    enabled: bool = True
    # HTTP MCP 的认证/租户 Header；使用 tuple 避免 frozen dataclass 内嵌可变 dict。
    headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class McpConfiguration:
    """解析后的 MCP Server 配置集合，按发现优先级（由近到远）排列。"""

    servers: tuple[McpServerConfig, ...]


def discover_mcp_files(
    start: Path,
    *,
    home: Path | None = None,
) -> tuple[Path, ...]:
    """返回按优先级排列的 ``.agent-desk/mcp.json`` 文件；一个都没有返回空元组。"""

    roots = config_search_roots(start.expanduser().resolve(strict=False), home=home)
    found: list[Path] = []
    for root in roots:
        candidate = root / AGENTDESK_DIR_NAME / MCP_FILE_NAME
        if not candidate.is_file():
            if candidate.exists():
                logger.warning("Ignoring MCP config path that is not a file: %s", candidate)
            continue
        found.append(candidate)
    return tuple(found)


def load_mcp_configuration(
    start: str | Path,
    *,
    home: Path | None = None,
) -> McpConfiguration:
    """发现并加载 ``.agent-desk/mcp.json``，跨层级按 Server 名合并（近者优先）。

    未配置不是错误：没有任何文件、或某一层文件坏掉时，只取能完整通过
    校验的层（坏层记 warning 后整体跳过），最坏返回零 Server，不抛异常。
    """

    merged: dict[str, McpServerConfig] = {}
    winners: dict[str, Path] = {}
    for path in discover_mcp_files(Path(start), home=home):
        try:
            entries = _read_file(path)
            configs = {
                server_id: _server_config(raw, server_id, path)
                for server_id, raw in entries.items()
            }
        except ValueError as exc:
            logger.warning("Ignoring MCP config file %s: %s", path, exc)
            continue
        for server_id, config in configs.items():
            if server_id in merged:
                logger.warning(
                    "MCP server '%s' shadowed by a nearer copy: %s (ignored: %s)",
                    server_id,
                    winners[server_id],
                    path,
                )
                continue
            merged[server_id] = config
            winners[server_id] = path
    return McpConfiguration(servers=tuple(merged.values()))


def _read_file(path: Path) -> dict[str, Any]:
    """读取单个 ``.agent-desk/mcp.json``，展开环境变量并校验顶层结构。"""

    try:
        raw = json.loads(
            _expand_environment(
                path.read_text(encoding="utf-8"),
                environment=_load_environment_files(),
            )
        )
    except OSError as exc:
        raise ValueError(f"Could not read MCP config file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MCP config JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"MCP config must be an object: {path}")
    if not set(raw) == {"mcpServers"}:
        raise ValueError(
            f"MCP config must contain exactly one 'mcpServers' object "
            f"(found: {sorted(raw) or 'nothing'}): {path}"
        )
    raw_servers = raw["mcpServers"]
    if not isinstance(raw_servers, dict):
        raise ValueError(f"MCP config 'mcpServers' must be an object: {path}")
    for server_id in raw_servers:
        if not isinstance(server_id, str) or not server_id.strip():
            raise ValueError(f"MCP server names must be non-empty strings: {path}")
    return {str(server_id).strip(): value for server_id, value in raw_servers.items()}


def _load_environment_files() -> dict[str, str]:
    """按 Settings 相同的优先级合并 dotenv 文件和进程环境。

    Returns:
        dict[str, str]: 所有分组 dotenv 文件和进程环境的合并结果。

    Note:
        ``BaseSettings`` 会读取 dotenv 文件，但不会把未声明的字段写回
        ``os.environ``。MCP 配置可以包含第三方 Server 自己的认证变量，
        因此这里显式读取 dotenv；真正的进程环境仍拥有最高优先级。
    """

    return load_dotenv_environment()


def _expand_environment(
    text: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """展开 ``${NAME}`` 和 ``${NAME:-default}`` 占位符（共享实现见 environment）。"""

    return expand_environment_values(
        text,
        environment=environment if environment is not None else _load_environment_files(),
    )


def _server_config(raw: Any, server_id: str, path: Path) -> McpServerConfig:
    """校验并构造单个 MCP Server 配置；错误信息带 server 名与文件路径。"""

    location = f"{server_id!r} in {path}"
    if not isinstance(raw, dict):
        raise ValueError(f"MCP server {location} must be an object")
    unknown = sorted(str(key) for key in raw if key not in _ENTRY_FIELDS)
    if unknown:
        raise ValueError(
            f"MCP server {location} has unsupported fields: {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(_ENTRY_FIELDS))}"
        )

    description = _required_string(raw, "description", location)
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(
            f"MCP server {location} description must not exceed "
            f"{MAX_DESCRIPTION_CHARS} characters"
        )
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"MCP server {location} enabled must be a boolean")

    transport = _string(raw, "type", "stdio", location).casefold()
    if transport in {"in-memory", "memory"}:
        transport = "in-memory"
    elif transport == "streamable-http":
        transport = "http"
    elif transport not in {"stdio", "http"}:
        raise ValueError(f"MCP server {location} uses unsupported type: {transport}")

    command = raw.get("command", "")
    if transport == "stdio":
        command = _required_string(raw, "command", location)
    elif not isinstance(command, str):
        raise ValueError(f"MCP server {location} command must be a string")
    args = raw.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError(f"MCP server {location} args must be a string list")
    timeout = raw.get("timeout_seconds", 30.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"MCP server {location} timeout_seconds must be positive")
    url = raw.get("url")
    if transport == "http":
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"MCP server {location} requires a non-empty url for HTTP")
        url = url.strip()
    elif url is not None and not isinstance(url, str):
        raise ValueError(f"MCP server {location} url must be a string")
    headers = _headers(raw.get("headers", {}), location, transport)
    return McpServerConfig(
        id=server_id,
        description=description,
        command=command,
        args=tuple(args),
        timeout_seconds=float(timeout),
        transport=transport,
        url=url,
        enabled=enabled,
        headers=headers,
    )


_PROTOCOL_HEADERS = frozenset(
    {
        "accept",
        "content-type",
        "content-length",
        "host",
        "mcp-protocol-version",
        "mcp-session-id",
    }
)


def _headers(
    raw: Any,
    location: str,
    transport: str,
) -> tuple[tuple[str, str], ...]:
    """校验 HTTP MCP 自定义 Header，并冻结为稳定的键值序列。"""

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"MCP server {location} headers must be an object")
    if transport != "http" and raw:
        raise ValueError(f"MCP server {location} headers are only supported for HTTP")

    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"MCP server {location} header names must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"MCP server {location} header values must be strings")
        header_name = name.strip()
        header_value = value
        if "\r" in header_name or "\n" in header_name:
            raise ValueError(f"MCP server {location} header name contains a newline")
        if "\r" in header_value or "\n" in header_value:
            raise ValueError(f"MCP server {location} header value contains a newline")
        folded_name = header_name.casefold()
        if folded_name in seen:
            raise ValueError(f"MCP server {location} contains duplicate header: {header_name}")
        if folded_name in _PROTOCOL_HEADERS:
            raise ValueError(
                f"MCP server {location} header is controlled by the MCP client: {header_name}"
            )
        seen.add(folded_name)
        normalized.append((header_name, header_value))
    return tuple(normalized)


def _string(raw: dict[str, Any], key: str, default: str, location: str) -> str:
    """读取可选字符串字段；缺失或空白时返回默认值。"""

    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MCP server {location} field '{key}' must be a non-empty string")
    return value.strip()


def _required_string(raw: dict[str, Any], key: str, location: str) -> str:
    """读取一个必须存在且非空的字符串配置字段。"""

    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MCP server {location} requires non-empty '{key}'")
    return value.strip()


__all__ = [
    "MCP_FILE_NAME",
    "McpConfiguration",
    "McpServerConfig",
    "discover_mcp_files",
    "load_mcp_configuration",
]
