"""配置驱动的 MCP Server 注册信息。

配置只选择内置的 transport/Server 命令，不允许 YAML 指定 Python import path。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.infrastructure.environment import load_dotenv_environment

_ENV_VALUE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


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
    """解析后的 MCP Server 配置集合。"""

    servers: tuple[McpServerConfig, ...]


def load_mcp_configuration(config_path: str | Path) -> McpConfiguration:
    """读取并严格校验应用自己的 MCP 配置。

    Args:
        config_path (str | Path): MCP YAML 文件路径。

    Returns:
        McpConfiguration: 展开环境变量、规范化 transport 并校验后的配置集合。

    Raises:
        RuntimeError: 配置文件无法读取时抛出。
        ValueError: YAML 无效、顶层结构错误、字段非法或 Server ID 重复时抛出。

    Note:
        ``command`` 只接受配置层预先允许的可执行入口，配置本身不会动态导入
        Python 模块；这样可以让 MCP 连接边界保持可审计。变量展开读取
        ``config/env`` 下的分组文件，进程环境拥有最高优先级。
    """

    path = Path(config_path)
    try:
        raw = yaml.safe_load(
            _expand_environment(
                path.read_text(encoding="utf-8"),
                environment=_load_environment_files(),
            )
        )
    except OSError as exc:
        raise RuntimeError(f"Could not read MCP config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid MCP config YAML: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("MCP config must be an object")

    raw_servers = raw.get("mcp_servers")
    if not isinstance(raw_servers, list):
        raise ValueError("MCP config must contain an mcp_servers list")
    servers = tuple(_server_config(item, index) for index, item in enumerate(raw_servers))
    ids = [server.id for server in servers]
    if len(set(ids)) != len(ids):
        raise ValueError("MCP config contains duplicate server ids")
    return McpConfiguration(servers=servers)


def _load_environment_files() -> dict[str, str]:
    """按 Settings 相同的优先级合并 dotenv 文件和进程环境。

    Returns:
        dict[str, str]: 所有分组 dotenv 文件和进程环境的合并结果。

    Note:
        ``BaseSettings`` 会读取 dotenv 文件，但不会把未声明的字段写回
        ``os.environ``。MCP YAML 可以包含第三方 Server 自己的认证变量，
        因此这里显式读取 dotenv；真正的进程环境仍拥有最高优先级。
    """

    return load_dotenv_environment()


def _expand_environment(
    text: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """展开 YAML 中的 ``${NAME}`` 和 ``${NAME:-default}`` 占位符。

    Args:
        text (str): 尚未解析的 YAML 文本。

    Returns:
        str: 替换环境变量后的 YAML 文本。

    Raises:
        ValueError: 占位符没有环境变量且没有默认值时抛出。
    """

    source = environment if environment is not None else _load_environment_files()

    def replace(match: re.Match[str]) -> str:
        """返回环境变量值，并在缺失时使用配置中的默认值。

        Args:
            match (re.Match[str]): 正则表达式匹配到的占位符；变量名位于第一个
                捕获组，``:-`` 后的默认值位于第二个捕获组。

        Returns:
            str: 环境变量的非空值，或占位符声明的默认值。

        Raises:
            ValueError: 环境变量为空且占位符没有默认值时抛出，避免把未展开的
                ``${NAME}`` 静默写入 MCP 连接配置。
        """

        value = source.get(match.group(1))
        if value is not None and value != "":
            return value
        default = match.group(2)
        if default is not None:
            return default
        raise ValueError(f"MCP config references unset environment variable: {match.group(1)}")

    return _ENV_VALUE.sub(replace, text)


def _server_config(raw: Any, index: int) -> McpServerConfig:
    """校验并构造列表中的单个 MCP Server 配置。

    Args:
        raw (Any): YAML 列表中的原始对象。
        index (int): 对外错误消息使用的列表下标。

    Returns:
        McpServerConfig: 规范化后的 Server 配置。

    Raises:
        ValueError: 字段缺失、类型错误、传输类型不支持或 URL/超时非法时抛出。
    """

    if not isinstance(raw, dict):
        raise ValueError(f"MCP server at index {index} must be an object")
    server_id = _required_string(raw, "id", index)
    description = _required_string(raw, "description", index)
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"MCP server '{server_id}' enabled must be a boolean")
    transport = _required_string(raw, "transport", index).casefold()
    if transport not in {"stdio", "http", "streamable-http", "in-memory", "memory"}:
        raise ValueError(f"MCP server '{server_id}' uses unsupported transport: {transport}")
    if transport in {"in-memory", "memory"}:
        transport = "in-memory"
    elif transport == "streamable-http":
        transport = "http"
    command = raw.get("command", "")
    if transport == "stdio":
        command = _required_string(raw, "command", index)
    elif not isinstance(command, str):
        raise ValueError(f"MCP server '{server_id}' command must be a string")
    args = raw.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError(f"MCP server '{server_id}' args must be a string list")
    timeout = raw.get("timeout_seconds", 30.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"MCP server '{server_id}' timeout_seconds must be positive")
    url = raw.get("url")
    if transport == "http":
        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                f"MCP server '{server_id}' requires a non-empty url for HTTP transport"
            )
        url = url.strip()
    elif url is not None and not isinstance(url, str):
        raise ValueError(f"MCP server '{server_id}' url must be a string")
    headers = _headers(raw.get("headers", {}), server_id, transport)
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
    server_id: str,
    transport: str,
) -> tuple[tuple[str, str], ...]:
    """校验 HTTP MCP 自定义 Header，并冻结为稳定的键值序列。"""

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"MCP server '{server_id}' headers must be an object")
    if transport != "http" and raw:
        raise ValueError(
            f"MCP server '{server_id}' headers are only supported for HTTP transport"
        )

    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"MCP server '{server_id}' header names must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"MCP server '{server_id}' header values must be strings")
        header_name = name.strip()
        header_value = value
        if "\r" in header_name or "\n" in header_name:
            raise ValueError(f"MCP server '{server_id}' header name contains a newline")
        if "\r" in header_value or "\n" in header_value:
            raise ValueError(f"MCP server '{server_id}' header value contains a newline")
        folded_name = header_name.casefold()
        if folded_name in seen:
            raise ValueError(f"MCP server '{server_id}' contains duplicate header: {header_name}")
        if folded_name in _PROTOCOL_HEADERS:
            raise ValueError(
                f"MCP server '{server_id}' header is controlled by the MCP client: {header_name}"
            )
        seen.add(folded_name)
        normalized.append((header_name, header_value))
    return tuple(normalized)


def _required_string(raw: dict[str, Any], key: str, index: int) -> str:
    """读取一个必须存在且非空的字符串配置字段。

    Args:
        raw (dict[str, Any]): 单个 Server 的原始配置。
        key (str): 要读取的字段名。
        index (int): 配置列表中的下标。

    Returns:
        str: 去除首尾空白后的字段值。

    Raises:
        ValueError: 字段不是非空字符串时抛出。
    """

    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MCP server at index {index} requires non-empty '{key}'")
    return value.strip()


__all__ = ["McpConfiguration", "McpServerConfig", "load_mcp_configuration"]
