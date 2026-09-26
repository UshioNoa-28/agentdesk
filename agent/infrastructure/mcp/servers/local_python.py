"""未沙箱化的本地 Python 官方 MCP Server。

工具处理、输入校验和 MCP JSON-RPC/session 由官方 low-level Server 负责；本
模块只保留 Python 子进程执行策略和工具注册。timeout 和输出截断仅控制资源
消耗，不提供文件、网络、进程或权限隔离。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Mapping

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

_SERVER = Server(
    "anna-local-python",
    version="0.1.0",
    instructions=(
        "Run explicitly requested Python calculations in an unsandboxed local "
        "subprocess. Timeout and output limits are not security boundaries."
    ),
)

_TOOL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "minLength": 1,
            "description": "Python source code to run in the local subprocess.",
        },
        "timeout_seconds": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 60,
            "default": 10,
            "description": "Maximum subprocess runtime in seconds.",
        },
    },
    "required": ["code"],
    "additionalProperties": False,
}


@_SERVER.list_tools()
async def _list_tools() -> list[Tool]:
    """返回本地 Python 工具的官方 MCP schema。"""

    return [
        Tool(
            name="run_python",
            description=(
                "Run Python code in an unsandboxed local subprocess for calculations, "
                "parsing, and small data transformations."
            ),
            inputSchema=_TOOL_SCHEMA,
        )
    ]


@_SERVER.call_tool()
async def _call_tool(name: str, arguments: Mapping[str, object]) -> object:
    """执行官方 Server 分发的本地 Python 工具调用。"""

    if name != "run_python":
        raise ValueError(f"Unknown tool: {name}")
    return await _run_python_async(
        str(arguments.get("code", "")),
        float(arguments.get("timeout_seconds", 10)),
    )


async def _run_python_async(code: str, timeout_seconds: float = 10) -> object:
    """异步运行本地 Python 子进程，不占用事件循环线程。"""

    if not code.strip():
        raise ValueError("code must be a non-empty string")
    timeout = timeout_seconds
    if not 0.1 <= timeout <= 60:
        raise ValueError("timeout_seconds must be between 0.1 and 60")
    max_bytes = int(os.environ.get("LOCAL_PYTHON_MAX_OUTPUT_BYTES", "65536"))
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=float(timeout))
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise RuntimeError(f"Python execution exceeded {float(timeout):g}s") from exc
    stdout, stdout_truncated = _bounded(stdout, max_bytes)
    stderr, stderr_truncated = _bounded(stderr, max_bytes)
    return {
        "exit_code": process.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "sandboxed": False,
    }


def run_python(code: str, timeout_seconds: float = 10) -> object:
    """在未沙箱化的子进程中运行 Python 代码。

    Args:
        code (str): 要执行的 Python 源码。
        timeout_seconds (float): 子进程最长运行时间，范围为 0.1 到 60 秒。

    Returns:
        object: 包含退出码、stdout/stderr、截断标记和 ``sandboxed=False`` 的字典。

    Raises:
        ValueError: 代码为空或超时时间不在允许范围内。
        RuntimeError: 子进程超时。

    Warning:
        该工具继承 Agent 进程的文件、网络和进程权限；超时和输出限制不是安全边界。
    """

    if not code.strip():
        raise ValueError("code must be a non-empty string")
    timeout = timeout_seconds
    if not 0.1 <= timeout <= 60:
        raise ValueError("timeout_seconds must be between 0.1 and 60")
    max_bytes = int(os.environ.get("LOCAL_PYTHON_MAX_OUTPUT_BYTES", "65536"))
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=False,
            timeout=float(timeout),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Python execution exceeded {float(timeout):g}s") from exc
    stdout, stdout_truncated = _bounded(completed.stdout, max_bytes)
    stderr, stderr_truncated = _bounded(completed.stderr, max_bytes)
    return {
        "exit_code": completed.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "sandboxed": False,
    }


def _bounded(value: bytes, limit: int) -> tuple[bytes, bool]:
    """按字节上限截取子进程输出，并返回是否发生截断。"""

    return value[:limit], len(value) > limit


async def _run_server() -> None:
    """在官方 MCP Server 上运行兼容 stdio transport。"""

    async with stdio_server() as streams:
        await _SERVER.run(
            *streams,
            _SERVER.create_initialization_options(),
        )


def main() -> int:
    """启动 stdio MCP Server，并在客户端关闭管道后返回。"""

    anyio.run(_run_server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
