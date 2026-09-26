"""固定 ``bash`` Meta Tool 与其权限拆段。

拆段判断标准是 AST 结构而非字符串猜测：能拆干净就逐段核验，拆不干净
（``command_targets`` 返回空元组）时工具层把**整条命令原样作为唯一一段**
返回——scoped 规则匹配不上一整条链式命令，必然落入询问，且 CLI 展示与
"always allow 落什么 scope"都自然成立。字符串本身分不出"命令还是路径"
（bash 语法里路径是合法的单词命令），所以拆段决策只属于知道自己领域的
那个工具。
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections import deque
from collections.abc import Mapping
from typing import Any

import bashlex

from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)

_MAX_OUTPUT_BYTES = 65_536
# 每路只保留头部与尾部各一半，溢出的字节继续排空但只计数不存储：
# 内存峰值恒定，且子进程永远不会因 pipe 写满而阻塞。
_OUTPUT_HEAD_BYTES = _MAX_OUTPUT_BYTES // 2
_OUTPUT_TAIL_BYTES = _MAX_OUTPUT_BYTES - _OUTPUT_HEAD_BYTES
_READ_CHUNK_BYTES = _MAX_OUTPUT_BYTES
_DEFAULT_TIMEOUT_SECONDS = 60.0
_MAX_TIMEOUT_SECONDS = 600.0

# 拆段容器：list 用 ; && || 分隔，pipeline 用 | 分隔，二者内部的命令才是段。
_CONTAINER_KINDS = frozenset({"list", "pipeline"})
_SEPARATOR_KINDS = frozenset({"operator", "pipe"})

# 引号剥离后的词里出现这些字符，意味着存在展开或 glob——运行时行为依赖
# 文件系统/环境变量，规则匹配的是用户书写的原文，两边对不上，拒绝给段。
# bashlex 不展开 ``{a,b}`` 也不留 AST 痕迹，只能对"完全无引号的词"保守拦截。
_UNSAFE_CHARS = frozenset("$`*?[")


class _OpaqueCommand(Exception):
    """命令行含无法逐段核验的构造。"""


def command_targets(command: object) -> tuple[str, ...]:
    """返回参与权限匹配的命令段；无法给出干净分段时返回空元组。

    每段是引号剥离后的 argv 以单空格 rejoin（``git commit -m "a b"`` →
    ``git commit -m a b``）。重定向、赋值前缀、命令/参数替换、glob、brace
    展开、compound 语句（if/while/…）及一切解析失败都返回 ``()``。
    """

    if not isinstance(command, str) or not command.strip():
        return ()
    try:
        nodes = bashlex.parse(command)
    except Exception:
        return ()
    segments: list[str] = []
    try:
        for node in nodes:
            _collect(node, command, segments)
    except _OpaqueCommand:
        return ()
    return tuple(segments)


def _collect(node: Any, source: str, segments: list[str]) -> None:
    kind = getattr(node, "kind", "")
    if kind in _CONTAINER_KINDS:
        for part in getattr(node, "parts", []) or []:
            if getattr(part, "kind", "") in _SEPARATOR_KINDS:
                continue
            _collect(part, source, segments)
        return
    if kind != "command":
        raise _OpaqueCommand  # compound / function / 算术展开等
    words: list[str] = []
    for part in getattr(node, "parts", []) or []:
        if getattr(part, "kind", "") != "word":
            raise _OpaqueCommand  # redirect / assignment / ...
        _check_word(part, source)
        words.append(part.word)
    if not words:
        raise _OpaqueCommand
    segments.append(" ".join(words))


def _check_word(word: Any, source: str) -> None:
    for child in getattr(word, "parts", []) or []:
        if getattr(child, "kind", "") != "tilde":
            raise _OpaqueCommand  # 命令替换、参数展开、进程替换……
    start, end = word.pos
    slice_text = source[start:end]
    if _UNSAFE_CHARS & (set(word.word) | set(slice_text)):
        raise _OpaqueCommand
    if word.word == slice_text and "{" in word.word:
        raise _OpaqueCommand  # 无引号 brace：可能触发花括号展开


BASH_DEFINITION = ToolDefinition(
    name="bash",
    description=(
        "Execute a bash command line locally (bash -c, unsandboxed: files, "
        "environment and network all accessible). Returns stdout, stderr and "
        "exit_code. Permission rules match each command segment split by &&, "
        "||, ; and | separately; constructs that cannot be statically verified "
        "(substitutions, redirections, globs, control flow) always require "
        "explicit approval. Each output stream is capped at 64 KiB (keeping "
        "its first and last 32 KiB when truncated)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "pattern": "\\S",
                "description": "The full bash command line to execute.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.1,
                "maximum": _MAX_TIMEOUT_SECONDS,
                "description": (
                    f"Maximum execution time in seconds; defaults to {_DEFAULT_TIMEOUT_SECONDS:g}."
                ),
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
)


class BashTool(AgentTool):
    """经 ``bash -c`` 执行命令行；权限段由 command_targets 提供。"""

    def __init__(self, *, definition: ToolDefinition = BASH_DEFINITION) -> None:
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def permission_targets(self, arguments: Mapping[str, object]) -> tuple[str, ...]:
        command = arguments["command"]
        return command_targets(command) or (command.strip(),)

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        command = str(arguments["command"])
        timeout = min(
            float(arguments.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)),
            _MAX_TIMEOUT_SECONDS,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except (OSError, ValueError) as exc:
            return error_result("bash_spawn_failed", str(exc))

        timed_out = False
        collection_completed = False
        stdout: tuple[bytes, bool] = (b"", False)
        stderr: tuple[bytes, bool] = (b"", False)
        try:
            stdout, stderr = await asyncio.wait_for(_collect_output(process), timeout=timeout)
            collection_completed = True
        except TimeoutError:
            timed_out = True
        finally:
            # 子进程清理的唯一属主：也覆盖 graph 取消与管道意外断开。
            if not collection_completed:
                await _terminate_and_reap(process)

        if timed_out:
            return error_result(
                "bash_timeout",
                f"Command timed out after {timeout:g}s and the process group was killed",
            )

        return ok_result(
            {
                "exit_code": process.returncode,
                "stdout": stdout[0].decode(errors="replace"),
                "stderr": stderr[0].decode(errors="replace"),
                "stdout_truncated": stdout[1],
                "stderr_truncated": stderr[1],
            },
            ok=process.returncode == 0,
        )


async def _collect_output(
    process: asyncio.subprocess.Process,
) -> tuple[tuple[bytes, bool], tuple[bytes, bool]]:
    """并行有界地排空两条管道，然后回收进程本体。"""

    stdout, stderr = await asyncio.gather(
        _drain_bounded(process.stdout),
        _drain_bounded(process.stderr),
    )
    await process.wait()
    return stdout, stderr


async def _drain_bounded(reader: asyncio.StreamReader | None) -> tuple[bytes, bool]:
    """分块读取一路管道：只保留头部与尾部各固定额度，其余计数后丢弃。"""

    if reader is None:
        return b"", False
    head = bytearray()
    tail: deque[bytes] = deque()
    tail_bytes = 0
    total = 0
    while chunk := await reader.read(_READ_CHUNK_BYTES):
        total += len(chunk)
        if len(head) < _OUTPUT_HEAD_BYTES:
            split = _OUTPUT_HEAD_BYTES - len(head)
            head += chunk[:split]
            chunk = chunk[split:]
        if not chunk:
            continue
        tail.append(chunk)
        tail_bytes += len(chunk)
        while tail_bytes > _OUTPUT_TAIL_BYTES:
            excess = tail_bytes - _OUTPUT_TAIL_BYTES
            oldest = tail[0]
            if len(oldest) <= excess:
                tail.popleft()
                tail_bytes -= len(oldest)
            else:
                tail[0] = oldest[excess:]
                tail_bytes -= excess
    return bytes(head) + b"".join(tail), total > _MAX_OUTPUT_BYTES


def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """结束超时进程及其同组子进程。"""

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass


async def _terminate_and_reap(process: asyncio.subprocess.Process) -> None:
    """终止进程组并回收退出状态。

    调用时 ``_collect_output`` 的排空任务已被取消，管道归事件循环收尾，这
    里只负责杀组与 reap——不再读任何输出。
    """

    _terminate_process(process)
    try:
        await asyncio.shield(process.wait())
    except (ProcessLookupError, RuntimeError):
        # 进程可能已经自行退出。
        return


__all__ = ["BASH_DEFINITION", "BashTool", "command_targets"]
