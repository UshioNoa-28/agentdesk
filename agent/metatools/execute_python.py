"""固定 ``execute_python`` Meta Tool。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping

from agent.domain.tools import AgentTool, ToolContext, ToolDefinition, ToolResult

EXECUTE_PYTHON_DEFINITION = ToolDefinition(
    name="execute_python",
    description=(
        "Run Python code in a local subprocess for calculations, parsing, and small "
        "data transformations. The subprocess is unsandboxed and runs with the "
        "agent's full permissions (files, environment, network, processes). "
        "Returns stdout, stderr, and exit_code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "minLength": 1,
                "description": "Python source code to execute.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 60,
                "default": 10,
                "description": "Maximum execution time in seconds.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
)


class ExecutePythonTool(AgentTool):
    """在未沙箱化的本地子进程中执行 Python 代码的 Meta Tool。"""

    def __init__(self, definition: ToolDefinition = EXECUTE_PYTHON_DEFINITION) -> None:
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        code = str(arguments.get("code", ""))
        if not code.strip():
            return ToolResult(
                content=json.dumps({"ok": False, "error": "code must be a non-empty string"})
            )

        timeout = float(arguments.get("timeout_seconds", 10))
        if not 0.1 <= timeout <= 60:
            return ToolResult(
                content=json.dumps(
                    {"ok": False, "error": "timeout_seconds must be between 0.1 and 60"}
                )
            )

        max_bytes = int(os.environ.get("LOCAL_PYTHON_MAX_OUTPUT_BYTES", "65536"))

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                process.kill()
                await process.communicate()
                return ToolResult(
                    content=json.dumps(
                        {"ok": False, "error": f"Python execution timed out after {timeout:g}s"}
                    )
                )

            stdout_trimmed, stdout_truncated = stdout[:max_bytes], len(stdout) > max_bytes
            stderr_trimmed, stderr_truncated = stderr[:max_bytes], len(stderr) > max_bytes

            return ToolResult(
                content=json.dumps(
                    {
                        "ok": process.returncode == 0,
                        "exit_code": process.returncode,
                        "stdout": stdout_trimmed.decode(errors="replace"),
                        "stderr": stderr_trimmed.decode(errors="replace"),
                        "stdout_truncated": stdout_truncated,
                        "stderr_truncated": stderr_truncated,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            return ToolResult(content=json.dumps({"ok": False, "error": str(exc)}))


__all__ = ["EXECUTE_PYTHON_DEFINITION", "ExecutePythonTool"]
