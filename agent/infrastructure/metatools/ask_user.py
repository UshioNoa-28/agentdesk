"""固定 ``ask_user`` Meta Tool（仅主控可用）。

人机澄清原语：模型拿不准时向用户提问并给出候选选项，而不是自行假设。
本工具经进程内 :class:`InterruptionBrokerPort` 发起一次中断——挂起本轮工具
协程，把 ``AskUserRequest`` 投递到活跃流，await 用户答复作为工具结果，模型
在**同一轮**内直接读到答案并继续。问题送达走独立的 ``ask_user_request``
流帧，与权限中断共用同一 broker 机制。
"""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

from agent.domain.interruption import (
    AskUserRequest,
    InterruptionResult,
    InterruptionStatus,
)
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
from agent.ports.runtime.interruption import InterruptionBrokerPort

# 与 CLI 选项框的美观对齐；自由输入永远允许，无需占位选项。
_MIN_OPTIONS = 2
_MAX_OPTIONS = 4

ASK_USER_DEFINITION = ToolDefinition(
    name="ask_user",
    description=(
        "Ask the human user a clarifying question with a few candidate choices "
        "when requirements are genuinely ambiguous and a wrong assumption would "
        "be costly. The user's answer is returned to you directly and you continue "
        "in the same turn. Do not use it for things you can reasonably assume."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "pattern": "\\S",
                "description": "The question to present to the user.",
            },
            "options": {
                "type": "array",
                "minItems": _MIN_OPTIONS,
                "maxItems": _MAX_OPTIONS,
                "uniqueItems": True,
                "items": {"type": "string", "pattern": "\\S"},
                "description": (
                    f"{_MIN_OPTIONS}-{_MAX_OPTIONS} distinct short choices. Free-form "
                    "answers are always possible on the client, so do not add filler "
                    "options like 'other'."
                ),
            },
            "recommended": {
                "type": "string",
                "description": "One of the provided options, highlighted as recommended.",
            },
        },
        "required": ["question", "options"],
        "additionalProperties": False,
    },
)


def _error(message: str) -> ToolResult:
    return error_result("invalid_arguments", message)


class AskUserTool(AgentTool):
    """向用户发起带选项的澄清提问；本轮内阻塞等待答复并作为工具结果返回。"""

    def __init__(self, broker: InterruptionBrokerPort) -> None:
        self._broker = broker

    @property
    def definition(self) -> ToolDefinition:
        return ASK_USER_DEFINITION

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        question = str(arguments["question"]).strip()
        options = [str(option).strip() for option in arguments["options"]]

        recommended = arguments.get("recommended")
        if recommended is not None:
            # 跨字段规则，schema 无法表达：推荐项必须是给出的选项之一。
            recommended_text = str(recommended).strip()
            if recommended_text not in options:
                return _error(
                    "recommended must be one of the provided options: "
                    + ", ".join(f"'{option}'" for option in options)
                )
            recommended = recommended_text

        result = await self._broker.request(
            AskUserRequest(
                interruption_id=str(uuid4()),
                session_id=context.caller_session_id,
                question=question,
                options=tuple(options),
                recommended=recommended,
            )
        )
        return _result_from_interruption(result)


def _result_from_interruption(result: InterruptionResult) -> ToolResult:
    """把中断结果转成工具载荷：RESOLVED 带答案，其余说明没问到用户。"""

    if result.status is InterruptionStatus.RESOLVED:
        return ok_result({"answer": str(result.payload.get("answer", ""))})
    if result.status is InterruptionStatus.UNAVAILABLE:
        return error_result(
            "ask_user_unavailable",
            "No active client could receive this question; the user was not reached. "
            "State your assumption and proceed, or end the turn.",
        )
    # CANCELLED：本轮在等待期间被打断。
    return error_result(
        "ask_user_cancelled",
        "The user did not answer this question before the turn ended.",
    )


__all__ = ["ASK_USER_DEFINITION", "AskUserTool"]
