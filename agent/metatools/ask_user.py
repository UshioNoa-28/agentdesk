"""固定 ``ask_user`` Meta Tool（仅主控可用）。

人机澄清原语：模型拿不准时向用户提问并给出候选选项，而不是自行假设。
本工具自身不等待——答案在语义上就是用户的下一条消息，走普通 ``POST
/messages`` 进入会话历史，由下一轮自然读到。轮中途终止（置位 +
``after_tool`` 条件边）尚未接入，当前由结果文案约束模型就此收尾。

问题送达交互客户端走的是已有的 ``tool_call`` SSE 帧：arguments 即题目，
CLI/Web 按 ``name == "ask_user"`` 特判渲染选项框，零协议新增。
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from agent.domain.tools import AgentTool, ToolContext, ToolDefinition, ToolResult

# 与 CLI 选项框的美观对齐；自由输入永远允许，无需占位选项。
_MIN_OPTIONS = 2
_MAX_OPTIONS = 4

ASK_USER_DEFINITION = ToolDefinition(
    name="ask_user",
    description=(
        "Ask the human user a clarifying question with a few candidate choices "
        "when requirements are genuinely ambiguous and a wrong assumption would "
        "be costly. State the full question in your reply text as well, then STOP: "
        "the user's answer arrives as their next message. Do NOT call any other "
        "tool in the same response as ask_user. Do not use it for things you can "
        "reasonably assume, and never answer your own question in a later turn "
        "before the user replies."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to present to the user.",
            },
            "options": {
                "type": "array",
                "minItems": _MIN_OPTIONS,
                "maxItems": _MAX_OPTIONS,
                "items": {"type": "string"},
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
    return ToolResult(content=json.dumps({"ok": False, "error": message}, ensure_ascii=False))


def _normalize_options(raw_options: object) -> list[str] | str:
    """校验并归一化 options；返回选项列表或错误文案。"""

    if not isinstance(raw_options, list) or not _MIN_OPTIONS <= len(raw_options) <= _MAX_OPTIONS:
        return f"options must be a list of {_MIN_OPTIONS}-{_MAX_OPTIONS} choices"

    normalized: list[str] = []
    for option in raw_options:
        if not isinstance(option, str) or not option.strip():
            return "each option must be a non-empty string"
        normalized.append(option.strip())

    if len(set(normalized)) != len(normalized):
        return "options must be distinct"
    return normalized


class AskUserTool(AgentTool):
    """向用户发起带选项的澄清提问；立即返回，答案由用户的下一条消息承载。"""

    @property
    def definition(self) -> ToolDefinition:
        return ASK_USER_DEFINITION

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            return _error("question must be a non-empty string")

        options = _normalize_options(arguments.get("options"))
        if isinstance(options, str):
            return _error(options)

        recommended = arguments.get("recommended")
        if recommended is not None:
            if not isinstance(recommended, str) or recommended.strip() not in options:
                return _error(
                    "recommended must be one of the provided options: "
                    + ", ".join(f"'{option}'" for option in options)
                )
            recommended = recommended.strip()

        return ToolResult(
            content=json.dumps(
                {
                    "ok": True,
                    "question": question.strip(),
                    "options": options,
                    "recommended": recommended,
                    "note": (
                        "The question has been presented to the user. End this turn "
                        "now: their reply will arrive as their next message, and you "
                        "will see it in your context then."
                    ),
                },
                ensure_ascii=False,
            )
        )


__all__ = ["ASK_USER_DEFINITION", "AskUserTool"]
