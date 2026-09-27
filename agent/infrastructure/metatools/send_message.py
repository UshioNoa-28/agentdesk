"""固定 ``send_message`` Meta Tool。"""

from __future__ import annotations

from collections.abc import Mapping

from agent.domain.exceptions import DomainValidationError
from agent.domain.multi_agent import MAIN_AGENT_NAME, MainAgentMessage, SubAgentMessage
from agent.domain.tools import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    error_result,
    ok_result,
)
from agent.ports.services import SessionServicePort
from agent.ports.tools import AgentRuntimePort

SEND_MESSAGE_DEFINITION = ToolDefinition(
    name="send_message",
    description=(
        "Send a message / assign a task to a specialized subagent asynchronously. "
        "Delivery is fire-and-forget: the tool returns immediately after the message "
        "is accepted by the runtime. The recipient processes it in its own session; "
        "its report or clarifying question will arrive later as a new incoming message."
    ),
    parameters={
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "pattern": "\\S",
                "description": "Target subagent name (e.g. 'coder', 'researcher', 'reviewer').",
            },
            "message": {
                "type": "string",
                "pattern": "\\S",
                "description": (
                    "The task instructions, context, or clarifying message to send to "
                    "the subagent. When reporting back to 'main_agent', include the "
                    "full deliverable or findings."
                ),
            },
        },
        "required": ["recipient", "message"],
        "additionalProperties": False,
    },
)


class SendMessageTool(AgentTool):
    """向指定目标非阻塞投递消息的 Meta Tool。

    发送方身份与消息归属一律以会话表为准：调用者会话没有
    ``main_session_id`` 即主控，否则视为子代理（显示名取标题后缀），
    消息路由锚点固定为其主会话。
    """

    def __init__(
        self,
        *,
        runtime: AgentRuntimePort,
        session_service: SessionServicePort,
        definition: ToolDefinition = SEND_MESSAGE_DEFINITION,
    ) -> None:
        self._runtime = runtime
        self._session_service = session_service
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
        content = str(arguments["message"]).strip()
        recipient = str(arguments["recipient"]).strip()

        try:
            caller_session = await self._session_service.get(context.caller_session_id)
        except Exception as exc:
            return error_result(
                "caller_session_unavailable",
                "Could not load caller session "
                f"'{context.caller_session_id}' to route messages: {exc}",
            )

        if caller_session.main_session_id is None:
            sender_name = MAIN_AGENT_NAME
            owner_session_id = caller_session.id
        else:
            marker = "_subagent_"
            sender_name = (
                caller_session.title.rsplit(marker, 1)[-1]
                if marker in caller_session.title
                else caller_session.title
            )
            owner_session_id = caller_session.main_session_id

        # 收件人决定消息类型；拓扑/自环/非空等规则由充血消息在构造期校验，
        # 这里只把领域错误原样翻译成工具错误，不含任何路由策略。
        try:
            if recipient == MAIN_AGENT_NAME:
                message = MainAgentMessage(
                    content=content,
                    sender=sender_name,
                    session_id=owner_session_id,
                )
            else:
                message = SubAgentMessage(
                    content=content,
                    sender=sender_name,
                    recipient=recipient,
                    session_id=owner_session_id,
                )
        except DomainValidationError as exc:
            return error_result(getattr(exc, "code", "domain_validation_error"), exc.message)

        try:
            await self._runtime.dispatch(message)
            return ok_result(
                {
                    "status": "dispatched",
                    "recipient": recipient,
                    "message": (
                        f"Message successfully dispatched to '{recipient}'. "
                        "Your turn should end now; the reply will arrive as a "
                        "new incoming message later."
                    ),
                }
            )
        except Exception as exc:
            return error_result(
                "dispatch_failed",
                f"Dispatching message to '{recipient}' failed: {exc}",
                details={"recipient": recipient},
            )


__all__ = ["SEND_MESSAGE_DEFINITION", "SendMessageTool"]
