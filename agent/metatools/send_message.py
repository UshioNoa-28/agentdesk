"""固定 ``send_message`` Meta Tool。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from agent.domain.multi_agent import MAIN_AGENT_NAME, AgentMessage
from agent.domain.tools import AgentTool, ToolContext, ToolDefinition, ToolResult
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
                "description": "Target subagent name (e.g. 'coder', 'researcher', 'reviewer').",
            },
            "message": {
                "type": "string",
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
        content = str(arguments.get("message", "")).strip()
        if not content:
            return ToolResult(
                content=json.dumps({"ok": False, "error": "Message content cannot be empty"})
            )

        recipient = str(arguments.get("recipient", "")).strip()
        if not recipient:
            return ToolResult(
                content=json.dumps(
                    {"ok": False, "error": "Recipient subagent name cannot be empty"}
                )
            )

        try:
            caller_session = await self._session_service.get(context.caller_session_id)
        except Exception as exc:
            return ToolResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "Could not load caller session "
                            f"'{context.caller_session_id}' to route messages: {exc}"
                        ),
                    }
                )
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

        # 自环拦截：主控发给自己的消息只会落库不驱动（工具却报成功），
        # 子代理发给自己同样毫无意义；一律显式拒绝并说明正确做法。
        if recipient == sender_name:
            return ToolResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            f"Cannot send a message to yourself ('{recipient}'). "
                            "The main agent should simply continue its turn; "
                            "subagents must report back with recipient='main_agent'."
                        ),
                    }
                )
            )

        # 星形拓扑硬约束：子代理（main_session_id 非空）只允许与主控通信。
        if caller_session.main_session_id is not None and recipient != MAIN_AGENT_NAME:
            return ToolResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            f"Subagents may only send messages to '{MAIN_AGENT_NAME}' "
                            "(star topology). Use recipient='main_agent'."
                        ),
                    }
                )
            )

        try:
            await self._runtime.dispatch(
                AgentMessage(
                    content=content,
                    sender=sender_name,
                    recipient=recipient,
                    session_id=owner_session_id,
                )
            )
            return ToolResult(
                content=json.dumps(
                    {
                        "ok": True,
                        "status": "dispatched",
                        "recipient": recipient,
                        "message": (
                            f"Message successfully dispatched to '{recipient}'. "
                            "Your turn should end now; the reply will arrive as a "
                            "new incoming message later."
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            return ToolResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "subagent": recipient,
                        "error": f"Dispatching message to '{recipient}' failed: {exc}",
                    },
                    ensure_ascii=False,
                )
            )


__all__ = ["SEND_MESSAGE_DEFINITION", "SendMessageTool"]
