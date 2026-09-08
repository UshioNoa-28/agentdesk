"""Agent 工作流节点共享的 TypedDict 状态契约。"""

from __future__ import annotations

from typing import TypedDict

from agent.domain.messages import Message
from agent.domain.model_messages import ModelMessage, ToolCall
from agent.domain.tools import AgentTool
from agent.ports.model import AgentModelPort
from agent.ports.runtime.stream import AgentStreamSink


class AgentState(TypedDict, total=False):
    """一次 Agent 工作流在 LangGraph 节点之间传递的运行时状态。"""

    messages: list[ModelMessage]
    # Hook 中间件的跨节点共享草稿；工具不读取它。
    scratch: dict[str, object]
    pending_tool_calls: list[ToolCall]
    turns: int
    model: AgentModelPort
    tools: list[AgentTool]
    # 流式事件旁路；为 None 时工作流走非流式路径。
    events: AgentStreamSink | None
    # 触发本次工作流的会话（= 执行者自己的会话），是工具上下文的唯一会话来源。
    caller_session_id: str | None
    persisted_assistant: Message | None
    # 本次运行中是否已成功向 main_agent 派发过 send_message（_execute_tools 标记）。
    sent_to_main: bool # 防止subagent未向main agent send message 汇报 
    # ask_user 是否已被调用 置位后 after_tool 直接 END 
    ask_user_called: bool # 答案在语义上就是用户的下一条消息 由下一轮自然读取 
    # 本轮最近一次 before_model 从库里读到的入站水位(prepare 提供,human-only max seq)。
    # 据此判断"读库之后是否又有人落了新消息"。
    inbound_watermark: int


__all__ = ["AgentState"]
