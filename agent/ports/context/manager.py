"""上下文准备、投影与压缩端口协议。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from agent.domain.messages import Message
from agent.domain.model_messages import ModelMessage
from agent.domain.tools import ToolDefinition


class ContextProjectorPort(Protocol):
    """把持久化非 System Message 投影为模型消息的纯应用端口。

    模型视角（LLM view）：冷压缩 checkpoint 生效——边界之前超过清除
    阈值的工具结果投影为占位文案。
    """

    def project(
        self,
        messages: Sequence[Message],
    ) -> list[ModelMessage]: ...

    def project_tool_result(
        self,
        message: Message,
    ) -> ModelMessage: ...

    @staticmethod
    def wrap_untrusted_content(content: str) -> str: ...

    @staticmethod
    def wrap_skill_guidance(content: str) -> str: ...

    @staticmethod
    def wrap_conversation_summary(summary: str) -> str: ...


class SummaryProjectorPort(Protocol):
    """把持久化 Message 投影为摘要源消息的纯应用端口。

    摘要视角（Summary view）：冷压缩状态对摘要不可见——摘要必须看到
    工具结果的全量内容（仍受单条截断约束），否则被摘要覆盖的历史会
    在唯一记录里被销毁两次。MAINTENANCE 消息直接跳过；SUMMARY 消息
    以 <conversation_summary> 边界参与摘要源（增量摘要）。
    """

    def project(
        self,
        messages: Sequence[Message],
    ) -> list[ModelMessage]: ...


class ContextCompactorPort(Protocol):
    """从持久化 Message 历史生成 Compact 更新。"""

    def summary_source(
        self,
        history: Sequence[Message],
        *,
        current_user_message_id: str | None,
    ) -> tuple[list[ModelMessage], Message] | None:
        """构造摘要源与推进边界。

        ``current_user_message_id`` 给定时锚在"当前用户问题"——它及其
        之后的消息（进行中的回答）不进摘要，服务于自动压缩的
        before_model 语义；传 ``None`` 锚在历史末尾，整段对话（含最后
        一轮问答）都可摘，服务于用户显式发起的手动压缩。
        """

        ...

    def cold_candidate(
        self,
        history: Sequence[Message],
        *,
        current_user_message_id: str,
    ) -> Message | None: ...

    async def summary_candidate(
        self,
        source_messages: Sequence[ModelMessage],
        *,
        boundary: Message,
        source_message_count: int,
    ) -> Message: ...


@dataclass(frozen=True, slots=True)
class CompactResult:
    """会话上下文压缩结果。"""

    session_id: str
    status: str  # "compacted" | "noop"
    kind: str | None = None  # "cold" | "summary" | None
    checkpoint_message_id: str | None = None
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0
    message: str = ""


class ContextManagerPort(Protocol):
    """上下文编排与压缩端口。"""

    @property
    def inbound_watermark(self) -> int:
        """供graph读取以判断当前轮次的max seq"""
        ...

    async def prepare(
        self,
        *,
        session_id: str,
        tools: Sequence[ToolDefinition],
    ) -> list[ModelMessage]: ...

    async def compact_session(
        self,
        session_id: str,
    ) -> CompactResult: ...


__all__ = [
    "CompactResult",
    "ContextCompactorPort",
    "ContextManagerPort",
    "ContextProjectorPort",
    "SummaryProjectorPort",
]
