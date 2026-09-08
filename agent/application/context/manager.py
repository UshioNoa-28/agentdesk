"""Persistent-message driven model context preparation."""

from __future__ import annotations

from collections.abc import Sequence

from agent.domain.exceptions import (
    ContextCompactionError,
    SystemPromptBudgetError,
)
from agent.domain.messages import Message, MessageKind, message_kind_from_value
from agent.domain.model_messages import ModelMessage
from agent.domain.tools import ToolDefinition
from agent.infrastructure.settings import ContextSettings
from agent.ports.context import (
    CompactResult,
    ContextCompactorPort,
    ContextManagerPort,
    ContextProjectorPort,
)
from agent.ports.services import (
    MessageServicePort,
    SessionServicePort,
    TokenCounterServicePort,
)
from agent.ports.tools import MetaToolRegistryPort


class ContextManager(ContextManagerPort):
    """Project, estimate, compact, persist and re-project one model turn."""

    def __init__(
        self,
        *,
        message_service: MessageServicePort,
        projector: ContextProjectorPort,
        compactor: ContextCompactorPort,
        token_counter: TokenCounterServicePort,
        settings: ContextSettings,
        session_service: SessionServicePort,
        meta_tools: MetaToolRegistryPort,
    ) -> None:
        self._messages = message_service
        self._projector = projector
        self._compactor = compactor
        self._counter = token_counter
        self._trigger = settings.compact_threshold_tokens
        self._cold_target = settings.cold_compact_target_tokens
        self._budget = settings.input_budget_tokens
        self._max_recent_tool_calls = settings.context_max_recent_tool_calls
        # 手动压缩路径按会话白名单解析工具定义，使估算口径与自动路径一致：
        # 白名单（授权了哪些名字）来自数据库，名字到定义的映射来自 registry。
        self._session_service = session_service
        self._meta_tools = meta_tools
        # 最近一次 prepare 从已读 history 推导的入站水位;无新入站不回退。
        self._inbound_watermark = 0

    @property
    def inbound_watermark(self) -> int:
        return self._inbound_watermark

    def _note_inbound_watermark(self, history: Sequence[Message]) -> None:
        """从刚读取的 history 就地推导入站水位——与本轮上下文同一次读取,
        不能再查库取数,否则 prepare 与取数之间的新消息会造成水位虚高。"""

        self._inbound_watermark = max(
            (message.seq for message in history if message.role == "human"),
            default=self._inbound_watermark,
        )

    def _project_and_estimate(
        self,
        history: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition],
    ) -> tuple[list[ModelMessage], int]:
        system_message = ModelMessage.system(history[0].content)
        system_tokens = self._counter.estimate_message(system_message)
        if system_tokens >= self._budget:
            raise SystemPromptBudgetError(
                f"System prompt alone ({system_tokens} tokens) exceeds "
                f"context budget ({self._budget} tokens)."
            )

        messages = [system_message, *self._projector.project(history[1:])]
        return messages, self._counter.measure_model_request(messages, tools=tools)

    def _estimate(
        self,
        messages: Sequence[ModelMessage],
        *,
        tools: Sequence[ToolDefinition],
    ) -> int:
        """估算模型消息与工具定义的总 token 数。"""

        return self._counter.measure_model_request(messages, tools=tools)

    def _project_full_history(self, history: Sequence[Message]) -> list[ModelMessage]:
        """system 消息 + 投影后的对话历史，用于不需要预算校验的路径。"""

        return [
            ModelMessage.system(history[0].content),
            *self._projector.project(history[1:]),
        ]

    async def prepare(
        self,
        *,
        session_id: str,
        tools: Sequence[ToolDefinition],
    ) -> list[ModelMessage]:
        raw_history = await self._messages.history(session_id)
        self._note_inbound_watermark(raw_history)
        projected, total_tokens = self._project_and_estimate(raw_history, tools=tools)
        if total_tokens <= self._trigger:
            return projected

        dialogue_history = raw_history[1:]
        current_user = self._latest_user(dialogue_history)

        # 尝试 Cold Compact
        candidate = self._compactor.cold_candidate(
            dialogue_history,
            current_user_message_id=current_user.id,
        )
        if candidate is not None:
            await self._persist_candidate(candidate, session_id=session_id)
            projected_after_cold = await self._reread_project(session_id, tools=tools)
            # 冷压缩时间内 本次turn的message的范围可能发生变化 需要重新 找到覆盖的最大的seq
            if self._estimate(projected_after_cold, tools=tools) <= self._cold_target:
                return projected_after_cold

        # 尝试 Summary Compact 截断逻辑在 compactor 内部完成
        # 超目标水位时从最老一侧按完整交换截断
        summary_source = self._compactor.summary_source(
            dialogue_history,
            current_user_message_id=current_user.id,
        )
        if summary_source is None:
            raise ContextCompactionError(
                "Context exceeded the compaction threshold, but there is no "
                "prior history that can be summarized."
            )

        source_messages, boundary = summary_source
        summary = await self._compactor.summary_candidate(
            source_messages,
            boundary=boundary,
            source_message_count=len(source_messages),
        )
        await self._persist_candidate(summary, session_id=session_id)
        # 同样范围可能发生变化 需要重新得到最新的 message
        return await self._reread_project(session_id, tools=tools)

    async def compact_session(
        self,
        session_id: str,
    ) -> CompactResult:
        """手动触发会话上下文压缩（直接调用 Summary 模型生成对话摘要）。"""

        history = await self._messages.history(session_id)
        dialogue_history = history[1:]
        projected_before = self._project_full_history(history)
        tools = await self._session_tools(session_id)
        tokens_before = self._estimate(projected_before, tools=tools)

        # 手动压缩：锚在历史末尾——没有"正在回答的问题"需要保护，整段
        # 对话（含最后一轮问答）都进入可摘范围；最近 N 组工具交换仍受
        # 保护区豁免，截断在 compactor 内部完成。
        summary_source = self._compactor.summary_source(
            dialogue_history,
            current_user_message_id=None,
        )
        if summary_source is None:
            return CompactResult(
                session_id=session_id,
                status="noop",
                estimated_tokens_before=tokens_before,
                estimated_tokens_after=tokens_before,
                message="No eligible history to summarize.",
            )

        source_messages, boundary = summary_source
        summary = await self._compactor.summary_candidate(
            source_messages,
            boundary=boundary,
            source_message_count=len(source_messages),
        )
        persisted = await self._persist_candidate(summary, session_id=session_id)
        new_history = await self._messages.history(session_id)
        projected_after = self._project_full_history(new_history)
        tokens_after = self._estimate(projected_after, tools=tools)
        return CompactResult(
            session_id=session_id,
            status="compacted",
            kind="summary",
            checkpoint_message_id=persisted.id,
            estimated_tokens_before=tokens_before,
            estimated_tokens_after=tokens_after,
            message=(
                "Successfully generated conversation summary covering "
                f"{len(source_messages)} messages."
            ),
        )

    async def _persist_candidate(self, candidate: Message, *, session_id: str) -> Message:
        return await self._messages.add(
            session_id=session_id,
            message=candidate.to_model(),
            metadata=candidate.metadata,
        )

    async def _session_tools(self, session_id: str) -> tuple[ToolDefinition, ...]:
        """按会话工具白名单解析定义：白名单读库，定义来自本地 registry。"""

        session = await self._session_service.get(session_id)
        tools = self._meta_tools.get_tools(session.allowed_tools)
        return tuple(tool.definition for tool in tools)

    async def _reread_project(
        self,
        session_id: str,
        *,
        tools: Sequence[ToolDefinition],
    ) -> list[ModelMessage]:
        history = await self._messages.history(session_id)
        # 压缩触发的重读才是模型最终看到的上下文,水位以这次为准。
        self._note_inbound_watermark(history)
        messages, estimated = self._project_and_estimate(history, tools=tools)
        if estimated > self._budget:
            raise ContextCompactionError(
                "Compacted context still exceeds the model input budget."
            )
        return messages

    @staticmethod
    def _latest_user(history: Sequence[Message]) -> Message:
        """找到最近的 user query。

        子代理回报也以 human/USER 落库，但它们的 ``source`` 是 ``agent``；
        只有 ``source == "user"`` 的消息才是真正的当前用户问题。优先取最近的
        真实用户消息，若历史里没有（例如只有 agent 回报），再退回最近一条
        USER 消息，保持旧行为。
        """

        fallback: Message | None = None
        for user_index in range(len(history) - 1, -1, -1):
            item = history[user_index]
            if (
                item.role != "human"
                or message_kind_from_value(item.metadata.get("kind"))
                != MessageKind.USER
            ):
                continue
            if item.metadata.get("source") == "user":
                return item
            if fallback is None:
                fallback = item
        if fallback is not None:
            return fallback
        raise ContextCompactionError(
            "Current user message is missing from context history."
        )


__all__ = ["ContextManager"]
