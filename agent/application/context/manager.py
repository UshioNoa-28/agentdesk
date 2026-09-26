"""Persistent-message driven model context preparation."""

from __future__ import annotations

from collections.abc import Sequence

from agent.domain.context_budget import derive_budget
from agent.domain.context_settings import ContextSettings
from agent.domain.exceptions import (
    ContextCompactionError,
    SystemPromptBudgetError,
)
from agent.domain.messages import Message, MessageKind, message_kind_from_value
from agent.domain.model_messages import ModelMessage
from agent.domain.tools import ToolDefinition
from agent.infrastructure.model.catalog import ModelCatalog
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
from agent.ports.tools import (
    McpRegistryPort,
    MemoryCatalogPort,
    MetaToolRegistryPort,
    SkillCatalogPort,
)
from agent.prompt.system import build_system_prompt


class ContextManager(ContextManagerPort):
    """Project, estimate, compact, persist and re-project one model turn."""

    def __init__(
        self,
        *,
        message_service: MessageServicePort,
        projector: ContextProjectorPort,
        compactor: ContextCompactorPort,
        token_counter: TokenCounterServicePort,
        catalog: ModelCatalog,
        settings: ContextSettings,
        session_service: SessionServicePort,
        meta_tools: MetaToolRegistryPort,
        mcp_registry: McpRegistryPort,
        skill_catalog: SkillCatalogPort,
        memory_catalog: MemoryCatalogPort,
    ) -> None:
        self._messages = message_service
        self._projector = projector
        self._compactor = compactor
        self._counter = token_counter
        # 预算锚定当前模型的 ModelProfile：每次 prepare 现取，切换模型后
        # 下一圈循环的水位与硬预算自动跟随，不存在构造期快照。
        # 派生公式收敛在 derive_budget 一处。
        self._catalog = catalog
        self._settings = settings
        # 手动压缩路径按会话白名单解析工具定义，使估算口径与自动路径一致：
        # 白名单（授权了哪些名字）来自数据库，名字到定义的映射来自 registry。
        self._session_service = session_service
        self._meta_tools = meta_tools
        # 默认 system prompt 的活配置来源：每轮现拼，不读入库快照。
        self._mcp_registry = mcp_registry
        self._skill_catalog = skill_catalog
        self._memory_catalog = memory_catalog
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

    async def _split_system(
        self,
        session_id: str,
        history: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition],
    ) -> tuple[ModelMessage, list[Message]]:
        """装配本轮 system 消息，并返回切掉 system 行之后的对话历史。

        默认策略是活配置（MCP/Skill/工具授权/记忆目录），每轮现拼；入库
        会把它们冻在建会话那一刻。存库的 seq=1 SYSTEM 行只服务自定义
        persona：非根会话（define_subagent 及其 resume 子会话）沿用存库
        文本；老版本落库的根会话默认 prompt 直接丢弃，换现拼。
        """

        session = await self._session_service.get(session_id)
        has_stored = len(history) > 0 and (
            message_kind_from_value(history[0].metadata.get("kind")) == MessageKind.SYSTEM
        )
        dialogue = list(history[1:] if has_stored else history)
        if session.main_session_id is not None and has_stored:
            # subagent 存 system prompt
            return ModelMessage.system(history[0].content), dialogue
        # main agent 现场拼
        system_content = build_system_prompt(
            server_descriptions=self._mcp_registry.server_descriptions(),
            skill_descriptions=self._skill_catalog.metadata(),
            meta_tools=tools,
            memory_layers=self._memory_catalog.load_prompt(),
        )
        return ModelMessage.system(system_content), dialogue

    async def _assemble(
        self,
        session_id: str,
        history: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition],
        budget: int,
    ) -> tuple[list[ModelMessage], list[Message], int]:
        """project 并 estimate token数量 返回 projected message original message token_count """
        system_message, dialogue = await self._split_system(session_id, history, tools=tools)
        system_tokens = self._counter.estimate_message(system_message)
        if system_tokens >= budget:
            raise SystemPromptBudgetError(
                f"System prompt alone ({system_tokens} tokens) exceeds "
                f"context budget ({budget} tokens)."
            )

        projected = [system_message, *self._projector.project(dialogue)]
        return projected, dialogue, self._counter.measure_model_request(projected, tools=tools)

    def _estimate(
        self,
        messages: Sequence[ModelMessage],
        *,
        tools: Sequence[ToolDefinition],
    ) -> int:
        """估算模型消息与工具定义的总 token 数。"""

        return self._counter.measure_model_request(messages, tools=tools)

    async def _project_full_history(
        self,
        session_id: str,
        history: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition],
    ) -> list[ModelMessage]:
        """system 消息 + 投影后的对话历史，用于不需要预算校验的路径。"""

        system_message, dialogue = await self._split_system(session_id, history, tools=tools)
        return [system_message, *self._projector.project(dialogue)]

    async def prepare(
        self,
        *,
        session_id: str,
        tools: Sequence[ToolDefinition],
    ) -> list[ModelMessage]:
        raw_history = await self._messages.history(session_id)
        self._note_inbound_watermark(raw_history)
        budget = derive_budget(self._catalog.current_profile(), self._settings)
        projected, dialogue_history, total_tokens = await self._assemble(
            session_id, raw_history, tools=tools, budget=budget.input
        )
        if total_tokens <= budget.compact_threshold:
            return projected

        # 尝试 Cold Compact
        candidate = self._compactor.cold_candidate(dialogue_history)
        if candidate is not None:
            await self._persist_candidate(candidate, session_id=session_id)
            projected_after_cold = await self._reread_project(
                session_id, tools=tools, budget=budget.input
            )
            # 冷压缩时间内 本次turn的message的范围可能发生变化 需要重新 找到覆盖的最大的seq
            if self._estimate(projected_after_cold, tools=tools) <= budget.cold_target:
                return projected_after_cold  # 达到token则不再需要summary压缩

        # 尝试 Summary Compact 截断逻辑在 compactor 内部完成
        # 超目标水位时从最老一侧按完整交换截断
        summary_source = self._compactor.summary_source(dialogue_history)
        if summary_source is None:
            raise ContextCompactionError(
                "Context exceeded the compaction threshold, but nothing new "
                "arrived since the last summary checkpoint; there is no "
                "history left to summarize."
            )

        source_messages, boundary = summary_source
        summary = await self._compactor.summary_candidate(
            source_messages,
            boundary=boundary,
            source_message_count=len(source_messages),
        )
        await self._persist_candidate(summary, session_id=session_id)
        # 同样范围可能发生变化 需要重新得到最新的 message
        return await self._reread_project(
            session_id, tools=tools, budget=budget.input
        )

    async def compact_session(
        self,
        session_id: str,
    ) -> CompactResult:
        """手动触发会话上下文压缩（直接调用 Summary 模型生成对话摘要）。"""

        history = await self._messages.history(session_id)
        tools = await self._session_tools(session_id)
        system_message, dialogue_history = await self._split_system(
            session_id, history, tools=tools
        )
        projected_before = [system_message, *self._projector.project(dialogue_history)]
        tokens_before = self._estimate(projected_before, tools=tools)

        # 手动压缩与自动压缩同语义：锚在历史末尾，最近 N 组工具交换受
        # 保护区豁免，截断在 compactor 内部完成。
        summary_source = self._compactor.summary_source(dialogue_history)
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
        projected_after = await self._project_full_history(session_id, new_history, tools=tools)
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
        budget: int,
    ) -> list[ModelMessage]:
        history = await self._messages.history(session_id)
        # 压缩触发的重读才是模型最终看到的上下文,水位以这次为准。
        self._note_inbound_watermark(history)
        projected, _, estimated = await self._assemble(
            session_id, history, tools=tools, budget=budget
        )
        if estimated > budget:
            raise ContextCompactionError("Compacted context still exceeds the model input budget.")
        return projected


__all__ = ["ContextManager"]
