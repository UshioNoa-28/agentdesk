"""Cold and summary candidate generation for the context manager."""

from __future__ import annotations

from collections.abc import Sequence

from agent.application.context.history_analysis import (
    collect_tool_exchanges,
    drop_wait_exchanges,
    first_exchange_end,
    latest_cold_boundary_index,
    latest_summary,
    message_index,
    protected_tail_start,
)
from agent.domain.exceptions import ContextCompactionError, SummaryInputBudgetError
from agent.domain.messages import (
    Message,
    MessageKind,
    message_kind_from_value,
)
from agent.domain.model_messages import ModelMessage, ModelTurn
from agent.infrastructure.settings import ContextSettings
from agent.ports.context import (
    ContextCompactorPort,
    ContextProjectorPort,
    SummaryProjectorPort,
)
from agent.ports.model import AgentModelPort
from agent.ports.services import TokenCounterServicePort
from agent.prompt.summary import SUMMARY_SYSTEM_PROMPT, build_summary_request_context


class ContextCompactor(ContextCompactorPort):
    """Generate candidates; the ContextManager decides whether to persist them.

    职责分工：范围选择（锚点、保护区、截断单元）建立在
    ``history_analysis`` 的纯函数之上；消息形态（全量工具结果、
    SUMMARY 边界、MAINTENANCE/SYSTEM 不可见）归 ``SummaryProjector``；
    本类只做候选生成编排与摘要请求构造。
    """

    def __init__(
        self,
        *,
        projector: ContextProjectorPort,
        summary_projector: SummaryProjectorPort,
        token_counter: TokenCounterServicePort,
        model: AgentModelPort,
        settings: ContextSettings,
    ) -> None:
        self._projector = projector
        self._summary_projector = summary_projector
        self._counter = token_counter
        self._model = model
        self._max_recent_tool_calls = settings.context_max_recent_tool_calls
        self._clear_threshold = settings.context_clear_tool_result_threshold_tokens
        self._summary_max_tokens = settings.context_summary_max_tokens
        self._summary_input_budget_tokens = (
            settings.input_budget_tokens - self._summary_max_tokens
        )
        if self._summary_input_budget_tokens <= 0:
            raise ValueError(
                "summary_max_output_tokens must be lower than the context input budget"
            )
        # 摘要源截断目标水位：与模型侧压缩触发水位同一 ratio。摘要源压到
        # 该水位之下，摘要请求就不可能把摘要输入预算吃满——也避免"摘要
        # 刚落库、下一轮马上又过阈值"的压缩空转。
        self._summary_source_target_tokens = int(
            self._summary_input_budget_tokens * settings.context_compact_threshold_ratio
        )
        if self._summary_source_target_tokens <= 0:
            raise ValueError(
                f"Summary source target {self._summary_source_target_tokens} must be "
                "positive: int(summary_input_budget_tokens="
                f"{self._summary_input_budget_tokens} * context_compact_threshold_ratio="
                f"{settings.context_compact_threshold_ratio}) truncated to 0. Raise "
                "context_compact_threshold_ratio, or raise context_window_tokens / "
                "lower context_summary_max_tokens until the product reaches 1."
            )

    def summary_source(
        self,
        history: Sequence[Message],
        *,
        current_user_message_id: str | None,
    ) -> tuple[list[ModelMessage], Message] | None:
        """Build ``summary + older tail`` and keep the newest K exchanges out.

        两种锚点语义：

        - current_user_message_id 给定 自动压缩 锚在当前用户
          问题 问题之前的进行summary
        - current_user_message_id=None 手动压缩 锚在历史末尾，
          整段对话作为summary来源
        """

        anchor_index = (
            message_index(history, current_user_message_id)
            if current_user_message_id is not None
            else len(history)
        )
        positions = {message.id: index for index, message in enumerate(history)}
        previous_summary, summary_boundary_index = latest_summary(
            history,
            positions=positions,
            anchor_index=anchor_index,
        )

        tail_start = (
            summary_boundary_index + 1
            if summary_boundary_index is not None
            else 0
        )
        # 纯 wait 轮次对摘要无信息量，属于范围选择，在这里剔除；SUMMARY
        # 消息也排除——它是"上次压缩的产物"而非可摘对话，已经通过
        # previous_summary 增量参与源。
        tail = [
            message
            for message in history[tail_start:anchor_index]
            if message_kind_from_value(message.metadata.get("kind"))
            != MessageKind.SUMMARY
        ]
        tail = drop_wait_exchanges(tail)
        older_tail = tail[: protected_tail_start(
            tail, max_recent_tool_calls=self._max_recent_tool_calls
        )]

        # older_tail 为空 = 没有新东西可摘（含"旧摘要之后全在保护区内"
        # 的无进展情形）：绝不让摘要重写自己，直接交还 None。
        if not older_tail:
            return None
        summary_projection = (
            self._project_summary_message(previous_summary)
            if previous_summary is not None
            else None
        )
        older_tail = self._truncate_older_tail(older_tail, summary_projection)
        if not older_tail:
            # 目标水位小到连一组交换都装不下（极端配置才可达）：本轮
            # no-op，不是错误——下一轮 before_model 会重新决策。
            return None

        source: list[ModelMessage] = []
        if summary_projection is not None:
            source.append(summary_projection)
        source.extend(self._summary_projector.project(older_tail))
        return source, older_tail[-1]

    def cold_candidate(
        self,
        history: Sequence[Message],
        *,
        current_user_message_id: str,
    ) -> Message | None:
        """Build a maintenance Message without writing it to the database."""

        anchor_index = message_index(history, current_user_message_id)
        before_current = drop_wait_exchanges(history[:anchor_index])
        exchanges = collect_tool_exchanges(before_current)
        protected_exchange_start = max(
            0,
            len(exchanges) - self._max_recent_tool_calls,
        )
        boundary: Message | None = None
        for exchange_index, exchange in enumerate(exchanges):
            if exchange_index >= protected_exchange_start:
                continue
            if any(
                item.role == "tool"
                and message_kind_from_value(item.metadata.get("kind"))
                == MessageKind.TOOL_RESULT
                and self._counter.estimate_text(
                    self._projector.wrap_untrusted_content(item.content)
                )
                > self._clear_threshold
                for item in exchange
            ):
                boundary = exchange[-1]
        if boundary is None:
            return None
        positions = {item.id: index for index, item in enumerate(history)}
        existing_boundary = latest_cold_boundary_index(history, positions)
        if existing_boundary is not None and positions[boundary.id] <= existing_boundary:
            return None
        return Message.create(
            session_id=boundary.session_id,
            seq=1,
            message=ModelMessage.assistant(content=""),
            metadata={
                "kind": MessageKind.MAINTENANCE,
                "maintenance_type": "cold_compact",
                "cold_cleared_through_message_id": boundary.id,
                "clear_tool_result_threshold_tokens": self._clear_threshold,
            },
        )

    async def summary_candidate(
        self,
        source_messages: Sequence[ModelMessage],
        *,
        boundary: Message,
        source_message_count: int,
    ) -> Message:
        """Summarize the already projected pre-cold context into a candidate."""

        # 请求构造归 compactor（摘要策略在这里）；体积测量归 service——
        # 同一变量先测后发，测量天然覆盖策略 prompt 与 continuation 指令。
        request = [
            ModelMessage.system(SUMMARY_SYSTEM_PROMPT),
            *source_messages,
            ModelMessage.human(build_summary_request_context()),
        ]
        input_tokens = self._counter.measure_request(request)
        if input_tokens > self._summary_input_budget_tokens:
            raise SummaryInputBudgetError(
                "Conversation history cannot fit the summary model input budget while "
                f"reserving {self._summary_max_tokens} output tokens and the normal "
                f"response reserve (estimated={input_tokens}, "
                f"budget={self._summary_input_budget_tokens})."
            )
        try:
            result = await self._model.ainvoke(
                messages=request,
                tools=(),
                tool_choice=None,
                max_output_tokens=self._summary_max_tokens,
            )
        except Exception as exc:
            raise ContextCompactionError(f"Summary model request failed: {exc}") from exc
        if result.tool_calls:
            raise ContextCompactionError(
                "Summary model returned a tool call instead of a plain summary."
            )
        summary = result.message.content.strip()
        output_tokens = self._counter.estimate_text(summary)
        if not summary:
            raise ContextCompactionError("Summary model returned an empty summary.")
        if output_tokens > self._summary_max_tokens:
            raise ContextCompactionError(
                "Summary model returned more tokens than the configured summary budget."
            )
        return Message.create(
            session_id=boundary.session_id,
            seq=1,
            message=ModelMessage.assistant(content=summary),
            metadata={
                "kind": MessageKind.SUMMARY,
                "covered_through_message_id": boundary.id,
                "source_message_count": source_message_count,
                "summary_input_tokens": input_tokens,
                "summary_output_tokens": output_tokens,
                "usage": _usage_payload(result),
            },
        )

    def _project_summary_message(self, summary: Message) -> ModelMessage:
        """把旧摘要投影为参与新摘要源的 <conversation_summary> 消息。"""

        return ModelMessage(
            role=summary.role,
            content=self._summary_projector.wrap_conversation_summary(
                summary.content
            ),
            tool_call_id=summary.tool_call_id,
            tool_name=summary.tool_name,
            tool_calls=summary.tool_calls,
        )

    def _truncate_older_tail(
        self,
        older_tail: Sequence[Message],
        summary_projection: ModelMessage | None,
    ) -> list[Message]:
        """从最老一侧按完整工具交换截断 older_tail 直至满足目标水位。

        截断优先级：保护区 > 旧摘要 > older_tail——只有 older_tail 可截。
        截断单位是完整交换（assistant(tool_calls) + 对应结果），孤儿
        tool_call 不允许进入摘要源。单条截断（投影器的
        max_tool_result_tokens）保证每组交换投影体积有上界，循环必然
        终止。
        """

        tail = list(older_tail)
        while not self._fits_summary_target(tail, summary_projection):
            cut = first_exchange_end(tail)
            if cut >= len(tail):
                return []
            tail = tail[cut:]
        return tail

    def _fits_summary_target(
        self,
        older_tail: Sequence[Message],
        summary_projection: ModelMessage | None,
    ) -> bool:
        """估算"旧摘要 + older_tail 摘要投影"是否在目标水位内。"""

        parts: list[ModelMessage] = []
        if summary_projection is not None:
            parts.append(summary_projection)
        parts.extend(self._summary_projector.project(older_tail))
        return (
            self._counter.estimate_messages(parts)
            <= self._summary_source_target_tokens
        )


def _usage_payload(result: ModelTurn) -> dict[str, int]:
    usage = result.usage
    if usage is None:
        return {}
    return {
        "input_tokens": int(usage.input_tokens),
        "output_tokens": int(usage.output_tokens),
        "total_tokens": int(usage.total_tokens),
    }


__all__ = ["ContextCompactor"]
