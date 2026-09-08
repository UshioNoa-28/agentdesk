from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase

from agent.application.context.compactor import ContextCompactor
from agent.application.context.projector import (
    MessageContextProjector,
    SummaryContextProjector,
)
from agent.application.services.token_counter_service import TokenCounterService
from agent.domain.entities import Session
from agent.domain.exceptions import SummaryInputBudgetError
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage, ModelTurn, ToolCall
from agent.infrastructure.settings import ContextSettings
from tests.tokenizer_support import budget_token_counter, budget_token_service


def _message(
    session_id: str,
    seq: int,
    message: ModelMessage,
    kind: MessageKind,
    **metadata: object,
) -> Message:
    return Message.create(
        session_id=session_id,
        seq=seq,
        message=message,
        metadata={"kind": kind, **metadata},
    )


class MessageProjectionTests(TestCase):
    """Message 历史的确定性投影测试。"""

    def test_projection_preserves_repository_order(self) -> None:
        first = _message("session-1", 1, ModelMessage.human("first"), MessageKind.USER)
        second = _message("session-1", 2, ModelMessage.human("second"), MessageKind.USER)

        projected = MessageContextProjector(
            token_counter=budget_token_service(),
        ).project([first, second])

        self.assertEqual(["first", "second"], [item.content for item in projected])

    def test_agent_message_between_call_and_result_is_deferred_after_result(self) -> None:
        """异步 agent 回报插在 assistant tool_calls 与 tool result 之间时，投影要
        把回报挪到 tool result 之后，保证 provider 协议顺序。"""

        call = ToolCall("call-1", "wait_for_replies", {})
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(content="", tool_calls=(call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        reply = _message(
            "session-1",
            2,
            ModelMessage.human("subagent report"),
            MessageKind.USER,
            source="agent",
        )
        result = _message(
            "session-1",
            3,
            ModelMessage.tool(name="wait_for_replies", tool_call_id=call.id, content="woke"),
            MessageKind.TOOL_RESULT,
        )

        projected = MessageContextProjector(
            token_counter=budget_token_service(),
        ).project([assistant, reply, result])

        self.assertEqual(["assistant", "tool", "human"], [item.role for item in projected])
        self.assertEqual("subagent report", projected[-1].content)

    def test_missing_tool_result_is_patched_with_view_only_synthetic_result(self) -> None:
        """崩溃残留的 open tool call 在视图中补合成结果，但不动持久化数据。"""

        call = ToolCall("call-1", "execute_mcp", {})
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(content="", tool_calls=(call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        user = _message("session-1", 2, ModelMessage.human("continue"), MessageKind.USER)

        projected = MessageContextProjector(
            token_counter=budget_token_service(),
        ).project([assistant, user])

        self.assertEqual(["assistant", "tool", "human"], [item.role for item in projected])
        self.assertEqual("call-1", projected[1].tool_call_id)
        self.assertIn("interrupted", projected[1].content)
        self.assertEqual("continue", projected[2].content)

    def test_multiple_calls_defer_all_interleaved_messages_until_every_result_lands(self) -> None:
        calls = (
            ToolCall("call-a", "execute_mcp", {}),
            ToolCall("call-b", "execute_mcp", {}),
        )
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(content="", tool_calls=calls),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        first_reply = _message(
            "session-1",
            2,
            ModelMessage.human("reply one"),
            MessageKind.USER,
            source="agent",
        )
        result_a = _message(
            "session-1",
            3,
            ModelMessage.tool(name="execute_mcp", tool_call_id="call-a", content="a"),
            MessageKind.TOOL_RESULT,
        )
        second_reply = _message(
            "session-1",
            4,
            ModelMessage.human("reply two"),
            MessageKind.USER,
            source="agent",
        )
        result_b = _message(
            "session-1",
            5,
            ModelMessage.tool(name="execute_mcp", tool_call_id="call-b", content="b"),
            MessageKind.TOOL_RESULT,
        )

        projected = MessageContextProjector(
            token_counter=budget_token_service(),
        ).project([assistant, first_reply, result_a, second_reply, result_b])

        self.assertEqual(
            ["assistant", "tool", "tool", "human", "human"],
            [item.role for item in projected],
        )
        self.assertEqual(["call-a", "call-b"], [item.tool_call_id for item in projected[1:3]])
        self.assertEqual(["reply one", "reply two"], [item.content for item in projected[3:]])


    def test_tool_result_is_wrapped_without_mutating_persisted_message(self) -> None:
        call = ToolCall("call-1", "search", {})
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(content="", tool_calls=(call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        stored = _message(
            "session-1",
            2,
            ModelMessage.tool(
                name="search",
                tool_call_id=call.id,
                content="external result",
            ),
            MessageKind.TOOL_RESULT,
        )

        projector = MessageContextProjector(
            token_counter=budget_token_service(),
        )
        projected = projector.project([assistant, stored])

        self.assertEqual(
            projector.wrap_untrusted_content("external result"),
            projected[-1].content,
        )
        self.assertEqual("external result", stored.content)

    def test_only_regular_tool_results_are_truncated_in_model_projection(self) -> None:
        calls = (
            ToolCall("call-tool", "execute_mcp", {}),
            ToolCall("call-skill", "load_skill", {}),
            ToolCall("call-definition", "search_mcp", {}),
        )
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(content="", tool_calls=calls),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        regular = _message(
            "session-1",
            2,
            ModelMessage.tool(
                name="execute_mcp",
                tool_call_id="call-tool",
                content="ordinary result " * 100,
            ),
            MessageKind.TOOL_RESULT,
        )
        skill = _message(
            "session-1",
            3,
            ModelMessage.tool(
                name="load_skill",
                tool_call_id="call-skill",
                content="skill guidance " * 100,
            ),
            MessageKind.SKILL_RESULT,
        )
        definition = _message(
            "session-1",
            4,
            ModelMessage.tool(
                name="search_mcp",
                tool_call_id="call-definition",
                content='{"tools":[' + '"definition"' * 100 + "]}",
            ),
            MessageKind.MCP_TOOL_DEFINITION,
        )
        projector = MessageContextProjector(
            token_counter=budget_token_service(),
            max_tool_result_tokens=40,
            clear_tool_result_threshold_tokens=100,
        )

        projected = projector.project([assistant, regular, skill, definition])

        self.assertIn("...[truncated; original result is retained]...", projected[1].content)
        self.assertEqual(projector.wrap_skill_guidance(skill.content), projected[2].content)
        self.assertEqual(
            projector.wrap_untrusted_content(definition.content),
            projected[3].content,
        )
        self.assertEqual(projected[1], projector.project_tool_result(regular))
        self.assertEqual("ordinary result " * 100, regular.content)

    def test_cold_checkpoint_is_derived_from_persisted_metadata(self) -> None:
        call = ToolCall("call-1", "search", {})
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(content="", tool_calls=(call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        stored = _message(
            "session-1",
            2,
            ModelMessage.tool(
                name="search",
                tool_call_id=call.id,
                content="external result",
            ),
            MessageKind.TOOL_RESULT,
        )
        checkpoint = _message(
            "session-1",
            3,
            ModelMessage.assistant(content=""),
            MessageKind.MAINTENANCE,
            maintenance_type="cold_compact",
            cold_cleared_through_message_id=stored.id,
            clear_tool_result_threshold_tokens=1,
        )

        projected = MessageContextProjector(
            token_counter=budget_token_service(),
            clear_tool_result_threshold_tokens=100,
        ).project([assistant, stored, checkpoint])

        self.assertEqual(
            "Tool result cleared during context compaction.\n"
            "The original result is not included in this model context.",
            projected[-1].content,
        )
        self.assertNotIn("external result", projected[-1].content)

    def test_summary_view_never_applies_cold_clearing(self) -> None:
        """双视角核心契约：cold checkpoint 只影响模型视角，摘要看全量。

        同一段"被 cold 清空的历史"，模型视角投影为占位文案，摘要视角
        必须保留原始结果——被摘要覆盖的历史从活跃视图移除后，摘要是
        唯一记录，清空即销毁两次。
        """
        call = ToolCall("call-1", "search", {})
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(content="", tool_calls=(call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        stored = _message(
            "session-1",
            2,
            ModelMessage.tool(
                name="search",
                tool_call_id=call.id,
                content="external result",
            ),
            MessageKind.TOOL_RESULT,
        )
        checkpoint = _message(
            "session-1",
            3,
            ModelMessage.assistant(content=""),
            MessageKind.MAINTENANCE,
            maintenance_type="cold_compact",
            cold_cleared_through_message_id=stored.id,
            clear_tool_result_threshold_tokens=1,
        )
        messages = [assistant, stored, checkpoint]

        model_view = MessageContextProjector(
            token_counter=budget_token_service(),
            clear_tool_result_threshold_tokens=100,
        ).project(messages)
        summary_view = SummaryContextProjector(
            token_counter=budget_token_service(),
            clear_tool_result_threshold_tokens=100,
        ).project(messages)

        self.assertIn("Tool result cleared during context compaction", model_view[-1].content)
        self.assertIn("external result", summary_view[-1].content)
        self.assertNotIn("Tool result cleared", summary_view[-1].content)

    def test_summary_view_hides_system_persona(self) -> None:
        """SYSTEM（会话 persona）不进入摘要源：摘要有自己的策略 prompt。"""

        messages = [
            _message("s", 1, ModelMessage.system("system persona"), MessageKind.SYSTEM),
            _message("s", 2, ModelMessage.human("user query"), MessageKind.USER),
        ]

        summary_view = SummaryContextProjector(
            token_counter=budget_token_service(),
        ).project(messages)

        self.assertEqual(["user query"], [item.content for item in summary_view])
        self.assertNotIn("system", [item.role for item in summary_view])

    def test_summary_view_keeps_summary_replacement_and_truncation(self) -> None:
        """摘要视角共享 SUMMARY 前缀替换与单条截断，行为与模型视角一致。"""

        covered = _message(
            "s", 1, ModelMessage.human("old question"), MessageKind.USER
        )
        summary = _message(
            "s",
            2,
            ModelMessage.assistant(content="previous summary"),
            MessageKind.SUMMARY,
            covered_through_message_id=covered.id,
        )
        follow_up = _message("s", 3, ModelMessage.human("new question"), MessageKind.USER)

        summary_view = SummaryContextProjector(
            token_counter=budget_token_service(),
        ).project([covered, summary, follow_up])

        self.assertEqual(
            [
                MessageContextProjector.wrap_conversation_summary("previous summary"),
                "new question",
            ],
            [item.content for item in summary_view],
        )

    def test_cold_checkpoint_uses_the_persisted_clear_threshold(self) -> None:
        call = ToolCall("call-1", "execute_mcp", {})
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(content="", tool_calls=(call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        stored = _message(
            "session-1",
            2,
            ModelMessage.tool(name=call.name, tool_call_id=call.id, content="short"),
            MessageKind.TOOL_RESULT,
        )
        checkpoint = _message(
            "session-1",
            3,
            ModelMessage.assistant(content=""),
            MessageKind.MAINTENANCE,
            maintenance_type="cold_compact",
            cold_cleared_through_message_id=stored.id,
            clear_tool_result_threshold_tokens=10_000,
        )

        projected = MessageContextProjector(
            token_counter=budget_token_service(),
            clear_tool_result_threshold_tokens=1,
        ).project([assistant, stored, checkpoint])

        self.assertIn("short", projected[-1].content)
        self.assertNotIn("Tool result cleared", projected[-1].content)

    def test_mcp_definition_is_never_cleared(self) -> None:
        call = ToolCall("call-1", "search_mcp", {})
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(content="", tool_calls=(call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        stored = _message(
            "session-1",
            2,
            ModelMessage.tool(
                name=call.name,
                tool_call_id=call.id,
                content='{"tools":[{"name":"search"}]}',
            ),
            MessageKind.MCP_TOOL_DEFINITION,
        )
        checkpoint = _message(
            "session-1",
            3,
            ModelMessage.assistant(content=""),
            MessageKind.MAINTENANCE,
            maintenance_type="cold_compact",
            cold_cleared_through_message_id=stored.id,
        )

        projected = MessageContextProjector(
            token_counter=budget_token_service(),
            clear_tool_result_threshold_tokens=100,
        ).project([assistant, stored, checkpoint])

        self.assertEqual(
            MessageContextProjector.wrap_untrusted_content(stored.content),
            projected[-1].content,
        )

    def test_projection_does_not_silently_drop_a_persisted_tool_result(self) -> None:
        assistant = _message(
            "session-1",
            1,
            ModelMessage.assistant(
                content="",
                tool_calls=(ToolCall("call-valid", "execute_mcp", {}),),
            ),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        valid = _message(
            "session-1",
            2,
            ModelMessage.tool(
                name="execute_mcp", tool_call_id="call-valid", content="valid"
            ),
            MessageKind.TOOL_RESULT,
        )
        orphan = _message(
            "session-1",
            3,
            ModelMessage.tool(
                name="execute_mcp", tool_call_id="call-missing", content="orphan"
            ),
            MessageKind.TOOL_RESULT,
        )

        projected = MessageContextProjector(
            token_counter=budget_token_service(),
        ).project([assistant, valid, orphan])

        self.assertEqual(["assistant", "tool", "tool"], [item.role for item in projected])
        self.assertIn("valid", projected[1].content)
        self.assertIn("orphan", projected[2].content)

    def test_summary_replaces_the_covered_prefix(self) -> None:
        old_user = _message("session-1", 1, ModelMessage.human("old"), MessageKind.USER)
        old_call = ToolCall("call-1", "execute_mcp", {})
        old_assistant = _message(
            "session-1",
            2,
            ModelMessage.assistant(content="", tool_calls=(old_call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        old_result = _message(
            "session-1",
            3,
            ModelMessage.tool(
                name=old_call.name, tool_call_id=old_call.id, content="old result"
            ),
            MessageKind.TOOL_RESULT,
        )
        current = _message("session-1", 4, ModelMessage.human("current"), MessageKind.USER)
        summary = _message(
            "session-1",
            5,
            ModelMessage.assistant(content="remember the old task"),
            MessageKind.SUMMARY,
            covered_through_message_id=old_result.id,
        )

        projected = MessageContextProjector(
            token_counter=budget_token_service(),
        ).project(
            [old_user, old_assistant, old_result, current, summary]
        )

        self.assertEqual(["assistant", "human"], [item.role for item in projected])
        self.assertIn("remember the old task", projected[0].content)
        self.assertEqual("current", projected[1].content)
        self.assertNotIn("old result", " ".join(item.content for item in projected))


class _SummaryModel:
    def __init__(self) -> None:
        self.calls = 0
        self.max_output_tokens: int | None = None

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self.calls += 1
        self.max_output_tokens = max_output_tokens
        assert tools == ()
        return ModelTurn(
            message=ModelMessage.assistant(content="GOALS\n- preserve the task"),
            tool_calls=(),
        )


class ContextCompactorCandidateTests(IsolatedAsyncioTestCase):
    @staticmethod
    def _history(
        session: Session,
        *,
        result_count: int = 3,
        result_chars: int = 100,
    ) -> list[Message]:
        history = [
            _message(session.id, 1, ModelMessage.human("old task"), MessageKind.USER)
        ]
        for index in range(result_count):
            call = ToolCall(f"call-{index}", "execute_mcp", {})
            history.append(
                _message(
                    session.id,
                    len(history) + 1,
                    ModelMessage.assistant(content="", tool_calls=(call,)),
                    MessageKind.ASSISTANT_TOOL_CALL,
                )
            )
            history.append(
                _message(
                    session.id,
                    len(history) + 1,
                    ModelMessage.tool(
                        name=call.name,
                        tool_call_id=call.id,
                        content="x" * result_chars,
                    ),
                    MessageKind.TOOL_RESULT,
                )
            )
        history.append(
            _message(
                session.id,
                len(history) + 1,
                ModelMessage.human("new task"),
                MessageKind.USER,
            )
        )
        return history

    @staticmethod
    def _compactor(
        model: _SummaryModel,
        *,
        max_recent: int = 1,
        context_window_tokens: int = 1_000,
    ) -> ContextCompactor:
        return ContextCompactor(
            projector=MessageContextProjector(
                token_counter=TokenCounterService(budget_token_counter()),
                max_tool_result_tokens=8_000,
                clear_tool_result_threshold_tokens=10,
            ),
            summary_projector=SummaryContextProjector(
                token_counter=TokenCounterService(budget_token_counter()),
                max_tool_result_tokens=8_000,
                clear_tool_result_threshold_tokens=10,
            ),
            token_counter=TokenCounterService(budget_token_counter()),
            model=model,
            settings=ContextSettings(
                context_window_tokens=context_window_tokens,
                context_reserve_ratio=0.1,
                context_compact_threshold_ratio=0.5,
                context_cold_compact_target_ratio=0.1,
                context_max_recent_tool_calls=max_recent,
                context_clear_tool_result_threshold_tokens=10,
                context_summary_max_tokens=77,
            ),
        )

    async def test_summary_source_truncates_oldest_exchanges_to_target(self) -> None:
        """前缀截断：摘要源超目标水位时从最老一组完整交换开始丢。"""
        history = self._history(
            Session.create(title="truncate"),
            result_count=4,
            result_chars=200,
        )
        current_user = history[-1]
        # window=550, reserve=0, summary_max=77 → summary_input_budget=473,
        # target=473*0.5=236。older_tail 全量投影 279 token 超目标；
        # 截掉最老一组交换（含开头 user 消息）后约 188 token，达标。
        compactor = self._compactor(
            _SummaryModel(),
            max_recent=1,
            context_window_tokens=550,
        )

        source = compactor.summary_source(
            history,
            current_user_message_id=current_user.id,
        )

        self.assertIsNotNone(source)
        assert source is not None
        source_messages, boundary = source
        # older_tail = user + 交换0..2（保护区=交换3）；截掉 user+交换0
        # 后剩交换 1、2，boundary 是交换 2 的 tool 结果（history[6]）。
        self.assertEqual(history[6].id, boundary.id)
        # 截断后源体积必须在目标水位内（无旧摘要，源即 older_tail 投影）。
        estimated = TokenCounterService(budget_token_counter()).estimate_messages(
            source_messages
        )
        self.assertLessEqual(estimated, 236)
        # 最老一组交换（call-0 的结果）已被截掉；被截的消息只是本轮
        # 不被摘要阅读，持久层原样保留。
        joined = " ".join(m.content for m in source_messages)
        self.assertEqual(2, joined.count("x" * 200))
        # 源不得以孤儿 tool 结果开头。
        self.assertNotEqual("tool", source_messages[0].role)

    async def test_summary_source_truncation_never_touches_summary_or_protected(self) -> None:
        """截断优先级：保护区 > 旧摘要 > older_tail；旧摘要永不丢。"""
        history = self._history(
            Session.create(title="prio"),
            result_count=3,
            result_chars=300,
        )
        # older_tail 里放一条旧摘要（覆盖到第一组交换的末尾）。
        old_summary = _message(
            history[0].session_id,
            len(history) + 1,
            ModelMessage.assistant(content="previous summary"),
            MessageKind.SUMMARY,
            covered_through_message_id=history[2].id,
        )
        # 重排：旧摘要必须在 current_user 之前才参与 summary_source。
        history.insert(len(history) - 1, old_summary)
        current_user = history[-1]
        # window=550 → target=236：旧摘要(~13) + 全量 older_tail(~208)
        # 超目标 → 截掉最老单元后达标，但旧摘要永不丢。
        compactor = self._compactor(
            _SummaryModel(),
            max_recent=1,
            context_window_tokens=550,
        )

        source = compactor.summary_source(
            history,
            current_user_message_id=current_user.id,
        )

        self.assertIsNotNone(source)
        assert source is not None
        source_messages, boundary = source
        # 旧摘要以 <conversation_summary> 参与源且永不丢失。
        self.assertTrue(
            any(
                "<conversation_summary>" in m.content
                and "previous summary" in m.content
                for m in source_messages
            )
        )
        # 截断只发生在 older_tail 内部：保护区（最后一组交换 = 交换 2，
        # 插入旧摘要后的 history[5:7]）不进源；全量 older_tail 超目标，
        # 截掉最老单元（user + 交换 0）后 boundary 落在交换 1 的
        # tool 结果（history[4]）。
        self.assertEqual(history[4].id, boundary.id)
        # 保护区内容不进摘要源。
        joined = " ".join(m.content for m in source_messages)
        self.assertEqual(1, joined.count("x" * 300))

    async def test_summary_source_returns_none_when_target_fits_nothing(self) -> None:
        """目标水位小到一组交换都装不下：no-op 而非错误/死循环。"""
        history = self._history(
            Session.create(title="tiny"),
            result_count=3,
            result_chars=400,
        )
        current_user = history[-1]
        # window=110, reserve=0, summary_max=40 → budget=70, target=35：
        # 一组 400 字符结果（约 100+ token）的交换装不下。
        compactor = self._compactor(
            _SummaryModel(),
            max_recent=1,
            context_window_tokens=110,
        )
        compactor._summary_max_tokens = 40
        compactor._summary_input_budget_tokens = 70
        compactor._summary_source_target_tokens = 35

        source = compactor.summary_source(
            history,
            current_user_message_id=current_user.id,
        )

        self.assertIsNone(source)

    async def test_manual_anchor_summarizes_the_latest_round(self) -> None:
        """手动锚点（None）：整段对话含最后一轮问答都可摘。

        自动锚点（最新 user）会把锚点之后的消息全部划出可摘范围——
        短会话里这几乎等于全部历史，手动 /compact 永远 noop。手动
        语义锚在末尾，最后一轮问答进入摘要源。
        """
        session = Session.create(title="manual")
        history = [
            _message(
                session.id, 1,
                ModelMessage.human("列出你的工具"),
                MessageKind.USER,
            ),
            _message(
                session.id, 2,
                ModelMessage.assistant(content="1. 联网搜索..."),
                MessageKind.ASSISTANT_ANSWER,
            ),
            _message(
                session.id, 3,
                ModelMessage.human("你不能执行python吗?"),
                MessageKind.USER,
            ),
            _message(
                session.id, 4,
                ModelMessage.assistant(content="不能。"),
                MessageKind.ASSISTANT_ANSWER,
            ),
        ]
        compactor = self._compactor(_SummaryModel(), max_recent=1)

        # 自动锚点：锚 = 最新 user（index 2）→ 源只含第一轮问答。
        auto_source = compactor.summary_source(
            history,
            current_user_message_id=history[2].id,
        )
        # 手动锚点：锚在末尾 → 源含全部两轮问答。
        manual_source = compactor.summary_source(
            history,
            current_user_message_id=None,
        )

        self.assertIsNotNone(auto_source)
        assert auto_source is not None
        auto_messages, auto_boundary = auto_source
        self.assertEqual(2, len(auto_messages))
        self.assertEqual(history[1].id, auto_boundary.id)

        self.assertIsNotNone(manual_source)
        assert manual_source is not None
        manual_messages, manual_boundary = manual_source
        self.assertEqual(4, len(manual_messages))
        self.assertEqual(history[3].id, manual_boundary.id)
        self.assertIn("不能。", " ".join(m.content for m in manual_messages))

    async def test_manual_anchor_still_respects_protection_zone(self) -> None:
        """手动锚点不豁免保护区：最近 N 组工具交换的原文仍不被摘要。"""
        history = self._history(
            Session.create(title="manual-protected"),
            result_count=3,
            result_chars=50,
        )
        compactor = self._compactor(_SummaryModel(), max_recent=1)

        source = compactor.summary_source(
            history,
            current_user_message_id=None,
        )

        self.assertIsNotNone(source)
        assert source is not None
        source_messages, boundary = source
        # 保护区 = 最后 1 组交换（交换 2，history[5:7]）；boundary 是
        # 保护区之前最后一条保留消息（交换 1 的 tool 结果，history[4]）。
        self.assertEqual(history[4].id, boundary.id)
        joined = " ".join(m.content for m in source_messages)
        # 最新交换（交换 2）的结果不进源；更早的交换在 older_tail 内可进。
        self.assertEqual(2, joined.count("x" * 50))
        """older_tail 为空（旧摘要之后全在保护区）：禁止摘要重写自己。"""
        session = Session.create(title="noprogress")
        # 历史：user + 一组交换 + 旧摘要（覆盖到交换末尾）+ current_user。
        # 保护 N=1：唯一的交换在保护区内 → older_tail 为空。
        call = ToolCall("call-0", "execute_mcp", {})
        history = [
            _message(session.id, 1, ModelMessage.human("task"), MessageKind.USER),
            _message(
                session.id,
                2,
                ModelMessage.assistant(content="", tool_calls=(call,)),
                MessageKind.ASSISTANT_TOOL_CALL,
            ),
            _message(
                session.id,
                3,
                ModelMessage.tool(
                    name=call.name,
                    tool_call_id=call.id,
                    content="result",
                ),
                MessageKind.TOOL_RESULT,
            ),
        ]
        old_summary = _message(
            session.id,
            4,
            ModelMessage.assistant(content="previous summary"),
            MessageKind.SUMMARY,
            covered_through_message_id=history[2].id,
        )
        history.append(old_summary)
        current_user = _message(
            session.id,
            5,
            ModelMessage.human("current task"),
            MessageKind.USER,
        )
        history.append(current_user)
        compactor = self._compactor(_SummaryModel(), max_recent=1)

        source = compactor.summary_source(
            history,
            current_user_message_id=current_user.id,
        )

        self.assertIsNone(source)

    async def test_wait_exchanges_do_not_consume_protected_seats(self) -> None:
        """纯 wait_for_replies 交换不占保护席、不进入 summary source。"""
        session = Session.create(title="wait")
        history = [
            _message(session.id, 1, ModelMessage.human("old task"), MessageKind.USER),
        ]
        history.append(
            _message(
                session.id, 2,
                ModelMessage.assistant(
                    content="",
                    tool_calls=(ToolCall("call-real", "execute_mcp", {}),),
                ),
                MessageKind.ASSISTANT_TOOL_CALL,
            )
        )
        history.append(
            _message(
                session.id, 3,
                ModelMessage.tool(name="execute_mcp", tool_call_id="call-real", content="x" * 100),
                MessageKind.TOOL_RESULT,
            )
        )
        for wait_idx in range(2):
            call = ToolCall(f"call-wait{wait_idx}", "wait_for_replies", {})
            history.append(
                _message(
                    session.id, len(history) + 1,
                    ModelMessage.assistant(content="", tool_calls=(call,)),
                    MessageKind.ASSISTANT_TOOL_CALL,
                )
            )
            history.append(
                _message(
                    session.id, len(history) + 1,
                    ModelMessage.tool(
                        name="wait_for_replies", tool_call_id=call.id, content='{"woke":false}'
                    ),
                    MessageKind.TOOL_RESULT,
                )
            )
        history.append(
            _message(session.id, len(history) + 1, ModelMessage.human("new task"), MessageKind.USER)
        )
        current = history[-1]
        compactor = self._compactor(_SummaryModel(), max_recent=1)

        # 唯一真实交换应受保护：冷清候选应为 None（无可清除项）。
        self.assertIsNone(
            compactor.cold_candidate(history, current_user_message_id=current.id)
        )

        # summary source 中不应出现 wait 结果，且真实工作的 100 字符结果应被保护。
        source = compactor.summary_source(history, current_user_message_id=current.id)
        self.assertIsNotNone(source)
        assert source is not None
        source_messages, _boundary = source
        combined = "".join(item.content for item in source_messages)
        self.assertNotIn("wait_for_replies", combined)
        self.assertNotIn("woke", combined)
        self.assertNotIn("x" * 100, combined)

    async def test_interleaved_agent_reply_does_not_break_wait_exchange_detection(self) -> None:
        """wait 调用与结果之间插入 agent 回报时，wait 交换仍应被整体剔除，
        而 agent 回报本身保留在 summary source 中。"""

        session = Session.create(title="wait-interleaved")
        history = [
            _message(session.id, 1, ModelMessage.human("old task"), MessageKind.USER),
        ]
        call = ToolCall("call-wait", "wait_for_replies", {})
        history.append(
            _message(
                session.id,
                2,
                ModelMessage.assistant(content="", tool_calls=(call,)),
                MessageKind.ASSISTANT_TOOL_CALL,
            )
        )
        history.append(
            _message(
                session.id,
                3,
                ModelMessage.human("subagent report"),
                MessageKind.USER,
                source="agent",
            )
        )
        history.append(
            _message(
                session.id,
                4,
                ModelMessage.tool(
                    name="wait_for_replies",
                    tool_call_id=call.id,
                    content='{"woke":false}',
                ),
                MessageKind.TOOL_RESULT,
            )
        )
        history.append(
            _message(session.id, 5, ModelMessage.human("new task"), MessageKind.USER)
        )
        current = history[-1]
        compactor = self._compactor(_SummaryModel(), max_recent=1)

        source = compactor.summary_source(history, current_user_message_id=current.id)
        self.assertIsNotNone(source)
        assert source is not None
        source_messages, _boundary = source
        combined = "".join(item.content for item in source_messages)
        self.assertNotIn("wait_for_replies", combined)
        self.assertNotIn("woke", combined)
        self.assertIn("subagent report", combined)


    async def test_cold_candidate_protects_the_latest_exchange_group(self) -> None:
        history = self._history(Session.create(title="cold"), result_count=3)
        compactor = self._compactor(_SummaryModel(), max_recent=1)

        candidate = compactor.cold_candidate(
            history,
            current_user_message_id=history[-1].id,
        )

        assert candidate is not None
        self.assertEqual(MessageKind.MAINTENANCE, candidate.metadata["kind"])
        self.assertEqual(history[4].id, candidate.metadata["cold_cleared_through_message_id"])

    async def test_cold_candidate_does_not_repeat_an_existing_boundary(self) -> None:
        history = self._history(Session.create(title="cold"), result_count=3)
        current_user_id = history[-1].id
        history.append(
            _message(
                history[0].session_id,
                len(history) + 1,
                ModelMessage.assistant(content=""),
                MessageKind.MAINTENANCE,
                maintenance_type="cold_compact",
                cold_cleared_through_message_id=history[4].id,
            )
        )
        compactor = self._compactor(_SummaryModel(), max_recent=1)

        candidate = compactor.cold_candidate(
            history,
            current_user_message_id=current_user_id,
        )

        self.assertIsNone(candidate)

    async def test_summary_source_keeps_summary_and_older_tail_only(self) -> None:
        session_id = "session-1"
        first_call = ToolCall("call-1", "execute_mcp", {})
        second_call = ToolCall("call-2", "execute_mcp", {})
        third_call = ToolCall("call-3", "execute_mcp", {})
        history = [
            _message(
                session_id,
                1,
                ModelMessage.human("old task"),
                MessageKind.USER,
            ),
            _message(
                session_id,
                2,
                ModelMessage.assistant(content="", tool_calls=(first_call,)),
                MessageKind.ASSISTANT_TOOL_CALL,
            ),
            _message(
                session_id,
                3,
                ModelMessage.tool(
                    name=first_call.name,
                    tool_call_id=first_call.id,
                    content="first result",
                ),
                MessageKind.TOOL_RESULT,
            ),
        ]
        summary = _message(
            session_id,
            4,
            ModelMessage.assistant(content="previous summary"),
            MessageKind.SUMMARY,
            covered_through_message_id=history[-1].id,
        )
        history.append(summary)
        second_assistant = _message(
            session_id,
            5,
            ModelMessage.assistant(content="", tool_calls=(second_call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        second_result = _message(
            session_id,
            6,
            ModelMessage.tool(
                name=second_call.name,
                tool_call_id=second_call.id,
                content="second result",
            ),
            MessageKind.TOOL_RESULT,
        )
        third_assistant = _message(
            session_id,
            7,
            ModelMessage.assistant(content="", tool_calls=(third_call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        third_result = _message(
            session_id,
            8,
            ModelMessage.tool(
                name=third_call.name,
                tool_call_id=third_call.id,
                content="third result",
            ),
            MessageKind.TOOL_RESULT,
        )
        current_user = _message(
            session_id,
            9,
            ModelMessage.human("current task"),
            MessageKind.USER,
        )
        cold_checkpoint = _message(
            session_id,
            10,
            ModelMessage.assistant(content=""),
            MessageKind.MAINTENANCE,
            maintenance_type="cold_compact",
            cold_cleared_through_message_id=second_result.id,
            clear_tool_result_threshold_tokens=1,
        )
        history.extend(
            [
                second_assistant,
                second_result,
                third_assistant,
                third_result,
                current_user,
                cold_checkpoint,
            ]
        )

        compactor = self._compactor(_SummaryModel(), max_recent=1)
        source = compactor.summary_source(
            history,
            current_user_message_id=current_user.id,
        )

        self.assertIsNotNone(source)
        assert source is not None
        source_messages, boundary = source
        self.assertEqual(second_result.id, boundary.id)
        self.assertEqual(
            [
                MessageContextProjector.wrap_conversation_summary(
                    "previous summary"
                ),
                "",
                MessageContextProjector.wrap_untrusted_content("second result"),
            ],
            [message.content for message in source_messages],
        )
        self.assertNotIn("third result", " ".join(message.content for message in source_messages))

    async def test_summary_source_without_summary_protects_latest_exchange(self) -> None:
        history = ContextCompactorCandidateTests._history(
            Session.create(title="summary"),
            result_count=3,
            result_chars=20,
        )
        current_user = history[-1]
        compactor = self._compactor(_SummaryModel(), max_recent=1)

        source = compactor.summary_source(
            history,
            current_user_message_id=current_user.id,
        )

        self.assertIsNotNone(source)
        assert source is not None
        source_messages, boundary = source
        self.assertEqual(history[4].id, boundary.id)
        self.assertEqual(5, len(source_messages))

    async def test_summary_source_accepts_checkpoint_appended_after_current_user(self) -> None:
        session_id = "session-1"
        call = ToolCall("call-1", "execute_mcp", {})
        old_user = _message(
            session_id,
            1,
            ModelMessage.human("old task"),
            MessageKind.USER,
        )
        old_assistant = _message(
            session_id,
            2,
            ModelMessage.assistant(content="", tool_calls=(call,)),
            MessageKind.ASSISTANT_TOOL_CALL,
        )
        old_result = _message(
            session_id,
            3,
            ModelMessage.tool(
                name=call.name,
                tool_call_id=call.id,
                content="old result",
            ),
            MessageKind.TOOL_RESULT,
        )
        tail_answer = _message(
            session_id,
            4,
            ModelMessage.assistant(content="tail answer"),
            MessageKind.ASSISTANT_ANSWER,
        )
        current_user = _message(
            session_id,
            5,
            ModelMessage.human("current task"),
            MessageKind.USER,
        )
        summary = _message(
            session_id,
            6,
            ModelMessage.assistant(content="previous summary"),
            MessageKind.SUMMARY,
            covered_through_message_id=old_result.id,
        )
        history = [old_user, old_assistant, old_result, tail_answer, current_user, summary]

        compactor = self._compactor(_SummaryModel(), max_recent=0)
        source = compactor.summary_source(
            history,
            current_user_message_id=current_user.id,
        )

        self.assertIsNotNone(source)
        assert source is not None
        source_messages, boundary = source
        self.assertEqual(tail_answer.id, boundary.id)
        self.assertEqual(
            [
                MessageContextProjector.wrap_conversation_summary("previous summary"),
                "tail answer",
            ],
            [message.content for message in source_messages],
        )

    async def test_summary_candidate_uses_the_configured_output_limit(self) -> None:
        model = _SummaryModel()
        compactor = self._compactor(model)
        history = self._history(Session.create(title="summary"), result_count=1)
        source = MessageContextProjector(
            token_counter=budget_token_service(),
        ).project(history[:-1])

        summary = await compactor.summary_candidate(
            source,
            boundary=history[-2],
            source_message_count=len(source),
        )

        self.assertEqual(1, model.calls)
        self.assertEqual(77, model.max_output_tokens)
        self.assertEqual(MessageKind.SUMMARY, summary.metadata["kind"])
        self.assertEqual(history[-2].id, summary.metadata["covered_through_message_id"])

    async def test_summary_input_budget_reserves_summary_output_tokens(self) -> None:
        model = _SummaryModel()
        compactor = ContextCompactor(
            projector=MessageContextProjector(
                token_counter=TokenCounterService(budget_token_counter()),
            ),
            summary_projector=SummaryContextProjector(
                token_counter=TokenCounterService(budget_token_counter()),
            ),
            token_counter=TokenCounterService(budget_token_counter()),
            model=model,
            settings=ContextSettings(
                context_window_tokens=100,
                context_reserve_ratio=0,
                context_compact_threshold_ratio=0.5,
                context_cold_compact_target_ratio=0.1,
                context_max_recent_tool_calls=1,
                context_summary_max_tokens=40,
            ),
        )

        with self.assertRaises(SummaryInputBudgetError):
            await compactor.summary_candidate(
                [ModelMessage.human("x" * 280)],
                boundary=_message(
                    "session-1",
                    1,
                    ModelMessage.human("boundary"),
                    MessageKind.USER,
                ),
                source_message_count=1,
            )

        self.assertEqual(0, model.calls)
