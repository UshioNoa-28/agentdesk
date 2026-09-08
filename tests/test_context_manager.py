from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from agent.application.context.compactor import ContextCompactor
from agent.application.context.manager import ContextManager
from agent.application.context.projector import (
    MessageContextProjector,
    SummaryContextProjector,
)
from agent.domain.exceptions import (
    ContextCompactionError,
    SystemPromptBudgetError,
)
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage, ToolCall
from agent.infrastructure.settings import ContextSettings
from tests.tokenizer_support import budget_token_service


class _Messages:
    def __init__(self, history: list[Message]) -> None:
        self.items = history
        self.added: list[Message] = []

    async def history(self, _session_id: str) -> list[Message]:
        return list(self.items)

    async def add(self, *, session_id, message, metadata=None) -> Message:
        stored = Message.create(
            session_id=session_id,
            seq=max((item.seq for item in self.items), default=0) + 1,
            message=message,
            metadata=metadata,
        )
        self.items.append(stored)
        self.added.append(stored)
        return stored


class _Compactor:
    def __init__(self, *, cold: Message | None, summary: Message) -> None:
        self.cold = cold
        self.summary = summary
        self.summary_source_messages: list[ModelMessage] | None = None
        self.summary_source_calls = 0
        self.summary_source_anchors: list[str | None] = []

    def summary_source(
        self,
        history,
        *,
        current_user_message_id,
    ):
        self.summary_source_calls += 1
        # 记录锚点语义：None = 手动压缩锚在末尾，非 None = 自动锚点。
        self.summary_source_anchors.append(current_user_message_id)
        if current_user_message_id is None:
            current_index = len(history)
        else:
            current_index = next(
                index
                for index, item in enumerate(history)
                if item.id == current_user_message_id
            )
        source = [
            item.to_model()
            for item in history[:current_index]
            if item.metadata["kind"]
            not in {MessageKind.MAINTENANCE, MessageKind.SUMMARY, MessageKind.SYSTEM}
        ]
        if not source:
            return None
        return source, history[current_index - 1]

    def cold_candidate(self, history, *, current_user_message_id):
        return self.cold

    async def summary_candidate(self, source_messages, *, boundary, source_message_count):
        self.summary_source_messages = list(source_messages)
        return self.summary


def _message(
    session_id: str,
    seq: int,
    message: ModelMessage,
    kind: MessageKind,
    **metadata,
) -> Message:
    return Message.create(
        session_id=session_id,
        seq=seq,
        message=message,
        metadata={"kind": kind, **metadata},
    )


class _SessionToolsDeps:
    """手动压缩路径的会话服务与工具注册表测试替身。"""

    def __init__(self, allowed_tools: tuple[str, ...] = ()) -> None:
        self.allowed_tools = allowed_tools
        self.get_calls: list[str] = []

    async def get(self, session_id: str):
        from agent.domain.entities import Session

        self.get_calls.append(session_id)
        return Session.create(title=f"session {session_id}", allowed_tools=self.allowed_tools)

    def get_tools(self, allowed_tools: tuple[str, ...]):
        return tuple(
            _StubTool(name) for name in allowed_tools if name in _STUB_TOOL_NAMES
        )


_STUB_TOOL_NAMES = frozenset(
    {
        "execute_python",
        "search_mcp",
        "execute_mcp",
        "load_skill",
        "search_memory",
        "define_subagent",
        "list_subagents",
        "send_message",
        "wait_for_replies",
    }
)


class _StubTool:
    def __init__(self, name: str) -> None:
        from agent.domain.tools import ToolDefinition

        self.definition = ToolDefinition(
            name=name,
            description=f"{name} stub",
            parameters={"type": "object", "properties": {}},
        )


class ContextManagerTests(IsolatedAsyncioTestCase):
    def _history(self) -> list[Message]:
        sid = "session-1"
        call = ToolCall("call-1", "execute_mcp", {})
        return [
            _message(sid, 1, ModelMessage.system("system persona"), MessageKind.SYSTEM),
            _message(sid, 2, ModelMessage.human("old"), MessageKind.USER),
            _message(
                sid,
                3,
                ModelMessage.assistant(content="", tool_calls=(call,)),
                MessageKind.ASSISTANT_TOOL_CALL,
            ),
            _message(
                sid,
                4,
                ModelMessage.tool(
                    name="execute_mcp", tool_call_id=call.id, content=("important raw result " * 30)
                ),
                MessageKind.TOOL_RESULT,
            ),
            _message(sid, 5, ModelMessage.human("current"), MessageKind.USER),
        ]

    def _settings(self, *, cold_target_ratio: float, compact_ratio: float = 0.5) -> ContextSettings:
        return ContextSettings(
            context_window_tokens=200,
            context_reserve_ratio=0,
            context_compact_threshold_ratio=compact_ratio,
            context_cold_compact_target_ratio=cold_target_ratio,
            context_max_recent_tool_calls=0,
        )

    def _manager(
        self,
        messages: _Messages,
        compactor: _Compactor,
        *,
        settings: ContextSettings | None = None,
    ) -> ContextManager:
        return ContextManager(
            message_service=messages,
            projector=MessageContextProjector(
                token_counter=budget_token_service(),
                max_tool_result_tokens=8_000,
                clear_tool_result_threshold_tokens=100,
            ),
            compactor=compactor,
            token_counter=budget_token_service(),
            settings=settings or self._settings(cold_target_ratio=0.0001, compact_ratio=0.01),
            session_service=_SessionToolsDeps(),
            meta_tools=_SessionToolsDeps(),
        )

    async def test_prepare_records_inbound_watermark_from_read_history(self) -> None:
        """水位 = 本次读到的 history 里 human 消息的最大 seq。

        assistant/tool 笔迹不抬水位;新的 human(含 source=agent 回报)才抬。
        """
        history = self._history()
        sid = history[0].session_id
        store = _Messages(history)
        manager = self._manager(
            store,
            _Compactor(cold=None, summary=None),
            settings=ContextSettings(
                context_window_tokens=2000,
                context_reserve_ratio=0,
                context_compact_threshold_ratio=0.9,
                context_cold_compact_target_ratio=0.5,
                context_max_recent_tool_calls=0,
            ),
        )

        await manager.prepare(session_id=sid, tools=())
        self.assertEqual(5, manager.inbound_watermark)

        store.items.append(
            _message(
                sid, 6, ModelMessage.assistant(content="own answer"), MessageKind.ASSISTANT_ANSWER
            )
        )
        await manager.prepare(session_id=sid, tools=())
        self.assertEqual(5, manager.inbound_watermark)

        store.items.append(
            _message(
                sid, 7, ModelMessage.human("late report"), MessageKind.USER, source="agent"
            )
        )
        await manager.prepare(session_id=sid, tools=())
        self.assertEqual(7, manager.inbound_watermark)

    async def test_failed_cold_candidate_is_not_saved_and_summary_sees_original_projection(self):
        history = self._history()
        sid = history[0].session_id
        cold = _message(
            sid,
            1,
            ModelMessage.assistant(content=""),
            MessageKind.MAINTENANCE,
            maintenance_type="cold_compact",
            cold_cleared_through_message_id=history[3].id,
            clear_tool_result_threshold_tokens=1,
        )
        summary = _message(
            sid,
            1,
            ModelMessage.assistant(content="short summary"),
            MessageKind.SUMMARY,
            covered_through_message_id=history[3].id,
        )
        messages = _Messages(history)
        compactor = _Compactor(cold=cold, summary=summary)
        manager = ContextManager(
            message_service=messages,
            projector=MessageContextProjector(
                token_counter=budget_token_service(),
                max_tool_result_tokens=8_000,
                clear_tool_result_threshold_tokens=100,
            ),
            compactor=compactor,
            token_counter=budget_token_service(),
            settings=self._settings(cold_target_ratio=0.0001, compact_ratio=0.01),
            session_service=_SessionToolsDeps(),
            meta_tools=_SessionToolsDeps(),
        )

        await manager.prepare(
            session_id=sid,
            tools=(),
        )

        self.assertEqual(
            [MessageKind.MAINTENANCE, MessageKind.SUMMARY],
            [item.metadata["kind"] for item in messages.added],
        )
        assert compactor.summary_source_messages is not None
        self.assertIn(
            "important raw result",
            " ".join(item.content for item in compactor.summary_source_messages),
        )

    async def test_successful_cold_candidate_is_saved_and_projected_from_reread_history(self):
        history = self._history()
        sid = history[0].session_id
        cold = _message(
            sid,
            1,
            ModelMessage.assistant(content=""),
            MessageKind.MAINTENANCE,
            maintenance_type="cold_compact",
            cold_cleared_through_message_id=history[3].id,
            clear_tool_result_threshold_tokens=1,
        )
        summary = _message(
            sid,
            1,
            ModelMessage.assistant(content="should not be used"),
            MessageKind.SUMMARY,
            covered_through_message_id=history[3].id,
        )
        messages = _Messages(history)
        compactor = _Compactor(cold=cold, summary=summary)
        manager = ContextManager(
            message_service=messages,
            projector=MessageContextProjector(
                token_counter=budget_token_service(),
                max_tool_result_tokens=1000,
                clear_tool_result_threshold_tokens=100,
            ),
            compactor=compactor,
            token_counter=budget_token_service(),
            settings=self._settings(cold_target_ratio=0.4, compact_ratio=0.5),
            session_service=_SessionToolsDeps(),
            meta_tools=_SessionToolsDeps(),
        )

        result = await manager.prepare(
            session_id=sid,
            tools=(),
        )

        self.assertEqual(
            [MessageKind.MAINTENANCE],
            [item.metadata["kind"] for item in messages.added],
        )
        self.assertIn("Tool result cleared during context compaction", result[3].content)
        self.assertEqual(0, compactor.summary_source_calls)

    async def test_summary_compact_is_single_shot_and_truncation_is_internal(self):
        """阶梯已删除：prepare 只调一次 summary_source，超预算截断在 compactor 内部。"""
        sid = "session-single-shot"
        history = [
            _message(sid, 1, ModelMessage.system("system prompt"), MessageKind.SYSTEM),
            _message(sid, 2, ModelMessage.human("old task"), MessageKind.USER),
        ]
        for index in range(5):
            call = ToolCall(f"call-{index}", "execute_mcp", {})
            history.extend(
                [
                    _message(
                        sid,
                        len(history) + 1,
                        ModelMessage.assistant(content="", tool_calls=(call,)),
                        MessageKind.ASSISTANT_TOOL_CALL,
                    ),
                    _message(
                        sid,
                        len(history) + 1,
                        ModelMessage.tool(
                            name=call.name,
                            tool_call_id=call.id,
                            content="tool output " * 30,
                        ),
                        MessageKind.TOOL_RESULT,
                    ),
                ]
            )
        current_user = _message(
            sid,
            len(history) + 1,
            ModelMessage.human("current task"),
            MessageKind.USER,
        )
        history.append(current_user)

        class _SingleShotCompactor(_Compactor):
            def __init__(self):
                super().__init__(
                    cold=None,
                    summary=_message(
                        sid,
                        1,
                        ModelMessage.assistant(content="short summary"),
                        MessageKind.SUMMARY,
                        # fake 的摘要必须声明覆盖边界，否则投影器无法用
                        # 它替换前缀，压完仍超预算（与真 compactor 的产出
                        # metadata 对齐）。
                        covered_through_message_id=history[-2].id,
                    ),
                )
                self.candidate_calls = 0

            def summary_source(self, history, *, current_user_message_id):
                self.summary_source_calls += 1
                current_index = next(
                    index
                    for index, item in enumerate(history)
                    if item.id == current_user_message_id
                )
                source = [item.to_model() for item in history[:current_index]
                          if item.metadata["kind"] is not MessageKind.SYSTEM]
                return source, history[current_index - 1]

            async def summary_candidate(self, source_messages, *, boundary, source_message_count):
                self.candidate_calls += 1
                return self.summary

        messages = _Messages(history)
        compactor = _SingleShotCompactor()
        settings = ContextSettings(
            context_window_tokens=400,
            context_reserve_ratio=0,
            context_compact_threshold_ratio=0.1,
            context_cold_compact_target_ratio=0.05,
            context_max_recent_tool_calls=4,
        )
        manager = ContextManager(
            message_service=messages,
            projector=MessageContextProjector(
                token_counter=budget_token_service(),
                max_tool_result_tokens=8_000,
                clear_tool_result_threshold_tokens=100,
            ),
            compactor=compactor,
            token_counter=budget_token_service(),
            settings=settings,
            session_service=_SessionToolsDeps(),
            meta_tools=_SessionToolsDeps(),
        )

        result = await manager.prepare(
            session_id=sid,
            tools=(),
        )

        self.assertEqual(1, compactor.summary_source_calls)
        self.assertEqual(1, compactor.candidate_calls)
        self.assertEqual([MessageKind.SUMMARY], [
            item.metadata["kind"] for item in messages.added
        ])
        self.assertEqual(1, len(messages.added))
        self.assertEqual(
            ["system", "assistant", "human"],
            [item.role for item in result],
        )

    async def test_summary_source_none_fails_the_turn_explicitly(self):
        """无新历史可摘时自动路径必须显式失败，不允许静默通过。"""
        history = self._history()
        sid = history[0].session_id

        class _NoSourceCompactor(_Compactor):
            def summary_source(self, history, *, current_user_message_id):
                self.summary_source_calls += 1
                return None

        messages = _Messages(history)
        manager = ContextManager(
            message_service=messages,
            projector=MessageContextProjector(
                token_counter=budget_token_service(),
                max_tool_result_tokens=8_000,
                clear_tool_result_threshold_tokens=100,
            ),
            compactor=_NoSourceCompactor(
                cold=None,
                summary=_message(
                    sid,
                    1,
                    ModelMessage.assistant(content="unused"),
                    MessageKind.SUMMARY,
                ),
            ),
            token_counter=budget_token_service(),
            settings=self._settings(cold_target_ratio=0.0001, compact_ratio=0.01),
            session_service=_SessionToolsDeps(),
            meta_tools=_SessionToolsDeps(),
        )

        with self.assertRaises(ContextCompactionError) as raised:
            await manager.prepare(session_id=sid, tools=())

        self.assertIn("no prior history that can be summarized", str(raised.exception))

    async def test_compact_session_performs_summary_compaction_directly(self) -> None:
        history = self._history()
        sid = "session-1"
        messages = _Messages(history)
        cold = _message(
            sid,
            6,
            ModelMessage.assistant(content=""),
            MessageKind.MAINTENANCE,
            maintenance_type="cold_compact",
            cold_cleared_through_message_id=history[3].id,
            clear_tool_result_threshold_tokens=100,
        )
        summary = _message(
            sid,
            7,
            ModelMessage.assistant(content="Summary"),
            MessageKind.SUMMARY,
            covered_through_message_id=history[3].id,
            source_message_count=3,
        )
        compactor = _Compactor(cold=cold, summary=summary)
        manager = self._manager(messages, compactor)

        result = await manager.compact_session(session_id=sid)

        self.assertEqual("compacted", result.status)
        self.assertEqual("summary", result.kind)
        self.assertIsNotNone(result.checkpoint_message_id)
        self.assertEqual(1, len(messages.added))
        self.assertEqual(MessageKind.SUMMARY, messages.added[0].metadata["kind"])

    async def test_compact_session_performs_summary_when_cold_is_none(self) -> None:
        history = self._history()
        sid = "session-1"
        messages = _Messages(history)
        summary = _message(
            sid,
            6,
            ModelMessage.assistant(content="Summary"),
            MessageKind.SUMMARY,
            covered_through_message_id=history[3].id,
            source_message_count=3,
        )
        compactor = _Compactor(cold=None, summary=summary)
        manager = self._manager(messages, compactor)

        result = await manager.compact_session(session_id=sid)

        self.assertEqual("compacted", result.status)
        self.assertEqual("summary", result.kind)
        self.assertIsNotNone(result.checkpoint_message_id)
        self.assertEqual(1, len(messages.added))
        self.assertEqual(MessageKind.SUMMARY, messages.added[0].metadata["kind"])

    async def test_compact_session_returns_noop_when_only_system_message_exists(self) -> None:
        sid = "fresh-session"
        history = [
            _message(sid, 1, ModelMessage.system("system prompt"), MessageKind.SYSTEM),
        ]
        messages = _Messages(history)
        compactor = _Compactor(
            cold=None,
            summary=_message(sid, 2, ModelMessage.assistant(content=""), MessageKind.SUMMARY),
        )
        manager = self._manager(messages, compactor)

        result = await manager.compact_session(session_id=sid)

        self.assertEqual("noop", result.status)
        self.assertIn("No eligible history", result.message)
        self.assertEqual(0, len(messages.added))
        # 手动压缩锚在末尾（不再是 user 锚点）。
        self.assertEqual([None], compactor.summary_source_anchors)

    async def test_compactor_never_summarizes_system_message(self) -> None:
        sid = "session-sys-compactor"
        history = [
            _message(sid, 1, ModelMessage.system("system prompt"), MessageKind.SYSTEM),
            _message(sid, 2, ModelMessage.human("user query"), MessageKind.USER),
            _message(
                sid, 3,
                ModelMessage.assistant(content="response"),
                MessageKind.ASSISTANT_ANSWER,
            ),
            _message(sid, 4, ModelMessage.human("current user"), MessageKind.USER),
        ]
        compactor = ContextCompactor(
            projector=MessageContextProjector(
                token_counter=budget_token_service(),
                max_tool_result_tokens=8_000,
                clear_tool_result_threshold_tokens=100,
            ),
            summary_projector=SummaryContextProjector(
                token_counter=budget_token_service(),
                max_tool_result_tokens=8_000,
                clear_tool_result_threshold_tokens=100,
            ),
            token_counter=budget_token_service(),
            model=None,
            settings=ContextSettings(
                context_window_tokens=20000,
                context_reserve_ratio=0.1,
                context_compact_threshold_ratio=0.8,
                context_cold_compact_target_ratio=0.4,
                context_max_recent_tool_calls=0,
                context_summary_max_tokens=1000,
            ),
        )
        source = compactor.summary_source(
            history,
            current_user_message_id=history[3].id,
        )
        self.assertIsNotNone(source)
        source_messages, _ = source
        for msg in source_messages:
            self.assertNotEqual("system", msg.role)
            self.assertNotIn("system prompt", msg.content)

    async def test_system_prompt_exceeding_budget_fails_explicitly(self) -> None:
        sid = "session-huge-sys"
        history = [
            _message(
                sid, 1,
                ModelMessage.system("huge system prompt content " * 100),
                MessageKind.SYSTEM,
            ),
            _message(sid, 2, ModelMessage.human("user query"), MessageKind.USER),
        ]
        manager = self._manager(
            _Messages(history),
            _Compactor(
                cold=None,
                summary=_message(sid, 3, ModelMessage.assistant(content=""), MessageKind.SUMMARY),
            ),
            settings=self._settings(cold_target_ratio=0.4, compact_ratio=0.8),
        )
        with self.assertRaises(SystemPromptBudgetError):
            await manager.prepare(session_id=sid, tools=())
