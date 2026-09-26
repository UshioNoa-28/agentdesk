from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from agent.application.context.compactor import ContextCompactor
from agent.application.context.manager import ContextManager
from agent.application.context.projector import (
    MessageContextProjector,
    SummaryContextProjector,
)
from agent.domain.context_settings import ContextSettings
from agent.domain.exceptions import (
    ContextCompactionError,
    SystemPromptBudgetError,
)
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage, ToolCall
from agent.domain.model_profile import ModelProfile
from tests.tokenizer_support import budget_token_service

_SETTINGS_FIELDS = frozenset(field.name for field in dataclasses.fields(ContextSettings))


def _source(profile: ModelProfile) -> SimpleNamespace:
    """把扁平档案包成 catalog 测试替身（每轮现取）。"""

    return SimpleNamespace(current_profile=lambda: profile)


def _profile(**overrides: object) -> ModelProfile:
    """小窗口测试档案：只剩事实字段，可按需覆盖。"""

    fields: dict[str, object] = {"context_window": 200, "max_output_tokens": 1}
    fields.update({key: value for key, value in overrides.items() if key not in _SETTINGS_FIELDS})
    return ModelProfile(**fields)  # type: ignore[arg-type]


def _settings(**overrides: object) -> ContextSettings:
    """小窗口下自洽的水位默认：触发 0.5、冷压 0.0001、R=1、C=1、无保护区。"""

    fields: dict[str, object] = {
        "compact_threshold_ratio": 0.5,
        "cold_compact_target_ratio": 0.0001,
        "max_recent_tool_calls": 0,
        "summary_max_tokens": 40,
        "max_tool_calls_per_turn": 1,
        "max_tool_result_tokens": 1,
    }
    fields.update({key: value for key, value in overrides.items() if key in _SETTINGS_FIELDS})
    return ContextSettings(**fields)  # type: ignore[arg-type]


# 投影器帽子（单条工具结果上限 8000、清除阈值 100）：大窗口下的字段默认值，
# 与被测的压缩水位分开注入。
_PROJECTOR_SETTINGS = ContextSettings(max_tool_result_tokens=8_000)


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

    def summary_source(self, history):
        self.summary_source_calls += 1
        # 锚在历史末尾（与真实 compactor 同语义）：整段对话都是可摘范围。
        source = [
            item.to_model()
            for item in history
            if item.metadata["kind"]
            not in {MessageKind.MAINTENANCE, MessageKind.SUMMARY, MessageKind.SYSTEM}
        ]
        if not source:
            return None
        return source, history[-1]

    def cold_candidate(self, history):
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

    def __init__(
        self,
        allowed_tools: tuple[str, ...] = (),
        *,
        main_session_id: str | None = None,
    ) -> None:
        self.allowed_tools = allowed_tools
        self.main_session_id = main_session_id
        self.get_calls: list[str] = []

    async def get(self, session_id: str):
        from agent.domain.entities import Session

        self.get_calls.append(session_id)
        return Session.create(
            title=f"session {session_id}",
            allowed_tools=self.allowed_tools,
            main_session_id=self.main_session_id,
        )

    def get_tools(self, allowed_tools: tuple[str, ...]):
        return tuple(_StubTool(name) for name in allowed_tools if name in _STUB_TOOL_NAMES)


class _McpRegistry:
    """默认 system prompt 活配置源之一：MCP 服务清单（测试固定为空）。"""

    def server_descriptions(self) -> tuple[tuple[str, str], ...]:
        return ()


class _SkillCatalog:
    """默认 system prompt 活配置源之一：Skill 元数据（测试固定为空）。"""

    def metadata(self) -> tuple:
        return ()


class _MemoryCatalog:
    """默认 system prompt 活配置源之一：记忆层路径清单，可按需撑大体积。"""

    def __init__(self, prompt: str = "user layer: /tmp/memory/user") -> None:
        self.prompt = prompt

    def load_prompt(self) -> str:
        return self.prompt


_STUB_TOOL_NAMES = frozenset(
    {
        "edit_file",
        "search_mcp",
        "execute_mcp",
        "load_skill",
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

    def _watermarks(self, *, cold_target_ratio: float, compact_ratio: float = 0.5):
        return _settings(
            compact_threshold_ratio=compact_ratio,
            cold_compact_target_ratio=cold_target_ratio,
        )

    def _manager(
        self,
        messages: _Messages,
        compactor: _Compactor,
        *,
        profile: ModelProfile | None = None,
        settings: ContextSettings | None = None,
        session_service: _SessionToolsDeps | None = None,
        memory_prompt: str | None = None,
    ) -> ContextManager:
        return ContextManager(
            message_service=messages,
            projector=MessageContextProjector(
                token_counter=budget_token_service(),
                settings=_PROJECTOR_SETTINGS,
            ),
            compactor=compactor,
            token_counter=budget_token_service(),
            catalog=_source(profile or _profile()),
            settings=settings or self._watermarks(cold_target_ratio=0.0001, compact_ratio=0.05),
            # 默认走 persona 重放分支：本文件的合成预算窗口只有 200~400 token，
            # 装不下 1000+ token 的活拼默认 prompt；活拼路径由专门测试显式覆盖。
            session_service=session_service
            or _SessionToolsDeps(main_session_id="main-1"),
            meta_tools=_SessionToolsDeps(),
            mcp_registry=_McpRegistry(),
            skill_catalog=_SkillCatalog(),
            memory_catalog=_MemoryCatalog(memory_prompt) if memory_prompt else _MemoryCatalog(),
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
            profile=_profile(context_window=2000),
            settings=_settings(
                compact_threshold_ratio=0.9,
                cold_compact_target_ratio=0.5,
                summary_max_tokens=200,
                max_tool_result_tokens=100,
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
            _message(sid, 7, ModelMessage.human("late report"), MessageKind.USER, source="agent")
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
        manager = self._manager(messages, compactor)

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
        manager = self._manager(
            messages,
            compactor,
            settings=self._watermarks(cold_target_ratio=0.4, compact_ratio=0.5),
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
                        covered_through_message_id=history[-1].id,
                    ),
                )
                self.candidate_calls = 0

            async def summary_candidate(self, source_messages, *, boundary, source_message_count):
                self.candidate_calls += 1
                return self.summary

        messages = _Messages(history)
        compactor = _SingleShotCompactor()
        manager = self._manager(
            messages,
            compactor,
            profile=_profile(context_window=400),
            settings=_settings(
                compact_threshold_ratio=0.1,
                cold_compact_target_ratio=0.05,
                max_recent_tool_calls=4,
                summary_max_tokens=40,
                max_tool_result_tokens=16,
            ),
        )

        result = await manager.prepare(
            session_id=sid,
            tools=(),
        )

        self.assertEqual(1, compactor.summary_source_calls)
        self.assertEqual(1, compactor.candidate_calls)
        self.assertEqual([MessageKind.SUMMARY], [item.metadata["kind"] for item in messages.added])
        self.assertEqual(1, len(messages.added))
        # fake 无保护区：摘要覆盖到最后一条消息（含当前问题），投影只剩
        # system + summary（summary 模型视角以 human 注入）。
        self.assertEqual(
            ["system", "human"],
            [item.role for item in result],
        )

    async def test_summary_source_none_fails_the_turn_explicitly(self):
        """无新历史可摘时自动路径必须显式失败，不允许静默通过。"""
        history = self._history()
        sid = history[0].session_id

        class _NoSourceCompactor(_Compactor):
            def summary_source(self, history):
                self.summary_source_calls += 1
                return None

        messages = _Messages(history)
        manager = self._manager(
            messages,
            _NoSourceCompactor(
                cold=None,
                summary=_message(
                    sid,
                    1,
                    ModelMessage.assistant(content="unused"),
                    MessageKind.SUMMARY,
                ),
            ),
        )

        with self.assertRaises(ContextCompactionError) as raised:
            await manager.prepare(session_id=sid, tools=())

        self.assertIn("nothing new arrived since the last summary", str(raised.exception))

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

    async def test_compactor_never_summarizes_system_message(self) -> None:
        sid = "session-sys-compactor"
        history = [
            _message(sid, 1, ModelMessage.system("system prompt"), MessageKind.SYSTEM),
            _message(sid, 2, ModelMessage.human("user query"), MessageKind.USER),
            _message(
                sid,
                3,
                ModelMessage.assistant(content="response"),
                MessageKind.ASSISTANT_ANSWER,
            ),
            _message(sid, 4, ModelMessage.human("current user"), MessageKind.USER),
        ]
        compactor_settings = _settings(
            compact_threshold_ratio=0.8,
            cold_compact_target_ratio=0.4,
            summary_max_tokens=1000,
            max_tool_result_tokens=1000,
        )
        compactor = ContextCompactor(
            projector=MessageContextProjector(
                token_counter=budget_token_service(),
                settings=compactor_settings,
            ),
            summary_projector=SummaryContextProjector(
                token_counter=budget_token_service(),
                settings=compactor_settings,
            ),
            token_counter=budget_token_service(),
            model=None,
            catalog=_source(_profile(context_window=20000, max_output_tokens=2000)),
            settings=compactor_settings,
        )
        source = compactor.summary_source(history)
        self.assertIsNotNone(source)
        source_messages, _ = source
        for msg in source_messages:
            self.assertNotEqual("system", msg.role)
            self.assertNotIn("system prompt", msg.content)

    async def test_system_prompt_exceeding_budget_fails_explicitly(self) -> None:
        """现拼的 system prompt 自身超预算时必须显式失败，而不是留给压缩循环。"""
        sid = "session-huge-sys"
        history = [
            _message(sid, 2, ModelMessage.human("user query"), MessageKind.USER),
        ]
        manager = self._manager(
            _Messages(history),
            _Compactor(
                cold=None,
                summary=_message(sid, 3, ModelMessage.assistant(content=""), MessageKind.SUMMARY),
            ),
            settings=self._watermarks(cold_target_ratio=0.4, compact_ratio=0.8),
            memory_prompt="huge memory layer path " * 100,
        )
        with self.assertRaises(SystemPromptBudgetError):
            await manager.prepare(session_id=sid, tools=())

    async def test_root_session_live_builds_system_prompt(self) -> None:
        """根会话每轮现拼默认 prompt：入库遗留的 SYSTEM 行不重放，只作对话切分。"""
        history = self._history()  # head 是 legacy "system persona" 行
        sid = history[0].session_id
        manager = self._manager(
            _Messages(history),
            _Compactor(cold=None, summary=None),
            profile=_profile(context_window=2000),
            settings=_settings(
                compact_threshold_ratio=0.9,
                cold_compact_target_ratio=0.5,
                summary_max_tokens=200,
                max_tool_result_tokens=100,
            ),
            session_service=_SessionToolsDeps(),
            memory_prompt="user layer: /tmp/memory/user",
        )

        result = await manager.prepare(session_id=sid, tools=())

        self.assertNotIn("system persona", result[0].content)
        self.assertIn("You are AgentDesk", result[0].content)
        self.assertIn("user layer: /tmp/memory/user", result[0].content)
        # 遗留行被切除，不进入对话投影
        self.assertEqual("human", result[1].role)
        self.assertEqual(5, len(result))

    async def test_subagent_session_replays_stored_persona(self) -> None:
        """非根会话（define_subagent 链）冻结重放入库 persona，不现拼活配置。"""
        history = self._history()
        sid = history[0].session_id
        manager = self._manager(
            _Messages(history),
            _Compactor(cold=None, summary=None),
            profile=_profile(context_window=2000),
            settings=_settings(
                compact_threshold_ratio=0.9,
                cold_compact_target_ratio=0.5,
                summary_max_tokens=200,
                max_tool_result_tokens=100,
            ),
            session_service=_SessionToolsDeps(main_session_id="main-1"),
        )

        result = await manager.prepare(session_id=sid, tools=())

        self.assertEqual("system persona", result[0].content)
        self.assertNotIn("You are AgentDesk", result[0].content)
        self.assertEqual(5, len(result))
