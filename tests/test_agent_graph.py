from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase

from agent.application.context.projector import MessageContextProjector
from agent.domain.messages import MessageKind
from agent.domain.model_messages import ModelMessage, ModelTurn, ModelUsage, ToolCall
from agent.domain.tools import ToolDefinition, ToolResult
from agent.graph.agent_graph import AgentGraph, AgentGraphExecutionError
from agent.graph.hook_registry import AgentHookRegistry
from agent.metatools import ExecuteMcpTool, SearchMcpTool


def _graph(hook_registry: AgentHookRegistry | None = None) -> AgentGraph:
    return AgentGraph(
        max_turns=4,
        hook_registry=hook_registry or AgentHookRegistry(),
    )


class _SeqSessions:
    """inbound_last_seq 恒返回固定水位的假 SessionService。"""

    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    async def inbound_last_seq(self, session_id: str) -> int:
        self.calls += 1
        return self.value


def _graph_with_sessions(sessions: _SeqSessions) -> AgentGraph:
    return AgentGraph(
        max_turns=4,
        hook_registry=AgentHookRegistry(),
        session_service=sessions,
    )


def _record(messages: list[tuple[ModelMessage, dict[str, object]]]):
    async def after_model(state):
        assistant = state["messages"][-1]
        metadata = {
            "kind": (
                MessageKind.ASSISTANT_TOOL_CALL
                if assistant.tool_calls
                else MessageKind.ASSISTANT_ANSWER
            )
        }
        messages.append((assistant, metadata))

    async def after_tool(state):
        recorded_ids = {m.tool_call_id for m, _ in messages if m.role == "tool"}
        for index, message in enumerate(state["messages"]):
            if message.role == "tool" and message.tool_call_id not in recorded_ids:
                kind = (
                    MessageKind.MCP_TOOL_DEFINITION
                    if "schema" in message.content or "ok" in message.content
                    else MessageKind.TOOL_RESULT
                )
                messages.append((message, {"kind": kind}))
                recorded_ids.add(message.tool_call_id)
                wrapped = (
                    MessageContextProjector.wrap_skill_guidance(message.content)
                    if kind == MessageKind.SKILL_RESULT
                    else MessageContextProjector.wrap_untrusted_content(message.content)
                )
                state["messages"][index] = ModelMessage.tool(
                    name=message.tool_name or "tool",
                    tool_call_id=message.tool_call_id or "unknown",
                    content=wrapped,
                )

    return after_model, after_tool


class AgentGraphTests(IsolatedAsyncioTestCase):
    async def test_model_can_answer_with_only_fixed_meta_tools(self) -> None:
        model = _AnswerOnlyModel()
        result = await _graph().ainvoke(
            messages=[ModelMessage.human("你好")],
            model=model,
            tools=(
                _ExampleTool("search_mcp"),
                _ExampleTool("load_skill"),
                _ExampleTool("execute_mcp"),
            ),
            caller_session_id="test-session",
        )

        self.assertEqual("普通回答", result["messages"][-1].content)
        self.assertEqual(
            [["search_mcp", "load_skill", "execute_mcp"]],
            model.tool_names,
        )

    async def test_search_definition_is_message_context_then_execute_mcp(self) -> None:
        registry = _FakeMcpRegistry()
        model = _DiscoveryModel()
        persisted: list[tuple[ModelMessage, dict[str, object]]] = []
        after_model, after_tool = _record(persisted)
        hook_registry = AgentHookRegistry()
        hook_registry.register_after_model(after_model)
        hook_registry.register_after_tool(after_tool)

        result = await _graph(hook_registry).ainvoke(
            messages=[ModelMessage.human("echo Anna")],
            model=model,
            tools=(SearchMcpTool(registry), _ExampleTool("load_skill"), ExecuteMcpTool(registry)),
            caller_session_id="test-session",
        )

        self.assertEqual("done", result["messages"][-1].content)
        self.assertEqual(
            [
                ["search_mcp", "load_skill", "execute_mcp"],
                ["search_mcp", "load_skill", "execute_mcp"],
                ["search_mcp", "load_skill", "execute_mcp"],
            ],
            model.tool_names,
        )
        tool_results = [message for message, _ in persisted if message.role == "tool"]
        self.assertTrue(json.loads(tool_results[0].content)["ok"])
        self.assertEqual("hello Anna", tool_results[1].content)
        expected_kind_index = 0 if persisted[0][0].role == "tool" else 1
        self.assertEqual(
            MessageKind.MCP_TOOL_DEFINITION,
            persisted[expected_kind_index][1]["kind"],
        )
        self.assertIn("<untrusted_content>", model.tool_contexts[-1])

    async def test_model_failure_keeps_completed_messages_but_creates_no_failure_message(
        self,
    ) -> None:
        persisted: list[tuple[ModelMessage, dict[str, object]]] = []
        after_model, after_tool = _record(persisted)
        hook_registry = AgentHookRegistry()
        hook_registry.register_after_model(after_model)
        hook_registry.register_after_tool(after_tool)
        with self.assertRaises(AgentGraphExecutionError) as raised:
            await _graph(hook_registry).ainvoke(
                messages=[ModelMessage.human("run")],
                model=_FailAfterToolModel(),
                tools=(_ExampleTool("execute_mcp"),),
                caller_session_id="test-session",
            )

        self.assertEqual("model failed", str(raised.exception))
        self.assertEqual(["assistant", "tool"], [message.role for message, _ in persisted])

    async def test_multiple_calls_stay_in_one_assistant_message_and_each_has_tool_result(
        self,
    ) -> None:
        persisted: list[tuple[ModelMessage, dict[str, object]]] = []
        after_model, after_tool = _record(persisted)
        hook_registry = AgentHookRegistry()
        hook_registry.register_after_model(after_model)
        hook_registry.register_after_tool(after_tool)
        result = await _graph(hook_registry).ainvoke(
            messages=[ModelMessage.human("two")],
            model=_TwoCallModel(),
            tools=(_ExampleTool("execute_mcp"),),
            caller_session_id="test-session",
        )

        assistant_calls = [message for message, _ in persisted if message.tool_calls]
        self.assertEqual(1, len(assistant_calls))
        self.assertEqual(("a", "b"), tuple(call.id for call in assistant_calls[0].tool_calls))
        tool_messages = [message for message, _ in persisted if message.role == "tool"]
        self.assertEqual(("a", "b"), tuple(message.tool_call_id for message in tool_messages))
        self.assertEqual("done", result["messages"][-1].content)

    async def test_all_hook_phases_run_in_order_and_share_state(self) -> None:
        events: list[tuple[str, object]] = []
        hook_registry = AgentHookRegistry()

        async def before_model(state):
            events.append(("before_model", state.get("turns", 0)))
            state["scratch"]["before_model"] = state.get("turns", 0)

        async def after_model(state):
            events.append(("after_model", state["messages"][-1].content))
            state["scratch"]["after_model"] = True

        async def before_tool(state):
            events.append(
                (
                    "before_tool",
                    (
                        state["pending_tool_calls"][0].id,
                        state["scratch"].copy(),
                        tuple(item.id for item in state["pending_tool_calls"]),
                    ),
                )
            )
            state["scratch"]["before_tool"] = True

        async def after_tool(state):
            events.append(
                (
                    "after_tool",
                    (
                        state["scratch"].copy(),
                        state["messages"][-1].content,
                    ),
                )
            )

        hook_registry.register_before_model(before_model)
        hook_registry.register_after_model(after_model)
        hook_registry.register_before_tool(before_tool)
        hook_registry.register_after_tool(after_tool)

        result = await _graph(hook_registry).ainvoke(
            messages=[ModelMessage.human("call once")],
            model=_OneCallThenAnswerModel(),
            tools=(_ExampleTool("execute_mcp"),),
            caller_session_id="test-session",
        )

        self.assertEqual(
            [
                "before_model",
                "after_model",
                "before_tool",
                "after_tool",
                "before_model",
                "after_model",
            ],
            [name for name, _value in events],
        )
        before_tool_event = events[2][1]
        self.assertEqual("execute-1", before_tool_event[0])
        self.assertEqual(
            {"before_model": 0, "after_model": True},
            before_tool_event[1],
        )
        self.assertEqual(("execute-1",), before_tool_event[2])
        after_tool_event = events[3][1]
        self.assertEqual(
            {"before_model": 0, "after_model": True, "before_tool": True},
            after_tool_event[0],
        )
        self.assertEqual("hello ", after_tool_event[1])
        self.assertEqual("done", result["messages"][-1].content)

    async def test_hook_updates_are_chained_and_mutate_shared_scratch(self) -> None:
        hook_registry = AgentHookRegistry()

        async def before_model_one(state):
            state["scratch"]["model_one"] = True

        async def before_model_two(state):
            self.assertTrue(state["scratch"]["model_one"])

        async def after_model_one(state):
            state["scratch"]["after_one"] = True

        async def after_model_two(state):
            self.assertTrue(state["scratch"]["after_one"])

        async def before_tool_one(state):
            state["scratch"]["tool_one"] = True

        async def before_tool_two(state):
            self.assertTrue(state["scratch"]["tool_one"])

        async def after_tool_one(state):
            state["scratch"]["after_tool_one"] = True

        observed_after_tool_two: dict[str, object] = {}

        async def after_tool_two(state):
            self.assertTrue(state["scratch"]["after_tool_one"])
            observed_after_tool_two.update(state["scratch"])

        hook_registry.register_before_model(before_model_one)
        hook_registry.register_before_model(before_model_two)
        hook_registry.register_after_model(after_model_one)
        hook_registry.register_after_model(after_model_two)
        hook_registry.register_before_tool(before_tool_one)
        hook_registry.register_before_tool(before_tool_two)
        hook_registry.register_after_tool(after_tool_one)
        hook_registry.register_after_tool(after_tool_two)

        model = _OneCallThenAnswerModel()
        result = await _graph(hook_registry).ainvoke(
            messages=[ModelMessage.human("call once")],
            model=model,
            tools=(_ExampleTool("execute_mcp"),),
            caller_session_id="test-session",
        )

        self.assertEqual("done", result["messages"][-1].content)
        self.assertTrue(observed_after_tool_two["after_tool_one"])

    async def test_hook_failure_is_wrapped_with_its_phase_and_index(self) -> None:
        hook_registry = AgentHookRegistry()

        async def broken(_state):
            raise RuntimeError("policy exploded")

        hook_registry.register_before_tool(broken)
        with self.assertRaises(AgentGraphExecutionError) as raised:
            await _graph(hook_registry).ainvoke(
                messages=[ModelMessage.human("run")],
                model=_OneCallThenAnswerModel(),
                tools=(_ExampleTool("execute_mcp"),),
                caller_session_id="test-session",
            )

        self.assertEqual("before_tool hook 0 failed: policy exploded", str(raised.exception))

    async def test_send_message_to_main_marks_sent_to_main(self) -> None:
        model = _ReportThenAnswerModel()
        result = await _graph().ainvoke(
            messages=[ModelMessage.human("report")],
            model=model,
            tools=(_SendMessageTool(ok=True),),
            caller_session_id="test-session",
        )

        self.assertTrue(result["sent_to_main"])

    async def test_failed_send_message_does_not_mark_sent_to_main(self) -> None:
        model = _ReportThenAnswerModel()
        result = await _graph().ainvoke(
            messages=[ModelMessage.human("report")],
            model=model,
            tools=(_SendMessageTool(ok=False),),
            caller_session_id="test-session",
        )

        self.assertFalse(result["sent_to_main"])

    async def test_plain_answer_without_send_message_leaves_sent_to_main_false(self) -> None:
        result = await _graph().ainvoke(
            messages=[ModelMessage.human("hello")],
            model=_AnswerOnlyModel(),
            tools=(_SendMessageTool(),),
            caller_session_id="test-session",
        )

        self.assertFalse(result["sent_to_main"])

    async def test_ask_user_ends_turn_without_calling_model_again(self) -> None:
        """ask_user 置位后 after_tool 直接 END:模型不再被调,答案由用户下一条消息承载。"""
        model = _AskThenContinueModel()
        result = await _graph().ainvoke(
            messages=[ModelMessage.human("help")],
            model=model,
            tools=(_ExampleTool("ask_user"), _ExampleTool("execute_mcp")),
            caller_session_id="test-session",
        )

        self.assertEqual(1, model.turn)
        last = result["messages"][-1]
        self.assertEqual("tool", last.role)
        self.assertEqual("ask_user", last.tool_name)

    async def test_finish_waits_when_inbound_watermark_moved(self) -> None:
        """收尾前水位被推进(human 入站)→ 回炉 before_model,而不是 END。"""
        graph = _graph_with_sessions(_SeqSessions(11))
        state: dict = {"pending_tool_calls": [], "inbound_watermark": 10, "caller_session_id": "s"}

        self.assertEqual("before_model", await graph._route_after_model(state))
        # 消息持续进来就该持续处理,不设回炉上限(仅受 max_turns 预算约束)。
        self.assertEqual("before_model", await graph._route_after_model(state))

    async def test_finish_when_watermark_unchanged(self) -> None:
        graph = _graph_with_sessions(_SeqSessions(10))
        state: dict = {"pending_tool_calls": [], "inbound_watermark": 10, "caller_session_id": "s"}
        self.assertEqual("finish", await graph._route_after_model(state))

    async def test_no_watermark_never_drains_and_tools_short_circuit_query(self) -> None:
        sessions = _SeqSessions(999)
        graph = _graph_with_sessions(sessions)

        no_wm: dict = {"pending_tool_calls": [], "caller_session_id": "s"}
        self.assertEqual("finish", await graph._route_after_model(no_wm))

        with_tools: dict = {"pending_tool_calls": [ToolCall("c", "execute_mcp", {})]}
        self.assertEqual("before_tool", await graph._route_after_model(with_tools))
        self.assertEqual(0, sessions.calls)  # 两条捷径都不该查库

    async def test_ask_user_flag_set_mid_batch_still_runs_remaining_calls(self) -> None:
        """同轮混调:ask_user 之后的调用照常执行完,再走 END,不产生孤儿 tool_call。"""
        model = _AskMixedWithAnotherModel()
        result = await _graph().ainvoke(
            messages=[ModelMessage.human("help")],
            model=model,
            tools=(_ExampleTool("ask_user"), _ExampleTool("execute_mcp")),
            caller_session_id="test-session",
        )

        self.assertEqual(1, model.turn)
        roles = [(message.role, message.tool_name) for message in result["messages"]]
        self.assertEqual(
            [("human", None), ("assistant", None), ("tool", "ask_user"), ("tool", "execute_mcp")],
            roles,
        )


class _AnswerOnlyModel:
    def __init__(self) -> None:
        self.tool_names: list[list[str]] = []

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self.tool_names.append([tool.name for tool in tools])
        return ModelTurn(message=ModelMessage.assistant(content="普通回答"), tool_calls=())


class _AskThenContinueModel:
    """第一轮调 ask_user，第二轮本不该被调用；被调则 turn 计数暴露。"""

    def __init__(self) -> None:
        self.turn = 0

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            call = ToolCall(
                id="ask-1",
                name="ask_user",
                arguments={"question": "q?", "options": ["a", "b"]},
            )
            return ModelTurn(
                message=ModelMessage.assistant(content="q?", tool_calls=(call,)),
                tool_calls=(call,),
            )
        return ModelTurn(message=ModelMessage.assistant(content="我不该被调用"), tool_calls=())


class _AskMixedWithAnotherModel:
    """一次回复里 ask_user 在前、普通工具在后：两者都必须执行完。"""

    def __init__(self) -> None:
        self.turn = 0

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            calls = (
                ToolCall(id="ask-1", name="ask_user", arguments={"question": "q?"}),
                ToolCall(id="exec-1", name="execute_mcp", arguments={"name": "x"}),
            )
            return ModelTurn(
                message=ModelMessage.assistant(content="q?", tool_calls=calls),
                tool_calls=calls,
            )
        return ModelTurn(message=ModelMessage.assistant(content="我不该被调用"), tool_calls=())


class _SendMessageTool:
    """模拟 send_message：ok 由构造参数决定，content 是可解析的 JSON。"""

    def __init__(self, ok: bool = True) -> None:
        self._ok = ok
        self._definition = ToolDefinition(
            name="send_message",
            description="send a message",
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def aexecute(self, arguments, *, context=None) -> ToolResult:
        return ToolResult(
            content=json.dumps({"ok": self._ok, "recipient": arguments.get("recipient")})
        )


class _ReportThenAnswerModel:
    """第一轮发 send_message(recipient='main_agent')，第二轮给出最终文本。"""

    def __init__(self) -> None:
        self._turn = 0

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self._turn += 1
        if self._turn == 1:
            call = ToolCall(
                id="report-1",
                name="send_message",
                arguments={"recipient": "main_agent", "message": "done"},
            )
            return ModelTurn(
                message=ModelMessage.assistant(content="", tool_calls=(call,)),
                tool_calls=(call,),
            )
        return ModelTurn(message=ModelMessage.assistant(content="final"), tool_calls=())


class _ExampleTool:
    def __init__(self, name: str) -> None:
        self._definition = ToolDefinition(
            name=name,
            description="Echo a name",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def aexecute(self, arguments, *, context=None) -> ToolResult:
        return ToolResult(content=f"hello {arguments.get('name', '')}")


class _OneCallThenAnswerModel:
    def __init__(self) -> None:
        self.turn = 0
        self.tool_contexts: list[str] = []

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self.tool_contexts.extend(
            message.content for message in messages if message.role == "tool"
        )
        self.turn += 1
        if self.turn == 1:
            call = ToolCall(id="execute-1", name="execute_mcp", arguments={})
            return ModelTurn(
                message=ModelMessage.assistant(content="", tool_calls=(call,)),
                tool_calls=(call,),
            )
        return ModelTurn(message=ModelMessage.assistant(content="done"), tool_calls=())


class _FakeMcpRegistry:
    def search_tools(self, *, mcp: str, query: str, limit: int):
        if mcp != "example":
            raise ValueError("Unknown MCP server")
        return (
            ToolDefinition(
                name="echo",
                description="Echo a name",
                parameters={"type": "object", "properties": {"name": {"type": "string"}}},
                use_cases=("echo text",),
            ),
        )[:limit]

    async def execute_tool(self, *, mcp: str, tool: str, arguments):
        if mcp != "example" or tool != "echo":
            return ToolResult(content="unknown")
        return ToolResult(content=f"hello {arguments['name']}")


class _DiscoveryModel:
    def __init__(self) -> None:
        self.turn = 0
        self.tool_names: list[list[str]] = []
        self.tool_contexts: list[str] = []

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self.tool_names.append([tool.name for tool in tools])
        self.tool_contexts.extend(
            message.content for message in messages if message.role == "tool"
        )
        self.turn += 1
        if self.turn == 1:
            call = ToolCall(
                id="search-1",
                name="search_mcp",
                arguments={"mcp": "example", "query": "echo a name"},
            )
        elif self.turn == 2:
            call = ToolCall(
                id="execute-1",
                name="execute_mcp",
                arguments={"mcp": "example", "tool": "echo", "arguments": {"name": "Anna"}},
            )
        else:
            return ModelTurn(message=ModelMessage.assistant(content="done"), tool_calls=())
        return ModelTurn(
            message=ModelMessage.assistant(content="", tool_calls=(call,)),
            tool_calls=(call,),
        )


class _FailAfterToolModel:
    def __init__(self) -> None:
        self.turn = 0

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            call = ToolCall(id="execute-1", name="execute_mcp", arguments={"name": "Anna"})
            return ModelTurn(
                message=ModelMessage.assistant(content="", tool_calls=(call,)),
                tool_calls=(call,),
            )
        raise RuntimeError("model failed")


class _TwoCallModel:
    def __init__(self) -> None:
        self.turn = 0

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            calls = (
                ToolCall(id="a", name="execute_mcp", arguments={"name": "A"}),
                ToolCall(id="b", name="execute_mcp", arguments={"name": "B"}),
            )
            return ModelTurn(
                message=ModelMessage.assistant(content="", tool_calls=calls),
                tool_calls=calls,
                usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            )
        return ModelTurn(
            message=ModelMessage.assistant(content="done"),
            tool_calls=(),
            usage=ModelUsage(input_tokens=20, output_tokens=3, total_tokens=23),
        )
