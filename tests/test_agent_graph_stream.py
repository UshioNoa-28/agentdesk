"""AgentGraph 流式路径：LangGraph messages 模式捕获增量与工具事件旁路。

流式增量的产生机制：节点内 ``model.ainvoke`` 在 LangGraph 回调链上时自动
内部转流，messages 模式逐 chunk 上抛。因此测试用"实现了 ``_astream`` 的
LangChain 假模型 + 普通端口包装"来忠实模拟生产里的
``LangChainAgentModel -> ChatOpenAI`` 结构。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar
from unittest import IsolatedAsyncioTestCase

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from agent.domain.model_messages import ModelMessage, ModelTurn, ToolCall
from agent.domain.streaming import TextDelta, ToolResultEvent
from agent.domain.tools import ToolDefinition, ToolResult
from agent.graph.agent_graph import AgentGraph
from agent.graph.hook_registry import AgentHookRegistry


class _TokenChatModel(BaseChatModel):
    """每次调用按序流式吐出预设 token，模拟 ChatOpenAI 内部转流。"""

    scripts: ClassVar[list[list[str]]] = [
        ["让我", "查一下"],
        ["完", "成"],
    ]
    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-token"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("sync path unused")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        tokens = self.scripts[min(self.call_index, len(self.scripts) - 1)]
        self.call_index += 1
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="".join(tokens)))]
        )

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator:
        tokens = self.scripts[min(self.call_index, len(self.scripts) - 1)]
        self.call_index += 1
        for token in tokens:
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))


class _PortModel:
    """模拟 LangChainAgentModel：包装底层 chat model，决定工具调用。"""

    def __init__(self, underlying: Any) -> None:
        self._underlying = underlying

    async def ainvoke(self, *, messages, tools=(), tool_choice=None, max_output_tokens=None):
        response = await self._underlying.ainvoke([])
        if self._underlying.call_index == 1:
            call = ToolCall(id="call-1", name="echo", arguments={"name": "x"})
            return ModelTurn(
                message=ModelMessage.assistant(content=response.content, tool_calls=(call,)),
                tool_calls=(call,),
            )
        return ModelTurn(message=ModelMessage.assistant(content=response.content), tool_calls=())


class _RecordingSink:
    """记录全部发布事件的测试 sink。"""

    def __init__(self) -> None:
        self.events: list[object] = []

    async def publish(self, event: object) -> None:
        self.events.append(event)


class _EchoTool:
    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="echo",
            description="Echo a name",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def aexecute(self, arguments, *, context=None) -> ToolResult:
        return ToolResult(content=f"hello {arguments.get('name', '')}")


class AgentGraphStreamTests(IsolatedAsyncioTestCase):
    async def test_astream_publishes_deltas_tool_events_and_final_state(self) -> None:
        sink = _RecordingSink()
        graph = AgentGraph(max_turns=4, hook_registry=AgentHookRegistry())

        result = await graph.astream(
            messages=[ModelMessage.human("hi")],
            model=_PortModel(_TokenChatModel()),
            tools=(_EchoTool(),),
            caller_session_id="session-1",
            events=sink,
        )

        self.assertEqual(
            [
                TextDelta(content="让我"),
                TextDelta(content="查一下"),
                ToolCall(id="call-1", name="echo", arguments={"name": "x"}),
                ToolResultEvent(
                    call=ToolCall(id="call-1", name="echo", arguments={"name": "x"}),
                    content="hello x",
                ),
                TextDelta(content="完"),
                TextDelta(content="成"),
            ],
            sink.events,
        )
        # 终态语义与 ainvoke 一致：最后一条 assistant 消息、turn 计数。
        self.assertEqual("完成", result["messages"][-1].content)
        self.assertEqual(2, result["turns"])

    async def test_ainvoke_path_is_unchanged_by_streaming_support(self) -> None:
        """无 sink 时 ainvoke 行为不变：一次性调用，不产生事件。"""

        graph = AgentGraph(max_turns=4, hook_registry=AgentHookRegistry())

        result = await graph.ainvoke(
            messages=[ModelMessage.human("hi")],
            model=_PortModel(_TokenChatModel()),
            tools=(_EchoTool(),),
            caller_session_id="session-1",
        )

        self.assertEqual("完成", result["messages"][-1].content)
        self.assertEqual(2, result["turns"])

    async def test_astream_without_events_is_rejected(self) -> None:
        """astream 没有无出口的用法：缺 sink 直接拒绝，调用方应走 ainvoke。"""

        graph = AgentGraph(max_turns=4, hook_registry=AgentHookRegistry())

        with self.assertRaises(ValueError):
            await graph.astream(
                messages=[ModelMessage.human("hi")],
                model=_PortModel(_TokenChatModel()),
                tools=(_EchoTool(),),
                caller_session_id="session-1",
                events=None,
            )
