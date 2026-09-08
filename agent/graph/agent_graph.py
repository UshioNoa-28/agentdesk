"""Agent 的模型-固定 Meta Tool 异步循环。"""

from __future__ import annotations

import json
from collections.abc import Sequence

from langgraph.graph import END, START, StateGraph

from agent.domain.model_messages import ModelMessage, ToolCall
from agent.domain.multi_agent import MAIN_AGENT_NAME
from agent.domain.streaming import TextDelta, ToolResultEvent
from agent.domain.tools import AgentTool, ToolContext, ToolResult
from agent.graph.hooks import AgentHook, HookRegistryPort
from agent.graph.state import AgentState
from agent.ports.context import ContextManagerPort, PersistenceManagerPort
from agent.ports.model import AgentModelPort, AgentWorkflow
from agent.ports.runtime.stream import AgentStreamSink
from agent.ports.services import SessionServicePort


class AgentGraphExecutionError(RuntimeError):
    """模型或固定 Meta Tool 循环失败。"""


class AgentGraph(AgentWorkflow):
    """负责模型与固定 Meta Tool 循环，委托两个流程 Manager 完成上下文和持久化。"""

    def __init__(
        self,
        *,
        max_turns: int,
        hook_registry: HookRegistryPort,
        context_manager: ContextManagerPort | None = None,
        persistence_manager: PersistenceManagerPort | None = None,
        session_service: SessionServicePort | None = None,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self._max_turns = max_turns
        self._hook_registry = hook_registry
        self._context_manager = context_manager
        self._sessions = session_service
        self._persistence_manager = persistence_manager
        self._graph = self._build()

    def _build(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("before_model", self._before_model)
        workflow.add_node("model_turn", self._model_turn)
        workflow.add_node("after_model", self._after_model)
        workflow.add_node("before_tool", self._before_tool)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("after_tool", self._after_tool)

        workflow.add_edge(START, "before_model")
        workflow.add_edge("before_model", "model_turn")
        workflow.add_edge("model_turn", "after_model")
        workflow.add_conditional_edges(
            "after_model",
            self._route_after_model,
            {"before_tool": "before_tool", "before_model": "before_model", "finish": END},
            # before tool 是正常的 判断是否有工具调用需要结束 
            # before model 是判断是否在存在未读human message的情况
        )
        workflow.add_edge("before_tool", "execute_tools")
        workflow.add_edge("execute_tools", "after_tool")
        workflow.add_conditional_edges(
            "after_tool",
            self._route_after_tool,
            {"before_model": "before_model", "finish": END},
        )# 专门为ask_user meta tool 添加的条件边 如果 调用了ask user 则 直接到END 
        return workflow.compile()

    async def _before_model(self, state: AgentState) -> AgentState:
        """准备上下文，并运行 before_model Hook。"""

        if self._context_manager is not None:
            caller_session_id = state.get("caller_session_id")
            if (
                not isinstance(caller_session_id, str)
                or not caller_session_id.strip()
            ):
                raise AgentGraphExecutionError(
                    "Context preparation requires a non-empty caller_session_id."
                )
            try:
                prepared_messages = await self._context_manager.prepare(
                    session_id=caller_session_id,
                    tools=tuple(tool.definition for tool in state["tools"]),
                )
            except AgentGraphExecutionError:
                raise
            except Exception as exc:
                raise AgentGraphExecutionError(str(exc)) from exc
            state["messages"] = list(prepared_messages)
            # 水位由 prepare 从它刚读的那份 history 就地推导,不另起查询——
            # "模型看到的"与"水位记录的"保证是同一次读取。
            state["inbound_watermark"] = self._context_manager.inbound_watermark

        await self._run_hooks(
            state,
            self._hook_registry.before_model_hooks(),
            phase="before_model",
        )
        return state

    async def _model_turn(self, state: AgentState) -> AgentState:
        """调用模型生成 Assistant 回复，并执行持久化。"""

        turns = state.get("turns", 0)
        messages = list(state.get("messages", []))
        if turns >= self._max_turns:
            raise AgentGraphExecutionError(
                f"Agent exceeded max turns: {self._max_turns}"
            )

        model = state["model"]
        tools = state["tools"]
        try:
            result = await model.ainvoke(
                messages=messages,
                tools=tuple(tool.definition for tool in tools),
                tool_choice=None,
            )
            messages.append(result.message)
        except AgentGraphExecutionError:
            raise
        except Exception as exc:
            raise AgentGraphExecutionError(str(exc)) from exc

        state["messages"] = messages
        state["pending_tool_calls"] = list(result.tool_calls)
        state["turns"] = turns + 1

        if self._persistence_manager is not None:
            try:
                stored = await self._persistence_manager.persist_assistant(
                    session_id=_caller_session_id(state),
                    turn=result,
                )
                state["persisted_assistant"] = stored
            except AgentGraphExecutionError:
                raise
            except Exception as exc:
                raise AgentGraphExecutionError(str(exc)) from exc

        return state

    async def _after_model(self, state: AgentState) -> AgentState:
        """运行 after_model Hook。"""

        await self._run_hooks(
            state,
            self._hook_registry.after_model_hooks(),
            phase="after_model",
        )
        return state

    async def _route_after_model(self, state: AgentState) -> str:
        """有工具先跑工具;想收尾前核对入站水位,晚到消息则回炉重读。

        消息无限进就一直处理——每轮回炉的 before_model 都会消费它们并刷新
        水位,风浪自息;仅受 max_turns 预算约束。ask_user 出口走
        ``_route_after_tool`` 不经过这里——等人就是等人,不因水位回炉。
        已知残留:本检查在仍持锁期间进行,"判完到 actor 释放锁"之间有亚毫秒
        窗口,单用户本地可忽略;要归零需 actor 层"先释放后复查"。
        """

        if state.get("pending_tool_calls"):
            return "before_tool"
        watermark = state.get("inbound_watermark")
        caller_session_id = state.get("caller_session_id")
        if (
            self._sessions is not None
            and isinstance(watermark, int)
            and isinstance(caller_session_id, str)
            and caller_session_id
        ):
            if await self._sessions.inbound_last_seq(caller_session_id) > watermark:
                return "before_model"
        return "finish"

    @staticmethod
    async def _route_after_tool(state: AgentState) -> str:
        """ask_user 调用即收尾:答案在语义上就是用户的下一条消息,不再回模型。"""

        return "finish" if state.get("ask_user_called") else "before_model"

    async def _before_tool(self, state: AgentState) -> AgentState:
        """运行 before_tool Hook。"""

        await self._run_hooks(
            state,
            self._hook_registry.before_tool_hooks(),
            phase="before_tool",
        )
        return state

    async def _execute_tools(self, state: AgentState) -> AgentState:
        """顺序执行所有待处理的固定 Meta Tool 调用，并按序完成持久化。

        注意：必须串行——工具共享请求作用域的 AsyncSession/UnitOfWork，
        并发执行会触发 SQLAlchemy "concurrent operations not permitted"。
        """

        messages = list(state.get("messages", []))
        calls = tuple(state.get("pending_tool_calls", ()))
        events = state.get("events")

        try:
            for call in calls:
                if events is not None:
                    await events.publish(call)
                result = await self._execute_tool(state, call)
                if _is_report_to_main(call, result):
                    state["sent_to_main"] = True
                if call.name == "ask_user":
                    state["ask_user_called"] = True
                if events is not None:
                    await events.publish(ToolResultEvent(call=call, content=result.content))
                model_message = ModelMessage.tool(
                    name=call.name,
                    tool_call_id=call.id,
                    content=result.content,
                )
                if self._persistence_manager is not None:
                    model_message = await self._persistence_manager.persist_tool(
                        session_id=_caller_session_id(state),
                        result=result,
                        message=model_message,
                    )
                messages.append(model_message)
        except AgentGraphExecutionError:
            raise
        except Exception as exc:
            raise AgentGraphExecutionError(str(exc)) from exc

        state["messages"] = messages
        state["pending_tool_calls"] = []
        return state

    async def _after_tool(self, state: AgentState) -> AgentState:
        """运行 after_tool Hook。"""

        await self._run_hooks(
            state,
            self._hook_registry.after_tool_hooks(),
            phase="after_tool",
        )
        return state

    async def _execute_tool(
        self,
        state: AgentState,
        call: ToolCall,
    ) -> ToolResult:
        """执行一个固定 Meta Tool。"""

        tools_by_name = {tool.definition.name: tool for tool in state["tools"]}
        tool = tools_by_name.get(call.name)
        if tool is None:
            return _error_result(
                code="unknown_meta_tool",
                message=f"Tool is not one of the fixed Meta Tools: {call.name}",
                tool_name=call.name,
            )
        tool_context = ToolContext(caller_session_id=_caller_session_id(state))
        try:
            result = await tool.aexecute(
                call.arguments,
                context=tool_context,
            )
        except Exception as exc:
            return _error_result(
                code="meta_tool_execution_failed",
                message=str(exc),
                tool_name=call.name,
            )
        if not isinstance(result, ToolResult):
            return _error_result(
                code="invalid_tool_result",
                message="Tool returned an invalid result object",
                tool_name=call.name,
            )
        return result

    @staticmethod
    async def _run_hooks(
        state: AgentState,
        hooks: Sequence[AgentHook],
        *,
        phase: str,
    ) -> None:
        """按注册顺序运行 Hook（中间件风格，传引用）。"""

        for index, hook in enumerate(hooks):
            try:
                outcome = await hook(state)
                if isinstance(outcome, dict):
                    state.update(outcome)
            except AgentGraphExecutionError:
                raise
            except Exception as exc:
                raise AgentGraphExecutionError(
                    f"{phase} hook {index} failed: {exc}"
                ) from exc

    async def ainvoke(
        self,
        *,
        messages: list[ModelMessage],
        model: AgentModelPort,
        tools: Sequence[AgentTool],
        caller_session_id: str | None = None,
    ) -> dict[str, object]:
        """运行模型和工具循环；caller_session_id 是工具上下文的唯一会话来源。

        非流式路径不携带事件 sink：state 里的 ``events`` 由 ``astream``
        负责写入，``_execute_tools`` 在缺失时自动跳过事件发布。
        """

        initial_state: AgentState = {
            "messages": list(messages),
            "scratch": {},
            "pending_tool_calls": [],
            "turns": 0,
            "model": model,
            "tools": list(tools),
            "caller_session_id": caller_session_id,
        }
        state = await self._graph.ainvoke(initial_state)
        return {
            "messages": state.get("messages", []),
            "turns": state.get("turns", 0),
            "persisted_assistant": state.get("persisted_assistant"),
            "sent_to_main": state.get("sent_to_main", False),
        }

    async def astream(
        self,
        *,
        messages: list[ModelMessage],
        model: AgentModelPort,
        tools: Sequence[AgentTool],
        caller_session_id: str | None = None,
        events: AgentStreamSink,
    ) -> dict[str, object]:
        """流式运行一次工作流，返回与 ``ainvoke`` 同形的结果 dict。

        LangGraph 双模式流在这里各司其职：

        - ``messages``：底层模型的 token chunk。``ainvoke`` 挂在 LangGraph
          回调链上时会自动内部转流，chunk 从这里逐个上抛 → 转 ``TextDelta``
          发布给 sink，供 SSE 增量输出；
        - ``values``：每个超步后的全量 state，循环结束时的最后一个即终态
          → 从中取 ``persisted_assistant`` 作为返回值，供 ask RPC 使用。

        ``events`` 是往 StreamHub 队列发布事件的 sink，必填——非流式调用
        应使用 ``ainvoke``。它进入 state 是因为 LangGraph 节点签名只有
        state 一个参数，``_execute_tools`` 要从这里拿它发布
        tool_call/tool_result 事件——与 state 里原有的 ``model``/``tools``
        是同一个先例。

        Raises:
            ValueError: ``events`` 为 ``None`` 时抛出；流式没有无出口的用法。
        """

        if events is None:
            raise ValueError(
                "astream requires an events sink; use ainvoke for non-streaming runs."
            )

        initial_state: AgentState = {
            "messages": list(messages),
            "scratch": {},
            "pending_tool_calls": [],
            "turns": 0,
            "model": model,
            "tools": list(tools),
            "events": events,
            "caller_session_id": caller_session_id,
        }
        final_state: AgentState | None = None
        async for mode, payload in self._graph.astream(
            initial_state,
            stream_mode=["messages", "values"],
        ):
            if mode == "values":
                final_state = payload
                continue
            # messages 模式的 payload 是 (message_chunk, metadata)。
            chunk = payload[0] if isinstance(payload, tuple) else payload
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                await events.publish(TextDelta(content=content))

        if final_state is None:
            raise AgentGraphExecutionError(
                "Agent graph stream finished without state values."
            )
        return {
            "messages": final_state.get("messages", []),
            "turns": final_state.get("turns", 0),
            "persisted_assistant": final_state.get("persisted_assistant"),
            "sent_to_main": final_state.get("sent_to_main", False),
        }


def _caller_session_id(state: AgentState) -> str:
    value = state.get("caller_session_id")
    if not isinstance(value, str) or not value.strip():
        raise AgentGraphExecutionError(
            "Tool execution and message persistence require caller_session_id."
        )
    return value


def _is_report_to_main(call: ToolCall, result: ToolResult) -> bool:
    """判断一次 tool call 是否"成功向 main_agent 派发了 send_message"。

    只有 recipient 指向 main_agent 且结果明确 ``ok: true`` 才算已汇报；
    派发失败（``ok: false``）不算，让子代理收尾时的兜底投递仍然生效。
    """

    if call.arguments.get("recipient") != MAIN_AGENT_NAME:
        return False
    try:
        payload = json.loads(result.content)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("ok") is True


def _error_result(*, code: str, message: str, tool_name: str) -> ToolResult:
    """把固定 Meta Tool 异常转成模型可读的普通文本结果。"""

    return ToolResult(
        content=json.dumps(
            {
                "ok": False,
                "error": {"code": code, "message": message, "tool_name": tool_name},
            },
            ensure_ascii=False,
        )
    )


__all__ = ["AgentGraph", "AgentGraphExecutionError"]
