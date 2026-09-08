"""FastAPI HTTP 边界的最小冒烟测试。

这些测试使用 ASGITransport 直接调用应用，不启动 lifespan，因此不会连接
PostgreSQL 或真实 LLM。它们只验证路由装配、OpenAPI 和统一异常处理。
"""

from __future__ import annotations

import asyncio
import json
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

import httpx
from dishka import make_async_container

from agent.api.controllers.agent_controller import (
    _detached_turns,
    _event_stream,
    _sse_frame,
)
from agent.container import providers
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage, ToolCall
from agent.domain.multi_agent import Answered, Queued
from agent.domain.streaming import TextDelta, ToolResultEvent, stream_frame
from agent.domain.tools import ALL_META_TOOL_NAMES
from agent.dto.session import CreateSessionRequest
from agent.infrastructure.model.chat_model import _safe_provider_error
from agent.main import create_app
from agent.ports.runtime import AgentStreamHubPort
from agent.ports.tools import AgentRuntimePort
from agent.runtime.manager import AgentRuntimeManager
from agent.runtime.stream_hub import StreamHub


class ApiSmokeTests(IsolatedAsyncioTestCase):
    """验证应用 HTTP 入口可以独立工作。"""

    async def asyncSetUp(self) -> None:
        """创建不触发应用 lifespan 的 ASGI 测试客户端。"""

        transport = httpx.ASGITransport(app=create_app())
        self.client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        """关闭测试客户端，释放 HTTP 连接资源。"""

        await self.client.aclose()

    async def test_health_endpoint(self) -> None:
        """健康接口应该返回固定的成功状态。"""

        response = await self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    async def test_openapi_contains_core_routes(self) -> None:
        """OpenAPI 只暴露会话、Agent、MCP 和 Skill 核心路由。"""

        response = await self.client.get("/openapi.json")
        paths = response.json()["paths"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/sessions", paths)
        self.assertIn("/api/sessions/{session_id}/resume", paths)
        self.assertIn("/api/sessions/{session_id}/messages", paths)
        self.assertIn("/api/sessions/{session_id}/messages/stream", paths)
        self.assertIn("/api/sessions/{session_id}/compact", paths)
        self.assertNotIn("/api/sessions/{session_id}/runs", paths)
        self.assertNotIn("/api/runs/{run_id}", paths)
        self.assertNotIn("/api/sessions/{session_id}/events", paths)
        self.assertIn("/api/mcps", paths)
        self.assertIn("/api/mcps/{server_id}/retry", paths)
        self.assertIn("/api/skills", paths)
        self.assertNotIn("/api/skills/reload", paths)
        self.assertNotIn("/api/documents", paths)

    async def test_skill_status_endpoint(self) -> None:
        """Skill 管理接口应该列出加载状态。"""

        listed = await self.client.get("/api/skills")

        self.assertEqual(listed.status_code, 200)
        self.assertTrue(any(item["name"] == "structured-answer" for item in listed.json()))
        self.assertTrue(
            all(item["state"] in {"loaded", "failed", "disabled"} for item in listed.json())
        )

    async def test_unknown_route_uses_unified_error_shape(self) -> None:
        """不存在的路由应该经过全局 HTTP 异常处理器。"""

        response = await self.client.get("/api/does-not-exist")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "http_error")

    async def test_provider_error_is_safe_and_diagnostic(self) -> None:
        message = _safe_provider_error(
            RuntimeError('404 model_not_found: api_key="secret-value"')
        )

        self.assertIn("model_not_found", message)
        self.assertNotIn("secret-value", message)

    async def test_app_lifespan_executes_initialization_and_cleanup(self) -> None:
        """测试应用 lifespan 能够正确获取依赖并执行初始化与关闭。"""

        app = create_app()
        mock_registry = AsyncMock()
        mock_registry.start_all_async = AsyncMock()
        mock_registry.close_async = AsyncMock()

        with (
            patch("agent.main.initialize_agent_database", new_callable=AsyncMock) as mock_init_db,
            patch("dishka.AsyncContainer.get", return_value=mock_registry),
        ):
            async with app.router.lifespan_context(app):
                mock_init_db.assert_called_once()
                mock_registry.start_all_async.assert_called_once()
            mock_registry.close_async.assert_called_once()

    async def test_runtime_port_is_resolvable_for_graceful_shutdown(self) -> None:
        """停机路径按 AgentRuntimePort 解析运行时；注册键漂移会在这里暴露。"""

        container = make_async_container(*providers())
        try:
            runtime = await container.get(AgentRuntimePort)
            self.assertIsInstance(runtime, AgentRuntimeManager)
            self.assertFalse(runtime.is_running)
        finally:
            await container.close()

    async def test_api_list_messages_filters_system_by_default(self) -> None:
        """调用 GET /sessions/{id}/messages 始终过滤内部系统提示词，防止泄露内部配置。"""
        from agent.api.controllers.session_controller import list_session_messages

        mock_service = AsyncMock()
        sys_msg = Message.create(
            session_id="s1",
            seq=1,
            message=ModelMessage.system("system prompt"),
            metadata={"kind": MessageKind.SYSTEM},
        )
        user_msg = Message.create(
            session_id="s1",
            seq=2,
            message=ModelMessage.human("user query"),
            metadata={"kind": MessageKind.USER},
        )
        mock_service.history = AsyncMock(return_value=[sys_msg, user_msg])

        fn = getattr(list_session_messages, "__dishka_orig_func__", list_session_messages)

        # 始终过滤系统消息
        filtered = await fn("s1", mock_service)
        self.assertEqual(1, len(filtered))
        self.assertEqual("human", filtered[0].role)
        self.assertEqual("user query", filtered[0].content)

    async def test_message_stream_emits_sse_frames_and_final_message(self) -> None:
        """流式接口应输出 SSE 帧并以 message_end 携带 RPC 最终消息。"""

        app = create_app()
        final = Message.create(
            session_id="sess-stream",
            seq=2,
            message=ModelMessage.assistant(content="done"),
            metadata={"kind": MessageKind.ASSISTANT_ANSWER},
        )
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
        try:
            with patch.object(
                AgentRuntimeManager,
                "send_message",
                new=AsyncMock(return_value=Answered(message=final)),
            ):
                async with client.stream(
                    "POST",
                    "/api/sessions/sess-stream/messages/stream",
                    json={"question": "hi"},
                ) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(
                        response.headers["content-type"].startswith("text/event-stream")
                    )
                    raw = b""
                    async for chunk in response.aiter_bytes():
                        raw += chunk

            frames = _parse_sse(raw.decode("utf-8"))
            self.assertEqual([], [f for f in frames if f["event"] == "text_delta"])
            ended = [f for f in frames if f["event"] == "message_end"]
            self.assertEqual(1, len(ended))
            self.assertEqual("done", ended[0]["data"]["message"]["content"])
        finally:
            await client.aclose()

    async def test_message_stream_queues_second_request_without_a_second_stream(
        self,
    ) -> None:
        """同一 session 已有活跃流时：不开第二条 SSE，直接入库并返回 202。"""

        app = create_app()
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
        try:
            hub = await app.state.dishka_container.get(AgentStreamHubPort)
            self.assertIsNotNone(hub.register("sess-busy"))

            with patch.object(
                AgentRuntimeManager,
                "send_message",
                new=AsyncMock(return_value=Queued(session_id="sess-busy")),
            ):
                response = await client.post(
                    "/api/sessions/sess-busy/messages/stream",
                    json={"question": "hi"},
                )

            self.assertEqual(response.status_code, 202)
            self.assertEqual(
                {"status": "queued", "session_id": "sess-busy"},
                response.json(),
            )
            self.assertFalse(
                response.headers["content-type"].startswith("text/event-stream")
            )
            # 活跃流的队列必须原样保留：第二个请求既没抢注也没回收它。
            self.assertIsNone(hub.register("sess-busy"))
        finally:
            await client.aclose()

    async def test_message_stream_pumps_produced_events_before_final(self) -> None:
        """actor 产出的增量事件应按序流式到达，message_end 收尾。"""

        app = create_app()
        final = Message.create(
            session_id="sess-live",
            seq=2,
            message=ModelMessage.assistant(content="done"),
            metadata={"kind": MessageKind.ASSISTANT_ANSWER},
        )

        async def _producing_ask(_self, message):
            hub = await app.state.dishka_container.get(AgentStreamHubPort)
            sink = hub.sink_for(message.session_id)
            assert sink is not None
            await sink.publish(TextDelta(content="你"))
            await asyncio.sleep(0)
            await sink.publish(TextDelta(content="好"))
            return Answered(message=final)

        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
        try:
            with patch.object(
                AgentRuntimeManager,
                "send_message",
                new=_producing_ask,
            ):
                async with client.stream(
                    "POST",
                    "/api/sessions/sess-live/messages/stream",
                    json={"question": "hi"},
                ) as response:
                    raw = b""
                    async for chunk in response.aiter_bytes():
                        raw += chunk

            frames = _parse_sse(raw.decode("utf-8"))
            self.assertEqual(
                ["text_delta", "text_delta", "message_end"],
                [frame["event"] for frame in frames],
            )
            self.assertEqual(
                "你好",
                "".join(
                    frame["data"]["content"]
                    for frame in frames
                    if frame["event"] == "text_delta"
                ),
            )
            self.assertEqual("done", frames[-1]["data"]["message"]["content"])

            # 流结束后注册表必须清空，后续可再次注册。
            hub = await app.state.dishka_container.get(AgentStreamHubPort)
            self.assertIsNotNone(hub.register("sess-live"))
        finally:
            await client.aclose()

    async def test_closing_the_stream_lets_the_turn_run_to_completion(self) -> None:
        """客户端提前收流：本轮照跑完，注册由轮次自己摘、不由断连摘。

        直接驱动 ``_event_stream`` 而不经 httpx：要验的是生成器被关闭时服务端
        的行为，与 ASGI transport 怎么传播断连无关。
        """

        hub = StreamHub()
        queue = hub.register("sess-hangup")
        self.assertIsNotNone(queue)
        turns_before = set(_detached_turns)
        resumed = asyncio.Event()
        final = Message.create(
            session_id="sess-hangup",
            seq=2,
            message=ModelMessage.assistant(content="done"),
            metadata={"kind": MessageKind.ASSISTANT_ANSWER},
        )

        class _HangingService:
            """发布一帧后卡在 RPC 里：图还在跑，听的人已经走了。"""

            async def ask(
                self, *, session_id: str, question: str, sender: str
            ) -> Answered:
                sink = hub.sink_for(session_id)
                assert sink is not None
                await sink.publish(TextDelta(content="你"))
                await resumed.wait()
                return Answered(message=final)

        stream = _event_stream(
            _HangingService(),
            hub,
            queue,
            session_id="sess-hangup",
            question="hi",
        )
        first = await anext(stream)
        self.assertTrue(first.startswith(b"event: text_delta"))

        await stream.aclose()

        # 断连不许回收注册：队列跟着这一轮，不跟着连接。
        self.assertIsNone(hub.register("sess-hangup"))
        pending_turns = _detached_turns - turns_before
        self.assertEqual(1, len(pending_turns))

        resumed.set()
        await asyncio.gather(*pending_turns)

        # 摘除发生在轮次自然结束时，且终止帧已经产出（只是没人读了）。
        # text_delta 那一帧已被 await anext 取走，队列里只剩收尾两帧。
        self.assertIsNotNone(hub.register("sess-hangup"))
        drained = []
        while not queue.empty():
            drained.append(queue.get_nowait())
        self.assertEqual(
            ["message_end", None],
            [frame[0] if frame else None for frame in drained],
        )

    async def test_failed_turn_still_deregisters_the_queue(self) -> None:
        """ask 抛错也必须摘掉注册，否则这个 session 从此开不了第二条流。"""

        hub = StreamHub()
        queue = hub.register("sess-boom")
        self.assertIsNotNone(queue)

        class _FailingService:
            async def ask(
                self, *, session_id: str, question: str, sender: str
            ) -> Answered:
                raise RuntimeError("graph exploded")

        stream = _event_stream(
            _FailingService(),
            hub,
            queue,
            session_id="sess-boom",
            question="hi",
        )
        frames = [frame async for frame in stream]

        self.assertEqual(1, len(frames))
        self.assertTrue(frames[0].startswith(b"event: error"))
        self.assertIsNotNone(hub.register("sess-boom"))

    async def test_cancelled_turn_still_deregisters_the_queue(self) -> None:
        """``ask`` 被取消时也要摘注册：CancelledError 不是 Exception。

        收尾回调里的 finally 就是为这一条存在的——注册漏一次，这个 session
        的流式就永久退化成 202。
        """

        hub = StreamHub()
        queue = hub.register("sess-cancel")
        self.assertIsNotNone(queue)
        stalled = asyncio.Event()

        class _StallingService:
            async def ask(
                self, *, session_id: str, question: str, sender: str
            ) -> Answered:
                await stalled.wait()
                raise AssertionError("unreachable: the turn is cancelled")

        turns_before = set(_detached_turns)
        stream = _event_stream(
            _StallingService(),
            hub,
            queue,
            session_id="sess-cancel",
            question="hi",
        )
        consuming = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0)  # 让生成器跑到 create_task

        (ask,) = _detached_turns - turns_before
        ask.cancel()
        # 没有 error 帧（不替取消编造结果），只有哨兵让读者自然收场。
        with self.assertRaises(StopAsyncIteration):
            await consuming
        self.assertIsNotNone(hub.register("sess-cancel"))

    async def test_message_endpoint_returns_final_answer(self) -> None:
        """非流式接口：本轮产出回答时返回 200 + 最终 AssistantMessage。"""

        final = Message.create(
            session_id="sess-ask",
            seq=2,
            message=ModelMessage.assistant(content="done"),
            metadata={"kind": MessageKind.ASSISTANT_ANSWER},
        )
        with patch.object(
            AgentRuntimeManager,
            "send_message",
            new=AsyncMock(return_value=Answered(message=final)),
        ):
            response = await self.client.post(
                "/api/sessions/sess-ask/messages", json={"question": "hi"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual("done", response.json()["content"])

    async def test_message_endpoint_returns_202_when_turn_is_queued(self) -> None:
        """会话已有工作流在跑：入库后返回 202，这是被接受的结果而不是错误。"""

        with patch.object(
            AgentRuntimeManager,
            "send_message",
            new=AsyncMock(return_value=Queued(session_id="sess-ask-busy")),
        ):
            response = await self.client.post(
                "/api/sessions/sess-ask-busy/messages", json={"question": "hi"}
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            {"status": "queued", "session_id": "sess-ask-busy"}, response.json()
        )
        self.assertNotIn("error", response.json())

    async def test_message_stream_emits_queued_frame_when_turn_lost_the_race(self) -> None:
        """注册到队列却没能拿到 RUNNING：以 queued 帧收尾，而不是 message_end/error。"""

        app = create_app()
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
        try:
            with patch.object(
                AgentRuntimeManager,
                "send_message",
                new=AsyncMock(return_value=Queued(session_id="sess-race")),
            ):
                async with client.stream(
                    "POST",
                    "/api/sessions/sess-race/messages/stream",
                    json={"question": "hi"},
                ) as response:
                    self.assertEqual(response.status_code, 200)
                    raw = b""
                    async for chunk in response.aiter_bytes():
                        raw += chunk

            frames = _parse_sse(raw.decode("utf-8"))
            self.assertEqual(
                [{"event": "queued", "data": {"session_id": "sess-race"}}], frames
            )
        finally:
            await client.aclose()


class StreamFrameEncodingTests(IsolatedAsyncioTestCase):
    """SSE 帧编码与领域事件映射。"""

    def test_text_delta_frame(self) -> None:
        raw = _sse_frame("text_delta", {"content": "你"}).decode("utf-8")

        self.assertEqual('event: text_delta\ndata: {"content": "你"}\n\n', raw)

    def test_tool_call_and_result_frames(self) -> None:
        call = ToolCall(id="c1", name="echo", arguments={"name": "x"})

        name, payload = stream_frame(call)
        call_frame = _sse_frame(name, payload)
        result_name, result_payload = stream_frame(
            ToolResultEvent(call=call, content="hello x")
        )
        result_frame = _sse_frame(result_name, result_payload)

        self.assertEqual(
            {"id": "c1", "name": "echo", "arguments": {"name": "x"}},
            json.loads(call_frame.decode("utf-8").split("data: ", 1)[1]),
        )
        self.assertIn('"content": "hello x"', result_frame.decode("utf-8"))


class CreateSessionRequestDefaultsTests(TestCase):
    """创建主会话请求的 allowed_tools 默认语义。"""

    def test_omitted_allowed_tools_defaults_to_all_meta_tools(self) -> None:
        request = CreateSessionRequest(title="main")

        self.assertEqual(list(ALL_META_TOOL_NAMES), request.allowed_tools)

    def test_explicit_empty_allowed_tools_means_no_tools(self) -> None:
        request = CreateSessionRequest(title="main", allowed_tools=[])

        self.assertEqual([], request.allowed_tools)


def _parse_sse(raw: str) -> list[dict[str, object]]:
    """把 SSE 文本解析成 [{"event": 名称, "data": 载荷}]。"""

    frames: list[dict[str, object]] = []
    event: str | None = None
    for line in raw.splitlines():
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            body = line[len("data:"):].strip()
            frames.append({"event": event, "data": json.loads(body)})
    return frames
