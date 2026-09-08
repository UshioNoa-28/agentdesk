"""StreamHub 的注册、读取与摘除语义。"""

from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase

from agent.domain.model_messages import ToolCall
from agent.domain.streaming import TextDelta, ToolResultEvent
from agent.runtime.stream_hub import StreamHub


class StreamHubTests(IsolatedAsyncioTestCase):
    async def test_register_returns_queue_and_publishes_frames_via_sink(self) -> None:
        hub = StreamHub()
        queue = hub.register("s1")

        self.assertIsNotNone(queue)
        sink = hub.sink_for("s1")
        self.assertIsNotNone(sink)

        await sink.publish(TextDelta(content="你好"))
        await sink.publish(ToolCall(id="c1", name="echo", arguments={"name": "x"}))
        await sink.publish(
            ToolResultEvent(call=ToolCall(id="c1", name="echo", arguments={}), content="hi")
        )

        self.assertEqual(
            ("text_delta", {"content": "你好"}),
            await queue.get(),
        )
        self.assertEqual(
            ("tool_call", {"id": "c1", "name": "echo", "arguments": {"name": "x"}}),
            await queue.get(),
        )
        self.assertEqual(
            (
                "tool_result",
                {
                    "tool_call_id": "c1",
                    "name": "echo",
                    "arguments": {},
                    "content": "hi",
                },
            ),
            await queue.get(),
        )

    async def test_second_register_is_rejected_while_stream_active(self) -> None:
        hub = StreamHub()
        self.assertIsNotNone(hub.register("s1"))

        self.assertIsNone(hub.register("s1"))

    async def test_sink_for_unknown_session_returns_none(self) -> None:
        hub = StreamHub()

        self.assertIsNone(hub.sink_for("missing"))

    async def test_remove_is_idempotent_and_allows_re_register(self) -> None:
        hub = StreamHub()
        hub.register("s1")
        hub.remove("s1")
        hub.remove("s1")

        self.assertIsNotNone(hub.register("s1"))

    async def test_queues_are_independent_between_sessions(self) -> None:
        hub = StreamHub()
        first = hub.register("s1")
        second = hub.register("s2")

        await hub.sink_for("s1").publish(TextDelta(content="a"))
        await hub.sink_for("s2").publish(TextDelta(content="b"))

        self.assertEqual(("text_delta", {"content": "a"}), await first.get())
        self.assertEqual(("text_delta", {"content": "b"}), await second.get())
        self.assertTrue(second.empty())

    async def test_publish_does_not_block_unbounded_queue(self) -> None:
        hub = StreamHub()
        queue = hub.register("s1")
        sink = hub.sink_for("s1")

        await asyncio.gather(
            *(sink.publish(TextDelta(content=str(index))) for index in range(1000))
        )

        self.assertEqual(1000, queue.qsize())
