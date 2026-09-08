"""TokenCounterService 请求测量方法的公式一致性测试。

测量方法是各消费方（manager/compactor）token 口径的唯一事实源；这里
锁死三个测量方法与原语加法的公式等价性。service 不持有请求构成的
知识：measure_request 测的是调用方构造好的完整请求。
"""

from __future__ import annotations

from unittest import TestCase

from agent.domain.model_messages import ModelMessage
from agent.domain.tools import ToolDefinition
from tests.tokenizer_support import budget_token_service


def _tool_definition(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} description",
        parameters={"type": "object", "properties": {}},
    )


class RequestMeasurementTests(TestCase):
    def setUp(self) -> None:
        self.service = budget_token_service()

    def test_measure_model_request_equals_messages_plus_tools(self) -> None:
        messages = [ModelMessage.system("sys"), ModelMessage.human("hello")]
        tools = (_tool_definition("execute_python"), _tool_definition("send_message"))

        measured = self.service.measure_model_request(messages, tools=tools)

        self.assertEqual(
            self.service.estimate_messages(messages)
            + self.service.estimate_tool_definitions(tools),
            measured,
        )

    def test_measure_model_request_without_tools_equals_messages(self) -> None:
        messages = [ModelMessage.human("hi")]

        self.assertEqual(
            self.service.estimate_messages(messages),
            self.service.measure_model_request(messages, tools=()),
        )

    def test_measure_request_equals_messages_estimate(self) -> None:
        """measure_request 是已构造请求的语义化测量，含全部请求开销。"""

        request = [
            ModelMessage.system("summary policy"),
            ModelMessage.human("x" * 200),
            ModelMessage.human("continuation instruction"),
        ]

        self.assertEqual(
            self.service.estimate_messages(request),
            self.service.measure_request(request),
        )


__all__ = ["RequestMeasurementTests"]
