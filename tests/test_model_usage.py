from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from agent.domain.model_messages import ModelUsage
from agent.domain.model_profile import ModelProfile
from agent.infrastructure.model.chat_model import (
    LangChainAgentModel,
    _model_usage,
    _safe_provider_error,
    _usage_summary,
)


def _profile(**overrides: object) -> ModelProfile:
    """小窗口 fixture 档案：身份与事实平铺在同一个 ModelProfile 上。"""

    fields: dict[str, object] = {
        "model_id": "m",
        "name": "M",
        "base_url": "https://u",
        "api_key": "test-key",
        "context_window": 1_000,
        "max_output_tokens": 100,
    }
    fields.update(overrides)
    return ModelProfile(**fields)  # type: ignore[arg-type]


class ModelUsageTests(TestCase):
    def test_langchain_standard_usage_metadata_is_normalized(self) -> None:
        response = AIMessage(
            content="done",
            usage_metadata={
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            },
        )

        usage = _model_usage(response)

        self.assertEqual(
            usage,
            ModelUsage(input_tokens=120, output_tokens=30, total_tokens=150),
        )
        self.assertEqual(_usage_summary(response), "input=120,output=30,total=150")

    def test_raw_chat_completions_usage_is_a_compatibility_fallback(self) -> None:
        response = SimpleNamespace(
            usage_metadata=None,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                }
            },
        )

        self.assertEqual(
            _model_usage(response),
            ModelUsage(input_tokens=11, output_tokens=7, total_tokens=18),
        )

    def test_missing_provider_usage_remains_explicitly_unknown(self) -> None:
        response = SimpleNamespace(usage_metadata=None, response_metadata={})

        self.assertIsNone(_model_usage(response))
        self.assertEqual(_usage_summary(response), "unknown")

    def test_provider_error_message_is_sanitized(self) -> None:
        message = _safe_provider_error(RuntimeError('404 model_not_found: api_key="secret-value"'))

        self.assertIn("model_not_found", message)
        self.assertNotIn("secret-value", message)

    def test_model_id_is_what_gets_sent(self) -> None:
        profile = _profile(model_id="the-real-id", name="Display Name")
        with self._build(profile) as chat_openai:
            pass

        self.assertEqual(chat_openai.call_args.kwargs["model"], "the-real-id")
        self.assertEqual(chat_openai.call_args.kwargs["base_url"], "https://u")

    def test_declared_output_limit_is_used(self) -> None:
        profile = _profile(context_window=10_000, max_output_tokens=777)
        with self._build(profile) as chat_openai:
            pass

        self.assertEqual(chat_openai.call_args.kwargs["max_tokens"], 777)

    def test_undeclared_output_limit_uses_field_default(self) -> None:
        profile = ModelProfile(model_id="m", base_url="https://u", api_key="test-key")
        with self._build(profile) as chat_openai:
            pass

        # 没有兜底公式：缺省就是 ModelProfile 字段的默认 8192。
        self.assertEqual(chat_openai.call_args.kwargs["max_tokens"], 8_192)

    def test_streaming_is_always_on(self) -> None:
        with self._build(_profile()) as chat_openai:
            pass

        # 所有调用一律走流式（ainvoke 内部转 astream 并聚合）。
        self.assertTrue(chat_openai.call_args.kwargs["streaming"])
        self.assertTrue(chat_openai.call_args.kwargs["stream_usage"])

    def test_request_timeout_is_forwarded_as_httpx_timeout(self) -> None:
        with self._build(_profile(request_timeout_seconds=600)) as co:
            pass

        timeout = co.call_args.kwargs["timeout"]
        self.assertEqual(600.0, timeout.read)
        # 连接建立与 provider 快慢无关：固定 10s，不占配置面。
        self.assertEqual(10.0, timeout.connect)

    def test_request_timeout_omitted_when_unset(self) -> None:
        with self._build(_profile()) as chat_openai:
            pass

        self.assertNotIn("timeout", chat_openai.call_args.kwargs)

    def test_max_retries_defaults_and_follows_profile(self) -> None:
        with self._build(_profile()) as chat_openai:
            pass
        self.assertEqual(chat_openai.call_args.kwargs["max_retries"], 2)

        with self._build(_profile(max_retries=5)) as chat_openai:
            pass
        self.assertEqual(chat_openai.call_args.kwargs["max_retries"], 5)

    def test_responses_api_switches_output_version(self) -> None:
        with self._build(_profile(use_responses_api=True)) as co:
            pass

        self.assertTrue(co.call_args.kwargs["use_responses_api"])
        self.assertEqual(co.call_args.kwargs["output_version"], "responses/v1")

    def _build(self, profile: ModelProfile):
        """在打桩 ChatOpenAI 与 http 客户端下，按模型现建一次基础模型。"""

        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            with (
                patch(
                    "agent.infrastructure.model.chat_model.build_provider_async_http_client",
                    return_value=Mock(),
                ),
                patch(
                    "agent.infrastructure.model.chat_model.ChatOpenAI",
                    return_value=Mock(),
                ) as chat_openai,
            ):
                model = LangChainAgentModel(
                    catalog=_StubCatalog(profile),
                    tool_adapter=Mock(),
                )
                model._build_model(profile)
                yield chat_openai

        return _ctx()


class _StubCatalog:
    def __init__(self, profile: ModelProfile) -> None:
        self._profile = profile

    def current_profile(self) -> ModelProfile:
        return self._profile


__all__ = ["ModelUsageTests"]
