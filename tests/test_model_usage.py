from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from agent.domain.model_messages import ModelUsage
from agent.infrastructure.model.chat_model import (
    LangChainAgentModel,
    _model_usage,
    _safe_provider_error,
    _usage_summary,
)
from agent.infrastructure.settings import ChatModelSettings, ContextSettings


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
        message = _safe_provider_error(
            RuntimeError('404 model_not_found: api_key="secret-value"')
        )

        self.assertIn("model_not_found", message)
        self.assertNotIn("secret-value", message)

    def test_context_reserve_is_fallback_output_limit(self) -> None:
        settings = ChatModelSettings(
            openai_api_key="test-key",
            openai_max_output_tokens=321,
        )
        context_settings = ContextSettings(
            context_window_tokens=1_000,
            context_reserve_ratio=0.321,
            context_compact_threshold_ratio=0.6,
        )
        http_client = Mock()
        model_instance = Mock()
        with (
            patch(
                "agent.infrastructure.model.chat_model.build_provider_async_http_client",
                return_value=http_client,
            ),
            patch(
                "agent.infrastructure.model.chat_model.ChatOpenAI",
                return_value=model_instance,
            ) as chat_openai,
        ):
            model = LangChainAgentModel(
                settings=settings,
                context_settings=context_settings,
                tool_adapter=Mock(),
            )

            self.assertIs(model._get_model(), model_instance)

        self.assertEqual(
            chat_openai.call_args.kwargs["max_tokens"],
            321,
        )

    def test_configured_model_output_limit_overrides_context_reserve(self) -> None:
        settings = ChatModelSettings(
            openai_api_key="test-key",
            openai_max_output_tokens=777,
        )
        context_settings = ContextSettings(
            context_window_tokens=1_000,
            context_reserve_ratio=0.321,
            context_compact_threshold_ratio=0.6,
        )
        http_client = Mock()
        model_instance = Mock()
        with (
            patch(
                "agent.infrastructure.model.chat_model.build_provider_async_http_client",
                return_value=http_client,
            ),
            patch(
                "agent.infrastructure.model.chat_model.ChatOpenAI",
                return_value=model_instance,
            ) as chat_openai,
        ):
            model = LangChainAgentModel(
                settings=settings,
                context_settings=context_settings,
                tool_adapter=Mock(),
            )
            model._get_model()

        self.assertEqual(chat_openai.call_args.kwargs["max_tokens"], 777)

    def test_stream_chunk_timeout_is_forwarded_to_chat_openai(self) -> None:
        settings = ChatModelSettings(
            openai_api_key="test-key",
            openai_stream_chunk_timeout_seconds=300,
        )
        context_settings = ContextSettings(
            context_window_tokens=1_000,
            context_reserve_ratio=0.321,
            context_compact_threshold_ratio=0.6,
        )
        http_client = Mock()
        model_instance = Mock()
        with (
            patch(
                "agent.infrastructure.model.chat_model.build_provider_async_http_client",
                return_value=http_client,
            ),
            patch(
                "agent.infrastructure.model.chat_model.ChatOpenAI",
                return_value=model_instance,
            ) as chat_openai,
        ):
            model = LangChainAgentModel(
                settings=settings,
                context_settings=context_settings,
                tool_adapter=Mock(),
            )
            model._get_model()

        self.assertEqual(chat_openai.call_args.kwargs["stream_chunk_timeout"], 300)

    def test_stream_chunk_timeout_omitted_when_unset(self) -> None:
        settings = ChatModelSettings(openai_api_key="test-key")
        context_settings = ContextSettings(
            context_window_tokens=1_000,
            context_reserve_ratio=0.321,
            context_compact_threshold_ratio=0.6,
        )
        http_client = Mock()
        model_instance = Mock()
        with (
            patch(
                "agent.infrastructure.model.chat_model.build_provider_async_http_client",
                return_value=http_client,
            ),
            patch(
                "agent.infrastructure.model.chat_model.ChatOpenAI",
                return_value=model_instance,
            ) as chat_openai,
        ):
            model = LangChainAgentModel(
                settings=settings,
                context_settings=context_settings,
                tool_adapter=Mock(),
            )
            model._get_model()

        self.assertNotIn("stream_chunk_timeout", chat_openai.call_args.kwargs)

    def test_request_timeout_is_forwarded_as_httpx_timeout(self) -> None:
        settings = ChatModelSettings(
            openai_api_key="test-key",
            openai_request_timeout_seconds=600,
        )
        context_settings = ContextSettings(
            context_window_tokens=1_000,
            context_reserve_ratio=0.321,
            context_compact_threshold_ratio=0.6,
        )
        http_client = Mock()
        model_instance = Mock()
        with (
            patch(
                "agent.infrastructure.model.chat_model.build_provider_async_http_client",
                return_value=http_client,
            ),
            patch(
                "agent.infrastructure.model.chat_model.ChatOpenAI",
                return_value=model_instance,
            ) as chat_openai,
        ):
            model = LangChainAgentModel(
                settings=settings,
                context_settings=context_settings,
                tool_adapter=Mock(),
            )
            model._get_model()

        timeout = chat_openai.call_args.kwargs["timeout"]
        self.assertEqual(600.0, timeout.read)
        self.assertEqual(600.0, timeout.connect)

    def test_request_timeout_omitted_when_unset(self) -> None:
        settings = ChatModelSettings(openai_api_key="test-key")
        context_settings = ContextSettings(
            context_window_tokens=1_000,
            context_reserve_ratio=0.321,
            context_compact_threshold_ratio=0.6,
        )
        http_client = Mock()
        model_instance = Mock()
        with (
            patch(
                "agent.infrastructure.model.chat_model.build_provider_async_http_client",
                return_value=http_client,
            ),
            patch(
                "agent.infrastructure.model.chat_model.ChatOpenAI",
                return_value=model_instance,
            ) as chat_openai,
        ):
            model = LangChainAgentModel(
                settings=settings,
                context_settings=context_settings,
                tool_adapter=Mock(),
            )
            model._get_model()

        self.assertNotIn("timeout", chat_openai.call_args.kwargs)
