"""纯事实 ModelProfile / ContextSettings 与唯一派生点 derive_budget 的测试。"""

from __future__ import annotations

from unittest import TestCase

from agent.domain.context_budget import derive_budget
from agent.domain.context_settings import ContextSettings
from agent.domain.model_profile import ModelProfile


class ModelProfileFactTests(TestCase):
    """档案只承载事实：一切字段可缺省，没有校验、没有公式。"""

    def test_defaults(self) -> None:
        profile = ModelProfile()

        self.assertEqual("model", profile.model_id)
        self.assertEqual("model", profile.name)
        self.assertEqual("", profile.base_url)
        self.assertEqual("", profile.api_key)
        self.assertEqual(200_000, profile.context_window)
        self.assertEqual(8_192, profile.max_output_tokens)
        self.assertFalse(profile.use_responses_api)
        self.assertIsNone(profile.request_timeout_seconds)
        self.assertEqual(2, profile.max_retries)

    def test_no_validation_even_for_absurd_values(self) -> None:
        profile = ModelProfile(
            context_window=0,
            max_output_tokens=-5,
            request_timeout_seconds=-1,
            max_retries=-1,
        )

        self.assertEqual(0, profile.context_window)
        self.assertEqual(-5, profile.max_output_tokens)


class ContextSettingsTests(TestCase):
    def test_defaults(self) -> None:
        settings = ContextSettings()

        self.assertEqual(0.80, settings.compact_threshold_ratio)
        self.assertEqual(0.50, settings.cold_compact_target_ratio)
        self.assertEqual(4, settings.max_recent_tool_calls)
        self.assertEqual(4_096, settings.max_tool_result_tokens)
        self.assertEqual(6, settings.max_tool_calls_per_turn)
        self.assertEqual(100, settings.clear_tool_result_threshold_tokens)
        self.assertEqual(2_048, settings.summary_max_tokens)


class DeriveBudgetTests(TestCase):
    """全部派生水位仅此一处：公式直白、数字固定。"""

    def test_small_window_numbers(self) -> None:
        budget = derive_budget(
            ModelProfile(context_window=1_000, max_output_tokens=100),
            ContextSettings(
                compact_threshold_ratio=0.5,
                cold_compact_target_ratio=0.1,
                summary_max_tokens=77,
            ),
        )

        self.assertEqual(900, budget.input)
        self.assertEqual(500, budget.compact_threshold)
        self.assertEqual(100, budget.cold_target)
        self.assertEqual(823, budget.summary_input)
        self.assertEqual(411, budget.summary_source_target)

    def test_oversized_output_declare_clamps_input_to_one(self) -> None:
        budget = derive_budget(
            ModelProfile(context_window=2_000, max_output_tokens=6_000),
            ContextSettings(compact_threshold_ratio=0.5, cold_compact_target_ratio=0.1),
        )

        # 输入预算夹紧到 1，触发/目标水位被输入预算收住——不报错也不收缩并行。
        self.assertEqual(1, budget.input)
        self.assertEqual(1, budget.compact_threshold)
        self.assertEqual(1, budget.cold_target)

    def test_defaults_at_full_window(self) -> None:
        budget = derive_budget(ModelProfile(context_window=100_000), ContextSettings())

        self.assertEqual(91_808, budget.input)
        self.assertEqual(80_000, budget.compact_threshold)
        self.assertEqual(50_000, budget.cold_target)
        self.assertEqual(89_760, budget.summary_input)
        self.assertEqual(71_808, budget.summary_source_target)


__all__ = [
    "ContextSettingsTests",
    "DeriveBudgetTests",
    "ModelProfileFactTests",
]
