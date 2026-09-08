from __future__ import annotations

from unittest import TestCase

from agent.infrastructure.tokenizer import HeuristicTokenCounter


class HeuristicTokenCounterTests(TestCase):
    """验证无网络计数器的分档规则与 token 预算截断不变量。"""

    def setUp(self) -> None:
        self.counter = HeuristicTokenCounter()

    def test_counts_ascii_in_four_char_units_and_other_scripts_per_char(self) -> None:
        self.assertEqual(0, self.counter.count_text(""))
        self.assertEqual(1, self.counter.count_text("hi"))
        self.assertEqual(1, self.counter.count_text("abcd"))
        self.assertEqual(2, self.counter.count_text("abcde"))
        self.assertEqual(2, self.counter.count_text("你好"))
        self.assertGreater(self.counter.count_text("你好"), self.counter.count_text("hi"))

    def test_mixed_text_counts_both_classes(self) -> None:
        self.assertEqual(2, self.counter.count_text("hi你"))

    def test_punctuation_costs_more_than_plain_letters(self) -> None:
        jsonish = '{"key":"value","count":12345}'

        self.assertGreater(
            self.counter.count_text(jsonish),
            self.counter.count_text("a" * len(jsonish)),
        )

    def test_text_within_budget_is_returned_unchanged(self) -> None:
        content = "short text"
        self.assertEqual(content, self.counter.truncate_text(content, 100, marker="[cut]"))

    def test_truncation_respects_budget_and_keeps_both_ends(self) -> None:
        content = "head-" + "a" * 400 + "-tail"
        result = self.counter.truncate_text(content, 40, marker="\n...[cut]...\n")

        self.assertLessEqual(self.counter.count_text(result), 40)
        self.assertIn("[cut]", result)
        self.assertTrue(result.startswith("head-"))
        self.assertTrue(result.endswith("-tail"))

    def test_truncation_of_cjk_text_respects_budget(self) -> None:
        content = "中" * 200
        result = self.counter.truncate_text(content, 30, marker="[cut]")

        self.assertLessEqual(self.counter.count_text(result), 30)
        self.assertIn("[cut]", result)

    def test_oversized_marker_is_clipped_to_budget(self) -> None:
        marker = "x" * 100
        result = self.counter.truncate_text("y" * 100, 5, marker=marker)

        self.assertLessEqual(self.counter.count_text(result), 5)

    def test_non_positive_budget_yields_empty_text(self) -> None:
        self.assertEqual("", self.counter.truncate_text("abc", 0, marker="[cut]"))
        self.assertEqual("", self.counter.truncate_text("abc", -1, marker="[cut]"))
