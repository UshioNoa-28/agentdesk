"""上下文预算：模型事实 × 调度参数的全部派生公式，仅此一处。

每轮 prepare 调一次 ``derive_budget``，manager/compactor 从返回的现成
数字里取水位，不再各自持有 profile+settings 现算。夹紧规则保留最小集：
输入预算下限 1，触发/目标水位不超过输入预算——除此之外没有任何校验与
兜底，配置填死就让它在计算处自然炸。
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.domain.context_settings import ContextSettings
from agent.domain.model_profile import ModelProfile


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """本轮上下文使用的全部派生水位（估算 token）。"""

    # 硬输入预算：system、工具 schema 与历史消息的上限。
    input: int
    # 压缩触发水位。
    compact_threshold: int
    # 冷压缩成功所需达到的更低水位。
    cold_target: int
    # 摘要请求的输入预算（从硬输入预算中再预留摘要输出）。
    summary_input: int
    # 摘要源截断目标水位：压到这里之下，摘要请求必装得下，也避免
    # "摘要刚落库、下一轮马上又过阈值"的压缩空转。
    summary_source_target: int


def derive_budget(profile: ModelProfile, settings: ContextSettings) -> ContextBudget:
    window = profile.context_window
    input_budget = max(1, window - profile.max_output_tokens)
    compact_threshold = min(input_budget, int(window * settings.compact_threshold_ratio))
    cold_target = min(input_budget, int(window * settings.cold_compact_target_ratio))
    summary_input = max(1, input_budget - settings.summary_max_tokens)
    return ContextBudget(
        input=input_budget,
        compact_threshold=compact_threshold,
        cold_target=cold_target,
        summary_input=summary_input,
        summary_source_target=int(summary_input * settings.compact_threshold_ratio),
    )


__all__ = ["ContextBudget", "derive_budget"]
