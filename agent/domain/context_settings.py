"""上下文层调度参数：我们系统的属性，不是模型的属性，全局一份注入。

全部是钉在字段上的固定数字，不开放 settings.json、不做校验；测试直接
构造覆盖。任何"事实 + 本参数"的派生水位都不在这里，收敛在
``agent.domain.context_budget.derive_budget`` 一处。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextSettings:
    """上下文层的调度旋钮。"""

    # 压缩触发水位：低于它时前缀原样不动，保住 KV cache 命中。
    compact_threshold_ratio: float = 0.80
    # 冷压缩成功水位：迟滞带，一次出手必须清到触发线明显之下，避免每圈
    # 清刚够又立刻重新越线、反复改写中段前缀打烂 KV cache。
    cold_compact_target_ratio: float = 0.50
    # 保护区上界：最近 N 组工具交换逐字保留，不参与摘要/冷压缩。
    max_recent_tool_calls: int = 4
    # 投影期单条工具结果的 token 截断上限（provider 无关的估算 token）。
    max_tool_result_tokens: int = 4_096
    # 单轮并行 tool call 上限，固定数字，直接用。
    max_tool_calls_per_turn: int = 6
    # 冷压缩清除工具结果原文的体积阈值：低于它的太小，清了不够抵占位符本身。
    clear_tool_result_threshold_tokens: int = 100
    # 摘要模型的输出上限。
    summary_max_tokens: int = 2_048


__all__ = ["ContextSettings"]
