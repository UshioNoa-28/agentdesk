"""测试使用的 token 计数支撑。

生产计数器本身是纯函数、无外部资源，因此策略测试直接复用它，
不再维护一份会漂移的测试替身。
"""

from __future__ import annotations

from agent.application.services.token_counter_service import TokenCounterService
from agent.infrastructure.tokenizer import HeuristicTokenCounter


def budget_token_counter() -> HeuristicTokenCounter:
    """返回上下文策略测试使用的计数器。"""

    return HeuristicTokenCounter()


def budget_token_service() -> TokenCounterService:
    """返回使用生产计数器的 token 应用服务。"""

    return TokenCounterService(budget_token_counter())


__all__ = ["budget_token_counter", "budget_token_service"]
