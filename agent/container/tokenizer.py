"""Token 计数 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.application.services.token_counter_service import TokenCounterService
from agent.infrastructure.tokenizer import HeuristicTokenCounter
from agent.ports.model import TokenCounter
from agent.ports.services import TokenCounterServicePort


class TokenizerProvider(Provider):
    """把 token 计数器适配为 Agent 消息预算服务。"""

    @provide(scope=Scope.APP)
    def provide_token_counter(self) -> TokenCounter:
        """创建进程级计数器；估算无状态、无外部资源，构造即完成。"""

        return HeuristicTokenCounter()

    @provide(scope=Scope.APP)
    def provide_token_counter_service(
        self,
        counter: TokenCounter,
    ) -> TokenCounterServicePort:
        """把计数器适配为 Agent 消息预算服务。"""

        return TokenCounterService(counter)


__all__ = ["TokenizerProvider"]
