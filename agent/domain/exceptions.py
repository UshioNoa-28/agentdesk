"""Agent 领域对象使用的规则异常。"""

from agent.exceptions import DomainError


class DomainValidationError(DomainError):
    """输入不满足 Agent 领域规则。"""

    code = "domain_validation_error"


class MessageRoutingError(DomainValidationError):
    """多智能体消息不满足路由/拓扑规则。

    领域消息在构造期完成校验，并携带具体 error code（如
    ``star_topology_violation`` / ``self_message_not_allowed``）；
    工具层只需捕获并原样转发，不再自行判断拓扑策略。
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ContextCompactionError(DomainError):
    """上下文无法在保护规则内压缩到模型输入预算。"""

    code = "context_compaction_failed"


class SummaryInputBudgetError(ContextCompactionError):
    """摘要请求本身超出摘要模型可用的输入预算。"""

    code = "summary_input_budget_exceeded"


class SystemPromptBudgetError(ContextCompactionError):
    """系统提示词本身超出模型上下文总预算。"""

    code = "system_prompt_budget_exceeded"


__all__ = [
    "ContextCompactionError",
    "DomainValidationError",
    "MessageRoutingError",
    "SummaryInputBudgetError",
    "SystemPromptBudgetError",
]
