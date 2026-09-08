"""Agent 领域对象使用的规则异常。"""

from agent.exceptions import DomainError


class DomainValidationError(DomainError):
    """输入不满足 Agent 领域规则。"""

    code = "domain_validation_error"


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
    "SummaryInputBudgetError",
    "SystemPromptBudgetError",
]
