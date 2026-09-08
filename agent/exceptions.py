"""Agent 应用层与领域层异常。

这些异常表达用例执行结果，但不携带 FastAPI 类型。API 层可以根据异常类型
把它们转换为 HTTP 响应；命令行、任务队列等其他入口也可以复用同一套错误。
"""

from __future__ import annotations

from collections.abc import Mapping


class DomainError(Exception):
    """所有领域规则异常的基类。"""

    code = "domain_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DomainStateError(DomainError):
    """领域对象当前状态不允许执行指定行为。"""

    code = "domain_state_error"


class ApplicationError(Exception):
    """应用用例可以安全向上层报告的异常。"""

    code = "application_error"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """保存对外可报告的错误消息和结构化详情。

        Args:
            message (str): 可安全返回给入口调用方的错误说明。
            details (Mapping[str, object] | None): 可选结构化诊断字段。
        """

        super().__init__(message)
        self.message = message
        self.details = dict(details or {})


class ResourceNotFoundError(ApplicationError):
    """请求的业务资源不存在。"""

    code = "resource_not_found"

    def __init__(self, *, resource: str, resource_id: str) -> None:
        """根据资源类型和 ID 构造统一的资源不存在异常。

        Args:
            resource (str): 资源类型名称。
            resource_id (str): 请求中找不到的资源 ID。
        """

        super().__init__(
            f"{resource} not found: {resource_id}",
            details={"resource": resource, "resource_id": resource_id},
        )


class ConflictError(ApplicationError):
    """请求与当前资源状态冲突，通常映射为 HTTP 409。"""

    code = "conflict"


class ExternalServiceError(ApplicationError):
    """模型、向量库等应用外部依赖执行失败。"""

    code = "external_service_error"


class SessionDeletionConflictError(ConflictError):
    """Session 因为仍有子会话而不能删除。"""

    code = "session_deletion_conflict"

    def __init__(self, *, session_id: str, reason: str) -> None:
        """构造可供 API 展示的删除冲突原因。

        Args:
            session_id (str): 无法删除的会话 ID。
            reason (str): 当前支持 ``children``。
        """

        messages = {
            "children": "Session has child sessions and cannot be deleted.",
        }
        message = messages.get(reason, "Session cannot be deleted in its current state.")
        super().__init__(
            message,
            details={"session_id": session_id, "reason": reason},
        )


class SessionNotFoundError(ResourceNotFoundError):
    """Session 不存在。"""

    code = "session_not_found"

    def __init__(self, session_id: str) -> None:
        """构造 Session 不存在异常。"""

        super().__init__(resource="Session", resource_id=session_id)


class AgentExecutionError(ExternalServiceError):
    """Agent 图执行失败；本轮未生成伪造的失败消息。"""

    code = "agent_execution_failed"

    def __init__(
        self,
        message: str = "Agent execution failed.",
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """构造可直接返回给入口层的 Agent 执行异常。"""

        super().__init__(message, details=details)


class AgentInternalError(ApplicationError):
    """Agent 图内部逻辑失败；非上游依赖问题，入口层映射为 HTTP 500。"""

    code = "agent_internal_error"

    def __init__(
        self,
        message: str = "Agent execution failed.",
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """构造内部逻辑失败异常，不向上游依赖甩锅。"""

        super().__init__(message, details=details)


__all__ = [
    "AgentExecutionError",
    "AgentInternalError",
    "ApplicationError",
    "ConflictError",
    "DomainError",
    "DomainStateError",
    "ExternalServiceError",
    "ResourceNotFoundError",
    "SessionDeletionConflictError",
    "SessionNotFoundError",
]
