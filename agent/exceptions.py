"""Agent 应用层与领域层异常。

这些异常表达用例执行结果，不携带任何传输层类型。入口层按异常的
``code``/类别把它们映射成各自的错误形状（如嵌入式传输的
``AgentApiError.status_code``），命令行与测试也可以复用同一套错误。
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


class SessionDriverBusyError(DomainStateError):
    """会话驾驶权被另一个进程占用(桌面多开的单驾驶守护)。

    多实例允许并存,但同一会话同一时刻只允许一个进程驱动工作流;
    锁未抢到即抛此错,入口层按 409 呈现。
    """

    code = "session_driver_busy"

    def __init__(self, session_id: str) -> None:
        super().__init__(
            "This session is being driven by another running instance; "
            "wait for that turn to finish or use that window."
        )


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


class PermissionNotFoundError(ResourceNotFoundError):
    """待处理权限请求不存在或已经离开内存 broker。"""

    code = "permission_not_found"

    def __init__(self, permission_id: str) -> None:
        super().__init__(resource="Permission", resource_id=permission_id)


class PermissionAlreadyResolvedError(ConflictError):
    """同一个权限请求已经被其它回复消费。"""

    code = "permission_already_resolved"

    def __init__(self, permission_id: str) -> None:
        super().__init__(
            "Permission request has already been answered.",
            details={"permission_id": permission_id},
        )


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
    "PermissionAlreadyResolvedError",
    "PermissionNotFoundError",
    "SessionDeletionConflictError",
    "SessionDriverBusyError",
    "SessionNotFoundError",
]
