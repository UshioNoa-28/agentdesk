from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from agent.domain.exceptions import DomainValidationError
from agent.domain.interruption import (
    InterruptionReplyStatus,
    InterruptionResult,
    InterruptionStatus,
    PermissionRequest,
)
from agent.domain.permissions import (
    PermissionAction,
    PermissionAuthorization,
    PermissionDecision,
    PermissionMode,
)
from agent.exceptions import (
    PermissionAlreadyResolvedError,
    PermissionNotFoundError,
    SessionNotFoundError,
)
from agent.ports.runtime.interruption import InterruptionBrokerPort
from agent.ports.services.permission import (
    PermissionManagerPort,
    PermissionServicePort,
)
from agent.ports.services.session import SessionServicePort


class PermissionService(PermissionServicePort):
    def __init__(
        self,
        *,
        broker: InterruptionBrokerPort,
        manager: PermissionManagerPort,
        session_service: SessionServicePort,
    ) -> None:
        self._broker = broker
        self._manager = manager
        self._sessions = session_service

    async def authorize(
        self,
        *,
        caller_session_id: str,
        tool_call_id: str,
        name: str,
        arguments: dict[str, object],
        targets: Sequence[str] | None = None,
    ) -> PermissionAuthorization:
        """决策表：deny > ask > allow；未匹配时看工具是否有权限面。

        ``targets=None`` 表示工具声明无权限面：默认放行，仅整工具的
        deny/ask 规则能把它拉回审核。非 None 段的折叠：deny/ask 任一段
        命中即生效，allow 必须全部段命中，未匹配走询问兜底。
        """

        no_surface = targets is None
        segments = tuple(targets) if targets else ()
        # 匹配探针：无权限面与零段折叠成单个空 target，只与整工具规则相遇。
        probes = segments or ("",)
        if self._manager.mode is PermissionMode.BYPASS:
            return PermissionAuthorization.ALLOW
        if any(self._manager.in_deny(name, probe) for probe in probes):
            return PermissionAuthorization.REJECT
        if self._manager.mode is PermissionMode.DONT_ASK:
            return PermissionAuthorization.ALLOW
        if any(self._manager.in_ask(name, probe) for probe in probes):
            return await self._request_cli_approval(
                caller_session_id=caller_session_id,
                tool_call_id=tool_call_id,
                name=name,
                arguments=arguments,
                targets=segments,
            )
        if no_surface or all(self._manager.in_allow(name, probe) for probe in probes):
            return PermissionAuthorization.ALLOW
        # 有权限面且规则全不命中：默认ask
        return await self._request_cli_approval(
            caller_session_id=caller_session_id,
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
            targets=segments,
        )

    async def _request_cli_approval(
        self,
        *,
        caller_session_id: str,
        tool_call_id: str,
        name: str,
        arguments: dict[str, object],
        targets: tuple[str, ...],
    ) -> PermissionAuthorization:
        normalized_session_id = caller_session_id.strip()
        if not normalized_session_id:
            return PermissionAuthorization.UNAVAILABLE
        try:
            session = await self._sessions.get(normalized_session_id)
        except SessionNotFoundError:
            return PermissionAuthorization.UNAVAILABLE

        target_session_id = (session.main_session_id or session.id).strip()
        if not target_session_id:
            return PermissionAuthorization.UNAVAILABLE
        result = await self._broker.request(
            PermissionRequest(
                interruption_id=str(uuid4()),
                session_id=target_session_id,
                tool_call_id=tool_call_id,
                name=name,
                arguments=dict(arguments),
                targets=targets,
            )
        )
        return _authorization_from_result(result)

    async def reply(
        self,
        *,
        permission_id: str,
        decision: str,
        scope: Sequence[str] | None = None,
    ) -> PermissionDecision:
        normalized_id = permission_id.strip()
        if not normalized_id:
            raise DomainValidationError("permission_id cannot be empty.")
        if decision not in tuple(PermissionDecision):
            raise DomainValidationError("decision must be 'allow' or 'reject'.")
        if scope is not None:
            if decision != str(PermissionDecision.ALLOW):
                raise DomainValidationError("scope is only accepted with decision 'allow'.")
            # 裸 str 自己就是 Sequence[str]，迭代出一串单字符规则：边界拒收。
            if isinstance(scope, str):
                raise DomainValidationError(
                    "scope must be a sequence of segments; wrap a single scope as ['x']."
                )
            if not scope or any(not isinstance(item, str) for item in scope):
                raise DomainValidationError("scope must be a non-empty sequence of strings.")
        # 唤醒前先反查原始请求，拿到落规则所需的工具名。
        request = self._broker.pending(normalized_id)
        tool_name = request.name if isinstance(request, PermissionRequest) else None
        status = self._broker.reply(
            interruption_id=normalized_id,
            payload={"decision": decision},
        )
        if status == InterruptionReplyStatus.NOT_FOUND:
            raise PermissionNotFoundError(normalized_id)
        if status == InterruptionReplyStatus.ALREADY_RESOLVED:
            raise PermissionAlreadyResolvedError(normalized_id)
        # scope 是入口回填的段集合（事件里 targets 的原文）：逐段落一条规则；
        # [""] 显式整工具放行，None 不落盘。段值自身的形状校验在 persist_rule。
        if decision == str(PermissionDecision.ALLOW) and scope is not None and tool_name:
            for segment in scope:
                self._manager.persist_rule(tool_name, PermissionAction.ALLOW, segment)
        return PermissionDecision(decision)

    def current_mode(self) -> PermissionMode:
        """读取当前生效的决策 mode，供 CLI 展示状态。"""

        return self._manager.mode

    def change_mode(self, mode: str) -> PermissionMode:
        """切换决策 mode：入口校验 CLI 传来的原始字符串，返回生效 mode。

        非法字符串抛 DomainValidationError；合法值经 manager 内存立即生效
        并持久化，持久化失败只告警不回滚（与 persist_rule 同一策略）。
        """

        try:
            target = PermissionMode(str(mode).strip())
        except ValueError:
            options = ", ".join(repr(str(value)) for value in PermissionMode)
            raise DomainValidationError(f"mode must be one of: {options}.") from None
        self._manager.change_mode(target)
        return target


def _authorization_from_result(result: InterruptionResult) -> PermissionAuthorization:
    """把中断结果映射回权限决策：RESOLVED 读 payload 里的 decision。"""

    if result.status is InterruptionStatus.RESOLVED:
        if str(result.payload.get("decision")) == str(PermissionDecision.ALLOW):
            return PermissionAuthorization.ALLOW
        return PermissionAuthorization.REJECT
    if result.status is InterruptionStatus.UNAVAILABLE:
        return PermissionAuthorization.UNAVAILABLE
    # CANCELLED：本轮被打断，未获批准。
    return PermissionAuthorization.REJECT


__all__ = ["PermissionService"]
