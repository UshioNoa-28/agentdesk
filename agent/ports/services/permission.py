"""Meta Tool 权限用例端口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent.domain.permissions import (
    PermissionAction,
    PermissionAuthorization,
    PermissionDecision,
    PermissionMode,
)


class PermissionManagerPort(Protocol):
    """按 (tool, scope) 规则与 mode 回答"这个调用落在哪个列表里"。

    ``target`` 是被匹配对象(如绝对路径、单个命令段、MCP server 名)。
    工具名本身就是语义域，不需要额外 style；target 为空时只有无 scope 的
    整工具规则参与匹配。多段(如 bash)由调用方逐段询问本端口。
    """

    @property
    def mode(self) -> PermissionMode: ...

    def in_allow(self, tool_name: str, target: str = "") -> bool: ...

    def in_deny(self, tool_name: str, target: str = "") -> bool: ...

    def in_ask(self, tool_name: str, target: str = "") -> bool: ...

    def persist_rule(
        self,
        tool_name: str,
        permission_action: PermissionAction,
        scope: str = "",
    ) -> None:
        """校验 (tool, scope) 形状后落规则；未知工具名或非法 scope 抛 ValueError。"""

        ...

    def change_mode(self, mode: PermissionMode) -> None:
        """切换决策 mode：内存立即生效并持久化。"""

        ...


class PermissionServicePort(Protocol):
    """发起和回复本地交互式权限请求的应用用例。"""

    async def authorize(
        self,
        *,
        caller_session_id: str,
        tool_call_id: str,
        name: str,
        arguments: dict[str, object],
        targets: Sequence[str] | None = None,
    ) -> PermissionAuthorization:
        """按静态策略直接放行，或等待 CLI 对本次调用作出决定。

        ``targets`` 由调用方从工具抽取，三态：
        - None：工具声明无权限面，deny/ask（整工具）之外默认放行；
        - 非空段：逐段独立走决策表，deny/ask 任一段命中即生效，
          allow 需要**全部**段命中，未匹配走询问兜底。
        参数不合法时工具钩子直接抛错，由 graph 转成参数错误，不经本端口。
        """

        ...

    async def reply(
        self,
        *,
        permission_id: str,
        decision: str,
        scope: Sequence[str] | None = None,
    ) -> PermissionDecision:
        """校验并消费权限决定；``scope`` 非 None 且 ``allow`` 时逐段落规则。

        段是事件里 ``targets`` 的原文，入口零加工：多段逐条各落一条规则；
        ``[""]`` 表示显式整工具放行，缺省 None 只放行本次调用、不落盘。
        裸 str（本身就是字符序列）、空序列、非字符串段以及非 ``allow`` 决定
        携带 scope 都是非法组合，在唤醒请求前直接拒绝。
        无效或过期请求抛出应用异常。
        """

        ...

    def current_mode(self) -> PermissionMode:
        """读取当前生效的决策 mode，供 CLI 展示状态。"""

        ...

    def change_mode(self, mode: str) -> PermissionMode:
        """以原始字符串切换决策 mode 并返回生效值；非法值抛领域校验错误。"""

        ...


__all__ = [
    "PermissionAuthorization",
    "PermissionManagerPort",
    "PermissionServicePort",
]
