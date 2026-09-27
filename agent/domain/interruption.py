"""进程内人机中断的领域值对象。

一次中断 = 运行中的协程挂起、把控制权交给入口、等一个带外回复唤醒自己。
权限审批与 ``ask_user`` 澄清共用这套机制，各自携带不同的 typed request 与
回填 payload；broker 只认 :data:`InterruptionRequest` 联合类型，不理解其语义，
按 ID 把入口回复路由回挂起的 Future。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias, Union


class InterruptionStatus(StrEnum):
    """一次 ``request``  await 的收尾性质。"""

    RESOLVED = "resolved"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"


class InterruptionReplyStatus(StrEnum):
    """``reply`` 同步投递的受理结果；语义校验由各 caller 在投递前完成。"""

    ACCEPTED = "accepted"
    NOT_FOUND = "not_found"
    ALREADY_RESOLVED = "already_resolved"


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """工具执行前等待入口审批的一次性权限中断。

    ``targets`` 是工具给出的逐段核验对象，随事件原样下发：入口展示、
    "always allow" 回填的 scope 都以此为唯一来源，入口不做任何提取。
    空元组表示本工具无权限面（默认放行的工具被整工具 ask/deny 拉回询问）。
    """

    interruption_id: str
    session_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, object]
    targets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AskUserRequest:
    """模型向用户发起带选项澄清提问、本轮内等待答复的中断。

    以 ``interruption_id`` 与入口回复关联；不需要 tool_call_id——ask_user
    自身的 tool_call 帧照常流式下发，澄清框凭本请求的 question/options 渲染。
    """

    interruption_id: str
    session_id: str
    question: str
    options: tuple[str, ...]
    recommended: str | None = None


InterruptionRequest: TypeAlias = Union[PermissionRequest, AskUserRequest]


@dataclass(frozen=True, slots=True)
class InterruptionResult:
    """中断收尾：``status`` 说明性质，``payload`` 是入口回填的原始映射。

    仅 ``RESOLVED`` 时 ``payload`` 携带约定键（权限 ``decision``、ask_user
    ``answer``），由发起方按自己请求的 kind 解读；broker 不解释其内容。
    其余状态没有回复，默认落在每实例独立的空 dict 上，消费方无需判空即可取键。
    """

    status: InterruptionStatus
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PendingInterruption:
    """broker 保存的一次在途中断：唤醒目标 + 原始请求（供 caller 反查语义）。"""

    future: asyncio.Future[InterruptionResult]
    request: InterruptionRequest


__all__ = [
    "AskUserRequest",
    "InterruptionReplyStatus",
    "InterruptionRequest",
    "InterruptionResult",
    "InterruptionStatus",
    "PendingInterruption",
    "PermissionRequest",
]
