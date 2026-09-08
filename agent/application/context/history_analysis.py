"""持久化 Message 序列的确定性结构分析。

这些函数只做"纯消息流分析"：找锚点、切工具交换、定位摘要/冷压缩
边界。不依赖任何实例状态、不做投影、不调 token 计数——上下文压缩的
范围选择（``ContextCompactor``）与截断策略都建立在这些原语之上。
"""

from __future__ import annotations

from collections.abc import Sequence

from agent.domain.exceptions import ContextCompactionError
from agent.domain.messages import Message, MessageKind, message_kind_from_value


def message_index(history: Sequence[Message], message_id: str) -> int:
    """根据消息 ID 返回它在历史中的下标。"""

    for index, message in enumerate(history):
        if message.id == message_id:
            return index
    raise ContextCompactionError(
        "Current user message is missing from the context history."
    )


def collect_tool_exchanges(
    messages: Sequence[Message],
) -> list[tuple[Message, ...]]:
    """收集tool call tool result 配对"""
    groups: list[tuple[Message, ...]] = []
    current: list[Message] | None = None
    for message in messages:
        if message.role == "human" and message.metadata.get("source") == "agent":
            continue
        if message.role == "assistant" and message.tool_calls:
            if current: # 如果 assistant 发起了新的工具调用且当前current有内容 则存入group
                groups.append(tuple(current))
            current = [message] # 如果没有则添加到current
        elif message.role == "tool" and current is not None:
            current.append(message) # 新的工具调用且当前没有待配对的tool call存入current
        else:
            if current: # 直到非工具结果再闭合
                groups.append(tuple(current))
                current = None
    if current:
        groups.append(tuple(current))
    return groups


def is_wait_exchange(exchange: tuple[Message, ...]) -> bool:
    """头部 assistant 的 tool_calls 全部是 wait_for_replies 即为纯等待轮。"""

    head = exchange[0]
    if head.role != "assistant" or not head.tool_calls:
        return False
    return all(call.name == "wait_for_replies" for call in head.tool_calls)


def drop_wait_exchanges(messages: Sequence[Message]) -> list[Message]:
    """从消息序列中剔除属于纯 wait 交换的成员（保持其余顺序）。"""

    dropped: set[str] = set()
    for exchange in collect_tool_exchanges(messages):
        if is_wait_exchange(exchange):
            dropped.update(item.id for item in exchange)
    return [item for item in messages if item.id not in dropped]


def latest_summary(
    history: Sequence[Message],
    *,
    positions: dict[str, int],
    anchor_index: int,
) -> tuple[Message | None, int | None]:
    """找最新一条有效 SUMMARY 及其覆盖边界下标。

    无效（边界缺失、边界越过锚点或摘要自身位置）的摘要被跳过；
    返回 ``(None, None)`` 表示没有可用的旧摘要。
    """

    latest: Message | None = None
    latest_boundary_index: int | None = None
    for position, message in enumerate(history):
        if (
            message_kind_from_value(message.metadata.get("kind"))
            != MessageKind.SUMMARY
        ):
            continue
        boundary_id = message.metadata.get("covered_through_message_id")
        if not isinstance(boundary_id, str):
            continue
        boundary_index = positions.get(boundary_id)
        if (
            boundary_index is None
            or boundary_index >= anchor_index
            or boundary_index >= position
        ):
            continue
        latest = message
        latest_boundary_index = boundary_index
    return latest, latest_boundary_index


def latest_cold_boundary_index(
    history: Sequence[Message],
    positions: dict[str, int],
) -> int | None:
    """Return the furthest valid cold checkpoint boundary in history."""

    latest: int | None = None
    for item in history:
        if (
            message_kind_from_value(item.metadata.get("kind"))
            != MessageKind.MAINTENANCE
            or item.metadata.get("maintenance_type") != "cold_compact"
        ):
            continue
        boundary_id = item.metadata.get("cold_cleared_through_message_id")
        if not isinstance(boundary_id, str):
            continue
        boundary_index = positions.get(boundary_id)
        if boundary_index is not None:
            latest = boundary_index if latest is None else max(latest, boundary_index)
    return latest


def protected_tail_start(
    tail: Sequence[Message],
    *,
    max_recent_tool_calls: int,
) -> int:
    """保护区起点：最近 ``max_recent_tool_calls`` 组交换之外。

    保护区语义：最近 N 组工具交换的原文必须留在模型上下文里（不进
    摘要、不被 cold 清空）。若锚点前移使保护区内容滚出豁免区，巨型
    工具结果即可在后续轮次被压缩——保护区卡死是暂态而非死锁。
    """

    if max_recent_tool_calls <= 0:
        return len(tail)
    exchanges = collect_tool_exchanges(tail)
    if not exchanges:
        return len(tail)
    first_protected = exchanges[max(0, len(exchanges) - max_recent_tool_calls)][0]
    for index, message in enumerate(tail):
        if message.id == first_protected.id:
            return index
    raise ContextCompactionError(
        "A protected tool exchange is missing from the summary tail."
    )


def first_exchange_end(tail: Sequence[Message]) -> int:
    """返回"最老一组截断单元"的结束下标（不含）。

    截断单元 = 开头连续的非交换消息 + 第一组完整交换。交换在
    tail 末尾闭合时，单元结束于该交换最后一个成员之后。没有
    交换时整个 tail 是一个单元（返回 len(tail)，由调用方判定
    截光）。子代理回报（human, source=agent）视为交换内透明
    插队者，不属于"开头连续非交换消息"。
    """

    first_exchange_start = len(tail)
    for index, message in enumerate(tail):
        if message.role == "assistant" and message.tool_calls:
            first_exchange_start = index
            break
    if first_exchange_start == len(tail):
        return len(tail)

    # 单元 = [0, 第一组交换最后一个成员下标 + 1)。用 collect_tool_exchanges
    # 的分组规则找到末成员位置（透明插队的子代理回报若紧跟在结果之后，
    # 不属于交换成员，单元在最后一个 tool 结果处闭合）。
    first_exchange = collect_tool_exchanges(tail)[0]
    member_ids = {member.id for member in first_exchange}
    for index in range(len(tail) - 1, -1, -1):
        if tail[index].id in member_ids:
            return index + 1
    return len(tail)


__all__ = [
    "collect_tool_exchanges",
    "drop_wait_exchanges",
    "first_exchange_end",
    "is_wait_exchange",
    "latest_cold_boundary_index",
    "latest_summary",
    "message_index",
    "protected_tail_start",
]
