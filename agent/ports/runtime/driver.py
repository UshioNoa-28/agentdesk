"""会话驾驶权的跨进程签出端口。

签出(checkout)发生在入口层打开会话时,句柄存活到该会话被切换/关闭;
只有持有句柄的进程应当驱动该会话的工作流。消息落库本身不需要锁
(seq 由单条语句原子分配),任何进程任何时候都可向任何会话 INSERT。
"""

from __future__ import annotations

from typing import Protocol


class SessionDriverHandle(Protocol):
    """一次签出的句柄。"""

    def release(self) -> None: ...


class SessionDriverLockPort(Protocol):
    """会话粒度的独占驾驶权注册表。"""

    def checkout(self, session_id: str) -> SessionDriverHandle:
        """非阻塞签出;已被占用时抛 ``SessionDriverBusyError``。"""
        ...


__all__ = ["SessionDriverHandle", "SessionDriverLockPort"]
