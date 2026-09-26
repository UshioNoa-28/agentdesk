"""会话驾驶锁(flock)与进程内空实现。

端侧允许多开应用,但同一会话同一时刻只允许一个窗口**签出**(checkout)
并驾驶它:actor 状态机(RUNNING 标志、drain、StreamHub 队列)全是进程
内存态,双进程驾驶同一会话会撕裂"一条消息 = 一次图运行"的契约。

签出发生在 UI 打开会话时(app.py),句柄存活到切换/退出:
- 进程退出(含崩溃)由内核自动释放 flock,不存在需要心跳/租约回收的
  僵尸持锁者;
- 非阻塞签出,失败立即抛 ``SessionDriverBusyError``,入口翻译为 409;
- 数据层并发安全不依赖此锁(seq 由单条语句原子分配),锁只保护驾驶权;
- 子会话不单独签出:其唯一入口是已签出主会话下的 /subagents,保护
  沿父子关系传递。
"""

from __future__ import annotations

import fcntl
from pathlib import Path

from agent.exceptions import SessionDriverBusyError


class FileSessionDriverHandle:
    """一次签出的句柄;release 前锁随 fd 存活。"""

    def __init__(self, handle) -> None:
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._handle.close()
        self._handle = None


class NullSessionDriverHandle:
    def release(self) -> None:
        return None


class FileSessionDriverLocks:
    """per-session advisory ``flock``;锁文件按会话惰性创建于 ``lock_dir``。"""

    def __init__(self, lock_dir: Path | str) -> None:
        directory = Path(lock_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory

    def checkout(self, session_id: str) -> FileSessionDriverHandle:
        """非阻塞签出会话驾驶权。

        Raises:
            SessionDriverBusyError: 已被其他进程(或本进程其他句柄)持有。
        """

        path = self._directory / f"{session_id}.lock"
        handle = path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise SessionDriverBusyError(session_id) from exc
        return FileSessionDriverHandle(handle)


class NullSessionDriverLocks:
    """测试/单实例形态的直通实现:``checkout`` 永不失败。"""

    def checkout(self, session_id: str) -> NullSessionDriverHandle:
        return NullSessionDriverHandle()


__all__ = [
    "FileSessionDriverHandle",
    "FileSessionDriverLocks",
    "NullSessionDriverHandle",
    "NullSessionDriverLocks",
]
