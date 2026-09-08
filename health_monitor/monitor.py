"""Health Monitor 的轮询生命周期和快照管理。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime

from health_monitor.health import DependencyHealthMonitor
from health_monitor.models import HealthSnapshot

logger = logging.getLogger(__name__)


class HealthMonitor:
    """在后台轮询依赖并保留最近一次探测结果。

    这个类只负责运行期状态和生命周期；探针的构造与具体检查逻辑由
    ``health_monitor.health`` 负责，HTTP 路由由 ``server.py`` 负责。
    """

    def __init__(
        self,
        *,
        dependencies: DependencyHealthMonitor,
        interval_seconds: float,
    ) -> None:
        """创建后台轮询器并初始化尚未 ready 的空快照。

        Args:
            dependencies (DependencyHealthMonitor): 并行执行依赖探针的集合。
            interval_seconds (float): 两次探测之间的等待秒数。

        Raises:
            ValueError: 轮询间隔不是正数时抛出。
        """

        if interval_seconds <= 0:
            raise ValueError("Health monitor interval must be greater than zero")
        self._dependencies = dependencies
        self._interval_seconds = interval_seconds
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._snapshot = HealthSnapshot(dependencies={})

    @property
    def snapshot(self) -> HealthSnapshot:
        """返回最近一次快照。

        Returns:
            HealthSnapshot: 当前内存中的不可变探测状态；首轮探测前 ``ready`` 为假。
        """

        return self._snapshot

    async def start(self) -> None:
        """启动后台轮询任务；重复启动不会创建第二个任务。

        Returns:
            None: 轮询任务已经安排，或发现已有活动任务后幂等返回。
        """

        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(),
            name="anna-health-monitor",
        )

    async def stop(self) -> None:
        """停止后台轮询并等待任务退出。

        Returns:
            None: 轮询协程退出后返回；没有启动任务时也保持幂等。
        """

        task = self._task
        if task is None:
            return
        self._stop_event.set()
        with suppress(asyncio.CancelledError):
            await task
        self._task = None

    async def check_once(self) -> HealthSnapshot:
        """执行一次完整探测并更新快照。

        单个探针异常由 ``DependencyHealthMonitor`` 转换为独立的 unhealthy
        结果；这里只处理探针集合本身的意外异常。

        Returns:
            HealthSnapshot: 更新后的依赖结果和本轮 UTC 检查时间。
        """

        try:
            dependencies = await self._dependencies.check_all()
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            logger.exception("Health Monitor dependency check task raised")
            previous = self._snapshot
            self._snapshot = HealthSnapshot(
                # 保留上一份可用快照，同时把本轮监控任务异常单独标出；
                # 这样调用方能区分“依赖刚刚失败”和“监控循环本身出错”。
                dependencies=previous.dependencies,
                checked_at=previous.checked_at,
                error=type(exception).__name__,
            )
            return self._snapshot

        self._snapshot = HealthSnapshot(
            dependencies=dict(dependencies),
            checked_at=datetime.now(UTC),
        )
        return self._snapshot

    async def _run(self) -> None:
        """立即执行一次探测，然后按固定间隔继续轮询。

        Note:
            使用 ``stop_event.wait`` 等待间隔，可以在 stop 时立即唤醒，而不必
            等待完整 sleep 周期。
        """

        while not self._stop_event.is_set():
            await self.check_once()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval_seconds,
                )
            except TimeoutError:
                continue


__all__ = ["HealthMonitor"]
