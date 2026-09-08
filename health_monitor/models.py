"""Health Monitor 使用的本地状态模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True, slots=True)
class HealthResult:
    """一次依赖健康检查的结果。"""

    name: str
    healthy: bool
    detail: str | None = None


HealthResults = Mapping[str, HealthResult]


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """最近一次依赖探测的不可变快照。

    ``checked_at`` 只在一次完整探测成功完成后写入。单个依赖失败不会让
    整次探测失败，具体失败会保存在 ``dependencies``；只有探针装配或监控
    任务本身异常时才会填充 ``error``。
    """

    dependencies: HealthResults
    checked_at: datetime | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        """是否至少完成过一次完整探测。

        Returns:
            bool: ``checked_at`` 已写入时为 ``True``，否则表示监控仍在启动。
        """

        return self.checked_at is not None

    @property
    def all_healthy(self) -> bool:
        """最近一次快照中的所有依赖是否健康。

        Returns:
            bool: 已完成探测、监控自身无错误且每个依赖均健康时为 ``True``。
        """

        return self.ready and not self.error and all(
            result.healthy for result in self.dependencies.values()
        )


__all__ = ["HealthResult", "HealthResults", "HealthSnapshot"]
