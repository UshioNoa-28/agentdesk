"""运行期依赖健康检查。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import httpx

from health_monitor.models import HealthResult
from health_monitor.settings import (
    DatabaseSettings,
    HealthSettings,
)

logger = logging.getLogger(__name__)


class HealthCheck(Protocol):
    """一个独立依赖探针的最小接口。"""

    name: str

    async def check(self) -> HealthResult:
        """执行一次探测并返回标准化健康结果。

        Returns:
            HealthResult: 包含探针名称、healthy 标志和可诊断详情的结果。
        """

        ...


@dataclass(frozen=True, slots=True)
class HealthCheckContext:
    """探针工厂需要的配置上下文。"""

    health: HealthSettings
    database: DatabaseSettings


HealthCheckFactory: TypeAlias = Callable[[HealthCheckContext], HealthCheck]


def build_health_monitor(
    *,
    health_settings: HealthSettings,
    database_settings: DatabaseSettings,
) -> DependencyHealthMonitor:
    """显式创建当前部署需要的依赖健康检查集合。

    探针数量很少且由本项目固定拥有；显式装配比全局注册表和动态 import
    更容易测试，也不会因为模块导入顺序或测试进程状态而改变结果。

    Args:
        health_settings (HealthSettings): 服务 health URL 和探针超时。
        database_settings (DatabaseSettings): PostgreSQL 连接地址。

    Returns:
        DependencyHealthMonitor: 包含 Agent 和 PostgreSQL 探针的监控器。
    """

    from health_monitor.health_checks.agent import build_agent_health_check
    from health_monitor.health_checks.postgres import build_postgres_health_check

    context = HealthCheckContext(
        health=health_settings,
        database=database_settings,
    )
    factories: tuple[HealthCheckFactory, ...] = (
        build_agent_health_check,
        build_postgres_health_check,
    )
    return DependencyHealthMonitor(factory(context) for factory in factories)


class DependencyHealthMonitor:
    """并行检查所有依赖并保留独立结果。"""

    def __init__(self, checks: Iterable[HealthCheck]) -> None:
        """保存探针集合，并校验每个探针都有唯一名称。

        Args:
            checks (Iterable[HealthCheck]): 要并行执行的依赖探针。构造时会复制
                为 tuple，使后续检查顺序和集合内容保持稳定。

        Raises:
            ValueError: 两个探针使用同一个 ``name`` 时抛出，因为重复名称会使
                健康快照互相覆盖，无法定位实际故障。
        """

        self._checks = tuple(checks)
        names = [check.name for check in self._checks]
        if len(names) != len(set(names)):
            raise ValueError("Health check names must be unique")

    async def check_all(self) -> dict[str, HealthResult]:
        """并行执行所有探针，单个探针失败时不影响其它探针。

        Returns:
            dict[str, HealthResult]: 以探针名称索引的本轮健康结果；探针抛出的
                异常会转换为 ``healthy=False``，取消异常仍会继续向上传播。
        """

        raw_results = await asyncio.gather(
            *(check.check() for check in self._checks),
            return_exceptions=True,
        )
        results: list[HealthResult] = []
        for check, result in zip(self._checks, raw_results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                result = HealthResult(
                    name=check.name,
                    healthy=False,
                    detail=type(result).__name__,
                )
            if not result.healthy:
                logger.warning(
                    "Dependency health check failed: %s (%s)", result.name, result.detail
                )
            results.append(result)
        return {result.name: result for result in results}


@dataclass(frozen=True, slots=True)
class HttpHealthCheck:
    """通过 HTTP liveness/ready endpoint 检查服务。"""

    name: str
    url: str
    timeout_seconds: float

    async def check(self) -> HealthResult:
        """访问配置的 HTTP endpoint，并把异常转换为 unhealthy 结果。

        Returns:
            HealthResult: HTTP 返回成功时 healthy，否则包含异常类型。
        """

        return await _run_check(self.name, self._probe, self.timeout_seconds)

    async def _probe(self) -> None:
        """执行一次不使用宿主代理的异步 GET 探测。

        Note:
            Health Monitor 只访问 Compose 内部地址，显式绕过宿主机代理；使用
            原生 async client，避免把 urllib 放进线程后在取消时留下线程资源。
        """

        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(self.url)
            response.raise_for_status()


@dataclass(frozen=True, slots=True)
class TcpHealthCheck:
    """检查一个 TCP 服务端口是否可以建立连接。"""

    name: str
    host: str
    port: int
    timeout_seconds: float

    async def check(self) -> HealthResult:
        """尝试建立并关闭 TCP 连接，返回标准化健康结果。

        Returns:
            HealthResult: 连接建立成功时 healthy，否则标记为 unhealthy。
        """

        return await _run_check(self.name, self._probe, self.timeout_seconds)

    async def _probe(self) -> None:
        """打开目标端口并立即关闭连接，只验证网络可达性。

        Returns:
            None: TCP 握手和关闭均成功后返回。

        Raises:
            OSError: DNS 解析、连接建立或关闭失败时抛出。
            TimeoutError: 在配置的超时时间内无法建立连接时抛出。
        """

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=self.timeout_seconds,
        )
        writer.close()
        await writer.wait_closed()
        del reader


async def _run_check(
    name: str,
    probe: Callable[[], Awaitable[None]],
    timeout_seconds: float,
) -> HealthResult:
    """把超时和底层异常转换成标准化健康结果。

    Args:
        name (str): 探针名称。
        probe (Callable[[], Awaitable[None]]): 成功时正常返回、失败时抛异常的异步探测。
        timeout_seconds (float): 单次探测的超时时间。

    Returns:
        HealthResult: 探测成功为 healthy，否则记录异常类型并标为 unhealthy。
    """

    try:
        await asyncio.wait_for(probe(), timeout=timeout_seconds)
    except asyncio.CancelledError:
        raise
    except Exception as exception:
        return HealthResult(name=name, healthy=False, detail=type(exception).__name__)
    return HealthResult(name=name, healthy=True)


__all__ = [
    "DependencyHealthMonitor",
    "HealthCheck",
    "HealthCheckContext",
    "HttpHealthCheck",
    "TcpHealthCheck",
    "build_health_monitor",
]
