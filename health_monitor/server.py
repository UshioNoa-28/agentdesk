"""Health Monitor 的 FastAPI 应用。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from health_monitor.health import build_health_monitor
from health_monitor.models import HealthSnapshot
from health_monitor.monitor import HealthMonitor
from health_monitor.settings import DatabaseSettings, HealthSettings


def create_app(
    *,
    health_settings: HealthSettings | None = None,
    database_settings: DatabaseSettings | None = None,
) -> FastAPI:
    """创建独立的健康聚合 API 和后台轮询任务。

    Args:
        health_settings (HealthSettings | None): 可选健康 URL、轮询和监听配置。
        database_settings (DatabaseSettings | None): 可选 PostgreSQL 连接配置。

    Returns:
        FastAPI: 注册了 liveness、readiness 和依赖快照路由的应用实例。

    Note:
        ``/livez`` 只回答进程是否存活，``/readyz`` 需要首轮探测完成；依赖
        细节通过 ``/health`` 查看。
    """

    resolved_health = health_settings or HealthSettings()
    resolved_database = database_settings or DatabaseSettings()
    monitor = HealthMonitor(
        dependencies=build_health_monitor(
            health_settings=resolved_health,
            database_settings=resolved_database,
        ),
        interval_seconds=resolved_health.health_monitor_check_interval_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """在 FastAPI 生命周期中启动并停止后台健康轮询任务。

        Yields:
            None: 应用处于可接收健康查询的运行阶段。
        """

        await monitor.start()
        try:
            yield
        finally:
            await monitor.stop()

    application = FastAPI(
        title="AgentDesk Health Monitor",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/livez")
    async def livez() -> dict[str, str]:
        """只报告监控进程自身是否能够处理请求。

        Returns:
            dict[str, str]: 固定的 ``{"status": "ok"}`` liveness 响应。
        """

        return {"status": "ok"}

    @application.get("/readyz")
    async def readyz() -> JSONResponse:
        """报告是否已经完成至少一次完整依赖探测。

        Returns:
            JSONResponse: 首轮探测前返回 503，完成后返回 200；响应同时包含
                最近检查时间和监控任务错误。
        """

        snapshot = monitor.snapshot
        status_code = 200 if snapshot.ready else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if snapshot.ready else "starting",
                "last_checked_at": _timestamp(snapshot),
                "error": snapshot.error,
            },
        )

    @application.get("/health")
    async def health() -> dict[str, object]:
        """返回监控进程状态及最近一次依赖快照。

        这个端点保持 liveness 语义：依赖异常会出现在响应体中，但不会让
        Health Monitor 自身被误判为已经退出。需要等待首轮探测完成时使用
        ``/readyz``。

        Returns:
            dict[str, object]: 进程状态、依赖状态、最近检查时间和监控错误。
        """

        snapshot = monitor.snapshot
        return {
            "status": "ok",
            "monitor": "ready" if snapshot.ready else "starting",
            "dependencies": {
                name: {
                    "status": "ok" if result.healthy else "unavailable",
                    "detail": result.detail,
                }
                for name, result in snapshot.dependencies.items()
            },
            "last_checked_at": _timestamp(snapshot),
            "error": snapshot.error,
        }

    return application


def _timestamp(snapshot: HealthSnapshot) -> str | None:
    """把快照时间转换成 JSON 可编码的 UTC 字符串。

    Args:
        snapshot (HealthSnapshot): 要读取时间字段的健康快照。

    Returns:
        str | None: ISO 8601 时间文本；尚未检查时返回 ``None``。
    """

    return snapshot.checked_at.isoformat() if snapshot.checked_at is not None else None


__all__ = ["create_app"]
