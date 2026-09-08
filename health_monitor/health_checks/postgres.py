"""PostgreSQL 健康探针。"""

from __future__ import annotations

from urllib.parse import urlparse

from health_monitor.health import HealthCheck, HealthCheckContext, TcpHealthCheck


def build_postgres_health_check(context: HealthCheckContext) -> HealthCheck:
    """从 DATABASE_URL 创建 PostgreSQL TCP 探针。

    Args:
        context (HealthCheckContext): 包含数据库连接 URL 和探针超时的上下文。

    Returns:
        HealthCheck: 指向数据库主机和端口的 TCP 探针。

    Raises:
        ValueError: DATABASE_URL 缺少主机名时抛出。
    """

    parsed = urlparse(context.database.database_url)
    if not parsed.hostname:
        raise ValueError("DATABASE_URL must contain a PostgreSQL host")
    return TcpHealthCheck(
        name="postgres",
        host=parsed.hostname,
        port=parsed.port or 5432,
        timeout_seconds=context.health.health_monitor_check_timeout_seconds,
    )


__all__ = ["build_postgres_health_check"]
