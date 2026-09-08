"""Agent Service 健康探针。"""

from __future__ import annotations

from health_monitor.health import HealthCheck, HealthCheckContext, HttpHealthCheck


def build_agent_health_check(context: HealthCheckContext) -> HealthCheck:
    """根据共享配置创建 Agent Service 的 HTTP 探针。

    Args:
        context (HealthCheckContext): 包含服务 URL 和超时配置的上下文。

    Returns:
        HealthCheck: 指向 Agent health endpoint 的探针。
    """

    return HttpHealthCheck(
        name="agent",
        url=context.health.agent_health_url,
        timeout_seconds=context.health.health_monitor_check_timeout_seconds,
    )


__all__ = ["build_agent_health_check"]
