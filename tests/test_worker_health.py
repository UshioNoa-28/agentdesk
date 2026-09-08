from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

import httpx

from health_monitor.health import DependencyHealthMonitor, build_health_monitor
from health_monitor.models import HealthResult
from health_monitor.monitor import HealthMonitor
from health_monitor.server import create_app
from health_monitor.settings import DatabaseSettings, HealthSettings


class HealthMonitorTests(IsolatedAsyncioTestCase):
    """验证 Health Monitor 的探测和快照边界。"""

    def test_deployment_monitor_contains_only_owned_core_services(self) -> None:
        """拆仓后不再把外部 MCP、RAG 或向量库当作核心容器依赖。"""

        dependencies = build_health_monitor(
            health_settings=HealthSettings(),
            database_settings=DatabaseSettings(),
        )

        self.assertEqual(
            ("agent", "postgres"),
            tuple(check.name for check in dependencies._checks),
        )

    async def test_livez_does_not_require_dependency_probe(self) -> None:
        """监控容器自身的 liveness 不应等待外部依赖或首轮轮询。"""

        application = create_app()
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/livez")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    async def test_dependency_results_remain_separate_when_one_probe_raises(self) -> None:
        """一个探针异常时，监控器仍保留其它容器的状态。"""

        statuses = await DependencyHealthMonitor(
            (_RaisingCheck(), _HealthyCheck())
        ).check_all()

        self.assertFalse(statuses["postgres"].healthy)
        self.assertEqual(statuses["postgres"].detail, "RuntimeError")
        self.assertTrue(statuses["model"].healthy)

    async def test_monitor_snapshot_is_not_ready_until_first_probe(self) -> None:
        """首轮探测前不能声称已经 ready。"""

        monitor = HealthMonitor(
            dependencies=DependencyHealthMonitor((_HealthyCheck(),)),
            interval_seconds=60,
        )
        self.assertFalse(monitor.snapshot.ready)
        snapshot = await monitor.check_once()
        self.assertTrue(snapshot.ready)
        self.assertTrue(snapshot.all_healthy)
        self.assertEqual(set(snapshot.dependencies), {"model"})

    async def test_monitor_preserves_unhealthy_dependency_in_snapshot(self) -> None:
        """单个依赖失败时，首轮探测仍然完成并保留其它结果。"""

        monitor = HealthMonitor(
            dependencies=DependencyHealthMonitor((_RaisingCheck(), _HealthyCheck())),
            interval_seconds=60,
        )
        snapshot = await monitor.check_once()
        self.assertTrue(snapshot.ready)
        self.assertFalse(snapshot.all_healthy)
        self.assertFalse(snapshot.dependencies["postgres"].healthy)
        self.assertTrue(snapshot.dependencies["model"].healthy)

    async def test_monitor_keeps_previous_snapshot_when_monitor_task_fails(self) -> None:
        """监控任务异常时保留旧快照，并单独暴露任务错误。"""

        monitor = HealthMonitor(
            dependencies=_FailingAfterFirstProbe(),
            interval_seconds=60,
        )
        await monitor.check_once()
        snapshot = await monitor.check_once()
        self.assertTrue(snapshot.ready)
        self.assertEqual(snapshot.error, "RuntimeError")
        self.assertEqual(set(snapshot.dependencies), {"model"})
        self.assertFalse(snapshot.all_healthy)


class _RaisingCheck:
    name = "postgres"

    async def check(self) -> HealthResult:
        raise RuntimeError("database probe failed")


class _HealthyCheck:
    name = "model"

    async def check(self) -> HealthResult:
        return HealthResult(name=self.name, healthy=True)


class _FailingAfterFirstProbe:
    """先返回一次成功快照，再模拟监控集合本身崩溃。"""

    def __init__(self) -> None:
        self.calls = 0

    async def check_all(self) -> dict[str, HealthResult]:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("monitor failed")
        return {"model": HealthResult(name="model", healthy=True)}
