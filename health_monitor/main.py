"""Health Monitor 进程入口。"""

from __future__ import annotations

import logging

import uvicorn

from health_monitor.server import create_app
from health_monitor.settings import DatabaseSettings, HealthSettings

logging.basicConfig(level=logging.INFO)
_health_settings = HealthSettings()
_database_settings = DatabaseSettings()
app = create_app(
    health_settings=_health_settings,
    database_settings=_database_settings,
)


def main() -> None:
    """使用配置的地址启动 Uvicorn。

    Returns:
        None: Uvicorn 服务器退出后返回。
    """

    uvicorn.run(
        app,
        host=_health_settings.health_monitor_host,
        port=_health_settings.health_monitor_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
