"""Health Monitor 自身的运行配置。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL 连接地址；compose 负责注入服务间地址。"""

    model_config = SettingsConfigDict(extra="ignore")

    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5434/agentdesk"


class HealthSettings(BaseSettings):
    """Health Monitor 自身和依赖探针配置；compose 负责注入服务间地址。"""

    model_config = SettingsConfigDict(extra="ignore")

    health_monitor_host: str = "0.0.0.0"
    health_monitor_port: int = 8081
    agent_health_url: str = "http://localhost:8000/api/health"
    health_monitor_check_interval_seconds: float = 10.0
    health_monitor_check_timeout_seconds: float = 3.0


__all__ = ["DatabaseSettings", "HealthSettings"]
