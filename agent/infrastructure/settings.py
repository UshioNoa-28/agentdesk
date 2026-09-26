"""Agent 私有的运行时配置。

每个 ``*Settings`` 类只描述一个基础设施或运行时边界。默认值一律写在字段
上；agent 边界保留 ``config/env/<scope>.env`` 文件承载真实增量
（密钥、开关、偏离默认值的部署参数），其余配置通过进程环境注入。
model 配置不在此列：模型身份（id、展示名、base_url、api_key）与模型档案
（窗口、输出上限、协议开关——即 ``ModelProfile`` 的可配面，全部可选、
缺键吃字段默认值；压缩水位、截断阈值等调度参数是代码里钉死的
``ContextSettings`` 固定数字，不开放配置）来自
``.agent-desk/settings.json`` 的 model 段，由 ``ModelCatalog`` 加载为扁平的
``ModelProfile``；**不存在全局窗口或全局协议参数**，一切随模型走。组合根
直接注入需要的窄配置，不提供跨边界的聚合 Settings。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.infrastructure.environment import env_files_for


class _AgentSettingsBase(BaseSettings):
    """Agent 专属配置共用的 dotenv 读取规则。"""

    model_config = SettingsConfigDict(
        env_file=env_files_for(),
        env_prefix="",
        extra="ignore",
    )


class DatabaseSettings(_AgentSettingsBase):
    """数据库连接配置；URL 只接受 SQLite 方言（aiosqlite 驱动）。"""

    database_url: str = "sqlite+aiosqlite:///data/agentdesk.db"


class ObservabilitySettings(_AgentSettingsBase):
    """内部 HTTP 和 provider wire 日志配置；默认关闭，按需经环境变量注入路径。"""

    # 记录 Agent 发给外部 OpenAI-compatible Provider 的完整 request/response
    # body。Authorization 等敏感 header 会被脱敏。
    provider_wire_log_path: str | None = None


class AgentRuntimeSettings(_AgentSettingsBase):
    """Agent 模型-工具循环的运行时上限。"""

    model_config = SettingsConfigDict(
        env_file=env_files_for("agent"),
        env_prefix="",
        extra="ignore",
    )

    max_turns: int = Field(default=32, gt=0)
    wait_default_timeout_seconds: float = Field(default=60.0, ge=1)
    wait_max_timeout_seconds: float = Field(default=600.0, gt=0)
    # .agent-desk 发现链（skills/mcp/settings/model）的工作目录根;缺省沿用进程 cwd(部署形态)。
    # 桌面形态应指向用户项目根,否则发现链找到的只是后端源码树;文件工具本身
    # 收绝对路径,靠权限规则约束,不再依赖此配置。
    workspace_root: str = ""
    session_lock_dir: str = "data/locks"

    @property
    def workspace_start(self) -> Path:
        """workspace 根起点（空则进程 cwd），供 skill/MCP/权限/模型发现共用。"""

        return Path(self.workspace_root) if self.workspace_root else Path.cwd()


__all__ = [
    "AgentRuntimeSettings",
    "DatabaseSettings",
    "ObservabilitySettings",
]
