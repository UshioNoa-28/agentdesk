"""进程内嵌 Agent 运行时的装配与生命周期(传输无关)。

启动/关闭序列原先长在 FastAPI lifespan 里;摘除 HTTP 层后
``AgentBootstrap`` 是这套序列的唯一来源,CLI 嵌入传输与
``python -m agent`` 手动入口都复用它。

关闭顺序是有意的:先唤醒在途中断 Future(否则 actor 永久卡在交互点),
再停 runtime、关 MCP、关模型客户端、关 Hub,最后关容器。
"""

from __future__ import annotations

import logging

from dishka import AsyncContainer, make_async_container

from agent.container import providers
from agent.infrastructure.db import migrate_database
from agent.ports.model import AgentModelPort
from agent.ports.runtime import AgentStreamHubPort, InterruptionBrokerPort
from agent.ports.services import PermissionManagerPort
from agent.ports.tools import AgentRuntimePort, McpRegistryPort

logger = logging.getLogger("app")


class AgentBootstrap:
    """持有 Dishka 容器;``start``/``stop`` 必须按序成对调用。"""

    def __init__(self) -> None:
        self.container: AsyncContainer = make_async_container(*providers())

    async def start(self) -> None:
        """预热配置与外部连接;任何一步失败都直接抛出,拒绝半启动状态。"""

        # Resolve the policy eagerly so a missing or stale tool entry fails
        # startup instead of surfacing during the first tool call.
        await self.container.get(PermissionManagerPort)
        # 模型目录是启动快照（settings.json 的 model 段）：在此构造一次，让
        # 缺 profile、窗口不合法等配置错误在 start 失败，而不是延迟到 stop
        # 关客户端时才暴露。
        await self.container.get(AgentModelPort)
        await migrate_database()
        registry = await self.container.get(McpRegistryPort)
        # MCP/Skill 在这里预热并暴露配置错误;Meta Tool 只在请求中
        # 使用当前已连接的注册表,不把数据库或 Session 状态提升到 APP scope。
        await registry.start_all_async()

    async def stop(self) -> None:
        """逆序释放资源;单个环节失败只记录,不阻塞后续清理。"""

        # 中断等待是进程内 Future;先唤醒它们,再停 Runtime,避免停机时
        # actor 协程永久卡在权限审批或 ask_user 等待点。
        try:
            broker = await self.container.get(InterruptionBrokerPort)
            broker.close()
        except Exception:
            logger.exception("Failed to close interruption broker")
        # 多智能体 Runtime 的工厂安装与启动由 AgentRuntimeManager 幂等负责,
        # 这里只做优雅停机;随后再关 MCP 与数据库资源。
        try:
            runtime = await self.container.get(AgentRuntimePort)
            if runtime.is_running:
                await runtime.stop()
        except Exception:
            logger.exception("Failed to stop multi-agent runtime")
        try:
            registry = await self.container.get(McpRegistryPort)
            await registry.close_async()
        except Exception:
            logger.exception("Failed to close MCP registry")
        try:
            model = await self.container.get(AgentModelPort)
            close = getattr(model, "aclose", None)
            if close is not None:
                await close()
        except Exception:
            logger.exception("Failed to close Agent model client")
        try:
            hub = await self.container.get(AgentStreamHubPort)
            hub.close()
        except Exception:
            logger.exception("Failed to close stream hub")
        await self.container.close()

    async def __aenter__(self) -> "AgentBootstrap":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()


__all__ = ["AgentBootstrap"]
