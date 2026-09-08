from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.api.exception_handlers import register_exception_handlers
from agent.api.routes import router
from agent.container import providers
from agent.infrastructure.db import initialize_agent_database
from agent.ports.model import AgentModelPort
from agent.ports.tools import AgentRuntimePort, McpRegistryPort

"""FastAPI 应用入口。

应用启动时创建 Dishka 容器、初始化 PostgreSQL 表；应用关闭时关闭容器，
由容器释放数据库引擎等应用级资源。
"""

def create_app() -> FastAPI:
    """创建并组装 FastAPI 应用实例。

    Returns:
        FastAPI: 已注册 Agent 路由、异常处理器和 Dishka 容器的应用实例。
    """

    _configure_application_logging()
    container = make_async_container(*providers())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        """只管理 Agent 自己的数据库初始化和 Dishka 容器生命周期。

        Args:
            _ (FastAPI): FastAPI 传入的应用实例；启动逻辑不需要读取其属性。

        Yields:
            None: Agent 数据库和 MCP 连接初始化完成后的运行阶段。

        Note:
            退出时先关闭 MCP Registry，再关闭 Dishka 容器，保证子进程和 HTTP
            客户端不会在数据库资源释放后继续运行。
        """

        try:
            await initialize_agent_database(await container.get(AsyncEngine))
            registry = await container.get(McpRegistryPort)
            # MCP/Skill 自身在这里预热并暴露配置错误；Meta Tool 只在请求中
            # 使用当前已连接的注册表，不把数据库或 Session 状态提升到 APP scope。
            await registry.start_all_async()
            yield
        finally:
            # Multi-Agent Runtime 的工厂安装与启动由 AgentRuntimeManager 幂等负责，
            # 这里只做优雅停机；随后再关闭 MCP 与数据库资源。
            try:
                runtime = await container.get(AgentRuntimePort)
                if runtime.is_running:
                    await runtime.stop()
            except Exception:
                logging.getLogger("app").exception("Failed to stop multi-agent runtime")
            try:
                registry = await container.get(McpRegistryPort)
                await registry.close_async()
            except Exception:
                logging.getLogger("app").exception("Failed to close MCP registry")
            try:
                model = await container.get(AgentModelPort)
                close = getattr(model, "aclose", None)
                if close is not None:
                    await close()
            except Exception:
                logging.getLogger("app").exception("Failed to close Agent model client")
            await container.close()

    application = FastAPI(title="AgentDesk", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(application)
    application.include_router(router)
    setup_dishka(container, application)
    return application


def _configure_application_logging() -> None:
    """让容器 stdout/stderr 能看到服务和 agent.* 的运行诊断日志。

    Returns:
        None: ``app`` 与 ``agent`` logger 已配置，重复调用不会重复添加 handler。
    """

    # Uvicorn 默认只给 uvicorn.* 配置 handler；服务和 agent.* 即使设为 INFO，
    # 没有自己的 handler 时也可能被根 logger 丢弃。给两个命名空间都挂上
    # handler，只配置一次，避免测试或重复创建 FastAPI app 时重复打印每条日志。
    for name in ("app", "agent"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.handlers:
            continue
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)


app = create_app()
