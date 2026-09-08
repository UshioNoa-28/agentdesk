from __future__ import annotations

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, HTTPException

from agent.dto.mcp import McpStatusResponse
from agent.ports.tools import McpRegistryPort

router = APIRouter(tags=["mcps"])


@router.get("/mcps", response_model=list[McpStatusResponse])
@inject
async def list_mcps(registry: FromDishka[McpRegistryPort]) -> list[McpStatusResponse]:
    """列出配置的 MCP Server 连接状态。

    Args:
        registry (McpRegistryPort): 由 Dishka 注入的 MCP 生命周期注册表。

    Returns:
        list[McpStatusResponse]: 每个配置项的连接状态、错误和已发现工具数量。
    """

    return [_response(status) for status in registry.list_statuses()]


@router.post("/mcps/{server_id}/retry", response_model=McpStatusResponse)
@inject
async def retry_mcp(
    server_id: str,
    registry: FromDishka[McpRegistryPort],
) -> McpStatusResponse:
    """重试一个连接失败的 MCP Server。

    Args:
        server_id (str): 配置中的 MCP Server 标识。
        registry (McpRegistryPort): 由 Dishka 注入的 MCP 生命周期注册表。

    Returns:
        McpStatusResponse: 重试后的最新连接状态。

    Raises:
        HTTPException: Server ID 不存在时返回 HTTP 404。
    """

    try:
        status = await registry.retry_async(server_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _response(status)


def _response(status: object) -> McpStatusResponse:
    """把注册表内部状态对象转换为 API DTO。

    Args:
        status (object): 具有 id、description、state、error 和 tool_count 属性的状态对象。

    Returns:
        McpStatusResponse: 不暴露连接对象的管理 API 响应。
    """

    return McpStatusResponse(
        id=status.id,
        description=status.description,
        state=str(status.state),
        error=status.error,
        tool_count=status.tool_count,
    )


__all__ = ["router"]
