from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """返回 Agent API 自己的 liveness 状态。

    Returns:
        dict[str, str]: 固定的 ``{"status": "ok"}`` 响应。依赖服务状态由
            独立 Health Monitor 负责，避免把外部故障混入 liveness。
    """

    return {"status": "ok"}
