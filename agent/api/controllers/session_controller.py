from __future__ import annotations

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, Response, status

from agent.api.message_mapper import message_response
from agent.domain.entities import Session
from agent.dto.agent import MessageResponse
from agent.dto.session import (
    CompactSessionResponse,
    CreateSessionRequest,
    RenameSessionRequest,
    ResumeSessionRequest,
    SessionResponse,
)
from agent.ports.context import ContextManagerPort
from agent.ports.services import SessionServicePort

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
@inject
async def create_session(
    request: CreateSessionRequest,
    service: FromDishka[SessionServicePort],
) -> SessionResponse:
    """创建一段长期会话。

    Args:
        request (CreateSessionRequest): 会话标题及可选父会话 ID。
        service (SessionServicePort): 由 Dishka 注入的会话应用服务。

    Returns:
        SessionResponse: 新会话的 ID、标题和时间字段。
    """

    session = await service.create(
        title=request.title,
        user_id=request.user_id,
        parent_session_id=request.parent_session_id,
        allowed_tools=tuple(request.allowed_tools),
    )
    return _session_response(session)


@router.get("/sessions", response_model=list[SessionResponse])
@inject
async def list_sessions(
    service: FromDishka[SessionServicePort],
) -> list[SessionResponse]:
    """读取会话列表。

    Args:
        service (SessionServicePort): 由 Dishka 注入的会话应用服务。

    Returns:
        list[SessionResponse]: 按最近更新时间排列的会话摘要。
    """

    sessions = await service.list()
    return [_session_response(session) for session in sessions]


@router.get("/sessions/{session_id}", response_model=SessionResponse)
@inject
async def get_session(
    session_id: str,
    service: FromDishka[SessionServicePort],
) -> SessionResponse:
    """读取一个会话。

    Args:
        session_id (str): 要查询的会话 ID。
        service (SessionServicePort): 由 Dishka 注入的会话应用服务。

    Returns:
        SessionResponse: 会话元数据。
    """

    session = await service.get(session_id)
    return _session_response(session)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
@inject
async def list_session_messages(
    session_id: str,
    service: FromDishka[SessionServicePort],
) -> list[MessageResponse]:
    """读取当前 Session 可继承的消息历史（跳过首条系统提示词）。"""

    messages = await service.history(session_id)
    return [message_response(message) for message in messages[1:]]


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
@inject
async def rename_session(
    session_id: str,
    request: RenameSessionRequest,
    service: FromDishka[SessionServicePort],
) -> SessionResponse:
    """修改会话标题。

    Args:
        session_id (str): 要修改的会话 ID。
        request (RenameSessionRequest): 新标题请求体。
        service (SessionServicePort): 由 Dishka 注入的会话应用服务。

    Returns:
        SessionResponse: 修改后的会话元数据。
    """

    session = await service.rename(session_id=session_id, title=request.title)
    return _session_response(session)


@router.post(
    "/sessions/{session_id}/resume",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
@inject
async def resume_session(
    session_id: str,
    service: FromDishka[SessionServicePort],
    request: ResumeSessionRequest | None = None,
) -> SessionResponse:
    """从指定会话分叉出一个新的可继续执行的会话。

    Args:
        session_id (str): 作为历史来源的父会话 ID。
        service (SessionServicePort): 由 Dishka 注入的会话应用服务。
        request (ResumeSessionRequest | None): 可选的新标题与父会话分叉 seq；
            省略时由服务生成标题并分叉到父会话最新 seq。

    Returns:
        SessionResponse: 新建子会话的元数据和父会话引用。
    """

    session = await service.resume(
        session_id=session_id,
        title=None if request is None else request.title,
        parent_last_seq=None if request is None else request.parent_last_seq,
    )
    return _session_response(session)


@router.post(
    "/sessions/{session_id}/compact",
    response_model=CompactSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
@inject
async def compact_session(
    session_id: str,
    context_manager: FromDishka[ContextManagerPort],
) -> CompactSessionResponse:
    """手动触发指定会话的历史上下文压缩。

    采用两阶段渐进式压缩策略：
    1. 优先尝试 Cold Maintenance（零开销清理失效历史工具输出）；
    2. 若无冷数据可清理，进入 Summary Compact（生成对话摘要）。

    Args:
        session_id (str): 要执行压缩的会话 ID。
        context_manager (ContextManagerPort): 由 Dishka 注入的上下文管理器。

    Returns:
        CompactSessionResponse: 包含压缩前后 Token 估算、生效类型及检查点消息 ID 的响应。
    """

    result = await context_manager.compact_session(session_id)
    return CompactSessionResponse(
        session_id=result.session_id,
        status=result.status,
        kind=result.kind,
        checkpoint_message_id=result.checkpoint_message_id,
        estimated_tokens_before=result.estimated_tokens_before,
        estimated_tokens_after=result.estimated_tokens_after,
        message=result.message,
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=None,
    status_code=status.HTTP_204_NO_CONTENT,
)
@inject
async def delete_session(
    session_id: str,
    service: FromDishka[SessionServicePort],
) -> Response:
    """删除没有子会话的 Session 聚合。

    Args:
        session_id (str): 要删除的会话 ID。
        service (SessionServicePort): 由 Dishka 注入的会话应用服务。

    Returns:
        Response: 删除成功时的 HTTP 204 空响应。
    """

    await service.delete(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _session_response(session: Session) -> SessionResponse:
    """把领域 Session 映射成 HTTP 响应 DTO。

    Args:
        session (Session): 应用服务返回的领域会话。

    Returns:
        SessionResponse: 可序列化的会话元数据。
    """

    return SessionResponse(
        id=session.id,
        title=session.title,
        user_id=session.user_id,
        parent_session_id=session.parent_session_id,
        main_session_id=session.main_session_id,
        allowed_tools=list(session.allowed_tools),
        parent_last_seq=session.parent_last_seq,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )
