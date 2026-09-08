from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse, StreamingResponse

from agent.api.message_mapper import message_response
from agent.domain.multi_agent import AskOutcome, Queued
from agent.domain.streaming import (
    STREAM_EVENT_ERROR,
    STREAM_EVENT_MESSAGE_END,
    STREAM_EVENT_QUEUED,
    StreamFrame,
)
from agent.dto.agent import AskRequest, MessageResponse, TurnQueuedResponse
from agent.ports.runtime.stream import AgentStreamHubPort, AgentStreamQueue
from agent.ports.services import AgentServicePort

router = APIRouter(tags=["messages"])
logger = logging.getLogger(__name__)

# 已经没人听、但仍在跑的轮次。asyncio 只弱引用任务，客户端断连后若不留这
# 一份强引用，驱动任务可能在下一次调度前就被 GC 掉。
_detached_turns: set[asyncio.Task[None]] = set()


@router.post(
    "/sessions/{session_id}/messages",
    response_model=MessageResponse,
    responses={status.HTTP_202_ACCEPTED: {"model": TurnQueuedResponse}},
)
@inject
async def create_message(
    session_id: str,
    request: AskRequest,
    service: FromDishka[AgentServicePort],
) -> MessageResponse | JSONResponse:
    """追加用户消息并运行 Agent。

    本轮产出回答时返回 200 + 最终 AssistantMessage；会话已有工作流在跑时消息
    照常入库并返回 202 + ``TurnQueuedResponse`` —— 本轮不再产出回复，也不要重发。
    """

    return _outcome_response(
        await service.ask(
            session_id=session_id,
            question=request.question,
            sender="user",
        )
    )


@router.post(
    "/sessions/{session_id}/messages/stream",
    response_model=None,
    responses={status.HTTP_202_ACCEPTED: {"model": TurnQueuedResponse}},
)
@inject
async def stream_message(
    session_id: str,
    request: AskRequest,
    service: FromDishka[AgentServicePort],
    hub: FromDishka[AgentStreamHubPort],
) -> StreamingResponse | JSONResponse:
    """流式追加用户消息：以 SSE 帧返回文本增量、工具事件和最终消息。

    ask 的 RPC 语义保持不变——``message_end`` 携带的正是 RPC 返回的最终
    持久化消息；流式只是消费 actor 发布到进程内 StreamHub 的旁路事件。

    队列注册标志着"这一轮还在跑"，与谁在听无关：拿不到队列说明本会话已有
    轮次在途，消息照常入库并直接返回本轮结果（通常是 202），不开第二条 SSE。
    客户端中途断连也不会中止本轮，图会跑到自然结束并把答案入库。
    """

    queue = hub.register(session_id)
    if queue is None:
        return _outcome_response(
            await service.ask(
                session_id=session_id,
                question=request.question,
                sender="user",
            )
        )
    return StreamingResponse(
        _event_stream(
            service,
            hub,
            queue,
            session_id=session_id,
            question=request.question,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _outcome_response(outcome: AskOutcome) -> MessageResponse | JSONResponse:
    """把一轮结果映射成 HTTP 响应：产出回答 200，仅入库排队 202。"""

    if isinstance(outcome, Queued):
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=TurnQueuedResponse(session_id=outcome.session_id).model_dump(
                mode="json"
            ),
        )
    return message_response(outcome.message)


async def _event_stream(
    service: AgentServicePort,
    hub: AgentStreamHubPort,
    queue: AgentStreamQueue,
    *,
    session_id: str,
    question: str,
) -> AsyncIterator[bytes]:
    """消费事件帧队列的单一循环；本生成器可以随时不再被读，本轮照跑完。

    "本轮"就是 ``service.ask`` 这一个任务，队列注册跟着它生死：ask 一结束
    （回答、排队或失败）就把终帧、``None`` 哨兵入队并摘掉注册。本生成器只是
    队列的一个读者，读者走了图继续跑到自然结束、答案照常入库，代价是几个再
    无人读取的帧。
    """

    def _finish_turn(task: asyncio.Task[AskOutcome]) -> None:
        """ask 的收尾回调：必须全同步，回调里没有任何可以 await 的东西。"""

        try:
            queue.put_nowait(_outcome_frame(task.result()))
        except Exception as exc:
            logger.exception("Streamed ask failed: session_id=%s", session_id)
            queue.put_nowait(
                (
                    STREAM_EVENT_ERROR,
                    {
                        "code": getattr(exc, "code", "internal_server_error"),
                        "message": getattr(exc, "message", str(exc)),
                    },
                )
            )
        finally:
            queue.put_nowait(None)
            hub.remove(session_id)

    ask = asyncio.create_task(
        service.ask(session_id=session_id, question=question, sender="user")
    )
    ask.add_done_callback(_finish_turn)
    # asyncio 只弱引用任务，而本生成器一被关掉，这个任务在源码里就再也无人
    # 引用它了——唯一持有者是下面这个集合。少了它，摘注册可能永远不会发生。
    _detached_turns.add(ask)
    ask.add_done_callback(_detached_turns.discard)

    while True:
        frame: StreamFrame | None = await queue.get()
        if frame is None:
            return
        yield _sse_frame(frame[0], frame[1])


def _outcome_frame(outcome: AskOutcome) -> StreamFrame:
    """把一轮结果映射成流式收尾帧：产出回答 ``message_end``，仅入库 ``queued``。"""

    if isinstance(outcome, Queued):
        # 消息已入库但本轮不回复：这是被接受的结果，用独立事件与 message_end
        # 区分，下游因此不会渲染出一条不存在的回答。
        return (STREAM_EVENT_QUEUED, {"session_id": outcome.session_id})
    return (
        STREAM_EVENT_MESSAGE_END,
        {"message": message_response(outcome.message).model_dump(mode="json")},
    )


def _sse_frame(event: str, payload: dict[str, object]) -> bytes:
    """编码一帧 SSE 文本；空行结尾表示帧结束。"""

    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode("utf-8")


__all__ = ["create_message", "router", "stream_message"]
