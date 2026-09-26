"""进程内传输:把 Agent 运行时直接嵌进 CLI。

一个专用线程持有 asyncio 事件循环与 ``AgentBootstrap`` 启停序列;同步
方法经 ``run_coroutine_threadsafe`` 进入独立请求作用域执行一次服务调用。
返回形状即本项目的传输契约(dict 载荷、JSON-safe 字段、时间戳 isoformat),
UI 层只认这一份形状。

当前覆盖读路径、CRUD、非流式 ``ask`` 与权限回复;``ask_stream`` 的
队列桥接是下一个里程碑,届时在此文件内补齐。
"""

from __future__ import annotations

import asyncio
import inspect
import queue
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import cli.ui as ui
from agent.bootstrap import AgentBootstrap
from agent.domain.messages import Message
from agent.domain.multi_agent import Answered, AskOutcome, Queued
from agent.domain.streaming import (
    STREAM_EVENT_ERROR,
    STREAM_EVENT_MESSAGE_END,
    STREAM_EVENT_QUEUED,
    StreamFrame,
)
from agent.domain.tools import ALL_META_TOOL_NAMES
from agent.exceptions import (
    AgentExecutionError,
    DomainError,
    PermissionAlreadyResolvedError,
    PermissionNotFoundError,
    ResourceNotFoundError,
    SessionDeletionConflictError,
    SessionDriverBusyError,
    SessionNotFoundError,
)
from agent.infrastructure.driver_locks import FileSessionDriverLocks
from agent.infrastructure.settings import AgentRuntimeSettings
from agent.ports.context import ContextManagerPort
from agent.ports.runtime.stream import AgentStreamHubPort
from agent.ports.services import (
    AgentServicePort,
    PermissionServicePort,
    SessionServicePort,
)
from agent.ports.tools import McpRegistryPort, MemoryCatalogPort, SkillCatalogPort
from cli.ports import AgentApiError

_STARTUP_TIMEOUT_SECONDS = 60.0
_SHUTDOWN_TIMEOUT_SECONDS = 30.0

# 状态码沿用原 HTTP 语义表(404/409/502),现为进程内错误契约。
_STATUS_BY_EXCEPTION: tuple[tuple[type[Exception], int], ...] = (
    ((SessionNotFoundError, PermissionNotFoundError, ResourceNotFoundError), 404),
    (
        (PermissionAlreadyResolvedError, SessionDeletionConflictError, SessionDriverBusyError),
        409,
    ),
    ((AgentExecutionError,), 502),
    ((DomainError,), 400),
)


def _translate(exc: Exception) -> AgentApiError:
    """把领域/应用异常映射成入口层统一的 AgentApiError。"""

    for types, status_code in _STATUS_BY_EXCEPTION:
        if isinstance(exc, types):
            message = getattr(exc, "message", None) or str(exc)
            return AgentApiError(str(message), status_code=status_code)
    return AgentApiError(str(exc), status_code=500)


def _message_payload(message: Message) -> dict[str, Any]:
    """Message 的 wire 形状:JSON-safe 键值,时间戳 isoformat。"""

    return {
        "id": message.id,
        "session_id": message.session_id,
        "seq": message.seq,
        "role": message.role,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
            for call in message.tool_calls
        ],
        "metadata": dict(message.metadata),
        "created_at": message.created_at.isoformat(),
    }


def _session_payload(session: Any) -> dict[str, Any]:
    return {
        "id": session.id,
        "title": session.title,
        "user_id": session.user_id,
        "parent_session_id": session.parent_session_id,
        "main_session_id": session.main_session_id,
        "allowed_tools": list(session.allowed_tools),
        "parent_last_seq": session.parent_last_seq,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


_TERMINAL_EVENTS = {STREAM_EVENT_MESSAGE_END, STREAM_EVENT_QUEUED, STREAM_EVENT_ERROR}


def _validate_message_payload(message: Any, session_id: str | None = None) -> None:
    """终止载荷的拒绝规则:显然畸形/损坏的消息不许进渲染层。"""

    if not isinstance(message, dict):
        raise AgentApiError("消息响应结构无效", details=message)
    required = ("id", "role", "content")
    if any(key not in message for key in required):
        raise AgentApiError(
            "消息响应缺少必要字段", details={"required": required, "message": message}
        )
    if not all(isinstance(message[key], str) for key in required):
        raise AgentApiError("消息响应字段类型无效", details=message)
    if session_id is not None and "session_id" in message and message["session_id"] != session_id:
        raise AgentApiError("消息响应 session_id 不匹配", details=message)
    if "seq" in message and (
        isinstance(message["seq"], bool)
        or not isinstance(message["seq"], int)
        or message["seq"] <= 0
    ):
        raise AgentApiError("消息响应 seq 无效", details=message)
    if "metadata" in message and not isinstance(message["metadata"], dict):
        raise AgentApiError("消息响应 metadata 类型无效", details=message)
    if "tool_calls" in message and not isinstance(message["tool_calls"], list):
        raise AgentApiError("消息响应 tool_calls 类型无效", details=message)


def _outcome_frame(outcome: AskOutcome) -> StreamFrame:
    """一轮结果的收尾帧:产出回答 ``message_end``,仅入库 ``queued``。"""

    if isinstance(outcome, Queued):
        return (STREAM_EVENT_QUEUED, {"session_id": outcome.session_id})
    return (STREAM_EVENT_MESSAGE_END, {"message": _message_payload(outcome.message)})


def _error_frame(exc: BaseException) -> StreamFrame:
    return (
        STREAM_EVENT_ERROR,
        {
            "code": getattr(exc, "code", "internal_server_error"),
            "message": getattr(exc, "message", None) or str(exc),
        },
    )


class _TurnBridge:
    """loop 线程 pump 协程 -> UI 消费线程的一次性帧管道。

    帧词汇、终止规则(message_end/queued/error 至多一个且即终结)与
    终止规则:message_end/queued/error 至多其一,读到即收尾。
    """

    def __init__(self) -> None:
        self._frames: queue.SimpleQueue[StreamFrame | None] = queue.SimpleQueue()

    def push(self, frame: StreamFrame | None) -> None:
        self._frames.put(frame)

    def events(self, session_id: str) -> Iterator[dict[str, Any]]:
        terminal_seen = False
        while True:
            frame = self._frames.get()
            if frame is None:
                return
            name, payload = frame
            if not isinstance(payload, dict):
                raise AgentApiError(f"{name} 事件 payload 无效", details=payload)
            if name == STREAM_EVENT_MESSAGE_END:
                message = payload.get("message")
                if not isinstance(message, dict):
                    raise AgentApiError("message_end 事件缺少有效的 message payload")
                _validate_message_payload(message, session_id)
            if name == STREAM_EVENT_QUEUED and payload.get("session_id") != session_id:
                raise AgentApiError("queued 事件 session_id 无效", details=payload)
            if name in _TERMINAL_EVENTS:
                if terminal_seen:
                    raise AgentApiError("流式响应包含多个终止事件")
                terminal_seen = True
            elif terminal_seen:
                raise AgentApiError("终止事件后收到额外事件")
            yield {"event": name, "data": payload}


class _AgentRuntimeThread:
    """专用线程 + 事件循环;启动失败在构造时立即抛出。"""

    def __init__(self) -> None:
        self.bootstrap = AgentBootstrap()
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="agent-runtime", daemon=True)
        self._thread.start()
        if not self._ready.wait(_STARTUP_TIMEOUT_SECONDS):
            raise AgentApiError("Embedded Agent runtime failed to start in time")
        if self._startup_error is not None:
            raise self._startup_error

    def _run(self) -> None:
        ui.silence_background_loggers()
        asyncio.set_event_loop(self.loop)

        async def _boot() -> None:
            try:
                await self.bootstrap.start()
            except BaseException as exc:  # noqa: BLE001 - 透传给构造方
                self._startup_error = exc
            finally:
                self._ready.set()

        self.loop.run_until_complete(_boot())
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def submit(self, coro: Any, timeout: float | None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout)

    def shutdown(self) -> None:
        if not self.loop.is_running():
            return

        async def _bye() -> None:
            await self.bootstrap.stop()

        try:
            asyncio.run_coroutine_threadsafe(_bye(), self.loop).result(_SHUTDOWN_TIMEOUT_SECONDS)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)


class LocalApiClient:
    """``AgentClientPort`` 的进程内实现(嵌入式后端)。"""

    def __init__(self) -> None:
        ui.silence_background_loggers()
        # 运行时线程在第一次服务调用时才拉起:UI 允许"构造但还没用"。
        self._runtime: _AgentRuntimeThread | None = None
        self._runtime_lock = threading.Lock()
        self._closed = False
        self._close_lock = threading.Lock()
        # 驾驶权签出表:session_id -> flock 句柄。UI 打开主会话时 checkout,
        # 切走/退出时 release;子会话经父主会话传递保护,不单独签出。
        self._driver_locks = FileSessionDriverLocks(AgentRuntimeSettings().session_lock_dir)
        self._checked_out: dict[str, Any] = {}
        self._checkout_lock = threading.Lock()
        # 在途流:request_id -> bridge;仅为 close() 唤醒保留,pump 持强引用。
        self._streams: dict[str, _TurnBridge] = {}
        self._streams_lock = threading.Lock()
        self._detached_turns: set[asyncio.Task[AskOutcome]] = set()

    # -- 通用调度 -------------------------------------------------------

    def _ensure_runtime(self) -> _AgentRuntimeThread:
        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = _AgentRuntimeThread()
            return self._runtime

    def _call(
        self,
        port: type,
        invoke: Callable[[Any], Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        runtime = self._ensure_runtime()

        async def _run() -> Any:
            async with runtime.bootstrap.container() as scope:
                service = await scope.get(port)
                res = invoke(service)
                if inspect.isawaitable(res):
                    return await res
                return res

        try:
            return runtime.submit(_run(), timeout)
        except AgentApiError:
            raise
        except Exception as exc:  # noqa: BLE001 - 边界翻译
            raise _translate(exc) from exc

    # -- 驾驶权签出 ------------------------------------------------------

    def checkout(self, session_id: str) -> None:
        with self._checkout_lock:
            if session_id in self._checked_out:
                return
            try:
                handle = self._driver_locks.checkout(session_id)
            except SessionDriverBusyError as exc:
                raise AgentApiError(str(exc), status_code=409) from exc
            self._checked_out[session_id] = handle

    def release(self, session_id: str) -> None:
        with self._checkout_lock:
            handle = self._checked_out.pop(session_id, None)
        if handle is not None:
            handle.release()

    # -- 传输面(与 AgentApiClient 同形状) ------------------------------

    def health(self, *, timeout: float | None = None) -> dict[str, Any]:
        async def _noop() -> str:
            await asyncio.sleep(0)
            return "ok"

        try:
            self._ensure_runtime().submit(_noop(), timeout if timeout is not None else 5.0)
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc) from exc
        return {"status": "ok"}

    def list_sessions(self, *, timeout: float | None = None) -> list[dict[str, Any]]:
        async def _list(service: SessionServicePort) -> list[dict[str, Any]]:
            return [_session_payload(s) for s in await service.list()]

        return self._call(SessionServicePort, _list, timeout=timeout)

    def get_session(self, session_id: str) -> dict[str, Any]:
        async def _get(service: SessionServicePort) -> dict[str, Any]:
            return _session_payload(await service.get(session_id))

        return self._call(SessionServicePort, _get)

    def create_session(
        self,
        title: str | None = None,
        allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        clean_title = title.strip() if title and title.strip() else None
        tools = tuple(allowed_tools) if allowed_tools is not None else tuple(ALL_META_TOOL_NAMES)

        async def _create(service: SessionServicePort) -> dict[str, Any]:
            session = await service.create(title=clean_title or "New Chat", allowed_tools=tools)
            return _session_payload(session)

        return self._call(SessionServicePort, _create)

    def rename_session(self, session_id: str, title: str) -> dict[str, Any]:
        async def _rename(service: SessionServicePort) -> dict[str, Any]:
            return _session_payload(await service.rename(session_id=session_id, title=title))

        return self._call(SessionServicePort, _rename)

    def resume_session(
        self,
        session_id: str,
        title: str | None = None,
        parent_last_seq: int | None = None,
    ) -> dict[str, Any]:
        async def _resume(service: SessionServicePort) -> dict[str, Any]:
            session = await service.resume(
                session_id=session_id,
                title=title,
                parent_last_seq=parent_last_seq,
            )
            return _session_payload(session)

        return self._call(SessionServicePort, _resume)

    def delete_session(self, session_id: str) -> None:
        async def _delete(service: SessionServicePort) -> None:
            await service.delete(session_id)

        self._call(SessionServicePort, _delete)

    def compact_session(self, session_id: str) -> dict[str, Any]:
        async def _compact(service: ContextManagerPort) -> dict[str, Any]:
            result = await service.compact_session(session_id)
            return {
                "session_id": result.session_id,
                "status": result.status,
                "kind": result.kind,
                "checkpoint_message_id": result.checkpoint_message_id,
                "estimated_tokens_before": result.estimated_tokens_before,
                "estimated_tokens_after": result.estimated_tokens_after,
                "message": result.message,
            }

        return self._call(ContextManagerPort, _compact)

    def list_messages(
        self,
        session_id: str,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        async def _history(service: SessionServicePort) -> list[dict[str, Any]]:
            messages = await service.history(session_id)
            return [_message_payload(m) for m in messages[1:]]

        return self._call(SessionServicePort, _history, timeout=timeout)

    def ask(self, session_id: str, question: str) -> dict[str, Any]:
        async def _ask(service: AgentServicePort) -> dict[str, Any]:
            outcome = await service.ask(
                session_id=session_id,
                question=question,
                sender="user",
            )
            if isinstance(outcome, Answered):
                return _message_payload(outcome.message)
            if isinstance(outcome, Queued):
                return {"status": "queued", "session_id": outcome.session_id}
            raise AgentApiError("未知问答结果类型", status_code=500)

        return self._call(AgentServicePort, _ask, timeout=600.0)

    def ask_stream(
        self,
        session_id: str,
        question: str,
        *,
        request_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        rid = request_id or str(uuid.uuid4())
        bridge = _TurnBridge()
        with self._streams_lock:
            self._streams[rid] = bridge
        runtime = self._ensure_runtime()
        try:
            asyncio.run_coroutine_threadsafe(
                self._begin_stream(session_id, question, bridge),
                runtime.loop,
            )
        except RuntimeError as exc:  # loop 已死/未运行
            with self._streams_lock:
                self._streams.pop(rid, None)
            raise AgentApiError(f"嵌入式运行时不可用: {exc}") from exc
        # 与 HTTP 语义一致:消费者弃读不中止本轮,答案照常入库。
        try:
            yield from bridge.events(session_id)
        finally:
            with self._streams_lock:
                self._streams.pop(rid, None)

    async def _begin_stream(
        self,
        session_id: str,
        question: str,
        bridge: _TurnBridge,
    ) -> None:
        """loop 线程内的单轮装配,结构镜像 agent_controller._event_stream。

        注册即"本轮开始",ask 任务收尾回调(全同步)压入终止帧 + ``None``
        哨兵并摘注册;pump 是队列唯一读者,把帧接力给线程侧管道。
        """

        try:
            runtime = self._runtime
            assert runtime is not None  # _begin_stream 只在 _ensure_runtime 后调度
            async with runtime.bootstrap.container() as scope:
                hub: AgentStreamHubPort = await scope.get(AgentStreamHubPort)
                service: AgentServicePort = await scope.get(AgentServicePort)
            queue = hub.register(session_id)
            if queue is None:
                # 已有在途轮次占着队列:与 HTTP 侧同样退化为同步 ask,
                # 用单个终止帧表达 Answered/Queued/失败。
                try:
                    outcome = await service.ask(
                        session_id=session_id, question=question, sender="user"
                    )
                except Exception as exc:  # noqa: BLE001
                    bridge.push(_error_frame(exc))
                    return
                bridge.push(_outcome_frame(outcome))
                return

            task = asyncio.create_task(
                service.ask(session_id=session_id, question=question, sender="user")
            )

            def _finish(finished: asyncio.Task[AskOutcome]) -> None:
                try:
                    queue.put_nowait(_outcome_frame(finished.result()))
                except asyncio.CancelledError:
                    # 取消不是业务失败,不伪造 error 帧;finally 仍发哨兵。
                    pass
                except Exception as exc:  # noqa: BLE001
                    queue.put_nowait(_error_frame(exc))
                finally:
                    queue.put_nowait(None)
                    hub.remove(session_id)

            task.add_done_callback(_finish)
            # asyncio 只弱引用任务;pump 之外必须有人持有它,收尾回调才可能永不触发。
            self._detached_turns.add(task)
            task.add_done_callback(self._detached_turns.discard)

            while True:
                frame = await queue.get()
                if frame is None:
                    return
                bridge.push(frame)
        except asyncio.CancelledError:
            pass
        finally:
            bridge.push(None)

    def abort_stream(self, request_id: str | None = None) -> None:
        # 弃读即终止于消费者一侧;本轮照跑完(与 HTTP 断连语义一致)。
        # 表里保留 bridge 只为 close() 能唤醒尚未见到哨兵的读取方。
        with self._streams_lock:
            if request_id is None:
                self._streams.clear()
            else:
                self._streams.pop(request_id, None)

    def cancel(self, session_id: str) -> None:
        normalized_id = session_id.strip()
        if not normalized_id:
            raise AgentApiError("session_id 不能为空", status_code=400)

        async def _cancel(service: AgentServicePort) -> None:
            if hasattr(service, "cancel"):
                res = service.cancel(session_id=normalized_id, sender="user")
                if inspect.isawaitable(res):
                    await res
                return
            # 兼容后端尚未在 AgentService 补充 cancel 期间直接经 AgentRuntimePort 派发
            from agent.domain.multi_agent import CancelMessage
            from agent.ports.tools import AgentRuntimePort

            runtime = self._ensure_runtime()
            async with runtime.bootstrap.container() as app_scope:
                rt: AgentRuntimePort = await app_scope.get(AgentRuntimePort)
                await rt.cancel(CancelMessage(session_id=normalized_id, sender="user"))

        self._call(AgentServicePort, _cancel)

    def reply_permission(
        self,
        permission_id: str,
        decision: str,
        scope: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        normalized_decision = decision.strip().lower()
        effective_scope: Sequence[str] | None = scope
        if normalized_decision == "once":
            normalized_decision = "allow"
            effective_scope = None
        elif normalized_decision == "allow_project":
            normalized_decision = "allow"
            effective_scope = scope if scope is not None else [""]
        elif normalized_decision not in {"allow", "reject"}:
            raise AgentApiError(
                "权限决定必须是 allow、reject (或兼容 once、allow_project)",
                status_code=400,
            )
        if isinstance(effective_scope, str):
            effective_scope = [effective_scope]
        if normalized_decision == "reject":
            effective_scope = None

        async def _reply(service: PermissionServicePort) -> dict[str, Any]:
            await service.reply(
                permission_id=permission_id,
                decision=normalized_decision,
                scope=effective_scope,
            )
            return {
                "permission_id": permission_id,
                "decision": normalized_decision,
                "status": "accepted",
            }

        return self._call(PermissionServicePort, _reply)

    def reply_ask_user(
        self,
        interruption_id: str,
        answer: str,
    ) -> dict[str, Any]:
        from agent.domain.interruption import InterruptionReplyStatus
        from agent.ports.runtime.interruption import InterruptionBrokerPort

        normalized_id = interruption_id.strip()
        if not normalized_id:
            raise AgentApiError("interruption_id 不能为空", status_code=400)

        def _reply(broker: InterruptionBrokerPort) -> dict[str, Any]:
            status = broker.reply(
                interruption_id=normalized_id,
                payload={"answer": answer},
            )
            if status == InterruptionReplyStatus.NOT_FOUND:
                raise AgentApiError(
                    f"Interruption request not found: {normalized_id}",
                    status_code=404,
                )
            if status == InterruptionReplyStatus.ALREADY_RESOLVED:
                raise AgentApiError(
                    f"Interruption request already resolved: {normalized_id}",
                    status_code=409,
                )
            return {
                "interruption_id": normalized_id,
                "answer": answer,
                "status": str(status),
            }

        return self._call(InterruptionBrokerPort, _reply)

    def get_permission_mode(self) -> str:
        def _get(service: PermissionServicePort) -> str:
            res = service.current_mode()
            return str(getattr(res, "value", res))

        return self._call(PermissionServicePort, _get)

    def set_permission_mode(self, mode: str) -> str:
        def _set(service: PermissionServicePort) -> str:
            res = service.change_mode(mode)
            return str(getattr(res, "value", res))

        return self._call(PermissionServicePort, _set)

    def list_mcps(self) -> list[dict[str, Any]]:
        async def _list(registry: McpRegistryPort) -> list[dict[str, Any]]:
            return [
                {
                    "id": status.id,
                    "description": status.description,
                    "state": str(status.state),
                    "error": status.error,
                    "tool_count": status.tool_count,
                }
                for status in registry.list_statuses()
            ]

        return self._call(McpRegistryPort, _list)

    def retry_mcp(self, server_id: str) -> dict[str, Any]:
        async def _retry(registry: McpRegistryPort) -> dict[str, Any]:
            try:
                status = await registry.retry_async(server_id)
            except ValueError as exc:
                raise AgentApiError(str(exc), status_code=404) from exc
            return {
                "id": status.id,
                "description": status.description,
                "state": str(status.state),
                "error": status.error,
                "tool_count": status.tool_count,
            }

        return self._call(McpRegistryPort, _retry)

    def list_skills(self) -> list[dict[str, Any]]:
        async def _list(catalog: SkillCatalogPort) -> list[dict[str, Any]]:
            return [
                {
                    "name": status.name,
                    "description": status.description,
                    "path": status.path,
                    "state": str(status.state),
                    "error": status.error,
                }
                for status in catalog.list_statuses()
            ]

        return self._call(SkillCatalogPort, _list)

    def list_memories(self) -> list[dict[str, Any]]:
        async def _list(catalog: MemoryCatalogPort) -> list[dict[str, Any]]:
            return [
                {
                    "layer": str(
                        entry.layer.value if hasattr(entry.layer, "value") else entry.layer
                    ),
                    "title": entry.title,
                    "description": entry.description,
                }
                for entry in catalog.list_entries()
            ]

        return self._call(MemoryCatalogPort, _list)

    def get_memory(self, layer: str, title: str) -> dict[str, Any] | None:
        async def _get(catalog: MemoryCatalogPort) -> dict[str, Any] | None:
            if hasattr(catalog, "read_note"):
                content = catalog.read_note(layer, title)
                if content is not None:
                    return {"layer": layer, "title": title, "content": content}
            return None

        return self._call(MemoryCatalogPort, _get)

    def list_models(self) -> list[dict[str, Any]]:
        from agent.infrastructure.model import ModelCatalog

        async def _list(catalog: ModelCatalog) -> list[dict[str, Any]]:
            return [
                {
                    "id": getattr(info, "model_id", None) or getattr(info, "profile", ""),
                    "name": getattr(info, "name", ""),
                }
                for info in catalog.list_models()
            ]

        return self._call(ModelCatalog, _list)

    def get_current_model(self) -> dict[str, Any]:
        from agent.infrastructure.model import ModelCatalog

        async def _get(catalog: ModelCatalog) -> dict[str, Any]:
            cur_id = getattr(catalog, "current_id", None) or getattr(catalog, "current_name", "")
            profile = (
                catalog.current_profile() if hasattr(catalog, "current_profile") else None
            )
            name = getattr(profile, "name", cur_id) if profile else cur_id
            return {"id": cur_id, "name": name}

        return self._call(ModelCatalog, _get)

    def select_model(self, model_id: str) -> dict[str, Any]:
        from agent.infrastructure.model import ModelCatalog

        async def _select(catalog: ModelCatalog) -> dict[str, Any]:
            chosen = catalog.select_model(model_id)
            profile = (
                catalog.current_profile() if hasattr(catalog, "current_profile") else None
            )
            name = getattr(profile, "name", chosen) if profile else chosen
            return {"id": chosen, "name": name}

        return self._call(ModelCatalog, _select)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        runtime = self._runtime
        if runtime is not None:
            runtime.shutdown()
        with self._checkout_lock:
            handles = list(self._checked_out.values())
            self._checked_out.clear()
        for handle in handles:
            handle.release()
        with self._streams_lock:
            bridges = list(self._streams.values())
            self._streams.clear()
        for bridge in bridges:  # 循环线程已停,唤醒任何还在等哨兵的读者
            bridge.push(None)


__all__ = ["LocalApiClient"]
