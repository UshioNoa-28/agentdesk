from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agent.domain.tools import ALL_META_TOOL_NAMES


class CreateSessionRequest(BaseModel):
    """创建主会话请求。

    子代理会话只能由 Agent 内部的 ``define_subagent`` 工具经 SessionService
    创建；API 层不暴露 ``main_session_id``，从本接口创建的永远是主会话。
    ``allowed_tools`` 省略时默认授予全部 Meta Tool；显式传 ``[]`` 才表示
    创建无工具会话。
    """

    title: str = Field(min_length=1, max_length=255)
    user_id: str = "default_user"
    parent_session_id: str | None = None
    allowed_tools: list[str] = Field(default_factory=lambda: list(ALL_META_TOOL_NAMES))


class RenameSessionRequest(BaseModel):
    """修改会话标题请求。"""

    title: str = Field(min_length=1, max_length=255)


class ResumeSessionRequest(BaseModel):
    """从已有会话分叉时可选的新标题与分叉点。

    ``parent_last_seq`` 指定父会话的分叉 seq：子会话历史将继承父会话
    ``1..parent_last_seq`` 的消息，子会话自己的第一条消息从
    ``parent_last_seq + 1`` 开始编号。省略时默认分叉到父会话最新 seq。
    """

    title: str | None = Field(default=None, max_length=255)
    parent_last_seq: int | None = Field(default=None, ge=1)


class SessionResponse(BaseModel):
    """会话响应。"""

    id: str
    title: str
    user_id: str = "default_user"
    parent_session_id: str | None
    main_session_id: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    parent_last_seq: int | None = None
    created_at: datetime
    updated_at: datetime


class CompactSessionResponse(BaseModel):
    """会话上下文压缩结果响应。"""

    session_id: str
    status: str
    kind: str | None = None
    checkpoint_message_id: str | None = None
    estimated_tokens_before: int
    estimated_tokens_after: int
    message: str
