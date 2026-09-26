"""LangChain 与 Agent 核心消息之间的适配器。

这里是供应商 SDK 的边界：上游只看到 ``AgentModelPort.ainvoke``、
``ToolDefinition`` 和 ``ModelMessage``，下游才使用 LangChain 的消息类、
``ChatOpenAI`` 和 tool calling 格式。更换 OpenAI 兼容服务或模型时，优先修改
这个文件。
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.domain.model_messages import ModelMessage, ModelTurn, ModelUsage, ToolCall
from agent.domain.model_profile import ModelProfile
from agent.domain.tools import ToolDefinition
from agent.infrastructure.model.catalog import ModelCatalog
from agent.infrastructure.model.langchain_adapter import LangChainToolAdapter
from agent.infrastructure.model.provider_http import build_provider_async_http_client
from agent.infrastructure.settings import (
    ObservabilitySettings,
)
from agent.ports.model import AgentModelPort, ModelCallPurpose

logger = logging.getLogger(__name__)

# LangGraph messages 模式的处理器（StreamMessagesHandler）在 on_chat_model_start
# 时跳过携带此 tag 的 run——该 run 的所有 token 增量从源头不进入 messages 流。
# 内部生成调用（摘要等）借此与用户可见流隔离，入口无需再按文本前缀猜内容。
_SUPPRESS_STREAM_TAG = "nostream"


class ModelProviderError(RuntimeError):
    """模型 provider 调用失败；异常消息已脱敏，可安全展示给调用方。"""


class LangChainAgentModel(AgentModelPort):
    """执行一次 LangChain 模型 turn 的适配器。

    它只负责消息转换和模型调用，是否循环、何时执行工具由
    ``agent.application.graph.AgentGraph`` 负责。模型配置（id、名称、base_url、
    api_key、输出上限、窗口与协议开关）平铺在 ``ModelProfile`` 里，从
    ``ModelCatalog`` 的当前项**每轮现取**：切换只改 catalog 的指针，这里每轮
    重建一个轻量的 ``ChatOpenAI`` 壳即可反映，无需缓存；provider 连接池
    (``httpx.AsyncClient``) 与模型无关，建一次缓存到 ``aclose``。
    """

    def __init__(
        self,
        *,
        catalog: ModelCatalog,
        tool_adapter: LangChainToolAdapter,
        observability: ObservabilitySettings | None = None,
    ) -> None:
        """保存模型目录、工具适配器与 wire 日志配置。

        Args:
            catalog (ModelCatalog): 多模型快照与当前选中来源（settings.json），
                窗口、输出上限与协议开关都在其 ``ModelProfile`` 里。
            tool_adapter (LangChainToolAdapter): 将工具 schema 包装成 LangChain 工具。
            observability (ObservabilitySettings | None): provider wire 日志配置。
        """

        self._catalog = catalog
        self._tool_adapter = tool_adapter
        self._observability = observability or ObservabilitySettings()
        self._async_http_client = None

    async def ainvoke(
        self,
        *,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition] = (),
        tool_choice: str | None = None,
        max_output_tokens: int | None = None,
        purpose: ModelCallPurpose = ModelCallPurpose.CHAT,
    ) -> ModelTurn:
        """异步执行一次模型调用；当前模型每轮现取，工具只绑定到本 turn
        """

        started_at = time.perf_counter()
        profile = self._catalog.current_profile()
        model = self._build_model(profile)
        model_tools = [self._tool_adapter.to_model_tool(tool) for tool in tools]
        api_mode = _api_mode(profile.use_responses_api)
        if model_tools:
            bind_kwargs: dict[str, object] = {}
            if tool_choice is not None:
                bind_kwargs["tool_choice"] = tool_choice
            model = model.bind_tools(model_tools, **bind_kwargs)
        if max_output_tokens is not None:
            model = model.bind(max_tokens=max_output_tokens)
        langchain_messages = [_to_langchain_message(message) for message in messages]
        input_chars = sum(len(message.content) for message in messages)
        run_config = None if purpose is ModelCallPurpose.CHAT else {"tags": [_SUPPRESS_STREAM_TAG]}
        try:
            response = await model.ainvoke(langchain_messages, config=run_config)
        except Exception as exc:
            logger.exception(
                "Async LLM invoke failed: api=%s model=%s purpose=%s "
                "messages=%d input_chars=%d tools=%d "
                "tool_choice=%s elapsed_ms=%.1f",
                api_mode,
                profile.model_id,
                purpose,
                len(messages),
                input_chars,
                len(model_tools),
                tool_choice or "auto",
                _elapsed_ms(started_at),
            )
            raise ModelProviderError(_safe_provider_error(exc)) from exc
        tool_calls = tuple(_tool_call_from_langchain(call) for call in response.tool_calls)
        content = _content_to_text(response.content)
        usage = _model_usage(response)
        logger.info(
            "Async LLM invoke completed: api=%s model=%s purpose=%s "
            "messages=%d input_chars=%d tools=%d "
            "tool_choice=%s output_chars=%d tool_calls=%d elapsed_ms=%.1f usage=%s",
            api_mode,
            profile.model_id,
            purpose,
            len(messages),
            input_chars,
            len(model_tools),
            tool_choice or "auto",
            len(content),
            len(tool_calls),
            _elapsed_ms(started_at),
            _usage_summary(response),
        )
        return ModelTurn(
            message=ModelMessage.assistant(content=content, tool_calls=tool_calls),
            tool_calls=tool_calls,
            usage=usage,
        )

    def _build_model(self, profile: ModelProfile):
        """按给定模型档案现建一个不带工具的基础模型；每轮调用一次，不缓存。

        重建 ``ChatOpenAI`` 只是拼配置对象，开销可忽略；provider 连接池
        ``httpx.AsyncClient`` 与模型无关，只在首次建、之后复用。全部协议
        参数取平铺的 ``ModelProfile``：输出上限显式声明则直接用，
        否则按其窗口与安全余量比例推导。

        Returns:
            ChatOpenAI: 已配置 provider、输出上限和 SDK 重试次数的基础模型。
        """

        model_kwargs: dict[str, object] = {
            "model": profile.model_id,
            "api_key": profile.api_key,
            "temperature": 0,
            # 保持 AgentService/Graph 每个 turn 只执行一次的语义，但允许底层
            # OpenAI SDK 重试短暂 provider 故障；显式配置避免 SDK 默认值变化
            # 改变请求预算：默认每个 turn 最多一次初始请求加两次 SDK 重试，
            # 链路不稳的 provider 可在 profile 里单独调大 max_retries。
            "max_retries": profile.max_retries,
            # 所有调用一律走流式：ainvoke 内部转 astream 并聚合结果（内容、
            # tool_calls、usage 均由 AIMessageChunk 累加而来）。摘要与无出口
            # 的 subagent 调用因此同样享受流式的字节级超时语义，无需任何
            # 调用方改动。显式开启 stream_usage 才能让聚合结果保留 token 计数。
            "streaming": True,
            "stream_usage": True,
            "use_responses_api": profile.use_responses_api,
        }
        if profile.request_timeout_seconds is not None:
            # HTTP 层超时：流式下 read 是"两个字节之间的静默"上限，抓挂起
            # 不抓慢速合法输出；连接建立走固定 10s，不占配置面。
            model_kwargs["timeout"] = httpx.Timeout(
                profile.request_timeout_seconds,
                connect=10.0,
            )
        # LangChain maps max_tokens to max_completion_tokens for Chat
        # Completions and max_output_tokens for the Responses API.
        model_kwargs["max_tokens"] = profile.max_output_tokens
        if profile.use_responses_api:
            model_kwargs["output_version"] = "responses/v1"
        model_kwargs["base_url"] = profile.base_url
        if self._async_http_client is None:
            self._async_http_client = build_provider_async_http_client(
                wire_log_path=self._observability.provider_wire_log_path,
            )
        model_kwargs["http_async_client"] = self._async_http_client
        return ChatOpenAI(**model_kwargs)

    async def aclose(self) -> None:
        """释放异步 Provider HTTP 连接池。"""

        if self._async_http_client is not None:
            await self._async_http_client.aclose()
            self._async_http_client = None


def _elapsed_ms(started_at: float) -> float:
    """把单调时钟起点转换成适合日志阅读的毫秒数。

    Args:
        started_at (float): ``time.perf_counter`` 记录的起始读数。

    Returns:
        float: 到当前时刻经过的毫秒数。
    """

    return (time.perf_counter() - started_at) * 1000


def _usage_summary(response: object) -> str:
    """只提取 token 计数，避免把模型响应或提示词写入日志。

    Args:
        response (object): LangChain 或兼容 provider 返回的消息对象。

    Returns:
        str: ``input/output/total`` 计数文本；没有 usage 时返回 ``unknown``。
    """

    usage = _model_usage(response)
    if usage is None:
        return "unknown"
    return (
        f"input={usage.input_tokens},output={usage.output_tokens},"
        f"total={usage.total_tokens}"
    )


def _model_usage(response: object) -> ModelUsage | None:
    """把 LangChain 或兼容 provider 的 usage 归一化为领域值对象。

    Args:
        response (object): LangChain 返回的 ``AIMessage`` 或兼容对象。

    Returns:
        ModelUsage | None: 供应商返回的完整 token 计数；没有可靠计数时为 ``None``。

    Note:
        LangChain 对 Chat Completions 和 Responses API 都会优先填充
    ``AIMessage.usage_metadata``。少数 OpenAI-compatible provider 只把原始
    Chat Completions usage 留在 ``response_metadata.token_usage``，这里保留
    一个兼容读取路径；完全不返回 usage 时则显式返回 ``None``。
    """

    usage = getattr(response, "usage_metadata", None)
    normalized = _usage_from_mapping(
        usage,
        input_key="input_tokens",
        output_key="output_tokens",
    )
    if normalized is not None:
        return normalized

    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    return _usage_from_mapping(
        metadata.get("token_usage"),
        input_key="prompt_tokens",
        output_key="completion_tokens",
    )


def _usage_from_mapping(
    usage: object,
    *,
    input_key: str,
    output_key: str,
) -> ModelUsage | None:
    """读取一组完整的非负 token 计数；不伪造 provider 未返回的数字。

    Args:
        usage (object): provider 返回的 usage mapping。
        input_key (str): 输入 token 对应的字段名。
        output_key (str): 输出 token 对应的字段名。

    Returns:
        ModelUsage | None: 完整且合法的 usage；字段缺失或类型非法时返回 ``None``。
    """

    if not isinstance(usage, Mapping):
        return None
    input_tokens = _token_count(usage.get(input_key))
    output_tokens = _token_count(usage.get(output_key))
    total_tokens = _token_count(usage.get("total_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _token_count(value: object) -> int | None:
    """只接受 provider 正常返回的非负整数 token 数。

    Args:
        value (object): usage mapping 中的原始字段。

    Returns:
        int | None: 非负整数，或表示 provider 未提供可靠计数的 ``None``。
    """

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _safe_provider_error(exception: BaseException) -> str:
    """生成不包含 traceback 或密钥的简短 provider 诊断。

    Args:
        exception (BaseException): provider 或 SDK 抛出的原始异常。

    Returns:
        str: 脱敏并截断后的错误文本，可安全放入对外错误详情。
    """

    message = str(exception).strip() or type(exception).__name__
    message = re.sub(
        r"(?i)(api[_-]?key|authorization|access[_-]?token)"
        r"(\s*[=:]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,]+)",
        r"\1\2redacted",
        message,
    )
    return f"Model provider request failed: {message[:1000]}"


def _api_mode(use_responses_api: bool) -> str:
    """返回当前模型适配器实际选择的 OpenAI API 模式。

    Args:
        use_responses_api (bool): 是否启用 Responses API。

    Returns:
        str: ``responses`` 或 ``chat_completions``。
    """

    return "responses" if use_responses_api else "chat_completions"


def _to_langchain_message(message: ModelMessage):
    """把项目内部消息映射为 LangChain 消息。

    Args:
        message (ModelMessage): Agent 内部的 system/human/assistant/tool 消息。

    Returns:
        BaseMessage: LangChain 对应消息类型。

    Note:
        这一步集中处理 SDK 的字段差异，例如 tool call 使用 ``args``，工具结果
        还需要携带对应的 ``tool_call_id``。
    """

    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "human":
        kwargs = {"name": message.name} if message.name else {}
        return HumanMessage(content=message.content, **kwargs)
    if message.role == "assistant":
        return AIMessage(
            content=message.content,
            tool_calls=[
                {
                    "id": call.id,
                    "name": call.name,
                    "args": call.arguments,
                }
                for call in message.tool_calls
            ],
        )
    if message.role == "tool":
        return ToolMessage(
            content=message.content,
            tool_call_id=message.tool_call_id or "unknown",
            name=message.tool_name,
        )
    raise ValueError(f"Unsupported model message role: {message.role}")


def _tool_call_from_langchain(call: object) -> ToolCall:
    """把 LangChain 返回的 tool call 归一化为内部 ToolCall。

    Args:
        call (object): LangChain 返回的 mapping-like 工具调用。

    Returns:
        ToolCall: 保留 ID、工具名和参数的领域对象。
    """

    call_id = str(getattr(call, "get", lambda *_: None)("id") or uuid4())
    name = str(getattr(call, "get", lambda *_: "")("name"))
    arguments = getattr(call, "get", lambda *_: {})("args") or {}
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _content_to_text(content: object) -> str:
    """兼容模型返回纯文本或多段内容的情况。

    Args:
        content (object): LangChain message 的内容字段。

    Returns:
        str: 原始文本，或稳定 JSON 编码后的多段内容。
    """

    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)
