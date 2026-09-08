"""模型调用基础设施：LangChain 适配器与 Provider wire 日志客户端。

这是供应商 SDK 的边界：上游只看到 ``AgentModelPort.ainvoke`` 和
``ToolDefinition``，下游才使用 LangChain 的消息类、``ChatOpenAI`` 和 tool
calling 格式。更换 OpenAI 兼容服务或模型时，优先修改这个包。
"""

from agent.infrastructure.model.chat_model import LangChainAgentModel, ModelProviderError
from agent.infrastructure.model.langchain_adapter import LangChainToolAdapter
from agent.infrastructure.model.provider_http import (
    AsyncProviderHttpClient,
    build_provider_async_http_client,
)

__all__ = [
    "AsyncProviderHttpClient",
    "LangChainAgentModel",
    "LangChainToolAdapter",
    "ModelProviderError",
    "build_provider_async_http_client",
]
