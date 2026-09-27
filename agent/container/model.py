"""进程内模型适配器 Provider。"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from agent.infrastructure.model import LangChainAgentModel, LangChainToolAdapter, ModelCatalog
from agent.infrastructure.settings import (
    AgentRuntimeSettings,
    ObservabilitySettings,
)
from agent.ports.model import AgentModelPort


class AgentModelProvider(Provider):
    """提供进程内聊天模型端口与多模型目录。"""

    @provide(scope=Scope.APP)
    def provide_tool_adapter(self) -> LangChainToolAdapter:
        """提供将工具 schema 绑定到 LangChain 的适配器。

        Returns:
            LangChainToolAdapter: 只负责 schema 包装、不执行 Agent 工具的适配器。
        """

        return LangChainToolAdapter()

    @provide(scope=Scope.APP)
    def provide_model_catalog(self, settings: AgentRuntimeSettings) -> ModelCatalog:
        """从 workspace 发现链加载 settings.json 的 model 段，得到启动快照。

        没有任何 model 段、模型缺 name/base_url/api_key、context_window 或
        max_output_tokens 非法（非正整数）、可配字段类型不对或出现未知字段时
        构造即抛错，拒绝半启动；预算派生不设交叉校验，任何窗口都能启动。
        """

        return ModelCatalog.from_root(settings.workspace_start)

    @provide(scope=Scope.APP)
    def provide_agent_model(
        self,
        catalog: ModelCatalog,
        tool_adapter: LangChainToolAdapter,
        observability: ObservabilitySettings,
    ) -> AgentModelPort:
        """装配单一聊天模型适配器：每轮向 catalog 取当前模型（窗口、输出上限
        与协议开关都在其 ``profile`` 里随模型走）；供 graph、摘要与子代理共用
        （切换模型会同时影响摘要）。"""

        return LangChainAgentModel(
            catalog=catalog,
            tool_adapter=tool_adapter,
            observability=observability,
        )


__all__ = ["AgentModelProvider"]
