"""Agent ToolDefinition 到 LangChain BaseTool 的适配器。"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from agent.domain.tools import ToolDefinition


class LangChainToolAdapter:
    """把 Agent 工具描述转换成 LangChain 可绑定的工具对象。"""

    def to_model_tool(self, definition: ToolDefinition) -> BaseTool:
        """将工具名称、说明和 JSON schema 包装成 StructuredTool。

        Args:
            definition (ToolDefinition): 本次模型 turn 绑定的完整工具定义。

        Returns:
            BaseTool: 可被 ChatOpenAI ``bind_tools`` 使用的 schema 工具。

        Note:
            返回对象的函数只是 schema 占位；真正执行仍由 AgentGraph 的工具节点
            负责，模型适配器不会在这里产生业务副作用。
        """

        return StructuredTool(
            name=definition.name,
            description=definition.description,
            args_schema=dict(definition.parameters),
            # AgentGraph 负责真正执行 Tool，这里的函数仅用于提供 schema。
            func=_schema_only_call,
        )


def _schema_only_call(**_: object) -> str:
    """提供 LangChain 所需的函数占位，不在模型适配器中执行应用 Tool。

    Args:
        **_ (object): LangChain 根据 schema 传入的参数；这里故意忽略。

    Returns:
        str: 空文本占位；实际工具执行路径不会调用该函数。
    """

    return ""
