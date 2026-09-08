"""Agent 自己拥有的框架无关 Tool 值对象。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from agent.domain.messages import MessageKind


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """描述一个 Tool；它不包含脚本执行细节。"""

    name: str
    description: str
    parameters: Mapping[str, object]
    keywords: tuple[str, ...] = ()
    use_cases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """复制并冻结配置输入，避免一次工作流中的定义被意外修改。

        Returns:
            None: 参数映射转换为只读视图，关键字和用例转换为元组。
        """

        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "keywords", tuple(self.keywords))
        object.__setattr__(self, "use_cases", tuple(self.use_cases))


@dataclass(frozen=True, slots=True)
class ToolContext:
    """工具执行时的调用上下文。

    只携带工具无法从参数推导、又不宜自行查询的调用者身份信息；
    其余会话事实（归属、显示名等）由工具按需从持久层推导，
    保证 sessions 表是身份的唯一事实来源。
    """

    caller_session_id: str


class AgentTool(Protocol):
    """Agent Graph 可调用的最小框架无关 Tool 协议。"""

    @property
    def definition(self) -> ToolDefinition:
        """返回模型请求使用的工具定义。"""

        ...

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> "ToolResult":
        """接收模型参数与调用上下文，返回框架无关的工具结果。"""

        ...


@dataclass(frozen=True, slots=True)
class ToolResult:
    """一次工具调用返回给 Agent 的文本和投影语义。"""

    content: str
    # kind 同时决定持久化 metadata 和模型上下文投影策略；调用方直接传入
    # 唯一的 MessageKind。
    kind: MessageKind = MessageKind.TOOL_RESULT

    def __post_init__(self) -> None:
        """把边界输入归一化为 MessageKind，避免运行时混用裸字符串。"""

        if isinstance(self.kind, MessageKind):
            return
        try:
            normalized = MessageKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported MessageKind for ToolResult: {self.kind!r}") from exc
        object.__setattr__(self, "kind", normalized)


ALL_META_TOOL_NAMES: tuple[str, ...] = (
    "search_mcp",
    "load_skill",
    "search_memory",
    "execute_mcp",
    "execute_python",
    "define_subagent",
    "send_message",
    "list_subagents",
    "wait_for_replies",
    "ask_user",
)

# 星形拓扑：子代理默认只授予对主控的回报通道，保证没有工具时至少能汇报。
DEFAULT_SUBAGENT_TOOL_NAMES: tuple[str, ...] = ("send_message",)

# 主控专用工具：子代理会话（main_session_id 非空）一律不许持有，
# 无论来自 define_subagent 还是任何其它创建路径。
MAIN_AGENT_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "define_subagent",
    "list_subagents",
    "search_memory",
    "wait_for_replies",
    "ask_user",
)

__all__ = [
    "ALL_META_TOOL_NAMES",
    "DEFAULT_SUBAGENT_TOOL_NAMES",
    "MAIN_AGENT_ONLY_TOOL_NAMES",
    "AgentTool",
    "ToolContext",
    "ToolDefinition",
    "ToolResult",
]
