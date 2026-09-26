"""Agent 自己拥有的框架无关 Tool 值对象。"""

from __future__ import annotations

import json
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

    def permission_targets(self, arguments: Mapping[str, object]) -> tuple[str, ...] | None:
        """从参数里抽出被权限规则匹配的对象段；command 类需要工具自己拆段。

        三态契约：
        - None（协议默认）：本工具无权限面，参数里没有可核验的对象，默认放行；
          用户仍可用整工具的 deny/ask 规则把它拉回审核。
        - 非空 tuple：段是工具自己的语义域里的可核验单元（路径、命令段、
          server 名……）；权限层要求**每一段**都命中 allow 才放行，任何一段
          落在 deny/ask 或未匹配都按各自的规则处理。
        - raise ValueError：参数给不出合法的段（非法路径等），由 graph 在
          授权前转成参数错误，不进询问也不执行。
        """

        return None


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


# 结果信封形状在此处定义一次。工具、AgentGraph 与 MCP registry 一律经由此处出口，
# 不得再手写 "ok" / "error" 字面量。形状与 api/exception_handlers 的错误体对齐：
# {"error": {"code", "message", "details"?}}，外加 "ok" 判别位。
#
# 两个判别面不要混用：
# - outcome_payload 表达"工具完成了请求的动作"，其中 ok=False 用于被调程序非零退出
#   这类**结果数据**（退出码、stdout、stderr 仍是载荷），不产生 error 字段；
# - error_payload 只表达"工具自身没能完成动作"（参数非法、路径越界、权限被拒），
#   此时 error.code 是稳定的机器判定依据。
#
# "ok" 与 "error" 的区分同时被 agent_graph 的 _is_report_to_main 和交互客户端依赖。


def outcome_payload(
    payload: Mapping[str, object] | None = None,
    *,
    ok: bool = True,
) -> dict[str, object]:
    """把工具载荷套上信封；``ok`` 表示工具是否完成了请求的动作。"""

    data = dict(payload or {})
    data["ok"] = ok
    return data


def error_payload(
    code: str,
    message: str,
    *,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """构造失败信封载荷；``code`` 供程序判定，``message`` 供模型阅读。"""

    error: dict[str, object] = {"code": code, "message": message}
    if details:
        error["details"] = dict(details)
    return {"ok": False, "error": error}


def ok_result(
    payload: Mapping[str, object] | None = None,
    *,
    ok: bool = True,
    kind: MessageKind = MessageKind.TOOL_RESULT,
) -> ToolResult:
    """把载荷序列化为 ToolResult。

    ``ok=False`` 表示"工具正常执行完毕，但请求的动作没有成功"
    （如命令非零退出），与 error_result 的"工具本身失败"相对。
    """

    return ToolResult(
        content=json.dumps(
            outcome_payload(payload, ok=ok),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        kind=kind,
    )


def error_result(
    code: str,
    message: str,
    *,
    details: Mapping[str, object] | None = None,
    kind: MessageKind = MessageKind.TOOL_RESULT,
) -> ToolResult:
    """把失败原因序列化为 ToolResult。"""

    return ToolResult(
        content=json.dumps(
            error_payload(code, message, details=details),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        kind=kind,
    )


ALL_META_TOOL_NAMES: tuple[str, ...] = (
    "search_mcp",
    "load_skill",
    "execute_mcp",
    "read_file",
    "search_text",
    "edit_file",
    "bash",
    "remember",
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
    "error_payload",
    "error_result",
    "ok_result",
    "outcome_payload",
]
