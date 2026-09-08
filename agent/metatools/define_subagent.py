"""固定 ``define_subagent`` Meta Tool。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from agent.domain.multi_agent import RESERVED_AGENT_NAMES
from agent.domain.tools import (
    DEFAULT_SUBAGENT_TOOL_NAMES,
    MAIN_AGENT_ONLY_TOOL_NAMES,
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from agent.metatools.execute_mcp import EXECUTE_MCP_DEFINITION
from agent.metatools.execute_python import EXECUTE_PYTHON_DEFINITION
from agent.metatools.load_skill import LOAD_SKILL_DEFINITION
from agent.metatools.search_mcp import SEARCH_MCP_DEFINITION
from agent.metatools.send_message import SEND_MESSAGE_DEFINITION
from agent.ports.services import SessionServicePort

GRANT_TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "execute_python": EXECUTE_PYTHON_DEFINITION,
    "search_mcp": SEARCH_MCP_DEFINITION,
    "execute_mcp": EXECUTE_MCP_DEFINITION,
    "load_skill": LOAD_SKILL_DEFINITION,
    "send_message": SEND_MESSAGE_DEFINITION,
}

# 这些是主控专用工具，子代理一律不许拿到。
FORBIDDEN_SUBAGENT_TOOLS = MAIN_AGENT_ONLY_TOOL_NAMES


def _build_allowed_tools_description() -> str:
    """从 worker 工具定义生成 allowed_tools 的描述，保持与定义单一来源同步。"""

    choices = "; ".join(
        f"'{name}' ({definition.description})"
        for name, definition in GRANT_TOOL_DEFINITIONS.items()
    )
    default_grants = ", ".join(f"'{name}'" for name in DEFAULT_SUBAGENT_TOOL_NAMES)
    return (
        "Explicit list of meta tools to grant to this subagent. Choose from: "
        f"{choices}. "
        f"Omit this field or pass an empty list to grant only {default_grants}: "
        "that is the default, and it guarantees the subagent can report back "
        "to 'main_agent'. define_subagent and list_subagents are main-agent-only "
        "and can never be granted to a subagent."
    )


DEFINE_SUBAGENT_DEFINITION = ToolDefinition(
    name="define_subagent",
    description=(
        "Define and dynamically register a specialized subagent with custom system prompt, "
        "persona, and authorized tool/MCP capabilities. Once defined, send tasks to it "
        "using send_message."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9_-]+$",
                "description": (
                    "Unique identifier/name of the subagent (e.g. 'coder', 'researcher')."
                ),
            },
            "description": {
                "type": "string",
                "description": "Brief description of the subagent's role and expertise.",
            },
            "system_prompt": {
                "type": "string",
                "description": (
                    "The dedicated system prompt / persona instructions for this subagent."
                ),
            },
            "allowed_tools": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(GRANT_TOOL_DEFINITIONS),
                },
                "description": _build_allowed_tools_description(),
            },
        },
        "required": ["name", "description", "system_prompt"],
        "additionalProperties": False,
    },
)


SUBAGENT_SYSTEM_PROMPT_TEMPLATE = """<reporting_contract>
You are subagent '{agent_name}' in a star topology coordinated by 'main_agent'.

- Plain text replies reach NOBODY. While your run is still going, the ONLY
  way to communicate with 'main_agent' is the 'send_message' tool with
  recipient='main_agent'.
- As soon as the task outcome exists (or a finding is worth sharing), call
  send_message(recipient='main_agent', ...) with the complete result, then
  end your turn. Never "report" by writing assistant text alone.
- Replies from 'main_agent' arrive later as new incoming messages in this
  session.
- Peer-to-peer messaging with other subagents is prohibited; all coordination
  goes through 'main_agent'.
</reporting_contract>

{raw_system_prompt}

<role_description>
Role: {description}
</role_description>

<authorized_meta_tools>
You have been granted access to the following Meta Tools:
{granted_tools_desc}
You MUST only call the tools granted above to accomplish your assigned tasks.
</authorized_meta_tools>"""


class DefineSubagentTool(AgentTool):
    """动态创建子智能体会话的 Meta Tool。

    子会话标题固定为 ``{主会话id}_subagent_{name}``，运行时按该格式把
    收件人裸名解析到子会话，因此创建会话即完成注册，无需额外的内存表。
    标题锚定在不可变的会话 ID 上，主会话改名不影响路由。
    """

    def __init__(
        self,
        *,
        session_service: SessionServicePort,
        definition: ToolDefinition = DEFINE_SUBAGENT_DEFINITION,
    ) -> None:
        self._session_service = session_service
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def aexecute(
        self,
        arguments: Mapping[str, object],
        *,
        context: ToolContext,
    ) -> ToolResult:
        agent_name = str(arguments.get("name", "")).strip()
        if not agent_name:
            return ToolResult(
                content=json.dumps({"ok": False, "error": "Subagent name cannot be empty"})
            )

        # 保留名拦截：main_agent 可伪造主控身份绕过星形约束，user 会污染
        # 消息来源判定；模型输出不做 schema 校验，必须在工具入口拒绝。
        if agent_name in RESERVED_AGENT_NAMES:
            return ToolResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            f"Subagent name '{agent_name}' is reserved. "
                            "Please choose a different name."
                        ),
                    }
                )
            )

        description = str(arguments.get("description", "")).strip()
        raw_system_prompt = str(arguments.get("system_prompt", "")).strip()
        raw_allowed_tools = arguments.get("allowed_tools")

        if isinstance(raw_allowed_tools, list):
            allowed_tools: tuple[str, ...] = tuple(
                str(t).strip() for t in raw_allowed_tools if str(t).strip()
            )
        else:
            allowed_tools = ()

        # 省略或空列表一律回退到默认白名单：子代理至少要有向主控回报的
        # 通道（send_message），否则会静默造出完全无法沟通的哑巴子代理。
        if not allowed_tools:
            allowed_tools = DEFAULT_SUBAGENT_TOOL_NAMES

        # 主控专用工具禁止下放给子代理：模型输出不做 schema 校验，
        # 因此必须在工具入口强制拦截，并用模型可读的方式说明原因。
        forbidden = [name for name in allowed_tools if name in FORBIDDEN_SUBAGENT_TOOLS]
        if forbidden:
            return ToolResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "Subagents cannot be granted these main-agent-only tools: "
                            f"{', '.join(forbidden)}. "
                            "Only the main agent may use them."
                        ),
                    },
                    ensure_ascii=False,
                )
            )

        # 未知工具名直接拒绝：放行会被 registry 静默丢弃，主控以为授权成功、
        # 子代理实际拿不到能力，且全程无任何报错。
        unknown = [name for name in allowed_tools if name not in GRANT_TOOL_DEFINITIONS]
        if unknown:
            return ToolResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            f"Unknown tools cannot be granted: {', '.join(unknown)}. "
                            "Valid choices: "
                            f"{', '.join(GRANT_TOOL_DEFINITIONS)}."
                        ),
                    },
                    ensure_ascii=False,
                )
            )

        # 所有者会话即调用者会话：define_subagent 是主控专用工具，
        # caller_session_id 就是主会话。查询失败必须显式上报；吞掉异常会在
        # 所有者会话不存在时继续创建出无法归属的孤儿会话。
        try:
            await self._session_service.get(context.caller_session_id)
        except Exception as exc:
            return ToolResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            f"Could not load owner session "
                            f"'{context.caller_session_id}' to define subagent "
                            f"'{agent_name}': {exc}"
                        ),
                    }
                )
            )

        sub_title = f"{context.caller_session_id}_subagent_{agent_name}"

        granted_tools_desc = "\n".join(
            f"- {name}: {GRANT_TOOL_DEFINITIONS[name].description}"
            for name in allowed_tools
        )

        system_prompt = SUBAGENT_SYSTEM_PROMPT_TEMPLATE.format_map(
            {
                "raw_system_prompt": raw_system_prompt,
                "description": description,
                "agent_name": agent_name,
                "granted_tools_desc": granted_tools_desc,
            }
        )

        session = await self._session_service.create(
            title=sub_title,
            main_session_id=context.caller_session_id,
            allowed_tools=allowed_tools,
            custom_system_prompt=system_prompt,
        )

        return ToolResult(
            content=json.dumps(
                {
                    "ok": True,
                    "status": "success",
                    "agent_name": agent_name,
                    "session_id": session.id,
                    "message": (
                        f"Subagent '{agent_name}' successfully registered "
                        f"(session_id: '{session.id}'). You can now assign tasks to it "
                        f"using send_message(recipient='{agent_name}', message=...)."
                    ),
                },
                ensure_ascii=False,
            )
        )


__all__ = ["DEFINE_SUBAGENT_DEFINITION", "DefineSubagentTool"]
