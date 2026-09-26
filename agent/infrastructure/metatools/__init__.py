"""Agent 进程内固定 Meta Tool。"""

from agent.infrastructure.metatools.ask_user import ASK_USER_DEFINITION, AskUserTool
from agent.infrastructure.metatools.define_subagent import (
    DEFINE_SUBAGENT_DEFINITION,
    DefineSubagentTool,
)
from agent.infrastructure.metatools.execute_mcp import EXECUTE_MCP_DEFINITION, ExecuteMcpTool
from agent.infrastructure.metatools.list_subagents import (
    LIST_SUBAGENTS_DEFINITION,
    ListSubagentsTool,
)
from agent.infrastructure.metatools.load_skill import LOAD_SKILL_DEFINITION, LoadSkillTool
from agent.infrastructure.metatools.registry import MetaToolRegistry
from agent.infrastructure.metatools.remember import REMEMBER_DEFINITION, RememberTool
from agent.infrastructure.metatools.search_mcp import SEARCH_MCP_DEFINITION, SearchMcpTool
from agent.infrastructure.metatools.send_message import SEND_MESSAGE_DEFINITION, SendMessageTool
from agent.infrastructure.metatools.wait_for_replies import (
    WAIT_FOR_REPLIES_DEFINITION,
    WaitForRepliesTool,
)
from agent.infrastructure.metatools.workspace import (
    EDIT_FILE_DEFINITION,
    READ_FILE_DEFINITION,
    SEARCH_TEXT_DEFINITION,
    EditFileTool,
    ReadFileTool,
    SearchTextTool,
)

__all__ = [
    "ASK_USER_DEFINITION",
    "AskUserTool",
    "DEFINE_SUBAGENT_DEFINITION",
    "DefineSubagentTool",
    "EXECUTE_MCP_DEFINITION",
    "ExecuteMcpTool",
    "EDIT_FILE_DEFINITION",
    "EditFileTool",
    "LIST_SUBAGENTS_DEFINITION",
    "ListSubagentsTool",
    "MetaToolRegistry",
    "REMEMBER_DEFINITION",
    "RememberTool",
    "SEARCH_MCP_DEFINITION",
    "READ_FILE_DEFINITION",
    "ReadFileTool",
    "SEARCH_TEXT_DEFINITION",
    "SearchTextTool",
    "LOAD_SKILL_DEFINITION",
    "SEND_MESSAGE_DEFINITION",
    "WAIT_FOR_REPLIES_DEFINITION",
    "SearchMcpTool",
    "LoadSkillTool",
    "SendMessageTool",
    "WaitForRepliesTool",
]
