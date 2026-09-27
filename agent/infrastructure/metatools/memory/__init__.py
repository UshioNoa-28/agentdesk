"""长期记忆整理者的 Meta Tool。

每个操作一个模块；这里把定义与工具类汇总出去。它们是普通 ``AgentTool``，与主
Agent 的工具同表注册，隔离靠名单：名字只进 ``MEMORY_TOOL_NAMES``，不进
``ALL_META_TOOL_NAMES``，所以任何会话的 ``allowed_tools`` 都筛不出它们。
"""

from agent.infrastructure.metatools.memory.delete_note import (
    DELETE_NOTE_DEFINITION,
    DeleteNoteTool,
)
from agent.infrastructure.metatools.memory.read_index import (
    READ_INDEX_DEFINITION,
    ReadIndexTool,
)
from agent.infrastructure.metatools.memory.read_note import READ_NOTE_DEFINITION, ReadNoteTool
from agent.infrastructure.metatools.memory.update_note import (
    UPDATE_NOTE_DEFINITION,
    UpdateNoteTool,
)
from agent.infrastructure.metatools.memory.write_note import (
    WRITE_NOTE_DEFINITION,
    WriteNoteTool,
)

__all__ = [
    "DELETE_NOTE_DEFINITION",
    "READ_INDEX_DEFINITION",
    "READ_NOTE_DEFINITION",
    "UPDATE_NOTE_DEFINITION",
    "WRITE_NOTE_DEFINITION",
    "DeleteNoteTool",
    "ReadIndexTool",
    "ReadNoteTool",
    "UpdateNoteTool",
    "WriteNoteTool",
]
