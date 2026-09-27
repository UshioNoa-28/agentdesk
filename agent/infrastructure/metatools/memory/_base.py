"""长期记忆整理者的 Meta Tool 共享细节。

这些工具就是普通 ``AgentTool``，注册进 ``MetaToolRegistry`` 与主 Agent 的工具同
一张表；隔离靠名字：它们只出现在 ``MEMORY_TOOL_NAMES``，不进
``ALL_META_TOOL_NAMES``，而会话工具表是 ``get_tools(session.allowed_tools)`` 按名
字过滤的结果，主 Agent 与子代理的名单里都没有它们，因此谁也调不到。参数只有
``layer`` 与 ``title``，没有路径，写不出记忆目录之外。

标题即身份：它同时是索引第一列和正文文件名 ``<标题>.md``。索引 ``MEMORY.md``
由 ``NoteBook`` 独家序列化，标题的合法形状也在那里校验（``_title_of``）。
"""

from __future__ import annotations

from agent.domain.memory import MemoryLayer, MemoryNote

LAYER_SCHEMA: dict[str, object] = {
    "type": "string",
    "enum": [layer.value for layer in MemoryLayer],
    "description": "Which memory layer the note lives in.",
}

TITLE_SCHEMA: dict[str, object] = {
    "type": "string",
    "description": "The note title shown by read_index; it names the file.",
}


def note_payload(note: MemoryNote) -> dict[str, object]:
    """把一条笔记投影成工具返回值；不带路径，模型不需要知道文件在哪。"""

    return {
        "title": note.entry.title,
        "description": note.entry.description,
        "timestamp": note.entry.timestamp.isoformat(),
        "body": note.body,
    }


def layer_of(value: object) -> MemoryLayer | None:
    try:
        return MemoryLayer(str(value).strip())
    except ValueError:
        return None


__all__ = ["LAYER_SCHEMA", "TITLE_SCHEMA", "layer_of", "note_payload"]
