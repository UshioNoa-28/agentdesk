"""后台长期记忆整理者：把待整理日志折叠成笔记。

主 Agent 只负责往 inbox 追加记录，笔记与索引一律由这里生成：读索引和积压记录
→ 交给模型逐条落笔（新建/改写/删除笔记）→ 全部成功后才把消化掉的记录摘出日志。
失败或超预算时记录原地保留，下一轮再试。

模型调用声明 ``purpose=NON_CHAT``，增量不进用户流；工具只给
``MEMORY_TOOL_NAMES`` 那五个，因此整理者碰不到会话历史、Shell 和文件系统，
写记忆的落盘面全在 ``NoteBook`` 里。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from jsonschema import ValidationError, validate

from agent.domain.memory import MemoryLayer, MemoryNoteEntry, PendingRecord
from agent.domain.model_messages import ModelMessage, ToolCall
from agent.domain.tools import (
    MEMORY_TOOL_NAMES,
    AgentTool,
    ToolContext,
    ToolResult,
    error_result,
)
from agent.infrastructure.prompt import PromptBuilder
from agent.ports.model import AgentModelPort, ModelCallPurpose
from agent.ports.services import (
    MemoryConsolidatorPort,
    MemoryNoteBookPort,
    MemoryStorePort,
)
from agent.ports.tools import MetaToolRegistryPort

logger = logging.getLogger(__name__)


class MemoryConsolidator(MemoryConsolidatorPort):
    """一个模型-记忆工具循环，输入是渲染进 prompt 的索引与积压记录。"""

    def __init__(
        self,
        *,
        store: MemoryStorePort,
        notebook: MemoryNoteBookPort,
        meta_tools: MetaToolRegistryPort,
        model: AgentModelPort,
        prompt_builder: PromptBuilder,
        max_turns: int,
    ) -> None:
        self._store = store
        self._notebook = notebook
        self._meta_tools = meta_tools
        self._model = model
        self._prompt_builder = prompt_builder
        self._max_turns = max_turns

    async def consolidate(self, *, session_id: str) -> None:
        """消化积压记录；任何失败只记日志，未消化的记录原地留给下一轮。"""

        if not self._store.enabled:
            return
        try:
            await self._digest(session_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Memory consolidation failed: session_id=%s", session_id)

    async def _digest(self, session_id: str) -> None:
        tools = self._meta_tools.get_tools(MEMORY_TOOL_NAMES)
        if len(tools) != len(MEMORY_TOOL_NAMES):
            # 少了工具就只能空谈，而空谈会被当成整理成功——必须先拦住。
            raise RuntimeError("memory tools are missing from the Meta Tool registry.")
        pending, sections = await asyncio.to_thread(self._survey)
        if not any(pending.values()):
            return

        report = await self._fold(session_id=session_id, sections=sections, tools=tools)
        for layer, records in pending.items():
            if records:
                await asyncio.to_thread(self._store.discard_pending, layer, records)
        logger.info(
            "Memory consolidated: session_id=%s records=%s report=%s",
            session_id,
            sum(len(records) for records in pending.values()),
            report,
        )

    def _survey(self) -> tuple[dict[MemoryLayer, tuple[PendingRecord, ...]], str]:
        """一次读盘：各层的积压记录，连同各层索引一起渲染成 prompt 片段。"""

        pending = {layer: self._store.pending_records(layer) for layer in MemoryLayer}
        sections = "\n".join(
            _layer_section(
                layer,
                self._notebook.list_entries(layer),
                pending[layer],
            )
            for layer in MemoryLayer
        )
        return pending, sections

    async def _fold(
        self,
        *,
        session_id: str,
        sections: str,
        tools: Sequence[AgentTool],
    ) -> str:
        """跑到模型不再调用工具为止，返回它那一句话的收尾报告。"""

        by_name = {tool.definition.name: tool for tool in tools}
        definitions = tuple(tool.definition for tool in tools)
        messages = [
            ModelMessage.system(self._prompt_builder.build_memory_system_prompt(sections)),
            ModelMessage.human(self._prompt_builder.build_memory_request()),
        ]
        for _ in range(self._max_turns):
            turn = await self._model.ainvoke(
                messages=messages,
                tools=definitions,
                tool_choice=None,
                purpose=ModelCallPurpose.NON_CHAT,
            )
            messages.append(turn.message)
            if not turn.tool_calls:
                return turn.message.content
            for call in turn.tool_calls:
                # 串行：NoteBook 的"先写正文再改索引"依赖同一层不并发落笔。
                result = await self._execute(by_name, call, session_id)
                messages.append(
                    ModelMessage.tool(
                        name=call.name,
                        tool_call_id=call.id,
                        content=result.content,
                    )
                )
        raise RuntimeError(f"memory consolidation exceeded {self._max_turns} model turns.")

    async def _execute(
        self,
        by_name: dict[str, AgentTool],
        call: ToolCall,
        session_id: str,
    ) -> ToolResult:
        tool = by_name.get(call.name)
        if tool is None:
            return error_result("unknown_memory_tool", f"Not a memory tool: {call.name}")
        try:
            validate(instance=call.arguments, schema=dict(tool.definition.parameters))
        except ValidationError as exc:
            return error_result("invalid_tool_arguments", exc.message)
        return await tool.aexecute(
            call.arguments,
            context=ToolContext(caller_session_id=session_id),
        )


def _layer_section(
    layer: MemoryLayer,
    entries: Sequence[MemoryNoteEntry],
    records: Sequence[PendingRecord],
) -> str:
    index = "\n".join(
        f"- {entry.title}\t{entry.timestamp.isoformat()}\t{entry.description}"
        for entry in entries
    ) or "(empty)"
    pending = "\n".join(
        f"- {record.created_at.isoformat()} {record.operation.value} {record.content}"
        for record in records
    ) or "(none)"
    return (
        f'<memory_layer name="{layer.value}">\n'
        f"<index>\n{index}\n</index>\n"
        f"<pending>\n{pending}\n</pending>\n"
        "</memory_layer>"
    )


__all__ = ["MemoryConsolidator"]
