"""供应商无关 Message 到模型上下文的确定性投影实现。"""

from __future__ import annotations

from collections.abc import Sequence
from xml.sax.saxutils import escape

from agent.domain.messages import (
    Message,
    MessageKind,
    message_kind_from_value,
)
from agent.domain.model_messages import ModelMessage, ToolCall
from agent.ports.services import TokenCounterServicePort

CLEARED_TOOL_RESULT_TEMPLATE = (
    "Tool result cleared during context compaction.\n"
    "The original result is not included in this model context."
)


class _BaseMessageProjector:
    """两个投影视角共享的确定性投影管道。

    共享变换（两个视角行为一致）：

    - 乱序修复：异步 agent 消息（如 wait_for_replies 期间落库的子代理
      回报）可能插在 tool call 与其结果之间，单遍双缓冲扫描恢复
      provider 协议合法顺序；未等到结果的调用补视图级合成结果。
    - 信任边界：工具结果包 <untrusted_content>、Skill 正文包
      <skill_guidance>、摘要包 <conversation_summary>，全部 XML 转义。
    - 单条截断：普通 TOOL_RESULT 按 max_tool_result_tokens 截断；
      Skill 正文和 MCP 定义是发现上下文，始终完整保留。
    - 视图替换：最新 SUMMARY 及其覆盖范围之后的 tail 替换被覆盖前缀；
      MAINTENANCE 不进入任何视图的输出。

    视角差异只有两个，均不提供缺省实现，由子类显式声明：

    - ``_cleared_message_ids``：cold clearing 是否生效；
    - ``_hidden_kinds``：MAINTENANCE 之外还有哪些 kind 不进入输出。
    """

    def __init__(
        self,
        *,
        token_counter: TokenCounterServicePort,
        max_tool_result_tokens: int | None = None,
        clear_tool_result_threshold_tokens: int | None = None,
    ) -> None:
        """创建投影器。

        ``token_counter`` 由组合根在运行时注入。Projector 不允许在缺少它时
        静默跳过 token 预算逻辑；``max_tool_result_tokens`` 和清除阈值可以
        由已有 checkpoint 的 metadata 或调用方配置提供。
        """

        if token_counter is None:
            raise ValueError("MessageContextProjector requires a token_counter")
        if max_tool_result_tokens is not None and max_tool_result_tokens <= 0:
            raise ValueError("max_tool_result_tokens must be positive")
        if (
            clear_tool_result_threshold_tokens is not None
            and clear_tool_result_threshold_tokens <= 0
        ):
            raise ValueError("clear_tool_result_threshold_tokens must be positive")
        self._token_counter = token_counter
        self._max_tool_result_tokens = max_tool_result_tokens
        self._clear_tool_result_threshold_tokens = clear_tool_result_threshold_tokens

    def _cleared_message_ids(self, messages: Sequence[Message]) -> frozenset[str]:
        """返回本视角下应替换为占位文案的工具结果 ID 集合（子类必须实现）。"""

        raise NotImplementedError

    @staticmethod
    def _hidden_kinds() -> frozenset[MessageKind]:
        """返回本视角下完全不进入输出的消息 kind（子类必须实现）。"""

        raise NotImplementedError

    def project(
        self,
        messages: Sequence[Message],
    ) -> list[ModelMessage]:
        """按序恢复可见历史，并保证 provider 工具调用协议合法。
        """

        # 本视角标记为"已清除"的工具结果 ID，投影时替换成占位文案。
        cleared_ids = self._cleared_message_ids(messages)
        ordered: list[ModelMessage] = [] # 最终返回给模型的输出列表，始终保持协议合法顺序
        open_calls: list[ToolCall] = [] # 还未配对的 tool result
        deferred: list[ModelMessage] = []# tool result还未到达时暂存的其他message

        def flush_deferred() -> None:
            """open call 全部闭合后，按原顺序加入暂存消息"""
            ordered.extend(deferred)
            deferred.clear()

        def close_open_calls() -> None:
            """如果数据异常 即没有配对的tool result 则手动添加一个用于填充"""
            for call in open_calls:
                ordered.append(self._synthetic_tool_result(call))
            open_calls.clear()
            flush_deferred()

        for message in self._visible_messages(messages):
            kind = message_kind_from_value(message.metadata.get("kind"))
            if kind == MessageKind.MAINTENANCE or kind in self._hidden_kinds():
                continue

            if message.role == "assistant" and message.tool_calls:
                # 新的工具调用组开始前，先闭合上一组
                close_open_calls()
                ordered.append(self._project_plain(message, kind=kind))
                open_calls = list(message.tool_calls)
                continue

            if message.role == "tool":
                result = self._project_tool_result(message, cleared_ids)
                if open_calls and message.tool_call_id == open_calls[0].id:
                    # for loop graph调用工具 数据没问题的话 能保证一一顺序对应
                    ordered.append(result)
                    open_calls.pop(0)
                    if not open_calls:
                        flush_deferred()
                else:
                    # 对不上 数据完整性不能被保证 直接填充让llm重试
                    if open_calls:
                        close_open_calls()
                    ordered.append(result)
                continue
            # human / 无 tool_calls 的 assistant / system
            if open_calls:
                # 如果当前有未闭合的tool call 则存到deferred等待下次追加
                deferred.append(self._project_plain(message, kind=kind))
            else:
                # 如果没有则直接追加
                ordered.append(self._project_plain(message, kind=kind))
        close_open_calls()
        return ordered

    def _project_tool_result(
        self,
        message: Message,
        cleared_message_ids: frozenset[str],
    ) -> ModelMessage:

        if message.role != "tool":
            raise ValueError("project_tool_result requires a tool Message")
        kind = message_kind_from_value(message.metadata.get("kind"))

        # 四种 tool 消息的投影策略
        if kind == MessageKind.SKILL_RESULT:
            # load_skill 返回的本地 Skill 正文
            content = self.wrap_skill_guidance(message.content)
        elif kind == MessageKind.MCP_TOOL_DEFINITION:
            # search_mcp 返回的远程工具 schema：外部数据，完整保留但包
            # untrusted_content 边界，防止其中的指令被模型当系统指令。
            content = self.wrap_untrusted_content(message.content)
        elif message.id in cleared_message_ids:
            # 被 cold compact 标记清除的普通工具结果：替换成占位文案
            content = self.cleared_tool_result()
        else:
            # 普通工具执行结果TOOL_RESULT 外部数据 按token上限截断 加上untrusted_content边界
            content = self.wrap_untrusted_content(
                self._token_counter.truncate_text(message.content,self._max_tool_result_tokens) 
                if self._max_tool_result_tokens is not None else message.content
            )

        return ModelMessage(
            role=message.role,
            content=content,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            tool_calls=message.tool_calls,
        )

    @staticmethod
    def _project_plain(message: Message, *, kind: MessageKind) -> ModelMessage:
        """投影非 tool 消息 SUMMARY 需要包上对话摘要边界"""

        content = (
            _BaseMessageProjector.wrap_conversation_summary(message.content)
            if kind == MessageKind.SUMMARY
            else message.content
        )
        name = message.metadata.get("name")
        return ModelMessage(
            role=message.role,
            content=content,
            name=str(name) if name else None,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            tool_calls=message.tool_calls,
        )

    @staticmethod
    def _synthetic_tool_result(call: ToolCall) -> ModelMessage:
        """为从未落库结果的 tool call 生成视图级合成结果（不写数据库）。"""

        return ModelMessage.tool(
            name=call.name,
            tool_call_id=call.id,
            content=(
                "Tool execution was interrupted before its result was persisted. "
                "The execution outcome is unknown. Retry if needed."
            ),
        )

    @staticmethod
    def _visible_messages(messages: Sequence[Message]) -> list[Message]:
        """用最近有效的 summary 及其对应的tail内容 替换它覆盖的历史前缀"""
        ordered = list(messages)
        positions = {message.id: index for index, message in enumerate(ordered)}
        latest: tuple[int, int] | None = None
        for index in range(len(ordered) - 1, -1, -1):
            candidate = ordered[index]
            if (
                message_kind_from_value(candidate.metadata.get("kind"))
                != MessageKind.SUMMARY
            ):
                continue
            boundary_id = candidate.metadata.get("covered_through_message_id")
            if not isinstance(boundary_id, str) or not boundary_id:
                continue
            found = positions.get(boundary_id)
            if found is None or found >= index:
                continue
            latest = (index, found)
            break

        if latest is None:
            # 没有 summary 时直接复制全量历史 maintenance 和 summary 不写入上下文
            return [
                item
                for item in ordered
                if message_kind_from_value(item.metadata.get("kind"))
                not in {MessageKind.MAINTENANCE, MessageKind.SUMMARY}
            ]

        latest_index, boundary_index = latest
        summary = ordered[latest_index]
        tail = [
            item
            for position, item in enumerate(ordered)
            if position > boundary_index # boundary 后面的 均为tail部分
            and position != latest_index
            and message_kind_from_value(item.metadata.get("kind"))
            not in {MessageKind.SUMMARY, MessageKind.MAINTENANCE}
        ]
        return [summary, *tail]

    @staticmethod
    def cleared_tool_result() -> str:
        """生成不包含外部恢复引用的工具结果清除提示。"""

        return CLEARED_TOOL_RESULT_TEMPLATE

    @staticmethod
    def wrap_untrusted_content(content: str) -> str:
        """为外部工具结果添加稳定的不可信内容边界。"""

        return f"<untrusted_content>\n{escape(content)}\n</untrusted_content>"

    @staticmethod
    def wrap_skill_guidance(content: str) -> str:
        """为 Skill 文档添加低优先级指导边界。"""

        return f"<skill_guidance>\n{escape(content)}\n</skill_guidance>"

    @staticmethod
    def wrap_conversation_summary(summary: str) -> str:
        """为 Compact 摘要添加边界。"""

        return f"<conversation_summary>\n{escape(summary)}\n</conversation_summary>"


class MessageContextProjector(_BaseMessageProjector):
    """模型视角投影器：cold clearing 生效，SYSTEM 由调用方切掉。

    实现 ContextProjectorPort（结构化匹配，无需显式继承 Protocol）。
    cold checkpoint（MAINTENANCE 消息）的边界之前、超过清除阈值的普通
    工具结果替换为占位文案——持久层不变，只是模型不再看到原文。
    会话 SYSTEM 消息由调用方以 ``history[1:]`` 切除，投影器不设防。
    """

    def _cleared_message_ids(self, messages: Sequence[Message]) -> frozenset[str]:
        """从最新 cold checkpoint 推导需要替换为 marker 的 Message ID。"""

        positions = {message.id: index for index, message in enumerate(messages)}
        checkpoint: tuple[Message, int] | None = None
        for message in messages:
            metadata = message.metadata
            if (
                message_kind_from_value(metadata.get("kind"))
                != MessageKind.MAINTENANCE
                or metadata.get("maintenance_type") != "cold_compact"
            ):
                continue
            boundary_id = metadata.get("cold_cleared_through_message_id")
            if not isinstance(boundary_id, str) or boundary_id not in positions:
                continue
            checkpoint = (message, positions[boundary_id])
        if checkpoint is None:
            return frozenset()
        checkpoint_message, boundary_index = checkpoint # index 记录 当前message list中的位置 
        threshold = checkpoint_message.metadata.get("clear_tool_result_threshold_tokens")
        if not isinstance(threshold, int):
            threshold = self._clear_tool_result_threshold_tokens
        if threshold is None:
            raise ValueError(
                "Cold checkpoint has no clear threshold and the projector was not "
                "given one"
            )
        cleared: set[str] = set()
        for item in messages[: boundary_index + 1]:
            if (
                item.role == "tool"
                and message_kind_from_value(item.metadata.get("kind"))
                == MessageKind.TOOL_RESULT
                and self._token_counter.estimate_text(
                    self.wrap_untrusted_content(item.content)
                )
                > threshold
            ):
                cleared.add(item.id)
        return frozenset(cleared)

    @staticmethod
    def _hidden_kinds() -> frozenset[MessageKind]:
        """模型视角没有额外隐藏的 kind；SYSTEM 由调用方切除。"""

        return frozenset()

    def project_tool_result(
        self,
        message: Message,
    ) -> ModelMessage:
        """投影单条新工具结果；调用方负责保证它已和 Assistant call 配对。"""

        return self._project_tool_result(message, frozenset())


class SummaryContextProjector(_BaseMessageProjector):
    """摘要视角投影器：cold clearing 永不生效，SYSTEM 不可见。

    实现 SummaryProjectorPort（结构化匹配）。摘要源必须看到工具结果的
    全量内容（仍受单条截断约束）。被摘要覆盖的历史从模型活跃视图移除
    后，摘要成为唯一记录；若摘要侧也应用 cold clearing，被清空的结果
    就等于被销毁两次——信息永久丢失。SYSTEM（会话 persona）也不进入
    摘要源：摘要请求有自己的策略 prompt，会话 persona 属于模型视角的
    每轮装配，不属于对话事实。
    """

    def _cleared_message_ids(self, messages: Sequence[Message]) -> frozenset[str]:
        """摘要视角永不应用 cold clearing 显式返回空集 即契约本身。"""

        return frozenset()

    @staticmethod
    def _hidden_kinds() -> frozenset[MessageKind]:
        """摘要视角额外隐藏 SYSTEM MAINTENANCE 对两个视角都不可见"""

        return frozenset({MessageKind.SYSTEM})


__all__ = [
    "CLEARED_TOOL_RESULT_TEMPLATE",
    "MessageContextProjector",
    "SummaryContextProjector",
]
