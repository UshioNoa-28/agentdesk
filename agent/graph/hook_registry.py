"""Agent Graph 可选 Hook 的集合实现。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.graph.hooks import (
    AfterModelHook,
    AfterToolHook,
    BeforeModelHook,
    BeforeToolHook,
    HookRegistryPort,
)


class AgentHookRegistry(HookRegistryPort):
    """保存一次 Agent 请求期间需要运行的生命周期 Hook。

    Registry 由组合根按请求创建并注入 ``AgentGraph``。Graph 直接从 Registry
    读取 Hook，因此无需把回调函数放进 LangGraph state；请求作用域结束后由
    Dishka 丢弃整个 Registry 实例。
    """

    def __init__(self) -> None:
        self._before_model: list[BeforeModelHook] = []
        self._after_model: list[AfterModelHook] = []
        self._before_tool: list[BeforeToolHook] = []
        self._after_tool: list[AfterToolHook] = []

    def register_before_model(self, hook: BeforeModelHook) -> None:
        """追加一个模型调用前 Hook。"""

        self._before_model.append(hook)

    def register_after_model(self, hook: AfterModelHook) -> None:
        """追加一个模型调用后 Hook。"""

        self._after_model.append(hook)

    def register_before_tool(self, hook: BeforeToolHook) -> None:
        """追加一个工具执行前 Hook。"""

        self._before_tool.append(hook)

    def register_after_tool(self, hook: AfterToolHook) -> None:
        """追加一个工具执行后 Hook。"""

        self._after_tool.append(hook)

    def before_model_hooks(self) -> tuple[BeforeModelHook, ...]:
        """返回当前模型前置 Hook 的快照。"""

        return tuple(self._before_model)

    def after_model_hooks(self) -> tuple[AfterModelHook, ...]:
        """返回当前模型后置 Hook 的快照。"""

        return tuple(self._after_model)

    def before_tool_hooks(self) -> tuple[BeforeToolHook, ...]:
        """返回当前工具执行前 Hook 的快照。"""

        return tuple(self._before_tool)

    def after_tool_hooks(self) -> tuple[AfterToolHook, ...]:
        """返回当前工具执行后 Hook 的快照。"""

        return tuple(self._after_tool)


@dataclass(frozen=True, slots=True)
class TurnHooks(HookRegistryPort):
    """构造后冻结的可选 Hook 集合。"""

    before_model: tuple[BeforeModelHook, ...] = ()
    after_model: tuple[AfterModelHook, ...] = ()
    before_tool: tuple[BeforeToolHook, ...] = ()
    after_tool: tuple[AfterToolHook, ...] = ()

    def before_model_hooks(self) -> tuple[BeforeModelHook, ...]:
        return self.before_model

    def after_model_hooks(self) -> tuple[AfterModelHook, ...]:
        return self.after_model

    def before_tool_hooks(self) -> tuple[BeforeToolHook, ...]:
        return self.before_tool

    def after_tool_hooks(self) -> tuple[AfterToolHook, ...]:
        return self.after_tool


__all__ = ["AgentHookRegistry", "TurnHooks"]
