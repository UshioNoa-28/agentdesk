"""Agent Graph 可选 Hook 的集合实现。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.application.graph.hooks import (
    AfterModelHook,
    AfterToolHook,
    BeforeModelHook,
    BeforeToolHook,
    HookRegistryPort,
)


class AgentHookRegistry(HookRegistryPort):
    """可在运行中动态登记的 Hook 集合。

    这是 ``HookRegistryPort`` 的可变实现：调用方按需 ``register_*`` 追加回调。
    生产组合根并不使用它——而是注入构造后冻结的 ``TurnHooks``；本类主要服务于
    测试，让用例逐条装配 Hook 后再注入 ``AgentGraph``。
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
    """构造后冻结的可选 Hook 集合：生产组合根注入的默认实现。"""

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
