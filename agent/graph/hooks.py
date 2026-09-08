"""Agent 工作流模型轮次和工具生命周期的扩展 Hook 协议。"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from agent.graph.state import AgentState


class AgentHook(Protocol):
    """一次 Agent 工作流节点生命周期扩展的异步 Hook（中间件风格，接收并按需更新 AgentState）。"""

    def __call__(self, state: AgentState) -> Awaitable[None]:
        """执行 Hook 逻辑，可通过修改 state['runtime'] 或 state 进行跨中间件/节点传递。"""

        ...


BeforeModelHook = AgentHook
AfterModelHook = AgentHook
BeforeToolHook = AgentHook
AfterToolHook = AgentHook


class HookRegistryPort(Protocol):
    """Agent Graph 读取的、运行期间不可变的 Hook 集合协议。"""

    def before_model_hooks(self) -> tuple[BeforeModelHook, ...]:
        """返回当前已注册的模型前置 Hook。"""

        ...

    def after_model_hooks(self) -> tuple[AfterModelHook, ...]:
        """返回当前已注册的模型后置 Hook。"""

        ...

    def before_tool_hooks(self) -> tuple[BeforeToolHook, ...]:
        """返回当前已注册的工具执行前 Hook。"""

        ...

    def after_tool_hooks(self) -> tuple[AfterToolHook, ...]:
        """返回当前已注册的工具执行后 Hook。"""

        ...


__all__ = [
    "AgentHook",
    "BeforeModelHook",
    "AfterModelHook",
    "BeforeToolHook",
    "AfterToolHook",
    "HookRegistryPort",
]
