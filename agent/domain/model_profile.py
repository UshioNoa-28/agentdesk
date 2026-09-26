"""模型档案：settings.json 里一个模型条目的完整镜像。

纯事实对象——身份、连接、窗口与输出上限，settings.json 的键与本类字段
一一对应，未知键忽略，缺键吃这里的默认值；不做任何校验。压缩水位、截断
阈值等调度参数是我们系统的属性，独立成全局注入的 ``ContextSettings``；
两者的派生预算全部收敛在 ``agent.domain.context_budget.derive_budget``
一处，本类不挂公式。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """一个模型的全部事实；一切字段可缺省，缺省即默认。"""

    # ---- 身份与连接（settings.json 的键与值对象字段一对一）。 ----
    # 真正发给 provider 的模型标识 = settings.json 里该条目的键。
    model_id: str = "model"
    # 展示名，仅供前端与日志。
    name: str = "model"
    base_url: str = ""
    api_key: str = ""
    # ---- 事实：窗口与输出上限。 ----
    context_window: int = 200_000
    max_output_tokens: int = 8_192
    # ---- 协议开关：随模型可配（provider 的部署属性）。 ----
    # 仅当 provider 明确支持 /v1/responses 时开启。
    use_responses_api: bool = False
    # HTTP 层超时（秒）。None 表示不显式设置（沿用 SDK 默认 600s）。
    request_timeout_seconds: float | None = None
    # SDK 层重试次数。
    max_retries: int = 2


__all__ = ["ModelProfile"]
