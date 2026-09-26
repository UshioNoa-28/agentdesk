"""多模型目录：构造期从 settings.json 加载一次、快照不可变（当前选中除外）。

与 ``SkillCatalog`` 同一套模式：从 workspace 根逐级向上扫描
``.agent-desk/settings.json`` 的 ``model`` 段，**JSON 的键即模型 id**（真正发给
provider 的标识），值对象的已知键与 ``ModelProfile`` 字段一对一平铺，
**不认识的键直接忽略、缺的键吃档案默认值**——没有任何逐键校验。调度参数
与配置无关（全局注入的 ``ContextSettings``）。同名由最近的层整体遮蔽，
不同模型并集。发现原语复用 ``project_settings``。

* **启动快照**：模型集合在构造期定下，之后只读；改了文件要重启进程。
  没有任何 model 条目时启动失败（目录至少要有当前模型可选）。
* **默认 = 发现顺序里的第一个模型**，不设 ``default`` 字段。
* **当前选中是唯一可变状态**：``select_model`` 移动活跃指针（仅进程内，
  不落盘）。一个窗口 = 一个进程 = 一个内嵌运行时，切换天然按实例隔离。
* 对外的 listing 只含模型 id 与展示名（``ModelInfo``），永不带出凭据。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from agent.domain.model_profile import ModelProfile
from agent.infrastructure.project_settings import (
    config_search_roots,
    project_settings_path,
    read_settings_object,
)

# ModelProfile 的字段名即 settings.json 的合法键，其余键忽略。
_PROFILE_KEYS: frozenset[str] = frozenset(field.name for field in fields(ModelProfile))


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """前端可见的模型摘要：仅 model_id 与展示名，绝不含凭据。"""

    model_id: str
    name: str


class ModelCatalog:
    """按模型 id 存放档案快照，并持有"当前选中"指针；供各处每轮现取。"""

    def __init__(self, models: Sequence[tuple[str, ModelProfile]]) -> None:
        """收录有序模型列表；空集合即抛错。默认取第一个。"""

        ordered = dict(models)
        if not ordered:
            raise ValueError("ModelCatalog requires at least one model in settings.json")
        self._models = ordered
        self._current = next(iter(ordered))

    @classmethod
    def from_root(cls, start: Path, *, home: Path | None = None) -> ModelCatalog:
        """沿发现链读取各层 ``model`` 段，构造启动快照。"""

        merged: dict[str, ModelProfile] = {}
        for root in config_search_roots(start.expanduser().resolve(strict=False), home=home):
            path = project_settings_path(root)
            if not path.exists():
                continue
            section = read_settings_object(path).get("model")
            if not isinstance(section, dict):
                continue
            for model_id, entry in section.items():
                merged.setdefault(str(model_id), _profile(str(model_id), entry))
        return cls(list(merged.items()))

    @property
    def current_id(self) -> str:
        """当前活跃的模型 id。"""

        return self._current

    def current_profile(self) -> ModelProfile:
        """返回当前活跃模型的档案。

        上下文层与 provider 适配层直接依赖本目录、每次现取，预算与协议
        参数天然跟随 ``select_model`` 的切换在下一圈循环生效。
        """

        return self._models[self._current]

    def list_models(self) -> tuple[ModelInfo, ...]:
        """按发现顺序列出全部模型（仅 id + 展示名，不含任何凭据字段）。"""

        return tuple(
            ModelInfo(model_id=model_id, name=profile.name)
            for model_id, profile in self._models.items()
        )

    def select_model(self, model_id: str) -> str:
        """切换活跃模型（仅进程内，摘要随之变化）。

        Raises:
            ValueError: 该模型 id 未声明。
        """

        chosen = model_id.strip()
        if chosen not in self._models:
            raise ValueError(f"Unknown model: {chosen!r}. Available: {list(self._models)}")
        self._current = chosen
        return chosen


def _profile(model_id: str, entry: Any) -> ModelProfile:
    """把一个模型的原始 JSON 平铺进 ``ModelProfile``；键不认识的丢弃。"""

    kwargs = entry if isinstance(entry, dict) else {}
    known = {key: value for key, value in kwargs.items() if key in _PROFILE_KEYS}
    known.pop("model_id", None)
    return ModelProfile(model_id=model_id, **known)


__all__ = ["ModelCatalog", "ModelInfo"]
