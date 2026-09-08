"""按配置边界读取 dotenv 文件，不修改当前进程环境。

真实配置放在 ``config/env``，而不是把所有服务的变量继续堆在仓库根目录。
进程环境仍可覆盖文件配置，方便容器编排和部署平台注入 secret。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

# 这些文件名是部署约定的一部分。不要在这里按字段再拆文件；同一边界内的
# 变量应当一起配置，避免出现 ``MAX_TURNS`` 这种单字段文件。
#
# 默认值统一放在各 Settings 类里；env 文件只承载真实增量（密钥、开关和
# 偏离默认值的部署参数），不在文件里复述默认值。
ENV_DIRECTORY = Path("config/env")

# 每个 Settings 类拥有自己的边界文件；Agent 专属字段集中在 agent.env。
ALL_SETTINGS_SCOPES = (
    "agent",
    "context",
    "memory",
    "model",
)


def env_files_for(*scopes: str) -> tuple[str, ...]:
    """返回 Settings 使用的 dotenv 文件顺序。

    ``config/env/<scope>.env.local`` 会覆盖同一边界的基础文件；进程环境仍由
    Pydantic 或调用方放在最高优先级。
    """

    files: list[str] = []
    for scope in scopes:
        normalized = scope.strip()
        if not normalized or normalized not in ALL_SETTINGS_SCOPES:
            raise ValueError(f"unsupported environment scope: {scope!r}")
        files.extend(
            (
                str(ENV_DIRECTORY / f"{normalized}.env"),
                str(ENV_DIRECTORY / f"{normalized}.env.local"),
            )
        )
    return tuple(files)


def load_dotenv_environment(
    *,
    scopes: tuple[str, ...] = ALL_SETTINGS_SCOPES,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """合并 scoped dotenv 文件和进程环境，供 YAML 等非-Pydantic配置使用。

    Args:
        scopes (tuple[str, ...]): 要读取的配置边界；默认读取全部边界。
        environ (Mapping[str, str] | None): 可注入的进程环境，主要用于测试。

    Returns:
        dict[str, str]: dotenv 值和进程环境的合并结果。

    Note:
        这里使用 ``dotenv_values`` 而不是 ``load_dotenv``，不会把密钥或配置
        写回全局 ``os.environ``。进程环境最后覆盖 dotenv 文件中的同名变量。
    """

    values: dict[str, str] = {}
    for file_name in env_files_for(*scopes):
        values.update(
            {
                key: value
                for key, value in dotenv_values(file_name).items()
                if value is not None
            }
        )
    values.update(dict(os.environ if environ is None else environ))
    return values


__all__ = [
    "ALL_SETTINGS_SCOPES",
    "env_files_for",
    "load_dotenv_environment",
]
