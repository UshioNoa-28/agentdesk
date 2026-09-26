"""``.agent-desk/settings.json`` 发现链与文件读写原语。

发现链沿用 Skill / MCP / model / permissions 的同一链条
（``config_search_roots``）：从 workspace 根（缺省进程 cwd）逐级向上直到
主目录。本模块只提供通用骨架——路径约定、逐层扫描、顶层 JSON 解析、
原子写入；各配置段（permissions、model…）的语义与校验住在使用方自己的
模块里，它们复用这里的原语各扫各的层级。

文件缺失是正常状态（等价于该层没有贡献）；坏 JSON / 非对象顶层是错误，
由 ``read_settings_object`` 抛 ``ValueError``，让拼写错误立刻可见。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_SETTINGS_FILE_NAME = "settings.json"

AGENTDESK_DIR_NAME = ".agent-desk"


def config_search_roots(start: Path, *, home: Path | None = None) -> tuple[Path, ...]:
    """返回按优先级从高到低排列的根"""

    home_dir = Path.home().resolve(strict=False) if home is None else home
    roots: list[Path] = []
    current = start
    while True:
        roots.append(current)
        if current == home_dir:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    if home_dir not in roots:
        roots.append(home_dir)
    return tuple(roots)


def project_settings_path(root: Path) -> Path:
    """返回某一层级的项目配置文件路径。"""

    return root / AGENTDESK_DIR_NAME / PROJECT_SETTINGS_FILE_NAME


def read_settings_object(path: Path) -> dict[str, Any]:
    """解析一个 settings.json 的顶层对象；坏 JSON / 非对象即抛 ``ValueError``。"""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read project settings file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid project settings JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Project settings must be an object: {path}")
    return data


def atomic_write(path: Path, text: str) -> None:
    """tmp + replace 原子写入；文件不存在时会自动创建。"""

    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


__all__ = [
    "AGENTDESK_DIR_NAME",
    "PROJECT_SETTINGS_FILE_NAME",
    "atomic_write",
    "config_search_roots",
    "project_settings_path",
    "read_settings_object",
]
