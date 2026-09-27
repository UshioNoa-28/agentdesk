"""应用启动时执行的 Alembic 迁移。

``command.upgrade`` 会运行 alembic_migrations/env.py,其中在线模式内部调用
``asyncio.run``;因此本函数只能在**没有运行事件循环的线程**里执行,
应用侧统一经 ``asyncio.to_thread`` 调进来。

迁移脚本随包一起分发(``agent/infrastructure/db/alembic_migrations/``),路径从
本模块位置解析,因此源码运行与 ``pip``/wheel 安装运行都能找到,不再依赖仓库
根的 ``alembic.ini``(那份只供 ``alembic`` CLI 手工生成 revision 用)。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config

# 迁移脚本随包分发；从模块自身位置解析，dev 检出与安装后的 site-packages 都正确。
_ALEMBIC_MIGRATIONS_DIR = Path(__file__).resolve().parent / "alembic_migrations"


def build_alembic_config() -> Config:
    """构造运行期 alembic 配置：不读 ini(避免依赖仓库根文件)，只指定脚本目录。"""

    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_MIGRATIONS_DIR))
    return config


def apply_migrations(revision: str = "head") -> None:
    """把数据库升级到指定 revision(默认 head)。失败直接抛异常。"""

    command.upgrade(build_alembic_config(), revision)


async def migrate_database() -> None:
    """asyncio 侧入口:迁移在无事件循环的工作线程里跑完再回来。"""

    await asyncio.to_thread(apply_migrations)


__all__ = ["apply_migrations", "build_alembic_config", "migrate_database"]
