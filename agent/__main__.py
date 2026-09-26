"""内嵌运行时手动入口:``python -m agent``。

不起任何传输,只把 AgentBootstrap 的完整启停序列跑起来(Ctrl+C 优雅退出),
用于验证生命周期与后续桌面壳复用;正式入口是 ``agentdesk`` CLI(同进程嵌入)。
"""

from __future__ import annotations

import asyncio
import logging

from agent.bootstrap import AgentBootstrap


async def _run() -> None:
    bootstrap = AgentBootstrap()
    await bootstrap.start()
    print("AgentDesk local runtime ready; press Ctrl+C to stop")
    try:
        await asyncio.Event().wait()
    finally:
        await bootstrap.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
