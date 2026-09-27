"""AgentDesk CLI 入口:UI 与 Agent 运行时同进程。"""

import asyncio

from cli.app import AnnaCliApp
from cli.ui import silence_background_loggers


def main() -> None:
    silence_background_loggers()
    try:
        app = AnnaCliApp()
        app.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    main()
