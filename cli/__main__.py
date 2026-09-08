import asyncio
from cli.app import AnnaCliApp


def main() -> None:
    try:
        app = AnnaCliApp()
        app.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    main()
