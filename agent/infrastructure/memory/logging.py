"""Memory 审计日志。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def log_memory_event(
    log_path: str | None,
    event: str,
    *,
    status: str = "INFO",
    user_id: str | None = None,
    duration_s: float | None = None,
    details: dict[str, Any] | None = None,
    error: str | None = None,
    exc_info: bool = False,
) -> None:
    """记录 Memory 审计事件：控制台只留一行摘要，完整明细进独立文件。"""

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status": status,
        "user_id": user_id,
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
        "details": details or {},
    }
    if error:
        record["error"] = error

    message = f"[Mem0] [{event}] status={status}"
    if user_id:
        message += f" user_id={user_id}"
    if duration_s is not None:
        message += f" duration={duration_s:.2f}s"
    if error:
        message += f" error={error}"

    if status == "ERROR":
        logger.error(message, exc_info=exc_info)
    else:
        logger.info(message)

    if not log_path:
        return

    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False))
            stream.write("\n")
    except OSError:
        logger.exception("Failed to write memory audit event to %s", log_path)


__all__ = ["log_memory_event"]
