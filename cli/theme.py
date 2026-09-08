"""AgentDesk CLI theme - clean & minimal dark terminal palette."""

from __future__ import annotations

# 基础色系（极简现代终端配色）
BLUE = "#3b82f6"
BLUE_LIGHT = "#60a5fa"
CYAN = "#38bdf8"
CYAN_BRIGHT = "#7dd3fc"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
YELLOW = "#eab308"

# 中性色系
WHITE = "#f8fafc"
WHITE_SOFT = "#f1f5f9"
FG = "#e2e8f0"
FG_MUTED = "#94a3b8"
DIM = "#64748b"
DIM_DARK = "#475569"

# 背景与边框
BG = "#0f172a"
BG_CARD = "#1e293b"
BG_SELECTION = "#334155"
BORDER = "#334155"
BORDER_FOCUS = "#38bdf8"

# 兼容别名（避免旧调用异常）
VIOLET = CYAN
VIOLET_LIGHT = CYAN
VIOLET_BRIGHT = BLUE_LIGHT
INDIGO = BLUE
INDIGO_DARK = "#1e293b"
INDIGO_DEEP = "#0f172a"
LIGHT_ORANGE = CYAN
LIGHT_ORANGE_BRIGHT = CYAN_BRIGHT
ORANGE = AMBER
CORAL = RED
PURPLE = CYAN

# 角色与语义映射
COLOR_USER = WHITE
COLOR_ASSISTANT = FG
COLOR_SUBAGENT = CYAN_BRIGHT
COLOR_TOOL = CYAN
COLOR_OK = GREEN
COLOR_ERROR = RED
COLOR_WARN = AMBER
COLOR_ACCENT = CYAN


def markup(color: str, text: str) -> str:
    """生成 rich markup 着色文本。"""
    return f"[{color}]{text}[/{color}]"


def dim(text: str) -> str:
    """生成 dim 文本。"""
    return f"[{DIM}]{text}[/{DIM}]"