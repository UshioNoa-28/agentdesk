FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Agent 和 Health Monitor 使用同一份依赖清单与运行时镜像。
COPY pyproject.toml README.md ./
COPY agent ./agent
COPY skills ./skills
COPY health_monitor ./health_monitor

# 默认使用官方 PyPI；网络受限时可通过 Compose 根 .env 中的
# PIP_INDEX_URL 或 ``docker compose build --build-arg`` 覆盖。
# BuildKit cache mount 保留 pip 下载的 wheel。即使源码变化导致这一层重新执行，
# pip 也会优先复用缓存；缓存不会进入最终镜像。
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN --mount=type=cache,id=anna-agent-pip,sharing=locked,target=/root/.cache/pip \
    python -m pip install --index-url "${PIP_INDEX_URL}" .

# 如果某次重命名只更新了 Compose command、却漏掉源码 COPY，构建阶段直接失败，
# 不要等容器启动时才得到 ModuleNotFoundError。
RUN python -c "import agent, health_monitor"
