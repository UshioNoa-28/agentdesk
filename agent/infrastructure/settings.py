"""Agent 私有的运行时配置。

每个 ``*Settings`` 类只描述一个基础设施或运行时边界。默认值一律写在字段
上；仅 model、agent、memory、context 四个边界保留
``config/env/<scope>.env`` 文件承载真实增量（密钥、开关、偏离默认值的部署
参数），其余配置通过进程环境注入。组合根直接注入需要的窄配置，不提供跨
边界的聚合 Settings。
"""

from __future__ import annotations

import math

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.infrastructure.environment import env_files_for


class _AgentSettingsBase(BaseSettings):
    """Agent 专属配置共用的 dotenv 读取规则。"""

    model_config = SettingsConfigDict(
        env_file=env_files_for(),
        env_prefix="",
        extra="ignore",
    )


class DatabaseSettings(_AgentSettingsBase):
    """PostgreSQL 连接池和数据库地址配置；无边界文件，env 只在偏离默认时注入。"""

    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5434/agentdesk"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout_seconds: float = 30.0


class ChatModelSettings(_AgentSettingsBase):
    """聊天模型 provider 和模型调用协议配置。"""

    model_config = SettingsConfigDict(
        env_file=env_files_for("model"),
        env_prefix="",
        extra="ignore",
    )

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    # Optional provider output cap. LangChain maps this to the protocol-specific
    # max_completion_tokens or max_output_tokens field.
    openai_max_output_tokens: int | None = Field(default=None, gt=0)
    openai_use_responses_api: bool = False
    # 流式响应"内容静默"上限（秒）：两次解析出的 chunk 之间的间隔。只对流式
    # 生效；抓的是"接单不干活"或生成中途卡死，不限制合法慢速输出的总时长。
    # None 表示沿用 LangChain 内置默认（LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S，
    # 120s）。非流式调用（子代理、摘要）不受此值影响。
    openai_stream_chunk_timeout_seconds: float | None = Field(default=None, gt=0)
    # HTTP 层超时（秒），透传给 httpx，四个维度统一取值。流式时等价于字节
    # 间隔上限（字节持续滴则永不触发）；非流式时退化为总时长硬切。None 表示
    # 不显式设置（沿用 SDK/httpx 默认）。
    openai_request_timeout_seconds: float | None = Field(default=None, gt=0)


class ObservabilitySettings(_AgentSettingsBase):
    """内部 HTTP 和 provider wire 日志配置；由 compose 按服务注入。"""

    # 仅用于调试；开启后记录内部 HTTP 的完整 request/response body，可能包含
    # prompt、文档正文和模型输出。Compose 会按服务写入项目根目录 logs/。
    http_wire_log_path: str | None = None
    # 记录 Agent 发给外部 OpenAI-compatible Provider 的完整 request/response
    # body。Authorization 等敏感 header 会被脱敏。
    provider_wire_log_path: str | None = None


class ContextSettings(_AgentSettingsBase):
    """Agent 上下文预算和压缩策略配置。

    上下文窗口通常只能从模型文档或部署配置中
    得到一个近似值，因此这里用安全余量计算可用输入预算，而不是假设
    provider 会在每次响应中返回最大窗口大小。
    """

    model_config = SettingsConfigDict(
        env_file=env_files_for("context"),
        env_prefix="",
        extra="ignore",
    )

    context_window_tokens: int = Field(default=200_000, gt=0)
    context_reserve_ratio: float = Field(default=0.20, ge=0, lt=1)
    context_compact_threshold_ratio: float = Field(default=0.80, gt=0, le=1)
    context_cold_compact_target_ratio: float = Field(default=0.50, gt=0, le=1)
    context_max_recent_tool_calls: int = Field(default=4, ge=0)
    context_max_tool_result_tokens: int = Field(default=8_000, gt=0)
    context_clear_tool_result_threshold_tokens: int = Field(default=100, gt=0)
    # 摘要模型（与对话模型同一 provider/模型）的输出 token 上限；同时从
    # input_budget 中预留该值得到摘要请求的输入预算。
    context_summary_max_tokens: int = Field(default=2_048, gt=0)

    @model_validator(mode="after")
    def validate_budget_ratios(self) -> ContextSettings:
        """确保压缩水位不会高于预留输出后的硬输入预算。

        Returns:
            ContextSettings: 校验通过后的配置对象。

        Raises:
            ValueError: 压缩阈值比例高于可用输入比例时抛出。
        """

        usable_ratio = 1.0 - self.context_reserve_ratio
        if self.context_compact_threshold_ratio > usable_ratio:
            raise ValueError(
                "context_compact_threshold_ratio cannot exceed the usable input ratio"
            )
        if self.context_cold_compact_target_ratio >= self.context_compact_threshold_ratio:
            raise ValueError(
                "context_cold_compact_target_ratio must be lower than the compact threshold"
            )
        return self

    @property
    def response_reserve_tokens(self) -> int:
        """返回按安全余量保留给模型输出和估算误差的 token 数。"""

        return math.ceil(self.context_window_tokens * self.context_reserve_ratio)

    @property
    def input_budget_tokens(self) -> int:
        """返回 system、工具 schema 和历史消息可使用的输入预算。"""

        return self.context_window_tokens - self.response_reserve_tokens

    @property
    def compact_threshold_tokens(self) -> int:
        """返回达到后才开始冷压缩或摘要的近似 token 水位。"""

        return min(
            self.input_budget_tokens,
            math.floor(self.context_window_tokens * self.context_compact_threshold_ratio),
        )

    @property
    def cold_compact_target_tokens(self) -> int:
        """返回 Cold Compact 成功所需达到的更低水位。"""

        return min(
            self.input_budget_tokens,
            math.floor(
                self.context_window_tokens * self.context_cold_compact_target_ratio
            ),
        )


class AgentRuntimeSettings(_AgentSettingsBase):
    """Agent 模型-工具循环的运行时上限。"""

    model_config = SettingsConfigDict(
        env_file=env_files_for("agent"),
        env_prefix="",
        extra="ignore",
    )

    max_turns: int = Field(default=32, gt=0)
    wait_default_timeout_seconds: float = Field(default=60.0, ge=1)
    wait_max_timeout_seconds: float = Field(default=600.0, gt=0)


class McpSettings(_AgentSettingsBase):
    """Agent MCP 配置文件路径。"""

    model_config = SettingsConfigDict(
        env_file=env_files_for("agent"),
        env_prefix="",
        extra="ignore",
    )

    mcp_config_path: str = "agent/mcp.yaml"


class SkillSettings(_AgentSettingsBase):
    """Agent Skill 配置文件路径。"""

    model_config = SettingsConfigDict(
        env_file=env_files_for("agent"),
        env_prefix="",
        extra="ignore",
    )

    skill_config_path: str = "agent/skills.yaml"


class MemorySettings(_AgentSettingsBase):
    """Agent Mem0 长期记忆配置。"""

    model_config = SettingsConfigDict(
        env_file=env_files_for("memory"),
        env_prefix="",
        extra="ignore",
    )

    memory_enabled: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "agent_memories"
    memory_search_max_limit: int = Field(default=40, ge=1, le=100)

    # LLM 事实提取与消歧模型配置
    memory_llm_provider: str = "openai"  # "openai", "ollama", 等
    memory_llm_model: str = "gpt-4o-mini"
    memory_llm_api_key: str | None = None
    memory_llm_base_url: str | None = None

    # Embedder 向量化模型配置 (支持本地 Ollama / BGE / OpenAI 等)
    memory_embedder_provider: str = "openai"  # "openai", "ollama", 等
    memory_embedder_model: str = "text-embedding-3-small"
    memory_embedder_api_key: str | None = None
    memory_embedder_base_url: str | None = None
    memory_embedder_dims: int | None = None

    # 独立记忆审计与诊断日志路径
    memory_log_path: str | None = "logs/memory.log"

    @model_validator(mode="after")
    def validate_memory_endpoints(self) -> MemorySettings:
        """开启 memory 时集中校验必填项，配置错误在构造期即失败。"""

        if not self.memory_enabled:
            return self
        if self.memory_embedder_dims is None:
            raise ValueError(
                "MEMORY_EMBEDDER_DIMS is required when MEMORY_ENABLED=true: "
                "the Qdrant collection dimension must be declared explicitly "
                "and match the embedder output"
            )
        endpoints = (
            (
                "MEMORY_LLM",
                self.memory_llm_provider,
                self.memory_llm_api_key,
                self.memory_llm_base_url,
            ),
            (
                "MEMORY_EMBEDDER",
                self.memory_embedder_provider,
                self.memory_embedder_api_key,
                self.memory_embedder_base_url,
            ),
        )
        for role, provider, api_key, base_url in endpoints:
            if provider.lower() != "ollama" and not api_key and not base_url:
                raise ValueError(
                    f"{role}_API_KEY or {role}_BASE_URL is required when "
                    f"MEMORY_ENABLED=true and {role}_PROVIDER={provider}"
                )
        return self


__all__ = [
    "AgentRuntimeSettings",
    "ChatModelSettings",
    "ContextSettings",
    "DatabaseSettings",
    "McpSettings",
    "MemorySettings",
    "ObservabilitySettings",
    "SkillSettings",
]
