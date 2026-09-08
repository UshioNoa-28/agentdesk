# AgentDesk

AgentDesk 是一个多智能体协作平台：FastAPI + LangGraph 驱动的 Agent 服务，
基于 AutoGen Core 的 Actor 运行时实现主控—子代理星形协作，负责 Session、
Message 历史、上下文压缩与 Mem0 长期记忆；模型调用由进程内 LangChain
适配器完成，检索等外部能力由独立 MCP Server 提供。

```text
Terminal CLI
    |
    v
Agent Service :8000 --LangChain--> model provider
    |
    +--AutoGen--> main_agent <-events-> subagent actors (define_subagent)
    |
    +----MCP----> Exa / local Python / other servers
    |
    +----Qdrant-> Mem0 (long-term memory)
    |
    +----------> PostgreSQL (sessions, messages)

Health Monitor ----> Agent / Qdrant / PostgreSQL
```


## Deploy

依赖：Docker Compose。

1. 生成配置并按需填写：

```bash
cp config/env/model.env.example config/env/model.env
cp config/env/agent.env.example config/env/agent.env
cp config/env/memory.env.example config/env/memory.env
cp config/env/context.env.example config/env/context.env
```

2. 构建并启动：

```bash
docker compose up -d --build
```

3. 验证：

```bash
curl http://127.0.0.1:8000/api/health
```

## CLI

服务启动后，在仓库根目录运行终端客户端：

```bash
python -m cli
```

默认连接 `http://127.0.0.1:8000/api`，可通过环境变量覆盖：

```bash
AGENT_API_URL=http://127.0.0.1:8000/api python -m cli
```

输入 `/` 可查看全部斜杠命令（会话管理、上下文压缩、多智能体监视等）。

