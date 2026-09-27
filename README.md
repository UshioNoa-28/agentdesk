# AgentDesk

单进程多智能体平台：LangGraph 运行时嵌入终端 CLI，主控与子代理以
AutoGen Core Actor 协作，带 Session/Message 历史与上下文压缩；模型走进程内
LangChain，外部能力走 MCP Server，持久化为单
文件 SQLite，默认无需任何外部服务。

```text
uv run agentdesk
  ├── LangChain ──▶ model provider
  ├── AutoGen   ──▶ main_agent ⇄ subagents
  ├── MCP       ──▶ Exa / …
  └── SQLite    ──▶ sessions / messages
```

## Quickstart

Python 3.12+，环境用 uv 管理。

```bash
git clone https://github.com/UshioNoa-28/agent.git
cd agent
uv run agentdesk
```

启动前请复制 `.agent-desk/settings.json.example` 到
`.agent-desk/settings.json` 然后填好 `model` 段。这是必须配置，不配
起不来。`mcp` 和 `skills` 都是可选的，没有就不加载，不再作为启动失败
处理。

## 配置

三处配置共用同一条发现链：从 workspace 根（缺省进程 cwd，可用
`WORKSPACE_ROOT` 指定）逐级向上直到 `~`，近层级优先。

| 路径 | 用途 | 缺失时 |
| --- | --- | --- |
| `.agent-desk/mcp.json` | MCP servers：`mcpServers` 映射（严格 JSON），扩展 `description`（路由用，必填）与 `enabled` | 缺失或坏文件均记 warning 跳过该层，最坏零 Server 启动 |
| `.agent-desk/settings.json` | `permissions.allow` 放行名单（未列出 = 一律审批）；`model` 以模型 id 为键、值为连接配置 + 窗口/输出上限 + 可选协议开关，平铺同级、全部可选（缺键吃默认，首个为默认） | 所有工具弹审批；没有任何 model 条目则启动失败 |
| `.agent-desk/skills/<name>/SKILL.md` | Skill：front matter 必填 `name`（=目录名）与 `description` | 该层无 Skill |

工具审批三选项：once / allow for project / reject；选第二项自动把
工具追加进 workspace 层的 `settings.json`。桌面可多开：同一会话同一
时刻只允许一个窗口驾驶（`data/locks/` 文件锁，进程退出自动释放）。

## 打包与安装（Linux）

项目是标准 setuptools 包，入口命令为 `agentdesk`（需 Python 3.11–3.12）。
Alembic 迁移脚本随包分发，安装到任意位置都能自动建库，无需源码检出。

```bash
# 构建（产物在 dist/：.whl + .tar.gz）
uv build

# 从本地 wheel 安装为独立命令行工具（推荐 pipx，隔离依赖）
pipx install dist/anna_agent-0.1.0-py3-none-any.whl
# 或在当前项目目录用 uv：uv tool install --from ./dist/*.whl agentdesk

# 运行：在放好配置的目录里执行
mkdir -p myproj && cd myproj
cp <repo>/.agent-desk/settings.json.example .agent-desk/settings.json  # 填 model 的 base_url/api_key
agentdesk
```

发行版以 GitHub Release 附上 `.whl` / `.tar.gz` 供下载；发布产物不入库
（`dist/` 已 gitignore）。

## Development

```bash
uv run python -m pytest     # 全量测试
uv run ruff check .         # lint
uv run python -m agent      # 只跑运行时生命周期（无 UI）
```

Schema 由 Alembic 管理：改 `agent/infrastructure/db/models.py` 后
`uv run python -m alembic revision --autogenerate -m "..."`，启动自动
升级到 head。
