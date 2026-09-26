"""Agent 微服务的组合根。

导入本包会先于任何 ``agent.*`` 子模块执行；这里统一为依赖库设好"import 期读取"
的环境默认值,避免它们各自刷屏或建多余的后台客户端:

* ``LANGCHAIN_OPENAI_TCP_KEEPALIVE``: langchain-openai 默认注入带 TCP keepalive
  的 socket options,会关掉 httpx 的代理自动探测并对 env 代理发 WARN。我们本就
  自带 http_async_client,置 0 放弃它的 keepalive 注入即可消警、行为不变。

均用 setdefault:调用方/CI 预先设过则以其为准。
"""

from __future__ import annotations

import os

os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")
