from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx

from agent.domain.messages import MessageKind
from agent.domain.tools import ToolContext
from agent.infrastructure.mcp import (
    McpRegistry,
    McpServerConfig,
    McpServerState,
    load_mcp_configuration,
)
from agent.infrastructure.mcp.servers.local_python import run_python
from agent.metatools import SearchMcpTool
from agent.prompt.system import build_system_prompt

_REAL_ASYNC_CLIENT = httpx.AsyncClient


class McpConfigTests(TestCase):
    def test_default_config_has_meta_tool_and_detailed_server_descriptions(self) -> None:
        with patch.dict(os.environ, {"EXA_MCP_ENABLED": "false"}):
            config = load_mcp_configuration(Path(__file__).parents[1] / "agent" / "mcp.yaml")
        self.assertEqual(
            ("exa-search",),
            tuple(server.id for server in config.servers),
        )
        self.assertFalse(config.servers[0].enabled)
        self.assertTrue(all(len(server.description) > 100 for server in config.servers))
        prompt = build_system_prompt(tuple((s.id, s.description) for s in config.servers))
        self.assertIn('server: "exa-search"', prompt)

    def test_http_server_config_requires_url_and_normalizes_streamable_http(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mcp.yaml"
            path.write_text(
                "mcp_servers:\n"
                "  - {id: rag, description: rag, transport: streamable-http, url: http://rag/mcp}\n",
                encoding="utf-8",
            )
            server = load_mcp_configuration(path).servers[0]
            self.assertEqual("http", server.transport)
            self.assertEqual("http://rag/mcp", server.url)

    def test_http_headers_expand_from_environment(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mcp.yaml"
            path.write_text(
                "mcp_servers:\n"
                "  - id: exa\n"
                "    description: Exa\n"
                "    transport: streamable-http\n"
                "    url: https://mcp.exa.ai/mcp\n"
                "    headers:\n"
                "      x-api-key: \"${EXA_API_KEY:-}\"\n",
                encoding="utf-8",
            )
            previous = os.environ.get("EXA_API_KEY")
            os.environ["EXA_API_KEY"] = "test-exa-key"
            try:
                server = load_mcp_configuration(path).servers[0]
            finally:
                if previous is None:
                    os.environ.pop("EXA_API_KEY", None)
                else:
                    os.environ["EXA_API_KEY"] = previous

        self.assertEqual((("x-api-key", "test-exa-key"),), server.headers)

    def test_http_headers_expand_from_dotenv_without_process_environment(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mcp.yaml"
            path.write_text(
                "mcp_servers:\n"
                "  - id: exa\n"
                "    description: Exa\n"
                "    transport: streamable-http\n"
                "    url: https://mcp.exa.ai/mcp\n"
                "    headers:\n"
                "      x-api-key: \"${EXA_API_KEY:-}\"\n",
                encoding="utf-8",
            )
            with patch(
                "agent.infrastructure.mcp.config.load_dotenv_environment",
                return_value={"EXA_API_KEY": "dotenv-exa-key"},
            ):
                previous = os.environ.pop("EXA_API_KEY", None)
                try:
                    server = load_mcp_configuration(path).servers[0]
                finally:
                    if previous is not None:
                        os.environ["EXA_API_KEY"] = previous

        self.assertEqual((("x-api-key", "dotenv-exa-key"),), server.headers)

    def test_stdio_headers_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mcp.yaml"
            path.write_text(
                "mcp_servers:\n"
                "  - id: local\n"
                "    description: local\n"
                "    transport: stdio\n"
                "    command: python\n"
                "    headers: {x-api-key: secret}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "only supported for HTTP"):
                load_mcp_configuration(path)

    def test_config_rejects_arbitrary_transport(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mcp.yaml"
            path.write_text(
                "mcp_servers:\n"
                "  - {id: bad, description: bad, transport: import, command: x}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported transport"):
                load_mcp_configuration(path)

    def test_config_requires_boolean_enabled_flag(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mcp.yaml"
            path.write_text(
                "mcp_servers:\n"
                "  - {id: disabled, description: disabled, "
                "enabled: false, transport: stdio, command: python}\n",
                encoding="utf-8",
            )
            server = load_mcp_configuration(path).servers[0]
            self.assertFalse(server.enabled)

            path.write_text(
                "mcp_servers:\n"
                "  - {id: invalid, description: invalid, "
                "enabled: 'false', transport: stdio, command: python}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "enabled must be a boolean"):
                load_mcp_configuration(path)


class McpRegistryTests(IsolatedAsyncioTestCase):
    async def test_disabled_server_is_visible_but_never_started_or_advertised(self) -> None:
        registry = McpRegistry(
            (
                McpServerConfig(
                    id="disabled",
                    description="disabled server",
                    command="/missing/anna-command",
                    args=(),
                    timeout_seconds=1,
                    enabled=False,
                ),
            )
        )
        self.addAsyncCleanup(registry.close_async)

        await registry.start_all_async()
        status = registry.list_statuses()[0]

        self.assertEqual(McpServerState.DISABLED, status.state)
        self.assertIsNone(status.error)
        self.assertEqual((), registry.server_descriptions())
        self.assertEqual(McpServerState.DISABLED, (await registry.retry_async("disabled")).state)
        with self.assertRaisesRegex(ValueError, "disabled"):
            registry.search_tools(mcp="disabled", query="anything", limit=1)

    async def test_start_all_replaces_a_previous_connection_without_leaving_it_running(
        self,
    ) -> None:
        registry = McpRegistry(
            (
                McpServerConfig(
                    "python",
                    "working server",
                    sys.executable,
                    ("-m", "agent.infrastructure.mcp.servers.local_python"),
                    5,
                ),
            )
        )
        self.addAsyncCleanup(registry.close_async)

        await registry.start_all_async()
        first = registry._connections["python"]  # type: ignore[attr-defined]
        await registry.start_all_async()
        second = registry._connections["python"]  # type: ignore[attr-defined]

        self.assertIsNot(first, second)
        self.assertEqual(McpServerState.CONNECTED, registry.list_statuses()[0].state)
        self.assertEqual(
            "42\n",
            json.loads(
                (
                    await registry.execute_tool(
                        mcp="python", tool="run_python", arguments={"code": "print(6 * 7)"}
                    )
                ).content
            )["stdout"],
        )

    async def test_stdio_server_connects_discovers_and_executes_tool(self) -> None:
        registry = McpRegistry(
            (
                McpServerConfig(
                    id="python",
                    description="test",
                    command=sys.executable,
                    args=("-m", "agent.infrastructure.mcp.servers.local_python"),
                    timeout_seconds=5,
                ),
            )
        )
        self.addAsyncCleanup(registry.close_async)

        await registry.start_all_async()
        status = registry.list_statuses()[0]
        tools = registry.search_tools(mcp="python", query="python code", limit=5)
        result = await registry.execute_tool(
            mcp="python", tool=tools[0].name, arguments={"code": "print(6 * 7)"}
        )

        self.assertEqual(McpServerState.CONNECTED, status.state)
        self.assertEqual(1, status.tool_count)
        self.assertEqual("run_python", tools[0].name)
        self.assertEqual("42\n", json.loads(result.content)["stdout"])

    async def test_failed_server_does_not_prevent_other_server_startup_and_can_retry(self) -> None:
        registry = McpRegistry(
            (
                McpServerConfig("bad", "bad server", "/missing/anna-command", (), 1),
                McpServerConfig(
                    "python",
                    "working server",
                    sys.executable,
                    ("-m", "agent.infrastructure.mcp.servers.local_python"),
                    5,
                ),
            )
        )
        self.addAsyncCleanup(registry.close_async)

        await registry.start_all_async()
        statuses = {status.id: status for status in registry.list_statuses()}

        self.assertEqual(McpServerState.FAILED, statuses["bad"].state)
        self.assertTrue(statuses["bad"].error)
        self.assertEqual(McpServerState.CONNECTED, statuses["python"].state)
        self.assertEqual(McpServerState.FAILED, (await registry.retry_async("bad")).state)

    async def test_search_mcp_returns_full_schema_without_binding_tools(self) -> None:
        registry = McpRegistry(
            (
                McpServerConfig(
                    "python",
                    "working server",
                    sys.executable,
                    ("-m", "agent.infrastructure.mcp.servers.local_python"),
                    5,
                ),
            )
        )
        self.addAsyncCleanup(registry.close_async)
        await registry.start_all_async()

        result = await SearchMcpTool(registry).aexecute(
            {"mcp": "python", "query": "run python code"},
            context=ToolContext(caller_session_id="s-1"),
        )
        payload = json.loads(result.content)

        self.assertTrue(payload["ok"])
        self.assertEqual("run_python", payload["tools"][0]["name"])
        self.assertIn("code", payload["tools"][0]["parameters"]["required"])
        self.assertEqual(MessageKind.MCP_TOOL_DEFINITION, result.kind)
        first = registry.search_tools(mcp="python", query="run python", limit=1)[0]
        second = registry.search_tools(mcp="python", query="run python", limit=1)[0]
        self.assertEqual(first.name, second.name)
        self.assertEqual(dict(first.parameters), dict(second.parameters))

    async def test_unknown_server_is_a_structured_search_error(self) -> None:
        payload = json.loads(
            (
                await SearchMcpTool(McpRegistry(())).aexecute(
                    {"mcp": "missing", "query": "anything"},
                    context=ToolContext(caller_session_id="s-1"),
                )
            ).content
        )
        self.assertFalse(payload["ok"])
        self.assertEqual("search_mcp_failed", payload["error"]["code"])

    async def test_http_server_negotiates_protocol_lists_and_calls_tool(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            payload = json.loads(request.content)
            method = payload["method"]
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "anna-rag", "version": "0.1.0"},
                }
            elif method == "notifications/initialized":
                return httpx.Response(202)
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "search_knowledge",
                            "description": "Search indexed knowledge",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        }
                    ]
                }
            else:
                self.assertEqual("tools/call", method)
                self.assertEqual("search_knowledge", payload["params"]["name"])
                result = {
                    "content": [{"type": "text", "text": '{"hits": []}'}],
                    "isError": False,
                }
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload.get("id"), "result": result},
            )

        config = McpServerConfig(
            id="rag",
            description="test RAG",
            command="",
            args=(),
            timeout_seconds=5,
            transport="http",
            url="http://rag:8200/mcp",
            headers=(("x-api-key", "test-key"),),
        )
        def client_factory(**kwargs):
            return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

        with patch(
            "agent.infrastructure.mcp.registry.httpx.AsyncClient",
            side_effect=client_factory,
        ) as factory:
            registry = McpRegistry((config,))
            self.addAsyncCleanup(registry.close_async)
            await registry.start_all_async()
            tools = registry.search_tools(mcp="rag", query="search knowledge", limit=5)
            result = await registry.execute_tool(
                mcp="rag", tool=tools[0].name, arguments={"query": "policy"}
            )

        self.assertEqual('{"hits": []}', result.content)
        self.assertEqual(4, len(requests))
        for request in requests:
            self.assertEqual("test-key", request.headers["x-api-key"])
        self.assertNotIn("MCP-Protocol-Version", requests[0].headers)
        for request in requests[1:]:
            self.assertEqual("2025-06-18", request.headers["MCP-Protocol-Version"])
        self.assertNotIn("Mcp-Session-Id", requests[-1].headers)
        factory.assert_called_once()
        self.assertEqual({"x-api-key": "test-key"}, factory.call_args.kwargs["headers"])
        self.assertFalse(factory.call_args.kwargs["trust_env"])

    async def test_json_rpc_tool_error_does_not_disconnect_server(self) -> None:
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            method = payload["method"]
            requests.append(method)
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "serverInfo": {"name": "mock", "version": "0.1.0"},
                }
            elif method == "notifications/initialized":
                return httpx.Response(202)
            elif method == "tools/list":
                result = {"tools": [{"name": "search", "inputSchema": {"type": "object"}}]}
            else:
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "error": {"code": -32602, "message": "bad arguments"},
                    },
                )
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload.get("id"), "result": result},
            )

        config = McpServerConfig("rag", "test", "", (), 5, "http", "http://rag/mcp")
        def client_factory(**kwargs):
            return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

        with patch(
            "agent.infrastructure.mcp.registry.httpx.AsyncClient",
            side_effect=client_factory,
        ):
            registry = McpRegistry((config,))
            self.addAsyncCleanup(registry.close_async)
            await registry.start_all_async()
            tool = registry.search_tools(mcp="rag", query="search", limit=1)[0]
            first = json.loads(
                (await registry.execute_tool(mcp="rag", tool=tool.name, arguments={})).content
            )
            second = json.loads(
                (await registry.execute_tool(mcp="rag", tool=tool.name, arguments={})).content
            )

        self.assertEqual("mcp_rpc_error", first["error"]["code"])
        self.assertEqual("mcp_rpc_error", second["error"]["code"])
        self.assertEqual(McpServerState.CONNECTED, registry.list_statuses()[0].state)
        self.assertEqual(2, requests.count("tools/call"))

    async def test_structured_content_is_kept_in_tool_text(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            method = payload["method"]
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "serverInfo": {"name": "mock", "version": "0.1.0"},
                }
            elif method == "notifications/initialized":
                return httpx.Response(202)
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "search",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                }
            else:
                hit = {
                    "document_id": "doc-1",
                    "document_name": "policy.md",
                    "chunk_index": 0,
                    "page_number": None,
                    "text": "expense policy",
                }
                result = {
                    "content": [{"type": "text", "text": json.dumps({"hits": [hit]})}],
                    "structuredContent": {"hits": [hit]},
                    "isError": False,
                }
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload.get("id"), "result": result},
            )

        config = McpServerConfig("rag", "test", "", (), 5, "http", "http://rag/mcp")
        def client_factory(**kwargs):
            return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

        with patch(
            "agent.infrastructure.mcp.registry.httpx.AsyncClient",
            side_effect=client_factory,
        ):
            registry = McpRegistry((config,))
            self.addAsyncCleanup(registry.close_async)
            await registry.start_all_async()
            tool = registry.search_tools(mcp="rag", query="search", limit=1)[0]
            result = await registry.execute_tool(mcp="rag", tool=tool.name, arguments={})

        self.assertIn("policy.md", result.content)
        self.assertIn('"hits"', result.content)


class LocalPythonTests(TestCase):
    def test_timeout_and_output_limit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            run_python("import time; time.sleep(1)", 0.1)

        import os
        previous = os.environ.get("LOCAL_PYTHON_MAX_OUTPUT_BYTES")
        os.environ["LOCAL_PYTHON_MAX_OUTPUT_BYTES"] = "4"
        try:
            result = run_python("print('123456')")
        finally:
            if previous is None:
                os.environ.pop("LOCAL_PYTHON_MAX_OUTPUT_BYTES", None)
            else:
                os.environ["LOCAL_PYTHON_MAX_OUTPUT_BYTES"] = previous
        self.assertEqual("1234", result["stdout"])
        self.assertTrue(result["stdout_truncated"])
        self.assertFalse(result["sandboxed"])
