from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

import httpx

from agent.infrastructure.model.provider_http import AsyncProviderHttpClient


class ProviderWireLogTests(IsolatedAsyncioTestCase):
    """模型发给外部 Provider 的最终 HTTP 内容必须可读。"""

    async def test_request_and_response_are_written_with_secret_headers_redacted(self) -> None:
        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "provider.jsonl"

            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    json={"id": "chatcmpl-test", "choices": []},
                    request=request,
                )

            client = AsyncProviderHttpClient(
                wire_log_path=log_path,
                trust_env=False,
                transport=httpx.MockTransport(handler),
                headers={"Authorization": "Bearer secret", "X-Debug": "yes"},
            )
            try:
                response = await client.post(
                    "https://relay.example/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "1+1"}],
                    },
                )
            finally:
                await client.aclose()

            self.assertEqual(200, response.status_code)
            record = json.loads(log_path.read_text(encoding="utf-8").strip())

        self.assertEqual("relay.example", record["provider"])
        self.assertEqual("POST", record["method"])
        self.assertEqual(
            "https://relay.example/v1/chat/completions",
            record["url"],
        )
        self.assertEqual(
            {"model": "test-model", "messages": [{"role": "user", "content": "1+1"}]},
            json.loads(record["request"]["body"]),
        )
        self.assertEqual("**REDACTED**", record["request"]["headers"]["authorization"])
        self.assertEqual("yes", record["request"]["headers"]["x-debug"])
        self.assertEqual(
            {"id": "chatcmpl-test", "choices": []},
            json.loads(record["response"]["body"]),
        )
