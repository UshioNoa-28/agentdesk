"""Web 异常处理器单测：状态码映射、错误体形状、内部信息不泄漏。"""

from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from agent.api.exception_handlers import (
    handle_application_error,
    handle_domain_error,
    handle_http_exception,
    handle_request_validation_error,
    handle_unexpected_error,
    register_exception_handlers,
)
from agent.domain.exceptions import DomainValidationError
from agent.exceptions import (
    AgentExecutionError,
    AgentInternalError,
    ApplicationError,
    DomainError,
    DomainStateError,
    SessionDeletionConflictError,
    SessionNotFoundError,
)


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/test",
            "scheme": "http",
            "server": ("testserver", 80),
            "headers": [],
            "query_string": b"",
        }
    )


async def _body(response) -> dict:
    return json.loads(response.body)


class ApplicationErrorHandlerTests(IsolatedAsyncioTestCase):
    async def test_status_code_mapping(self) -> None:
        cases = [
            (SessionNotFoundError("s1"), 404, "session_not_found"),
            (
                SessionDeletionConflictError(session_id="s1", reason="children"),
                409,
                "session_deletion_conflict",
            ),
            (AgentExecutionError("model exploded"), 502, "agent_execution_failed"),
            (AgentInternalError("graph bug"), 500, "agent_internal_error"),
            (ApplicationError("boom"), 500, "application_error"),
        ]
        for exception, status, code in cases:
            response = await handle_application_error(_request(), exception)
            body = await _body(response)
            self.assertEqual(status, response.status_code, code)
            self.assertEqual(code, body["error"]["code"])
            self.assertEqual(exception.message, body["error"]["message"])


class DomainErrorHandlerTests(IsolatedAsyncioTestCase):
    async def test_validation_error_maps_to_400(self) -> None:
        response = await handle_domain_error(_request(), DomainValidationError("bad input"))
        body = await _body(response)
        self.assertEqual(400, response.status_code)
        self.assertEqual("domain_validation_error", body["error"]["code"])
        self.assertEqual("bad input", body["error"]["message"])

    async def test_state_error_maps_to_409(self) -> None:
        response = await handle_domain_error(_request(), DomainStateError("not allowed"))
        body = await _body(response)
        self.assertEqual(409, response.status_code)
        self.assertEqual("domain_state_error", body["error"]["code"])


class RequestValidationErrorHandlerTests(IsolatedAsyncioTestCase):
    async def test_hides_user_input_but_keeps_error_shape(self) -> None:
        exception = RequestValidationError(
            [
                {
                    "type": "missing",
                    "loc": ("body", "title"),
                    "msg": "Field required",
                    "input": {"title": None, "secret": "hunter2"},
                }
            ],
            body={"title": None},
        )
        response = await handle_request_validation_error(_request(), exception)
        body = await _body(response)

        self.assertEqual(422, response.status_code)
        self.assertEqual("request_validation_error", body["error"]["code"])
        entry = body["error"]["details"]["errors"][0]
        self.assertNotIn("input", entry)
        self.assertEqual(["body", "title"], entry["loc"])
        self.assertEqual("Field required", entry["msg"])
        self.assertNotIn("hunter2", json.dumps(body))


class HttpExceptionHandlerTests(IsolatedAsyncioTestCase):
    async def test_preserves_status_and_detail(self) -> None:
        response = await handle_http_exception(
            _request(), StarletteHTTPException(404, "Route missing")
        )
        body = await _body(response)
        self.assertEqual(404, response.status_code)
        self.assertEqual("http_error", body["error"]["code"])
        self.assertEqual("Route missing", body["error"]["message"])

    async def test_mapping_detail_flattens_message_and_keeps_details(self) -> None:
        response = await handle_http_exception(
            _request(), StarletteHTTPException(418, {"message": "teapot", "extra": 1})
        )
        body = await _body(response)
        self.assertEqual(418, response.status_code)
        self.assertEqual("teapot", body["error"]["message"])
        self.assertEqual(1, body["error"]["details"]["extra"])


class UnexpectedErrorHandlerTests(IsolatedAsyncioTestCase):
    async def test_does_not_leak_internal_exception_text(self) -> None:
        response = await handle_unexpected_error(_request(), RuntimeError("secret-db-password"))
        body = await _body(response)

        self.assertEqual(500, response.status_code)
        self.assertEqual("internal_server_error", body["error"]["code"])
        self.assertEqual("Internal server error.", body["error"]["message"])
        self.assertNotIn("secret-db-password", response.body.decode("utf-8"))


class RegistrationTests(IsolatedAsyncioTestCase):
    def test_register_exception_handlers_wires_all_five(self) -> None:
        app = FastAPI()
        register_exception_handlers(app)
        for key in (
            ApplicationError,
            DomainError,
            RequestValidationError,
            StarletteHTTPException,
            Exception,
        ):
            self.assertIn(key, app.exception_handlers)
