from __future__ import annotations

import logging
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent.exceptions import (
    ApplicationError,
    ConflictError,
    DomainError,
    DomainStateError,
    ExternalServiceError,
    ResourceNotFoundError,
)

logger = logging.getLogger(__name__)


def register_exception_handlers(application: FastAPI) -> None:
    """把所有入口异常处理器注册到 FastAPI 应用。

    Args:
        application (FastAPI): 要安装统一错误响应策略的应用实例。
    """

    application.add_exception_handler(ApplicationError, handle_application_error)
    application.add_exception_handler(DomainError, handle_domain_error)
    application.add_exception_handler(RequestValidationError, handle_request_validation_error)
    application.add_exception_handler(StarletteHTTPException, handle_http_exception)
    application.add_exception_handler(Exception, handle_unexpected_error)


async def handle_application_error(
    _: Request,
    exception: ApplicationError,
) -> JSONResponse:
    """处理应用服务主动报告的业务异常。

    Args:
        _ (Request): 当前 HTTP 请求；这里只用于 FastAPI handler 签名。
        exception (ApplicationError): 应用层可安全报告的异常。

    Returns:
        JSONResponse: 按异常类别映射状态码的统一 error body。
    """

    if isinstance(exception, ExternalServiceError):
        _log_exception("External dependency failed", exception)
    else:
        logger.warning("Application use case failed: %s", exception.message)
    return _error_response(
        status_code=_application_status_code(exception),
        code=exception.code,
        message=exception.message,
        details=exception.details,
    )


async def handle_domain_error(_: Request, exception: DomainError) -> JSONResponse:
    """处理领域对象拒绝非法输入或状态变更。

    Args:
        _ (Request): 当前 HTTP 请求。
        exception (DomainError): 领域校验或状态异常。

    Returns:
        JSONResponse: 400/409 的统一错误响应。
    """

    status_code = 409 if isinstance(exception, DomainStateError) else 400
    return _error_response(
        status_code=status_code,
        code=exception.code,
        message=exception.message,
    )


async def handle_request_validation_error(
    _: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    """把 Pydantic 请求校验错误统一成项目错误格式。

    Args:
        _ (Request): 当前 HTTP 请求。
        exception (RequestValidationError): FastAPI/Pydantic 校验错误。

    Returns:
        JSONResponse: HTTP 422 和可序列化 errors 详情。
    """

    return _error_response(
        status_code=422,
        code="request_validation_error",
        message="Request validation failed.",
        details={"errors": jsonable_encoder(_sanitize_validation_errors(exception.errors()))},
    )


async def handle_http_exception(
    _: Request,
    exception: StarletteHTTPException,
) -> JSONResponse:
    """统一处理路由不存在等 Starlette/FastAPI HTTP 异常。

    Args:
        _ (Request): 当前 HTTP 请求。
        exception (StarletteHTTPException): Starlette HTTP 异常。

    Returns:
        JSONResponse: 保留原状态码的统一 error body。
    """

    detail = exception.detail
    if isinstance(detail, Mapping):
        message = str(detail.get("message", "HTTP request failed."))
        details = detail
    else:
        message = str(detail)
        details = None
    return _error_response(
        status_code=exception.status_code,
        code="http_error",
        message=message,
        details=details,
    )


async def handle_unexpected_error(request: Request, exception: Exception) -> JSONResponse:
    """兜底处理未分类异常，不把内部堆栈泄漏给客户端。

    Args:
        request (Request): 当前请求，用于日志定位。
        exception (Exception): 未分类内部异常。

    Returns:
        JSONResponse: HTTP 500 的通用错误响应。
    """

    _log_exception(
        f"Unhandled exception: {request.method} {request.url.path}",
        exception,
    )
    return _error_response(
        status_code=500,
        code="internal_server_error",
        message="Internal server error.",
    )


def _sanitize_validation_errors(
    errors: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    """去掉 Pydantic 错误条目里的 ``input``，不回显用户原始提交内容。"""

    return [
        {key: value for key, value in error.items() if key != "input"}
        for error in errors
    ]


def _application_status_code(exception: ApplicationError) -> int:
    """把应用异常映射为 HTTP 状态码，避免应用层依赖 HTTP。

    Args:
        exception (ApplicationError): 应用层异常及其稳定 code。

    Returns:
        int: 对外响应的 HTTP 状态码。
    """

    if isinstance(exception, ResourceNotFoundError):
        return 404
    if isinstance(exception, ConflictError):
        return 409
    if isinstance(exception, ExternalServiceError):
        return 502
    return 500


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: Mapping[str, object] | None = None,
) -> JSONResponse:
    """生成统一错误响应。

    Args:
        status_code (int): HTTP 状态码。
        code (str): 稳定应用错误码。
        message (str): 面向调用方的错误文本。
        details (Mapping[str, object] | None): 可选结构化详情。

    Returns:
        JSONResponse: ``{"error": ...}`` 形状的响应。
    """

    error_body: dict[str, object] = {
        "code": code,
        "message": message,
    }
    if details:
        error_body["details"] = dict(details)
    return JSONResponse(
        status_code=status_code,
        content={"error": error_body},
    )


def _log_exception(message: str, exception: BaseException) -> None:
    """保留服务端堆栈，同时避免把底层异常文本返回给客户端。

    Args:
        message (str): 日志上下文说明。
        exception (BaseException): 要记录 traceback 的原始异常。
    """

    logger.error(
        message,
        exc_info=(type(exception), exception, exception.__traceback__),
    )
