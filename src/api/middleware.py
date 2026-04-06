from __future__ import annotations

from time import perf_counter

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        started = perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = perf_counter() - started
            status_code = response.status_code if response is not None else 500
            if response is not None:
                response.headers["X-Process-Time"] = f"{elapsed:.6f}"
            logger.info(
                "{} {} -> {} ({:.4f}s)",
                request.method,
                request.url.path,
                status_code,
                elapsed,
            )
