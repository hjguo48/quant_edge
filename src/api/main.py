from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.deps import close_resources, init_resources
from src.api.middleware import RequestTimingMiddleware
from src.api.routers import ROUTERS
from src.api.schemas.common import HealthResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_resources()
    try:
        yield
    finally:
        await close_resources()


app = FastAPI(
    title="QuantEdge API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Compress responses > 1KB. Critical when accessed via WSL2 portproxy
# (large uncompressed responses tend to ERR_CONNECTION_RESET through NAT).
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(RequestTimingMiddleware)


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
    )


for router in ROUTERS:
    app.include_router(router)


# Serve frontend production build (dist/) from the same origin as /api/*.
# Eliminates CORS, vite proxy, and WSL2 portproxy multi-port complexity.
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_FRONTEND_DIST / "assets")),
        name="assets",
    )

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        """SPA fallback: serve index.html for any non-/api/* path."""
        candidate = _FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
