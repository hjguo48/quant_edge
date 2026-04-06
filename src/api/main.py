from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
app.add_middleware(RequestTimingMiddleware)


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
    )


for router in ROUTERS:
    app.include_router(router)
