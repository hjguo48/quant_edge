from __future__ import annotations

from collections.abc import AsyncGenerator

from loguru import logger
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from src.config import Settings, get_settings as _get_settings

_async_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None
_redis_client: Redis | None = None


def get_settings() -> Settings:
    return _get_settings()


def init_resources() -> None:
    global _async_engine, _async_session_factory, _redis_client

    if _async_engine is None or _async_session_factory is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.async_database_url,
            pool_pre_ping=True,
        )
        _async_session_factory = async_sessionmaker(
            bind=_async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("initialized async database engine for QuantEdge API")

    if _redis_client is None:
        settings = get_settings()
        _redis_client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
        )
        logger.info("initialized Redis client for QuantEdge API")


async def close_resources() -> None:
    global _async_engine, _async_session_factory, _redis_client

    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("closed Redis client for QuantEdge API")

    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
        logger.info("disposed async database engine for QuantEdge API")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    if _async_session_factory is None:
        init_resources()
    if _async_session_factory is None:
        raise RuntimeError("Async database session factory is not available.")

    async with _async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_redis() -> Redis:
    if _redis_client is None:
        init_resources()
    if _redis_client is None:
        raise RuntimeError("Redis client is not available.")
    return _redis_client
