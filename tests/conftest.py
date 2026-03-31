from __future__ import annotations

import time

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.db.session import get_engine


@pytest.fixture(scope="session")
def db_engine() -> Engine:
    get_engine.cache_clear()
    engine = get_engine()
    deadline = time.time() + 60
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            break
        except Exception as exc:  # pragma: no cover - exercised only when services lag.
            last_error = exc
            time.sleep(2)
    else:
        raise RuntimeError("Database did not become ready within 60 seconds.") from last_error

    yield engine

    engine.dispose()
    get_engine.cache_clear()


@pytest.fixture
def db_connection(db_engine: Engine):
    with db_engine.connect() as connection:
        yield connection
