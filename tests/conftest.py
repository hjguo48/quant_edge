from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import time

import pytest
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

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


@contextmanager
def _isolated_session_factory(
    db_engine: Engine,
    tables: Iterable[sa.Table],
) -> Iterator[sessionmaker]:
    connection = db_engine.connect()
    transaction = connection.begin()
    try:
        for table in tables:
            if db_engine.dialect.name == "postgresql":
                table.create(bind=db_engine, checkfirst=True)
                connection.execute(
                    text(
                        f"CREATE TEMP TABLE {table.name} "
                        f"(LIKE public.{table.name} INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES)"
                    ),
                )
            else:
                table.create(bind=db_engine, checkfirst=True)
                connection.execute(table.delete())
        yield sessionmaker(bind=connection, expire_on_commit=False)
    finally:
        transaction.rollback()
        connection.close()


@pytest.fixture
def isolated_session_factory(db_engine: Engine):
    return lambda tables: _isolated_session_factory(db_engine, tables)


@pytest.fixture
def short_sale_session_factory(isolated_session_factory):
    from src.data.finra_short_sale import ShortSaleVolume

    with isolated_session_factory([ShortSaleVolume.__table__]) as session_factory:
        yield session_factory
