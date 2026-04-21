from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import subprocess
import sys

from alembic import command
from alembic.config import Config
import pytest
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from src.data.polygon_trades import stable_sequence_fallback


ROOT_DIR = Path(__file__).resolve().parents[2]
ALEMBIC_INI = ROOT_DIR / "alembic.ini"


def _alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))


def _sample_row(
    *,
    ticker: str = "AAPL",
    exchange: int | None = 4,
    sequence_number: int = 1,
    sip_timestamp: datetime | None = None,
    price: Decimal = Decimal("100.000000"),
    size: Decimal = Decimal("100.0000"),
    trade_id: str | None = "T1",
) -> dict:
    ts = sip_timestamp or datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    row = {
        "ticker": ticker,
        "trading_date": date(2026, 1, 5),
        "sip_timestamp": ts,
        "participant_timestamp": ts,
        "trf_timestamp": None,
        "knowledge_time": datetime(2026, 1, 5, 14, 45, tzinfo=timezone.utc),
        "price": price,
        "size": size,
        "decimal_size": None,
        "tape": 1,
        "conditions": [0],
        "correction": 0,
        "sequence_number": sequence_number,
        "trade_id": trade_id,
        "trf_id": None,
        "sampled_reason": "earnings",
    }
    if exchange is not None:
        row["exchange"] = exchange
    return row


def _insert_trade(engine: Engine, row: dict) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                insert into stock_trades_sampled (
                    ticker, trading_date, sip_timestamp, participant_timestamp, trf_timestamp,
                    knowledge_time, price, size, decimal_size, exchange, tape, conditions,
                    correction, sequence_number, trade_id, trf_id, sampled_reason
                )
                values (
                    :ticker, :trading_date, :sip_timestamp, :participant_timestamp, :trf_timestamp,
                    :knowledge_time, :price, :size, :decimal_size, :exchange, :tape, :conditions,
                    :correction, :sequence_number, :trade_id, :trf_id, :sampled_reason
                )
                """,
            ),
            row,
        )


def test_migration_007_upgrade_downgrade_reupgrade_and_constraints(db_engine: Engine) -> None:
    cfg = _alembic_config()
    command.downgrade(cfg, "006")
    command.upgrade(cfg, "007")

    inspector = inspect(db_engine)
    assert "stock_trades_sampled" in inspector.get_table_names()
    assert "trades_sampling_state" in inspector.get_table_names()
    columns = {column["name"]: column for column in inspector.get_columns("stock_trades_sampled")}
    assert {"sip_timestamp", "participant_timestamp", "trf_timestamp", "trf_id"} <= set(columns)
    assert columns["exchange"]["nullable"] is False
    assert columns["sequence_number"]["nullable"] is False
    pk_columns = inspector.get_pk_constraint("stock_trades_sampled")["constrained_columns"]
    assert pk_columns == ["ticker", "sip_timestamp", "exchange", "sequence_number"]

    base_ts = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    _insert_trade(db_engine, _sample_row(sip_timestamp=base_ts, exchange=4, sequence_number=100))
    with pytest.raises(IntegrityError):
        _insert_trade(db_engine, _sample_row(sip_timestamp=base_ts, exchange=4, sequence_number=100))

    _insert_trade(
        db_engine,
        _sample_row(
            sip_timestamp=base_ts,
            exchange=5,
            sequence_number=100,
            trade_id="T1-cross-venue",
        ),
    )

    row_without_exchange = _sample_row(
        sip_timestamp=datetime(2026, 1, 5, 14, 31, tzinfo=timezone.utc),
        exchange=None,
        sequence_number=101,
        trade_id="missing-exchange",
    )
    with db_engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                insert into stock_trades_sampled (
                    ticker, trading_date, sip_timestamp, participant_timestamp, trf_timestamp,
                    knowledge_time, price, size, decimal_size, tape, conditions, correction,
                    sequence_number, trade_id, trf_id, sampled_reason
                )
                values (
                    :ticker, :trading_date, :sip_timestamp, :participant_timestamp, :trf_timestamp,
                    :knowledge_time, :price, :size, :decimal_size, :tape, :conditions, :correction,
                    :sequence_number, :trade_id, :trf_id, :sampled_reason
                )
                """,
            ),
            row_without_exchange,
        )
    with db_engine.connect() as conn:
        exchange = conn.execute(
            sa.text("select exchange from stock_trades_sampled where trade_id = 'missing-exchange'"),
        ).scalar_one()
    assert exchange == -1

    fallback_a = stable_sequence_fallback(
        trade_id=None,
        price=Decimal("101.000000"),
        size=Decimal("100.0000"),
        participant_timestamp_ns=1_767_624_060_000_000_000,
        conditions=(0,),
    )
    fallback_b = stable_sequence_fallback(
        trade_id=None,
        price=Decimal("101.010000"),
        size=Decimal("100.0000"),
        participant_timestamp_ns=1_767_624_060_000_000_000,
        conditions=(0,),
    )
    assert fallback_a < 0
    assert fallback_b < 0
    assert fallback_a != fallback_b
    _insert_trade(
        db_engine,
        _sample_row(
            sip_timestamp=datetime(2026, 1, 5, 14, 32, tzinfo=timezone.utc),
            exchange=4,
            sequence_number=fallback_a,
            price=Decimal("101.000000"),
            trade_id=None,
        ),
    )
    _insert_trade(
        db_engine,
        _sample_row(
            sip_timestamp=datetime(2026, 1, 5, 14, 32, tzinfo=timezone.utc),
            exchange=4,
            sequence_number=fallback_b,
            price=Decimal("101.010000"),
            trade_id=None,
        ),
    )

    with db_engine.connect() as conn:
        count = conn.execute(sa.text("select count(*) from stock_trades_sampled")).scalar_one()
    assert count == 5

    command.downgrade(cfg, "006")
    inspector = inspect(db_engine)
    assert "stock_trades_sampled" not in inspector.get_table_names()
    assert "trades_sampling_state" not in inspector.get_table_names()

    command.upgrade(cfg, "007")
    inspector = inspect(db_engine)
    assert "stock_trades_sampled" in inspector.get_table_names()
    assert "trades_sampling_state" in inspector.get_table_names()


def test_stable_sequence_fallback_is_stable_across_python_hash_seeds() -> None:
    code = """
import json
from decimal import Decimal
from src.data.polygon_trades import stable_sequence_fallback
print(json.dumps({
    "value": stable_sequence_fallback(
        trade_id="abc",
        price=Decimal("123.456789"),
        size=Decimal("321.1234"),
        participant_timestamp_ns=1767624060000000000,
        conditions=(0, 37),
    )
}))
"""
    values: set[int] = set()
    for seed in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT_DIR)}
        output = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
        values.add(json.loads(output)["value"])
    assert len(values) == 1
    (value,) = values
    assert value < 0
