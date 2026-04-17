"""Core ORM models; time-series tables require create_hypertable()."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB
try:
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()
    Base.__allow_unmapped__ = True

    class Mapped:
        @classmethod
        def __class_getitem__(cls, _item: Any) -> Any:
            return Any

    def mapped_column(*args: Any, **kwargs: Any) -> sa.Column[Any]:
        return sa.Column(*args, **kwargs)
else:
    class Base(DeclarativeBase):
        __allow_unmapped__ = True


class TimestampMixin:
    __allow_unmapped__ = True

    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
    )


class Stock(TimestampMixin, Base):
    __tablename__ = "stocks"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    company_name: Mapped[str] = mapped_column(String(200), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(50))
    industry: Mapped[str | None] = mapped_column(String(100))
    ipo_date: Mapped[date | None] = mapped_column(Date)
    delist_date: Mapped[date | None] = mapped_column(Date)
    delist_reason: Mapped[str | None] = mapped_column(String(50))
    shares_outstanding: Mapped[int | None] = mapped_column(BigInteger)


class StockPrice(Base):
    __tablename__ = "stock_prices"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    high: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    low: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    close: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    adj_close: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    volume: Mapped[int | None] = mapped_column(BigInteger)
    knowledge_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str | None] = mapped_column(String(20))


class StockMinuteAggs(Base):
    __tablename__ = "stock_minute_aggs"
    __table_args__ = (
        sa.Index("idx_stock_minute_aggs_trade_date", "trade_date"),
        sa.Index("idx_stock_minute_aggs_knowledge_time", "knowledge_time"),
    )

    # TimescaleDB hypertables require any unique constraint to include the
    # partition key, so the ORM identity is the `(ticker, minute_ts)` pair
    # rather than a standalone surrogate `id`.
    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    minute_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    id: Mapped[int | None] = mapped_column(BigInteger, sa.Identity(), nullable=False, index=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    high: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    low: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    close: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    volume: Mapped[int | None] = mapped_column(BigInteger)
    vwap: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    transactions: Mapped[int | None] = mapped_column(BigInteger)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    knowledge_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False)


class PriceReconciliationEvent(Base):
    __tablename__ = "price_reconciliation_events"
    __table_args__ = (
        sa.Index("idx_price_reconciliation_events_trade_date", "trade_date"),
        sa.Index("idx_price_reconciliation_events_severity", "severity"),
        sa.Index("idx_price_reconciliation_events_batch_id", "batch_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    field: Mapped[str] = mapped_column(String(20), nullable=False)
    stock_prices_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    minute_agg_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    delta_bp: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False)


class FundamentalsPIT(Base):
    __tablename__ = "fundamentals_pit"
    __table_args__ = (
        sa.UniqueConstraint(
            "ticker",
            "fiscal_period",
            "metric_name",
            "knowledge_time",
            name="uq_fundamentals_pit_version",
        ),
        sa.Index(
            "idx_fundamentals_pit_lookup",
            "ticker",
            "knowledge_time",
            "metric_name",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    fiscal_period: Mapped[str] = mapped_column(String(10), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 6))
    event_time: Mapped[date] = mapped_column(Date, nullable=False)
    knowledge_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_restated: Mapped[bool | None] = mapped_column(
        Boolean,
        server_default=sa.text("FALSE"),
    )
    source: Mapped[str | None] = mapped_column(String(20))


class UniverseMembership(Base):
    __tablename__ = "universe_membership"
    __table_args__ = (
        sa.UniqueConstraint(
            "ticker",
            "index_name",
            "effective_date",
            name="uq_universe_membership_entry",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    index_name: Mapped[str] = mapped_column(String(20), nullable=False)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date)
    reason: Mapped[str | None] = mapped_column(String(50))


class CorporateAction(Base):
    __tablename__ = "corporate_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    action_type: Mapped[str] = mapped_column(String(20), nullable=False)
    ex_date: Mapped[date] = mapped_column(Date, nullable=False)
    ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    old_ticker: Mapped[str | None] = mapped_column(String(10))
    new_ticker: Mapped[str | None] = mapped_column(String(10))
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class FeatureStore(Base):
    __tablename__ = "feature_store"
    __table_args__ = (
        sa.UniqueConstraint(
            "ticker",
            "calc_date",
            "feature_name",
            "batch_id",
            name="uq_feature_store_batch",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    calc_date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_name: Mapped[str] = mapped_column(String(50), nullable=False)
    feature_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    is_filled: Mapped[bool | None] = mapped_column(
        Boolean,
        server_default=sa.text("FALSE"),
    )
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    model_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)
    train_start: Mapped[date] = mapped_column(Date, nullable=False)
    train_end: Mapped[date] = mapped_column(Date, nullable=False)
    val_start: Mapped[date | None] = mapped_column(Date)
    val_end: Mapped[date | None] = mapped_column(Date)
    features_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    hyperparams_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metrics_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    status: Mapped[str | None] = mapped_column(
        String(20),
        server_default=sa.text("'staging'"),
    )
    mlflow_run_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=sa.func.now(),
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        sa.UniqueConstraint(
            "ticker",
            "signal_date",
            "model_version_id",
            name="uq_prediction_model_signal",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)
    model_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("model_registry.model_id"),
        nullable=False,
    )
    feature_batch_id: Mapped[str] = mapped_column(String(36), nullable=False)
    pred_score: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    pred_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    pred_decile: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))


class Portfolio(Base):
    __tablename__ = "portfolios"

    portfolio_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(36))
    strategy_name: Mapped[str | None] = mapped_column(String(100))
    weighting_scheme: Mapped[str | None] = mapped_column(String(20))
    config_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=sa.func.now(),
    )


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    backtest_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_version_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("model_registry.model_id"),
    )
    config_snapshot_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    metrics_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    equity_curve: Mapped[Any] = mapped_column(JSONB, nullable=False)
    stat_tests_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=sa.func.now(),
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    actor: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version_id: Mapped[str | None] = mapped_column(String(36))
    feature_batch_id: Mapped[str | None] = mapped_column(String(36))
    details_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


sa.Index("idx_audit_log_time", AuditLog.timestamp.desc())
