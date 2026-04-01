from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
from loguru import logger

from src.data.db.pit import get_fundamentals_pit

FUNDAMENTAL_FEATURE_NAMES = (
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "ev_ebitda",
    "fcf_yield",
    "dividend_yield",
    "roe",
    "roa",
    "gross_margin",
    "operating_margin",
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "debt_to_equity",
    "current_ratio",
    "eps_surprise",
)

_PIT_METRIC_NAMES = (
    "eps",
    "weighted_average_shares_outstanding",
    "book_value_per_share",
    "revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_debt",
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    "ebitda",
    "cash",
    "cash_and_cash_equivalents",
    "annual_dividend",
    "dividend_per_share",
    "gross_profit",
    "operating_income",
    "current_assets",
    "current_liabilities",
    "consensus_eps",
    "eps_consensus",
)


def compute_fundamental_features(
    ticker: str,
    as_of: date | datetime,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    if prices_df.empty:
        return _empty_feature_frame()

    max_as_of = _coerce_as_of_date(as_of)
    prepared_prices = prices_df.copy()
    prepared_prices["ticker"] = prepared_prices["ticker"].astype(str).str.upper()
    prepared_prices["trade_date"] = pd.to_datetime(prepared_prices["trade_date"]).dt.date
    prepared_prices["close"] = pd.to_numeric(prepared_prices["close"], errors="coerce")
    prepared_prices = prepared_prices.loc[
        (prepared_prices["ticker"] == ticker.upper()) & (prepared_prices["trade_date"] <= max_as_of)
    ].sort_values("trade_date")
    if prepared_prices.empty:
        return _empty_feature_frame()

    pit_cache: dict[date, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for price_row in prepared_prices[["trade_date", "close"]].itertuples(index=False):
        trade_date = price_row.trade_date
        pit_frame = pit_cache.get(trade_date)
        if pit_frame is None:
            pit_frame = get_fundamentals_pit(
                ticker=ticker.upper(),
                as_of=trade_date,
                metric_names=_PIT_METRIC_NAMES,
            )
            pit_cache[trade_date] = pit_frame

        features = _calculate_feature_snapshot(
            pit_frame=pit_frame,
            price=float(price_row.close) if not pd.isna(price_row.close) else np.nan,
        )
        for feature_name, feature_value in features.items():
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "trade_date": trade_date,
                    "feature_name": feature_name,
                    "feature_value": feature_value,
                },
            )

    feature_frame = pd.DataFrame(rows)
    logger.info(
        "computed {} PIT fundamental feature rows for {} across {} dates",
        len(feature_frame),
        ticker.upper(),
        prepared_prices["trade_date"].nunique(),
    )
    return feature_frame


def _calculate_feature_snapshot(
    *,
    pit_frame: pd.DataFrame,
    price: float,
) -> dict[str, float]:
    features = {feature_name: np.nan for feature_name in FUNDAMENTAL_FEATURE_NAMES}
    if pit_frame.empty:
        return features

    history = _build_pit_history(pit_frame)
    if history.empty:
        return features

    latest = history.iloc[-1]
    shares_outstanding = _latest_metric(history, "weighted_average_shares_outstanding")
    market_cap = _market_cap(price, shares_outstanding)
    equity = _safe_subtract(latest.get("total_assets"), latest.get("total_liabilities"))
    revenue_ttm = _ttm(history, "revenue")
    eps_ttm = _ttm(history, "eps")
    operating_cash_flow_ttm = _ttm(history, "operating_cash_flow")
    free_cash_flow_ttm = _free_cash_flow_ttm(history, operating_cash_flow_ttm)
    ebitda_ttm = _ttm(history, "ebitda")
    dividend_per_share = _first_non_nan(latest.get("annual_dividend"), latest.get("dividend_per_share"))
    if pd.notna(dividend_per_share) and pd.isna(latest.get("annual_dividend")):
        dividend_per_share = dividend_per_share * 4
    cash = _first_non_nan(latest.get("cash"), latest.get("cash_and_cash_equivalents"))
    consensus_eps = _first_non_nan(latest.get("consensus_eps"), latest.get("eps_consensus"))
    total_debt = _first_non_nan(latest.get("total_debt"), latest.get("total_liabilities"))

    revenue_per_share = (
        revenue_ttm / shares_outstanding
        if pd.notna(revenue_ttm) and shares_outstanding is not None and shares_outstanding > 0
        else np.nan
    )

    features["pe_ratio"] = _safe_divide(price, eps_ttm)
    features["pb_ratio"] = _safe_divide(price, latest.get("book_value_per_share"))
    features["ps_ratio"] = _safe_divide(price, revenue_per_share)
    enterprise_value = _safe_add(market_cap, total_debt)
    enterprise_value = _safe_subtract(enterprise_value, cash)
    features["ev_ebitda"] = _safe_divide(enterprise_value, ebitda_ttm)
    features["fcf_yield"] = _safe_divide(free_cash_flow_ttm, market_cap)
    features["dividend_yield"] = _safe_divide(dividend_per_share, price)
    features["roe"] = _safe_divide(latest.get("net_income"), equity)
    features["roa"] = _safe_divide(latest.get("net_income"), latest.get("total_assets"))
    features["gross_margin"] = _safe_divide(latest.get("gross_profit"), latest.get("revenue"))
    features["operating_margin"] = _safe_divide(latest.get("operating_income"), latest.get("revenue"))
    features["revenue_growth_yoy"] = _yoy_growth(history, "revenue")
    features["earnings_growth_yoy"] = _yoy_growth(history, "net_income")
    features["debt_to_equity"] = _safe_divide(total_debt, equity)
    features["current_ratio"] = _safe_divide(latest.get("current_assets"), latest.get("current_liabilities"))
    eps_surprise_denom = abs(consensus_eps) if pd.notna(consensus_eps) else np.nan
    features["eps_surprise"] = (
        _safe_divide(latest.get("eps") - consensus_eps, eps_surprise_denom)
        if pd.notna(latest.get("eps")) and pd.notna(consensus_eps)
        else np.nan
    )

    return features


def _build_pit_history(pit_frame: pd.DataFrame) -> pd.DataFrame:
    history = pit_frame.copy()
    history["event_time"] = pd.to_datetime(history["event_time"]).dt.date
    history["metric_value"] = pd.to_numeric(history["metric_value"], errors="coerce")
    wide = (
        history.pivot_table(
            index=["fiscal_period", "event_time"],
            columns="metric_name",
            values="metric_value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["event_time", "fiscal_period"])
        .reset_index(drop=True)
    )
    return wide


def _ttm(history: pd.DataFrame, metric_name: str) -> float:
    if metric_name not in history.columns or len(history) < 4:
        return np.nan
    tail = pd.to_numeric(history[metric_name], errors="coerce").tail(4)
    if tail.count() < 4:
        return np.nan
    return float(tail.sum())


def _yoy_growth(history: pd.DataFrame, metric_name: str) -> float:
    if metric_name not in history.columns or len(history) < 5:
        return np.nan
    current = pd.to_numeric(history[metric_name], errors="coerce").iloc[-1]
    prior = pd.to_numeric(history[metric_name], errors="coerce").iloc[-5]
    if pd.isna(current) or pd.isna(prior) or prior == 0:
        return np.nan
    return float((current - prior) / abs(prior))


def _latest_metric(history: pd.DataFrame, metric_name: str) -> float:
    if metric_name not in history.columns:
        return np.nan
    values = pd.to_numeric(history[metric_name], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def _free_cash_flow_ttm(history: pd.DataFrame, operating_cash_flow_ttm: float) -> float:
    direct_fcf_ttm = _ttm(history, "free_cash_flow")
    if pd.notna(direct_fcf_ttm):
        return direct_fcf_ttm
    if pd.isna(operating_cash_flow_ttm):
        return np.nan

    capital_expenditure_ttm = _ttm(history, "capital_expenditure")
    if pd.isna(capital_expenditure_ttm):
        return np.nan
    if capital_expenditure_ttm <= 0:
        return float(operating_cash_flow_ttm + capital_expenditure_ttm)
    return float(operating_cash_flow_ttm - capital_expenditure_ttm)


def _market_cap(price: float, shares_outstanding: float | None) -> float:
    if pd.isna(price) or shares_outstanding is None or shares_outstanding <= 0:
        return np.nan
    return float(price * shares_outstanding)


def _safe_divide(numerator: float | int | None, denominator: float | int | None) -> float:
    if numerator is None or denominator is None:
        return np.nan
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _safe_add(left: float | int | None, right: float | int | None) -> float:
    left_value = np.nan if left is None or pd.isna(left) else float(left)
    right_value = np.nan if right is None or pd.isna(right) else float(right)
    if pd.isna(left_value) or pd.isna(right_value):
        return np.nan
    return float(left_value + right_value)


def _safe_subtract(left: float | int | None, right: float | int | None) -> float:
    left_value = np.nan if left is None or pd.isna(left) else float(left)
    right_value = np.nan if right is None or pd.isna(right) else float(right)
    if pd.isna(left_value) or pd.isna(right_value):
        return np.nan
    return float(left_value - right_value)


def _first_non_nan(*values: object) -> float:
    for value in values:
        if value is not None and not pd.isna(value):
            return float(value)
    return np.nan


def _coerce_as_of_date(as_of: date | datetime) -> date:
    if isinstance(as_of, datetime):
        return as_of.date()
    return as_of


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
