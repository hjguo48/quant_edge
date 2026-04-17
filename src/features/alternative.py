"""Alternative data features — earnings, analyst, short interest, insider trading.

All features are PIT-safe: only use data where knowledge_time <= as_of.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.data.db.session import get_engine
from src.data.db.pit import get_fundamentals_pit
from src.features.sector import load_sector_map_pit


def _as_of_utc(as_of: date) -> datetime:
    return datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(converted) or np.isinf(converted):
        return None
    return converted


def _current_price_from_prices(prices_df: pd.DataFrame | None, ticker: str, as_of: date) -> float | None:
    if prices_df is None or prices_df.empty:
        return None
    frame = prices_df.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    sub = frame.loc[
        (frame["ticker"] == ticker.upper()) & (frame["trade_date"] <= as_of),
        ["trade_date", "close"],
    ].sort_values("trade_date")
    if sub.empty:
        return None
    return _safe_float(sub.iloc[-1]["close"])


def _market_cap_for_ticker(ticker: str, as_of: date, current_price: float | None) -> float | None:
    if current_price is None or current_price <= 0:
        return None
    history = get_fundamentals_pit(
        ticker=ticker,
        as_of=as_of,
        metric_names=("weighted_average_shares_outstanding",),
    )
    if history.empty:
        return None
    shares = pd.to_numeric(
        history.loc[history["metric_name"] == "weighted_average_shares_outstanding", "metric_value"],
        errors="coerce",
    ).dropna()
    if shares.empty:
        return None
    latest_shares = float(shares.iloc[-1])
    if latest_shares <= 0:
        return None
    return current_price * latest_shares


def _role_weight(type_of_owner: str | None) -> float:
    normalized = str(type_of_owner or "").strip().lower()
    if any(token in normalized for token in ("chief executive", "ceo", "chief financial", "cfo")):
        return 1.5
    if "director" in normalized and "officer" not in normalized:
        return 0.75
    return 1.0


def _is_ceo_cfo_owner(type_of_owner: str | None) -> bool:
    normalized = str(type_of_owner or "").strip().lower()
    return any(token in normalized for token in ("chief executive", "ceo", "chief financial", "cfo"))


def _is_director_or_officer_owner(type_of_owner: str | None) -> bool:
    normalized = str(type_of_owner or "").strip().lower()
    return ("director" in normalized) or ("officer" in normalized)


# ---------------------------------------------------------------------------
# Earnings Surprise Features
# ---------------------------------------------------------------------------

def compute_earnings_surprise(
    ticker: str,
    as_of: date,
    *,
    lookback_quarters: int = 4,
) -> dict[str, float]:
    """Compute earnings surprise features from earnings_estimates table.

    Returns:
        earnings_surprise_latest: (actual - estimated) / |estimated| for most recent quarter
        earnings_surprise_avg_4q: average surprise over last 4 quarters
        earnings_beat_streak: consecutive quarters of positive surprise
        earnings_surprise_recency: latest surprise with a 30-day exponential decay
        earnings_beat_recency: beat streak with the same 30-day exponential decay
        earnings_surprise_recency_20d: latest surprise with a 20-day exponential decay
        earnings_beat_recency_30d: beat streak with a 30-day exponential decay
        surprise_flip_qoq: latest surprise minus the prior quarter surprise
        surprise_vs_history: latest surprise minus the recent 4-quarter average
    """
    engine = get_engine()
    as_of_dt = _as_of_utc(as_of)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT fiscal_date, eps_actual, eps_estimated, revenue_actual, revenue_estimated
            FROM earnings_estimates
            WHERE ticker = :ticker
              AND knowledge_time <= :as_of
              AND eps_actual IS NOT NULL
              AND eps_estimated IS NOT NULL
              AND eps_estimated != 0
            ORDER BY fiscal_date DESC
            LIMIT :limit
        """), {"ticker": ticker, "as_of": as_of_dt, "limit": lookback_quarters}).fetchall()

    result = {
        "earnings_surprise_latest": np.nan,
        "earnings_surprise_avg_4q": np.nan,
        "earnings_beat_streak": np.nan,
        "earnings_surprise_recency": np.nan,
        "earnings_beat_recency": np.nan,
        "earnings_surprise_recency_20d": np.nan,
        "earnings_beat_recency_30d": np.nan,
        "surprise_flip_qoq": np.nan,
        "surprise_vs_history": np.nan,
    }

    if not rows:
        return result

    surprises = []
    for row in rows:
        surprise = float(row[1] - row[2]) / abs(float(row[2]))
        surprises.append(surprise)

    result["earnings_surprise_latest"] = surprises[0]
    result["earnings_surprise_avg_4q"] = float(np.mean(surprises))

    streak = 0
    for s in surprises:
        if s > 0:
            streak += 1
        else:
            break
    result["earnings_beat_streak"] = float(streak)
    latest_fiscal_date = rows[0][0]
    if latest_fiscal_date is not None:
        fiscal_date = pd.to_datetime(latest_fiscal_date).date()
        days_since = max((as_of - fiscal_date).days, 0)
        recency_decay_30d = float(np.exp(-days_since / 30.0))
        recency_decay_20d = float(np.exp(-days_since / 20.0))
        result["earnings_surprise_recency"] = result["earnings_surprise_latest"] * recency_decay_30d
        result["earnings_beat_recency"] = result["earnings_beat_streak"] * recency_decay_30d
        result["earnings_surprise_recency_20d"] = result["earnings_surprise_latest"] * recency_decay_20d
        result["earnings_beat_recency_30d"] = result["earnings_beat_streak"] * recency_decay_30d
    if len(surprises) >= 2:
        result["surprise_flip_qoq"] = surprises[0] - surprises[1]
    result["surprise_vs_history"] = result["earnings_surprise_latest"] - result["earnings_surprise_avg_4q"]

    return result


# ---------------------------------------------------------------------------
# Analyst Estimate Revision Features
# ---------------------------------------------------------------------------

def compute_earnings_revision(
    ticker: str,
    as_of: date,
) -> dict[str, float]:
    """Compute analyst estimate revision features.

    Returns:
        eps_revision_direction: +1 if latest EPS estimate > prior, -1 if <, 0 if same/unavailable
        revenue_revision_pct: (latest revenue_avg - prior revenue_avg) / prior
        analyst_coverage: number of analysts covering (numAnalystsEps)
    """
    engine = get_engine()
    as_of_dt = datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT fiscal_date, eps_avg, revenue_avg, num_analysts_eps
            FROM analyst_estimates
            WHERE ticker = :ticker
              AND knowledge_time <= :as_of
              AND period = 'quarter'
            ORDER BY fiscal_date DESC
            LIMIT 2
        """), {"ticker": ticker, "as_of": as_of_dt}).fetchall()

    result = {
        "eps_revision_direction": 0.0,
        "revenue_revision_pct": np.nan,
        "analyst_coverage": np.nan,
    }

    if not rows:
        return result

    latest = rows[0]
    result["analyst_coverage"] = float(latest[3]) if latest[3] is not None else np.nan

    if len(rows) >= 2:
        prior = rows[1]
        if latest[1] is not None and prior[1] is not None and float(prior[1]) != 0:
            diff = float(latest[1]) - float(prior[1])
            result["eps_revision_direction"] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        if latest[2] is not None and prior[2] is not None and prior[2] != 0:
            result["revenue_revision_pct"] = (float(latest[2]) - float(prior[2])) / abs(float(prior[2]))

    return result


# ---------------------------------------------------------------------------
# Short Interest Features
# ---------------------------------------------------------------------------

def compute_short_interest_features(
    ticker: str,
    as_of: date,
    *,
    prices_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute short interest features from short_interest table.

    Returns:
        short_interest_ratio: short_interest / avg_daily_volume (= days_to_cover)
        short_interest_change: (current - prior) / prior
    """
    engine = get_engine()
    as_of_dt = _as_of_utc(as_of)
    lookback_start = as_of - timedelta(days=365)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT settlement_date, short_interest, avg_daily_volume, days_to_cover
            FROM short_interest
            WHERE ticker = :ticker
              AND knowledge_time <= :as_of
              AND settlement_date >= :start_date
            ORDER BY settlement_date DESC
        """), {"ticker": ticker, "as_of": as_of_dt, "start_date": lookback_start}).fetchall()

    result = {
        "short_interest_ratio": np.nan,
        "short_interest_change": np.nan,
        "short_interest_sector_rel": np.nan,
        "short_interest_change_20d": np.nan,
        "short_interest_abnormal_1y": np.nan,
        "short_squeeze_setup": np.nan,
        "crowding_unwind_risk": np.nan,
    }

    if not rows:
        return result

    latest = rows[0]
    if latest[3] is not None:
        result["short_interest_ratio"] = float(latest[3])
    elif latest[1] is not None and latest[2] is not None and latest[2] > 0:
        result["short_interest_ratio"] = float(latest[1]) / float(latest[2])

    if len(rows) >= 2:
        prior = rows[1]
        if latest[1] is not None and prior[1] is not None and prior[1] > 0:
            result["short_interest_change"] = (float(latest[1]) - float(prior[1])) / float(prior[1])
            result["short_interest_change_20d"] = result["short_interest_change"]
    history_ratios = []
    for settlement_date, short_interest, avg_daily_volume, days_to_cover in rows:
        if days_to_cover is not None:
            history_ratios.append(float(days_to_cover))
        elif short_interest is not None and avg_daily_volume is not None and float(avg_daily_volume) > 0:
            history_ratios.append(float(short_interest) / float(avg_daily_volume))
    if history_ratios:
        latest_ratio = history_ratios[0]
        median = float(np.nanmedian(history_ratios))
        std = float(np.nanstd(history_ratios))
        if std > 0:
            result["short_interest_abnormal_1y"] = (latest_ratio - median) / std
    current_sector = load_sector_map_pit(as_of).get(ticker.upper())
    latest_settlement = latest[0]
    if current_sector and latest_settlement is not None:
        with engine.connect() as conn:
            sector_rows = conn.execute(
                text(
                    """
                    with latest_rows as (
                        select distinct on (ticker)
                            ticker,
                            settlement_date,
                            short_interest,
                            avg_daily_volume,
                            days_to_cover
                        from short_interest
                        where settlement_date = :settlement_date
                          and knowledge_time <= :as_of
                        order by ticker, knowledge_time desc
                    )
                    select lr.ticker, lr.short_interest, lr.avg_daily_volume, lr.days_to_cover
                    from latest_rows lr
                    join stocks s on s.ticker = lr.ticker
                    where s.sector = :sector
                    """,
                ),
                {
                    "settlement_date": latest_settlement,
                    "as_of": as_of_dt,
                    "sector": current_sector,
                },
            ).fetchall()
        sector_values = []
        for _, short_interest, avg_daily_volume, days_to_cover in sector_rows:
            if days_to_cover is not None:
                sector_values.append(float(days_to_cover))
            elif short_interest is not None and avg_daily_volume is not None and float(avg_daily_volume) > 0:
                sector_values.append(float(short_interest) / float(avg_daily_volume))
        if sector_values and pd.notna(result["short_interest_ratio"]):
            sector_std = float(np.nanstd(sector_values))
            if sector_std > 0:
                result["short_interest_sector_rel"] = (
                    float(result["short_interest_ratio"]) - float(np.nanmean(sector_values))
                ) / sector_std
    current_price = _current_price_from_prices(prices_df, ticker, as_of)
    ret_5d = np.nan
    ret_20d = np.nan
    volume_surge = np.nan
    if prices_df is not None and not prices_df.empty:
        sub = prices_df.loc[prices_df["ticker"].astype(str).str.upper() == ticker.upper()].copy()
        if not sub.empty:
            sub["trade_date"] = pd.to_datetime(sub["trade_date"]).dt.date
            sub.sort_values("trade_date", inplace=True)
            sub["close"] = pd.to_numeric(sub["close"], errors="coerce")
            sub["volume"] = pd.to_numeric(sub["volume"], errors="coerce")
            sub["ret_5d"] = sub["close"].pct_change(5)
            sub["ret_20d"] = sub["close"].pct_change(20)
            sub["volume_surge"] = sub["volume"] / sub["volume"].shift(1).rolling(20, min_periods=20).mean()
            latest_row = sub.loc[sub["trade_date"] <= as_of].tail(1)
            if not latest_row.empty:
                ret_5d = float(latest_row["ret_5d"].iloc[0]) if pd.notna(latest_row["ret_5d"].iloc[0]) else np.nan
                ret_20d = float(latest_row["ret_20d"].iloc[0]) if pd.notna(latest_row["ret_20d"].iloc[0]) else np.nan
                volume_surge = (
                    float(latest_row["volume_surge"].iloc[0])
                    if pd.notna(latest_row["volume_surge"].iloc[0])
                    else np.nan
                )
    if pd.notna(result["short_interest_sector_rel"]):
        result["short_squeeze_setup"] = (
            float(result["short_interest_sector_rel"]) * max(0.0, 0.0 if pd.isna(ret_5d) else ret_5d)
            * (1.0 if pd.isna(volume_surge) else volume_surge)
        )
        result["crowding_unwind_risk"] = (
            float(result["short_interest_sector_rel"]) * min(0.0, 0.0 if pd.isna(ret_20d) else ret_20d)
        )

    return result


# ---------------------------------------------------------------------------
# Insider Trading Features
# ---------------------------------------------------------------------------

def compute_insider_features(
    ticker: str,
    as_of: date,
    *,
    lookback_days: int = 90,
    prices_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute insider trading features from insider_trades table.

    Returns:
        insider_net_buy_ratio: (buy_count - sell_count) / total_count over lookback
        insider_buy_value: total dollar value of purchases
        insider_cluster_buy: 1 if >= 3 distinct insiders bought, else 0
    """
    engine = get_engine()
    as_of_dt = _as_of_utc(as_of)
    lookback_start = as_of - timedelta(days=730)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT filing_date, transaction_type, securities_transacted, price, reporting_cik,
                   acquisition_or_disposition, type_of_owner
            FROM insider_trades
            WHERE ticker = :ticker
              AND knowledge_time <= :as_of
              AND filing_date >= :start_date
        """), {"ticker": ticker, "as_of": as_of_dt, "start_date": lookback_start}).fetchall()

    result = {
        "insider_net_buy_ratio": np.nan,
        "insider_buy_value": np.nan,
        "insider_cluster_buy": 0.0,
        "insider_buy_intensity_20d": np.nan,
        "insider_net_intensity_60d": np.nan,
        "insider_cluster_buy_30d_w": np.nan,
        "insider_abnormal_buy_90d": np.nan,
        "insider_role_skew_30d": np.nan,
    }

    if not rows:
        return result

    buys = 0
    sells = 0
    buy_value = 0.0
    buy_insiders = set()
    buy_value_20d = 0.0
    buy_value_60d = 0.0
    sell_value_60d = 0.0
    cluster_weight_30d = 0.0
    ceo_cfo_buy_30d = 0.0
    director_officer_sell_30d = 0.0
    buy_value_90d = 0.0
    historical_annual_buy_intensities: list[float] = []
    current_price = _current_price_from_prices(prices_df, ticker, as_of)
    market_cap = _market_cap_for_ticker(ticker, as_of, current_price)

    for filing_date, tx_type, shares, price, cik, acq_disp, type_of_owner in rows:
        tx_type = tx_type or ""
        acq_disp = acq_disp or ""
        filing_date = pd.to_datetime(filing_date).date()
        shares_value = float(shares) if shares is not None else 0.0
        price_value = float(price) if price is not None else 0.0
        transaction_value = shares_value * price_value
        age = max((as_of - filing_date).days, 0)
        role_weight = _role_weight(type_of_owner)

        is_purchase = "P-Purchase" in tx_type or (acq_disp == "A" and "Award" not in tx_type)
        is_sale = "S-Sale" in tx_type or acq_disp == "D"

        if is_purchase:
            buys += 1
            buy_value += transaction_value
            buy_insiders.add(cik or "")
            if age <= 20:
                buy_value_20d += transaction_value * role_weight * float(np.exp(-age / 20.0))
            if age <= 60:
                buy_value_60d += transaction_value * role_weight * float(np.exp(-age / 60.0))
            if age <= 30:
                cluster_weight_30d += float(np.exp(-age / 30.0))
                if _is_ceo_cfo_owner(type_of_owner):
                    ceo_cfo_buy_30d += transaction_value * role_weight * float(np.exp(-age / 30.0))
            if age <= 90:
                buy_value_90d += transaction_value * role_weight
        elif is_sale:
            sells += 1
            if age <= 60:
                sell_value_60d += transaction_value * role_weight * float(np.exp(-age / 60.0))
            if age <= 30 and _is_director_or_officer_owner(type_of_owner):
                director_officer_sell_30d += transaction_value * role_weight * float(np.exp(-age / 30.0))

    total = buys + sells
    if total > 0:
        result["insider_net_buy_ratio"] = (buys - sells) / total
    result["insider_buy_value"] = buy_value
    result["insider_cluster_buy"] = 1.0 if len(buy_insiders) >= 3 else 0.0
    if market_cap is not None and market_cap > 0:
        result["insider_buy_intensity_20d"] = buy_value_20d / market_cap
        result["insider_net_intensity_60d"] = (buy_value_60d - sell_value_60d) / market_cap
        result["insider_abnormal_buy_90d"] = np.nan
        annual_bins: dict[int, float] = {}
        for filing_date, tx_type, shares, price, _, acq_disp, type_of_owner in rows:
            tx_type = tx_type or ""
            acq_disp = acq_disp or ""
            if not ("P-Purchase" in tx_type or (acq_disp == "A" and "Award" not in tx_type)):
                continue
            filing_year = pd.to_datetime(filing_date).date().year
            role_weight = _role_weight(type_of_owner)
            annual_bins[filing_year] = annual_bins.get(filing_year, 0.0) + (
                (float(shares) if shares is not None else 0.0)
                * (float(price) if price is not None else 0.0)
                * role_weight
            ) / market_cap
        historical_annual_buy_intensities = [value for year, value in sorted(annual_bins.items())]
        if historical_annual_buy_intensities:
            baseline = float(np.nanmedian(historical_annual_buy_intensities))
            if baseline > 0:
                current_intensity_90d = buy_value_90d / market_cap
                result["insider_abnormal_buy_90d"] = current_intensity_90d / baseline
    result["insider_cluster_buy_30d_w"] = cluster_weight_30d
    result["insider_role_skew_30d"] = ceo_cfo_buy_30d - director_officer_sell_30d

    return result


# ---------------------------------------------------------------------------
# Price-based Daily Features
# ---------------------------------------------------------------------------

def compute_daily_features(
    prices_df: pd.DataFrame,
    ticker: str,
    as_of: date,
) -> dict[str, float]:
    """Compute daily price-based features.

    Args:
        prices_df: DataFrame with columns [ticker, trade_date, open, close, volume]

    Returns:
        overnight_gap: (open[T] - close[T-1]) / close[T-1]
        volume_surge: volume[T] / rolling_20d_avg_volume
    """
    result = {
        "overnight_gap": np.nan,
        "volume_surge": np.nan,
    }

    mask = (prices_df["ticker"] == ticker) & (prices_df["trade_date"] <= as_of)
    sub = prices_df.loc[mask].sort_values("trade_date").tail(25)

    if len(sub) < 2:
        return result

    latest = sub.iloc[-1]
    prev = sub.iloc[-2]

    # Overnight gap
    prev_close = float(prev["close"]) if pd.notna(prev["close"]) else None
    curr_open = float(latest["open"]) if pd.notna(latest["open"]) else None
    if prev_close and prev_close != 0 and curr_open:
        result["overnight_gap"] = (curr_open - prev_close) / prev_close

    # Volume surge
    curr_vol = float(latest["volume"]) if pd.notna(latest["volume"]) else None
    if len(sub) >= 21 and curr_vol:
        avg_vol = sub["volume"].iloc[-21:-1].mean()
        if avg_vol and avg_vol > 0:
            result["volume_surge"] = curr_vol / avg_vol

    return result


def compute_sec_filing_features(
    ticker: str,
    as_of: date,
) -> dict[str, float]:
    """Compute PIT-safe SEC filing metadata features."""
    engine = get_engine()
    as_of_dt = _as_of_utc(as_of)
    lookback_start = as_of - timedelta(days=60)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                select filing_date, accepted_date, form_type
                from sec_filings
                where ticker = :ticker
                  and knowledge_time <= :as_of
                  and accepted_date <= :as_of
                  and filing_date >= :lookback_start
                order by accepted_date desc
                """,
            ),
            {
                "ticker": ticker,
                "as_of": as_of_dt,
                "lookback_start": lookback_start,
            },
        ).fetchall()

    result = {
        "days_since_last_8k": np.nan,
        "days_since_last_10q": np.nan,
        "days_since_last_10k": np.nan,
        "recent_8k_count_5d": 0.0,
        "recent_8k_count_20d": 0.0,
        "recent_8k_count_60d": 0.0,
        "has_recent_8k_5d": 0.0,
        "filing_burst_20d": 0.0,
    }
    if not rows:
        return result

    latest_8k: date | None = None
    latest_10q: date | None = None
    latest_10k: date | None = None
    for filing_date, accepted_date, form_type in rows:
        filing_day = pd.to_datetime(filing_date or accepted_date).date()
        normalized_form = str(form_type or "").upper()
        age = max((as_of - filing_day).days, 0)
        if normalized_form.startswith("8-K"):
            result["recent_8k_count_60d"] += 1.0
            if age <= 20:
                result["recent_8k_count_20d"] += 1.0
            if age <= 5:
                result["recent_8k_count_5d"] += 1.0
                result["has_recent_8k_5d"] = 1.0
            if latest_8k is None:
                latest_8k = filing_day
        if normalized_form.startswith("10-Q") and latest_10q is None:
            latest_10q = filing_day
        if normalized_form.startswith("10-K") and latest_10k is None:
            latest_10k = filing_day
        if age <= 20:
            result["filing_burst_20d"] += 1.0

    if latest_8k is not None:
        result["days_since_last_8k"] = float(max((as_of - latest_8k).days, 0))
    if latest_10q is not None:
        result["days_since_last_10q"] = float(max((as_of - latest_10q).days, 0))
    if latest_10k is not None:
        result["days_since_last_10k"] = float(max((as_of - latest_10k).days, 0))
    return result


# ---------------------------------------------------------------------------
# Price Target Features
# ---------------------------------------------------------------------------

def compute_price_target_features(
    ticker: str,
    as_of: date,
    current_price: float | None = None,
) -> dict[str, float]:
    """Compute price target upside from FMP price-target-consensus.

    Note: This queries live API since we don't store historical consensus snapshots.
    For backtesting, this feature may not be available and should return NaN.
    """
    result = {
        "price_target_upside": np.nan,
    }

    if current_price is None or current_price <= 0:
        return result

    try:
        from src.config import settings
        import requests

        r = requests.get(
            f"https://financialmodelingprep.com/stable/price-target-consensus",
            params={"symbol": ticker, "apikey": settings.FMP_API_KEY},
            timeout=10,
        )
        if r.ok:
            data = r.json()
            if isinstance(data, list) and data:
                consensus = data[0].get("targetConsensus")
                if consensus and consensus > 0:
                    result["price_target_upside"] = (consensus - current_price) / current_price
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Batch Feature Computation (for IC screening / walk-forward)
# ---------------------------------------------------------------------------

ALTERNATIVE_FEATURE_NAMES = [
    "earnings_surprise_latest",
    "earnings_surprise_avg_4q",
    "earnings_beat_streak",
    "earnings_surprise_recency",
    "earnings_beat_recency",
    "earnings_surprise_recency_20d",
    "earnings_beat_recency_30d",
    "surprise_flip_qoq",
    "surprise_vs_history",
    "pead_setup",
    "eps_revision_direction",
    "revenue_revision_pct",
    "analyst_coverage",
    "short_interest_ratio",
    "short_interest_change",
    "short_interest_sector_rel",
    "short_interest_change_20d",
    "short_interest_abnormal_1y",
    "short_squeeze_setup",
    "crowding_unwind_risk",
    "insider_net_buy_ratio",
    "insider_buy_value",
    "insider_cluster_buy",
    "insider_buy_intensity_20d",
    "insider_net_intensity_60d",
    "insider_cluster_buy_30d_w",
    "insider_abnormal_buy_90d",
    "insider_role_skew_30d",
    "days_since_last_8k",
    "days_since_last_10q",
    "days_since_last_10k",
    "recent_8k_count_5d",
    "recent_8k_count_20d",
    "recent_8k_count_60d",
    "has_recent_8k_5d",
    "filing_burst_20d",
    "overnight_gap",
    "volume_surge",
]


def compute_alternative_features_for_ticker(
    ticker: str,
    as_of: date,
    prices_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute all alternative features for a single ticker at a point in time."""
    features = {}
    features.update(compute_earnings_surprise(ticker, as_of))
    features.update(compute_earnings_revision(ticker, as_of))
    features.update(compute_short_interest_features(ticker, as_of, prices_df=prices_df))
    features.update(compute_insider_features(ticker, as_of, prices_df=prices_df))
    features.update(compute_sec_filing_features(ticker, as_of))

    if prices_df is not None:
        features.update(compute_daily_features(prices_df, ticker, as_of))
    else:
        features["overnight_gap"] = np.nan
        features["volume_surge"] = np.nan

    if pd.notna(features.get("earnings_surprise_recency_20d")):
        prices = prices_df if prices_df is not None else pd.DataFrame()
        ret_5d = np.nan
        if not prices.empty:
            sub = prices.loc[prices["ticker"].astype(str).str.upper() == ticker.upper()].copy()
            if not sub.empty:
                sub["trade_date"] = pd.to_datetime(sub["trade_date"]).dt.date
                sub.sort_values("trade_date", inplace=True)
                sub["close"] = pd.to_numeric(sub["close"], errors="coerce")
                sub["ret_5d"] = sub["close"].pct_change(5)
                latest_row = sub.loc[sub["trade_date"] <= as_of].tail(1)
                if not latest_row.empty and pd.notna(latest_row["ret_5d"].iloc[0]):
                    ret_5d = float(latest_row["ret_5d"].iloc[0])
        features["pead_setup"] = float(features["earnings_surprise_recency_20d"]) * max(0.0, 0.0 if pd.isna(ret_5d) else ret_5d)
    else:
        features["pead_setup"] = np.nan

    return features
