from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
import math

from loguru import logger
import numpy as np
import pandas as pd

from src.backtest.cost_model import AlmgrenChrissCostModel
from src.portfolio.black_litterman import black_litterman_portfolio
from src.portfolio.constraints import (
    PortfolioConstraints,
    apply_turnover_buffer,
    apply_weight_constraints,
)
from src.portfolio.equal_weight import equal_weight_portfolio
from src.portfolio.vol_weighted import vol_inverse_portfolio

SUPPORTED_WEIGHTING_SCHEMES = {
    "equal_weight",
    "vol_inverse",
    "black_litterman",
}


@dataclass(frozen=True)
class PortfolioPeriodResult:
    signal_date: str
    execution_date: str
    exit_date: str
    universe_size: int
    selected_count: int
    turnover: float
    cost_rate: float
    gross_return: float
    net_return: float
    benchmark_return: float
    gross_excess_return: float
    net_excess_return: float
    avg_gap: float
    cost_breakdown: dict[str, float]
    selected_tickers: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioBacktestResult:
    periods: list[PortfolioPeriodResult]
    gross_return: float
    net_return: float
    benchmark_return: float
    gross_excess_return: float
    net_excess_return: float
    annualized_gross_return: float
    annualized_net_return: float
    annualized_benchmark_return: float
    annualized_excess_gross: float
    annualized_excess_net: float
    total_cost_drag: float
    average_turnover: float
    cost_breakdown: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["periods"] = [period.to_dict() for period in self.periods]
        return payload


def prepare_execution_price_frame(prices_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume"}
    missing_columns = sorted(required_columns - set(prices_df.columns))
    if missing_columns:
        raise ValueError(f"prices_df is missing required columns: {missing_columns}")

    frame = prices_df.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])

    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.sort_values(["ticker", "trade_date"], inplace=True)
    frame["adjustment_factor"] = np.where(
        frame["close"].abs() > 1e-12,
        frame["adj_close"] / frame["close"],
        np.nan,
    )
    frame["adjustment_factor"] = (
        frame["adjustment_factor"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )
    frame["close_px"] = frame["adj_close"].fillna(frame["close"])
    frame["open_px"] = (frame["open"] * frame["adjustment_factor"]).fillna(frame["close_px"])
    frame["high_px"] = (frame["high"] * frame["adjustment_factor"]).fillna(frame["close_px"])
    frame["low_px"] = (frame["low"] * frame["adjustment_factor"]).fillna(frame["close_px"])
    frame["typical_price"] = (
        frame["open_px"] + frame["high_px"] + frame["low_px"] + frame["close_px"]
    ) / 4.0
    frame["execution_price"] = ((frame["open_px"] * 0.5) + (frame["typical_price"] * 0.5)).fillna(frame["close_px"])

    grouped = frame.groupby("ticker", sort=False)
    frame["prev_close"] = grouped["close_px"].shift(1)
    frame["open_gap"] = np.where(
        frame["prev_close"] > 0,
        (frame["open_px"] - frame["prev_close"]) / frame["prev_close"],
        0.0,
    )
    frame["daily_return"] = grouped["close_px"].pct_change()
    frame["sigma_20d"] = (
        grouped["daily_return"]
        .transform(lambda series: series.shift(1).rolling(20, min_periods=5).std())
        .fillna(0.02)
    )
    frame["adv_20d_shares"] = (
        grouped["volume"]
        .transform(lambda series: series.shift(1).rolling(20, min_periods=5).mean())
        .fillna(frame["volume"])
        .fillna(0.0)
    )
    frame["volume_ratio"] = np.where(
        frame["adv_20d_shares"] > 0,
        frame["volume"] / frame["adv_20d_shares"],
        1.0,
    )

    prepared = frame[
        [
            "ticker",
            "trade_date",
            "execution_price",
            "close_px",
            "open_px",
            "volume",
            "adv_20d_shares",
            "sigma_20d",
            "open_gap",
            "volume_ratio",
            "daily_return",
        ]
    ].copy()
    prepared.set_index(["trade_date", "ticker"], inplace=True)
    return prepared.sort_index()


def build_execution_schedule(
    signal_dates: Sequence[pd.Timestamp | date],
    available_trade_dates: Sequence[pd.Timestamp | date],
) -> dict[pd.Timestamp, pd.Timestamp]:
    signal_index = pd.DatetimeIndex(pd.to_datetime(signal_dates)).sort_values().unique()
    trade_index = pd.DatetimeIndex(pd.to_datetime(available_trade_dates)).sort_values().unique()
    schedule: dict[pd.Timestamp, pd.Timestamp] = {}

    for signal_date in signal_index:
        insert_at = trade_index.searchsorted(signal_date, side="right")
        if insert_at >= len(trade_index):
            continue
        schedule[pd.Timestamp(signal_date)] = pd.Timestamp(trade_index[insert_at])

    return schedule


def simulate_top_decile_portfolio(
    *,
    predictions: pd.Series,
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str = "SPY",
    universe_by_date: Mapping[pd.Timestamp, set[str]] | None = None,
    initial_capital: float = 1_000_000.0,
    min_external_universe_overlap: int = 100,
) -> PortfolioBacktestResult:
    return simulate_portfolio(
        predictions=predictions,
        prices=prices,
        cost_model=cost_model,
        weighting_scheme="equal_weight",
        benchmark_ticker=benchmark_ticker,
        universe_by_date=universe_by_date,
        initial_capital=initial_capital,
        selection_pct=0.10,
        sell_buffer_pct=None,
        min_trade_weight=0.0,
        min_external_universe_overlap=min_external_universe_overlap,
    )


def simulate_portfolio(
    *,
    predictions: pd.Series,
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    weighting_scheme: str,
    benchmark_ticker: str = "SPY",
    universe_by_date: Mapping[pd.Timestamp, set[str]] | None = None,
    initial_capital: float = 1_000_000.0,
    selection_pct: float = 0.10,
    sell_buffer_pct: float | None = None,
    min_trade_weight: float = 0.0,
    max_weight: float = 0.05,
    min_holdings: int = 20,
    min_external_universe_overlap: int = 100,
    bl_lookback_days: int = 60,
) -> PortfolioBacktestResult:
    if weighting_scheme not in SUPPORTED_WEIGHTING_SCHEMES:
        raise ValueError(f"Unsupported weighting scheme: {weighting_scheme}")

    if predictions.empty:
        return _empty_backtest_result()

    execution = prepare_execution_price_frame(prices)
    if execution.empty:
        raise RuntimeError("Execution price frame is empty.")

    returns_history = (
        execution["daily_return"]
        .unstack("ticker")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )

    benchmark = benchmark_ticker.upper()
    signal_dates = pd.DatetimeIndex(
        pd.to_datetime(predictions.index.get_level_values("trade_date")),
    ).sort_values().unique()
    trade_dates = execution.index.get_level_values("trade_date").unique().sort_values()
    schedule = build_execution_schedule(signal_dates, trade_dates)

    constraints = PortfolioConstraints(
        max_weight=max_weight,
        min_holdings=min_holdings,
        turnover_buffer=min_trade_weight,
    )

    current_weights: dict[str, float] = {}
    portfolio_value = float(initial_capital)
    periods: list[PortfolioPeriodResult] = []
    cost_totals = _empty_cost_breakdown()
    cumulative_gross = 1.0
    cumulative_net = 1.0
    cumulative_benchmark = 1.0
    universe_fallback_warnings = 0

    for signal_date, next_signal_date in zip(signal_dates[:-1], signal_dates[1:]):
        execution_date = schedule.get(pd.Timestamp(signal_date))
        exit_date = schedule.get(pd.Timestamp(next_signal_date))
        if execution_date is None or exit_date is None or exit_date <= execution_date:
            continue

        score_frame = predictions.xs(signal_date, level="trade_date").dropna().astype(float).sort_values(ascending=False)
        eligible = set(score_frame.index.astype(str))
        if universe_by_date is not None:
            external_universe = set(universe_by_date.get(pd.Timestamp(signal_date), set()))
            overlap = eligible & external_universe
            if len(overlap) >= min_external_universe_overlap:
                eligible = overlap
            elif external_universe and universe_fallback_warnings < 5:
                logger.warning(
                    "external universe overlap too small on {} (overlap={} candidates={}); using signal cross-section",
                    pd.Timestamp(signal_date).date(),
                    len(overlap),
                    len(eligible),
                )
                universe_fallback_warnings += 1

        if (execution_date, benchmark) not in execution.index or (exit_date, benchmark) not in execution.index:
            continue

        entry_slice = execution.xs(execution_date, level="trade_date")
        exit_slice = execution.xs(exit_date, level="trade_date")
        eligible &= set(entry_slice.index.astype(str)) & set(exit_slice.index.astype(str))
        eligible.discard(benchmark)
        if len(eligible) < max(min_holdings, 2):
            continue

        filtered_scores = score_frame.loc[score_frame.index.astype(str).isin(eligible)].sort_values(ascending=False)
        if filtered_scores.empty:
            continue

        ranking = filtered_scores.index.astype(str).tolist()
        candidate_tickers = select_candidate_tickers(
            ranking=ranking,
            current_weights=current_weights,
            selection_pct=selection_pct,
            sell_buffer_pct=sell_buffer_pct,
            min_holdings=min_holdings,
            max_weight=max_weight,
        )
        candidate_scores = filtered_scores.reindex(candidate_tickers).dropna()
        if candidate_scores.empty:
            continue

        target_weights = build_target_weights(
            weighting_scheme=weighting_scheme,
            scores=candidate_scores,
            entry_slice=entry_slice,
            returns_history=returns_history.loc[:execution_date].iloc[:-1],
            constraints=constraints,
            bl_lookback_days=bl_lookback_days,
        )
        if min_trade_weight > 0.0:
            buffer_reference_weights = {
                ticker: weight
                for ticker, weight in current_weights.items()
                if ticker in ranking
            }
            target_weights = apply_turnover_buffer(
                target_weights,
                current_weights=buffer_reference_weights,
                min_trade_weight=min_trade_weight,
                ranking=ranking,
                constraints=constraints,
            )
        else:
            target_weights = apply_weight_constraints(
                target_weights,
                ranking=ranking,
                constraints=constraints,
            )

        if not target_weights:
            continue

        previous_weights = current_weights.copy()
        selected_tickers = list(target_weights)

        period_costs = _empty_cost_breakdown()
        all_trade_tickers = set(previous_weights) | set(target_weights)
        for ticker in all_trade_tickers:
            delta_weight = target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)
            if math.isclose(delta_weight, 0.0, abs_tol=1e-12):
                continue
            bar = entry_slice.loc[ticker]
            order_notional = abs(delta_weight) * portfolio_value
            order_shares = order_notional / max(float(bar["execution_price"]), 1e-12)
            estimate = cost_model.estimate_trade(
                order_shares=order_shares,
                execution_price=float(bar["execution_price"]),
                sigma_20d=float(bar["sigma_20d"]),
                adv_20d_shares=float(bar["adv_20d_shares"]),
                open_gap=float(bar["open_gap"]),
                execution_volume_ratio=float(bar["volume_ratio"]),
            )
            period_costs["commission"] += estimate.commission_cost
            period_costs["spread"] += estimate.spread_cost
            period_costs["temporary_impact"] += estimate.temporary_cost
            period_costs["permanent_impact"] += estimate.permanent_cost
            period_costs["gap_penalty"] += estimate.gap_cost
            period_costs["total"] += estimate.total_cost

        cost_rate = period_costs["total"] / portfolio_value if portfolio_value else 0.0

        asset_returns: dict[str, float] = {}
        gross_return = 0.0
        gaps: list[float] = []
        for ticker, weight in target_weights.items():
            entry_bar = entry_slice.loc[ticker]
            exit_bar = exit_slice.loc[ticker]
            entry_price = max(float(entry_bar["execution_price"]), 1e-12)
            exit_price = float(exit_bar["execution_price"])
            realized = (exit_price / entry_price) - 1.0
            asset_returns[ticker] = realized
            gross_return += weight * realized
            gaps.append(abs(float(entry_bar["open_gap"])))

        bench_entry = entry_slice.loc[benchmark]
        bench_exit = exit_slice.loc[benchmark]
        benchmark_return = (
            float(bench_exit["execution_price"]) / max(float(bench_entry["execution_price"]), 1e-12)
        ) - 1.0
        net_return = ((1.0 - cost_rate) * (1.0 + gross_return)) - 1.0

        cumulative_gross *= 1.0 + gross_return
        cumulative_net *= 1.0 + net_return
        cumulative_benchmark *= 1.0 + benchmark_return
        portfolio_value *= 1.0 + net_return

        drift_denominator = 1.0 + gross_return
        if math.isclose(drift_denominator, 0.0, abs_tol=1e-12):
            current_weights = {}
        else:
            current_weights = {
                ticker: float((weight * (1.0 + asset_returns[ticker])) / drift_denominator)
                for ticker, weight in target_weights.items()
                if (weight * (1.0 + asset_returns[ticker])) > 1e-8
            }

        turnover = 0.5 * sum(
            abs(target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0))
            for ticker in set(target_weights) | set(previous_weights)
        )
        period = PortfolioPeriodResult(
            signal_date=pd.Timestamp(signal_date).date().isoformat(),
            execution_date=pd.Timestamp(execution_date).date().isoformat(),
            exit_date=pd.Timestamp(exit_date).date().isoformat(),
            universe_size=int(len(filtered_scores)),
            selected_count=int(len(selected_tickers)),
            turnover=float(turnover),
            cost_rate=float(cost_rate),
            gross_return=float(gross_return),
            net_return=float(net_return),
            benchmark_return=float(benchmark_return),
            gross_excess_return=float(gross_return - benchmark_return),
            net_excess_return=float(net_return - benchmark_return),
            avg_gap=float(np.mean(gaps)) if gaps else 0.0,
            cost_breakdown={key: float(value) for key, value in period_costs.items()},
            selected_tickers=selected_tickers,
        )
        periods.append(period)

        for key, value in period_costs.items():
            cost_totals[key] += float(value)

    if not periods:
        return PortfolioBacktestResult(
            periods=[],
            gross_return=0.0,
            net_return=0.0,
            benchmark_return=0.0,
            gross_excess_return=0.0,
            net_excess_return=0.0,
            annualized_gross_return=0.0,
            annualized_net_return=0.0,
            annualized_benchmark_return=0.0,
            annualized_excess_gross=0.0,
            annualized_excess_net=0.0,
            total_cost_drag=0.0,
            average_turnover=0.0,
            cost_breakdown=cost_totals,
        )

    start_date = pd.Timestamp(periods[0].execution_date)
    end_date = pd.Timestamp(periods[-1].exit_date)
    total_days = max((end_date - start_date).days, 1)

    gross_return_total = cumulative_gross - 1.0
    net_return_total = cumulative_net - 1.0
    benchmark_total = cumulative_benchmark - 1.0
    annualized_gross = _annualize(total_return=gross_return_total, total_days=total_days)
    annualized_net = _annualize(total_return=net_return_total, total_days=total_days)
    annualized_benchmark = _annualize(total_return=benchmark_total, total_days=total_days)

    return PortfolioBacktestResult(
        periods=periods,
        gross_return=float(gross_return_total),
        net_return=float(net_return_total),
        benchmark_return=float(benchmark_total),
        gross_excess_return=float(gross_return_total - benchmark_total),
        net_excess_return=float(net_return_total - benchmark_total),
        annualized_gross_return=float(annualized_gross),
        annualized_net_return=float(annualized_net),
        annualized_benchmark_return=float(annualized_benchmark),
        annualized_excess_gross=float(annualized_gross - annualized_benchmark),
        annualized_excess_net=float(annualized_net - annualized_benchmark),
        total_cost_drag=float(annualized_gross - annualized_net),
        average_turnover=float(np.mean([period.turnover for period in periods])),
        cost_breakdown={key: float(value) for key, value in cost_totals.items()},
    )


def select_candidate_tickers(
    *,
    ranking: Sequence[str],
    current_weights: Mapping[str, float],
    selection_pct: float,
    sell_buffer_pct: float | None,
    min_holdings: int,
    max_weight: float,
) -> list[str]:
    if not ranking:
        return []

    min_by_cap = int(np.ceil(1.0 / max_weight)) if max_weight > 0.0 else 1
    entry_count = max(min_holdings, min_by_cap, int(math.ceil(len(ranking) * selection_pct)))
    entry_count = min(len(ranking), entry_count)
    selected = list(ranking[:entry_count])

    if sell_buffer_pct is not None and current_weights:
        sell_count = max(entry_count, int(math.ceil(len(ranking) * sell_buffer_pct)))
        sell_count = min(len(ranking), sell_count)
        sell_zone = set(ranking[:sell_count])
        retained = [ticker for ticker in current_weights if ticker in sell_zone]
        for ticker in retained:
            if ticker not in selected:
                selected.append(ticker)

    return selected


def build_target_weights(
    *,
    weighting_scheme: str,
    scores: pd.Series,
    entry_slice: pd.DataFrame,
    returns_history: pd.DataFrame,
    constraints: PortfolioConstraints,
    bl_lookback_days: int,
) -> dict[str, float]:
    tickers = scores.index.astype(str)
    if weighting_scheme == "equal_weight":
        return equal_weight_portfolio(
            scores,
            n_stocks=len(scores),
            selection_pct=1.0,
            constraints=constraints,
        )

    if weighting_scheme == "vol_inverse":
        sigma = entry_slice.loc[tickers, "sigma_20d"]
        return vol_inverse_portfolio(
            scores,
            volatilities=sigma,
            n_stocks=len(scores),
            selection_pct=1.0,
            constraints=constraints,
        )

    if weighting_scheme == "black_litterman":
        trailing_returns = returns_history.reindex(columns=tickers).tail(bl_lookback_days)
        dollar_liquidity = (
            entry_slice.loc[tickers, "adv_20d_shares"]
            * entry_slice.loc[tickers, "execution_price"]
        )
        return black_litterman_portfolio(
            scores,
            trailing_returns=trailing_returns,
            dollar_liquidity=dollar_liquidity,
            n_stocks=len(scores),
            selection_pct=1.0,
            constraints=constraints,
            lookback_days=bl_lookback_days,
        )

    raise ValueError(f"Unsupported weighting scheme: {weighting_scheme}")


def _annualize(*, total_return: float, total_days: int) -> float:
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    return float(base ** (365.25 / max(total_days, 1)) - 1.0)


def _empty_cost_breakdown() -> dict[str, float]:
    return {
        "commission": 0.0,
        "spread": 0.0,
        "temporary_impact": 0.0,
        "permanent_impact": 0.0,
        "gap_penalty": 0.0,
        "total": 0.0,
    }


def _empty_backtest_result() -> PortfolioBacktestResult:
    return PortfolioBacktestResult(
        periods=[],
        gross_return=0.0,
        net_return=0.0,
        benchmark_return=0.0,
        gross_excess_return=0.0,
        net_excess_return=0.0,
        annualized_gross_return=0.0,
        annualized_net_return=0.0,
        annualized_benchmark_return=0.0,
        annualized_excess_gross=0.0,
        annualized_excess_net=0.0,
        total_cost_drag=0.0,
        average_turnover=0.0,
        cost_breakdown=_empty_cost_breakdown(),
    )
