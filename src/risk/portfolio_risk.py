from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mstats


@dataclass(frozen=True)
class ConstraintAuditEntry:
    priority: str
    rule_name: str
    triggered: bool
    action: str
    message: str
    before: dict[str, Any]
    after: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstrainedPortfolio:
    weights: dict[str, float]
    cash_weight: float
    gross_exposure: float
    holding_count: int
    turnover: float
    portfolio_beta: float | None
    cvar_99: float | None
    sector_weights: dict[str, float]
    warnings: list[str]
    audit_trail: list[ConstraintAuditEntry]
    beta_contributions: dict[str, float] = field(default_factory=dict)
    top_beta_contributors: list[dict[str, float | str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "cash_weight": float(self.cash_weight),
            "gross_exposure": float(self.gross_exposure),
            "holding_count": int(self.holding_count),
            "turnover": float(self.turnover),
            "portfolio_beta": None if self.portfolio_beta is None else float(self.portfolio_beta),
            "cvar_99": None if self.cvar_99 is None else float(self.cvar_99),
            "sector_weights": dict(self.sector_weights),
            "warnings": list(self.warnings),
            "audit_trail": [entry.to_dict() for entry in self.audit_trail],
            "beta_contributions": dict(self.beta_contributions),
            "top_beta_contributors": [dict(item) for item in self.top_beta_contributors],
        }


class PortfolioRiskEngine:
    """Layer 3: Ordered portfolio constraint engine with audit trail."""

    def apply_all_constraints(
        self,
        *,
        weights: Mapping[str, float] | pd.Series,
        benchmark_weights: Mapping[str, float] | pd.Series | None = None,
        sector_map: Mapping[str, str] | None = None,
        beta_map: Mapping[str, float] | None = None,
        return_history: pd.DataFrame | None = None,
        spy_returns: pd.Series | None = None,
        current_weights: Mapping[str, float] | pd.Series | None = None,
        candidate_ranking: Sequence[str] | pd.Index | None = None,
        max_single_stock_weight: float = 0.10,
        max_sector_deviation: float = 0.15,
        cvar_confidence: float = 0.99,
        cvar_floor: float = -0.05,
        beta_hard_bounds: tuple[float, float] = (0.7, 1.3),
        beta_target_bounds: tuple[float, float] = (0.8, 1.2),
        correlation_threshold: float = 0.85,
        turnover_cap: float = 0.40,
        min_holdings: int = 20,
        stress_test_shock: float = -0.10,
        stress_warning_threshold: float = -0.12,
    ) -> ConstrainedPortfolio:
        target = _coerce_weights(weights, normalize_to=1.0)
        current = _coerce_weights(current_weights, normalize_to=None)
        benchmark = _coerce_weights(benchmark_weights, normalize_to=1.0)
        sectors = _normalize_sector_map(sector_map or {})
        ranking = _normalize_ranking(candidate_ranking, fallback=list(target.sort_values(ascending=False).index))
        beta_lookup = _resolve_beta_map(beta_map=beta_map, return_history=return_history, spy_returns=spy_returns)

        audit: list[ConstraintAuditEntry] = []
        warnings: list[str] = []

        if target.empty:
            return ConstrainedPortfolio(
                weights={},
                cash_weight=1.0,
                gross_exposure=0.0,
                holding_count=0,
                turnover=0.0,
                portfolio_beta=None,
                cvar_99=None,
                sector_weights={},
                warnings=["Target portfolio was empty; no constraints were applied."],
                audit_trail=[],
            )

        # P0: single-name cap.
        before_max_weight = float(target.max())
        capped = _cap_position_weights(target, max_weight=max_single_stock_weight)
        audit.append(
            ConstraintAuditEntry(
                priority="P0",
                rule_name="single_stock_cap",
                triggered=before_max_weight > max_single_stock_weight + 1e-12,
                action="cap_weights",
                message="Cap each name at the maximum single-stock weight.",
                before={"max_weight": before_max_weight, "limit": float(max_single_stock_weight)},
                after={"max_weight": float(capped.max()) if not capped.empty else 0.0},
            ),
        )
        target = capped

        # P0: sector concentration vs benchmark.
        before_sector_weights = compute_sector_weights(target, sectors)
        before_sector_deviation = sector_weight_deviation(before_sector_weights, compute_sector_weights(benchmark, sectors))
        sector_adjusted = _apply_sector_constraint(
            weights=target,
            benchmark_weights=benchmark,
            sector_map=sectors,
            candidate_ranking=ranking,
            max_sector_deviation=max_sector_deviation,
        )
        sector_adjusted = _cap_position_weights(sector_adjusted, max_weight=max_single_stock_weight)
        after_sector_weights = compute_sector_weights(sector_adjusted, sectors)
        after_sector_deviation = sector_weight_deviation(after_sector_weights, compute_sector_weights(benchmark, sectors))
        audit.append(
            ConstraintAuditEntry(
                priority="P0",
                rule_name="sector_deviation_cap",
                triggered=_max_abs_value(before_sector_deviation) > max_sector_deviation + 1e-12,
                action="trim_overweight_sectors",
                message="Keep sector deviation versus benchmark within the configured band.",
                before={
                    "max_abs_deviation": _max_abs_value(before_sector_deviation),
                    "sector_weights": before_sector_weights,
                },
                after={
                    "max_abs_deviation": _max_abs_value(after_sector_deviation),
                    "sector_weights": after_sector_weights,
                },
            ),
        )
        target = sector_adjusted

        # P0: tail risk haircut.
        cvar_before = compute_portfolio_cvar(target, return_history=return_history, confidence=cvar_confidence)
        cvar_adjusted = target.copy()
        cvar_triggered = cvar_before is not None and cvar_before < cvar_floor
        cvar_iterations = 0
        if cvar_triggered:
            cvar_after = cvar_before
            while cvar_after is not None and cvar_after < cvar_floor and cvar_iterations < 3:
                cvar_adjusted = cvar_adjusted * 0.8
                cvar_iterations += 1
                cvar_after = compute_portfolio_cvar(
                    cvar_adjusted,
                    return_history=return_history,
                    confidence=cvar_confidence,
                )
            warnings.append(
                f"CVaR {cvar_before:.4f} breached the floor {cvar_floor:.4f}; "
                f"applied {cvar_iterations} 20% gross haircut round(s)."
            )
            if cvar_after is not None and cvar_after < cvar_floor and cvar_iterations >= 3:
                warnings.append(
                    f"CVaR {cvar_after:.4f} remained below the floor {cvar_floor:.4f} after "
                    f"{cvar_iterations} haircut rounds; no further scaling was applied."
                )
        else:
            cvar_after = compute_portfolio_cvar(
                cvar_adjusted,
                return_history=return_history,
                confidence=cvar_confidence,
            )
        audit.append(
            ConstraintAuditEntry(
                priority="P0",
                rule_name="cvar_haircut",
                triggered=cvar_triggered,
                action="scale_down_gross",
                message="Reduce gross exposure by 20% per round for up to three rounds if 99% CVaR breaches the floor.",
                before={"cvar_99": cvar_before, "floor": float(cvar_floor), "gross_exposure": float(target.sum())},
                after={
                    "cvar_99": cvar_after,
                    "gross_exposure": float(cvar_adjusted.sum()),
                    "iterations": int(cvar_iterations),
                },
            ),
        )
        target = cvar_adjusted

        # P1: beta alignment.
        beta_before = compute_portfolio_beta(target, beta_lookup)
        beta_adjusted = target.copy()
        beta_triggered = beta_before is not None and (
            beta_before < beta_hard_bounds[0] or beta_before > beta_hard_bounds[1]
        )
        if beta_triggered:
            beta_adjusted = _adjust_beta_exposure(
                weights=target,
                beta_lookup=beta_lookup,
                target_bounds=beta_target_bounds,
                max_weight=max_single_stock_weight,
            )
        beta_after = compute_portfolio_beta(beta_adjusted, beta_lookup)
        audit.append(
            ConstraintAuditEntry(
                priority="P1",
                rule_name="beta_alignment",
                triggered=beta_triggered,
                action="tilt_toward_target_beta",
                message="Nudge the book into the target beta band when it breaches the hard bounds.",
                before={"portfolio_beta": beta_before, "hard_bounds": tuple(map(float, beta_hard_bounds))},
                after={"portfolio_beta": beta_after, "target_bounds": tuple(map(float, beta_target_bounds))},
            ),
        )
        target = beta_adjusted

        # P1: correlation deduplication.
        corr_pairs_before = identify_high_correlation_pairs(
            return_history=return_history,
            tickers=target.index,
            threshold=correlation_threshold,
        )
        corr_adjusted = target.copy()
        if corr_pairs_before:
            corr_adjusted = _deduplicate_correlated_positions(
                weights=target,
                return_history=return_history,
                ranking=ranking,
                threshold=correlation_threshold,
                max_weight=max_single_stock_weight,
            )
        corr_pairs_after = identify_high_correlation_pairs(
            return_history=return_history,
            tickers=corr_adjusted.index,
            threshold=correlation_threshold,
        )
        audit.append(
            ConstraintAuditEntry(
                priority="P1",
                rule_name="correlation_dedup",
                triggered=bool(corr_pairs_before),
                action="remove_redundant_names",
                message="Trim highly correlated duplicate positions.",
                before={"high_corr_pairs": len(corr_pairs_before)},
                after={"high_corr_pairs": len(corr_pairs_after)},
            ),
        )
        target = corr_adjusted

        # P1: turnover cap.
        turnover_before = compute_turnover(target, current)
        turnover_adjusted = target.copy()
        if turnover_before > turnover_cap:
            turnover_adjusted = _apply_turnover_cap(
                target_weights=target,
                current_weights=current,
                turnover_cap=turnover_cap,
            )
        turnover_after = compute_turnover(turnover_adjusted, current)
        audit.append(
            ConstraintAuditEntry(
                priority="P1",
                rule_name="turnover_cap",
                triggered=turnover_before > turnover_cap + 1e-12,
                action="blend_with_current_weights",
                message="Cap one-period turnover at the configured limit.",
                before={"turnover": turnover_before, "limit": float(turnover_cap)},
                after={"turnover": turnover_after},
            ),
        )
        target = turnover_adjusted

        # P2: minimum holdings.
        holdings_before = int((target > 1e-12).sum())
        min_holdings_adjusted = target.copy()
        if holdings_before < min_holdings:
            min_holdings_adjusted = _ensure_minimum_holdings(
                weights=target,
                candidate_ranking=ranking,
                benchmark_weights=benchmark,
                min_holdings=min_holdings,
                max_weight=max_single_stock_weight,
            )
        holdings_after = int((min_holdings_adjusted > 1e-12).sum())
        audit.append(
            ConstraintAuditEntry(
                priority="P2",
                rule_name="minimum_holdings",
                triggered=holdings_before < min_holdings,
                action="add_small_satellite_positions",
                message="Maintain a diversified minimum number of holdings.",
                before={"holding_count": holdings_before, "minimum": int(min_holdings)},
                after={"holding_count": holdings_after},
            ),
        )
        target = min_holdings_adjusted

        # P2: stress test warning only.
        stress_return = compute_stress_return(
            weights=target,
            beta_lookup=beta_lookup,
            shock=stress_test_shock,
        )
        stress_triggered = stress_return < stress_warning_threshold
        if stress_triggered:
            warnings.append(
                f"Stress scenario return {stress_return:.4f} breached the warning threshold {stress_warning_threshold:.4f}."
            )
        audit.append(
            ConstraintAuditEntry(
                priority="P2",
                rule_name="stress_test",
                triggered=stress_triggered,
                action="warning_only",
                message="Run a simple market shock stress test without modifying weights.",
                before={"shock": float(stress_test_shock), "warning_threshold": float(stress_warning_threshold)},
                after={"stressed_return": float(stress_return)},
            ),
        )

        final_weights = _clip_negative(target)
        cash_weight = max(0.0, 1.0 - float(final_weights.sum()))
        beta_contributions = compute_beta_contributions(final_weights, beta_lookup)
        top_beta_contributors = build_top_beta_contributors(final_weights, beta_lookup, limit=10)
        return ConstrainedPortfolio(
            weights={str(ticker): float(weight) for ticker, weight in final_weights.items() if weight > 0.0},
            cash_weight=float(cash_weight),
            gross_exposure=float(final_weights.sum()),
            holding_count=int((final_weights > 1e-12).sum()),
            turnover=float(compute_turnover(final_weights, current)),
            portfolio_beta=compute_portfolio_beta(final_weights, beta_lookup),
            cvar_99=compute_portfolio_cvar(final_weights, return_history=return_history, confidence=cvar_confidence),
            sector_weights=compute_sector_weights(final_weights, sectors),
            warnings=warnings,
            audit_trail=audit,
            beta_contributions=beta_contributions,
            top_beta_contributors=top_beta_contributors,
        )

    @staticmethod
    def compute_sector_weights(
        weights: Mapping[str, float] | pd.Series,
        sector_map: Mapping[str, str],
    ) -> dict[str, float]:
        return compute_sector_weights(_coerce_weights(weights, normalize_to=None), sector_map)


def compute_sector_weights(
    weights: Mapping[str, float] | pd.Series,
    sector_map: Mapping[str, str],
) -> dict[str, float]:
    series = _coerce_weights(weights, normalize_to=None)
    if series.empty:
        return {}
    sectors = pd.Series(
        [sector_map.get(str(ticker).upper(), "Unknown") for ticker in series.index],
        index=series.index,
        dtype=object,
    )
    grouped = series.groupby(sectors).sum().sort_values(ascending=False)
    return {str(sector): float(weight) for sector, weight in grouped.items()}


def sector_weight_deviation(
    portfolio_sector_weights: Mapping[str, float],
    benchmark_sector_weights: Mapping[str, float],
) -> dict[str, float]:
    sectors = sorted(set(portfolio_sector_weights) | set(benchmark_sector_weights))
    return {
        sector: float(portfolio_sector_weights.get(sector, 0.0) - benchmark_sector_weights.get(sector, 0.0))
        for sector in sectors
    }


def compute_turnover(
    target_weights: Mapping[str, float] | pd.Series,
    current_weights: Mapping[str, float] | pd.Series,
) -> float:
    target = _coerce_weights(target_weights, normalize_to=None)
    current = _coerce_weights(current_weights, normalize_to=None)
    tickers = sorted(set(target.index) | set(current.index))
    if not tickers:
        return 0.0
    target_series = target.reindex(tickers).fillna(0.0)
    current_series = current.reindex(tickers).fillna(0.0)
    return float(0.5 * (target_series - current_series).abs().sum())


def compute_portfolio_beta(
    weights: Mapping[str, float] | pd.Series,
    beta_lookup: Mapping[str, float] | None,
) -> float | None:
    if not beta_lookup:
        return None
    series = _coerce_weights(weights, normalize_to=None)
    beta_series = pd.Series(beta_lookup, dtype=float)
    aligned = beta_series.reindex(series.index).dropna()
    if aligned.empty:
        return None
    return float(series.reindex(aligned.index).fillna(0.0).dot(aligned))


def compute_beta_contributions(
    weights: Mapping[str, float] | pd.Series,
    beta_lookup: Mapping[str, float] | None,
) -> dict[str, float]:
    if not beta_lookup:
        return {}
    series = _coerce_weights(weights, normalize_to=None)
    if series.empty:
        return {}
    beta_series = pd.Series(beta_lookup, dtype=float).reindex(series.index)
    contributions = series.mul(beta_series).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        str(ticker): float(value)
        for ticker, value in contributions.items()
        if np.isfinite(value)
    }


def build_top_beta_contributors(
    weights: Mapping[str, float] | pd.Series,
    beta_lookup: Mapping[str, float] | None,
    *,
    limit: int = 10,
) -> list[dict[str, float | str]]:
    if not beta_lookup:
        return []
    series = _coerce_weights(weights, normalize_to=None)
    if series.empty:
        return []
    beta_series = pd.Series(beta_lookup, dtype=float).reindex(series.index)
    frame = pd.DataFrame({"weight": series, "beta": beta_series}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return []
    frame["contribution"] = frame["weight"] * frame["beta"]
    frame["abs_contribution"] = frame["contribution"].abs()
    ranked = frame.sort_values(["abs_contribution", "contribution"], ascending=[False, False]).head(int(limit))
    return [
        {
            "ticker": str(ticker),
            "weight": float(row.weight),
            "beta": float(row.beta),
            "contribution": float(row.contribution),
        }
        for ticker, row in ranked.iterrows()
    ]


def compute_portfolio_cvar(
    weights: Mapping[str, float] | pd.Series,
    *,
    return_history: pd.DataFrame | None,
    confidence: float = 0.99,
) -> float | None:
    if return_history is None or return_history.empty:
        return None
    series = _coerce_weights(weights, normalize_to=None)
    if series.empty:
        return None

    history = return_history.copy()
    history.columns = history.columns.astype(str).str.upper()
    aligned = history.reindex(columns=series.index).dropna(how="all")
    if aligned.empty:
        return None

    portfolio_returns = aligned.fillna(0.0).mul(series.reindex(aligned.columns).fillna(0.0), axis=1).sum(axis=1)
    portfolio_returns = pd.to_numeric(portfolio_returns, errors="coerce").dropna()
    if portfolio_returns.empty:
        return None

    cutoff = float(portfolio_returns.quantile(1.0 - float(confidence)))
    tail = portfolio_returns.loc[portfolio_returns <= cutoff]
    if tail.empty:
        return cutoff
    return float(tail.mean())


def identify_high_correlation_pairs(
    *,
    return_history: pd.DataFrame | None,
    tickers: Sequence[str] | pd.Index,
    threshold: float,
    lookback: int = 60,
) -> list[tuple[str, str, float]]:
    if return_history is None or return_history.empty:
        return []

    tickers = [str(ticker).upper() for ticker in tickers]
    frame = return_history.copy()
    frame.columns = frame.columns.astype(str).str.upper()
    frame = frame.reindex(columns=tickers).tail(int(lookback)).dropna(axis=1, how="all")
    if frame.shape[1] < 2:
        return []

    corr = frame.corr().fillna(0.0)
    pairs: list[tuple[str, str, float]] = []
    columns = list(corr.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = float(corr.loc[left, right])
            if value > threshold:
                pairs.append((left, right, value))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs


def compute_stress_return(
    *,
    weights: Mapping[str, float] | pd.Series,
    beta_lookup: Mapping[str, float] | None,
    shock: float,
) -> float:
    series = _coerce_weights(weights, normalize_to=None)
    gross = float(series.sum())
    beta = compute_portfolio_beta(series, beta_lookup)
    effective_beta = 1.0 if beta is None else beta
    return float(gross * effective_beta * float(shock))


def _apply_sector_constraint(
    *,
    weights: pd.Series,
    benchmark_weights: pd.Series,
    sector_map: Mapping[str, str],
    candidate_ranking: list[str],
    max_sector_deviation: float,
) -> pd.Series:
    if weights.empty or benchmark_weights.empty or not sector_map:
        return weights.copy()

    adjusted = weights.copy()
    gross_target = float(adjusted.sum())
    benchmark_sector_weights = compute_sector_weights(benchmark_weights, sector_map)
    tolerance = 1e-12

    for _ in range(max(1, len(benchmark_sector_weights) * 2)):
        current_sector_weights = compute_sector_weights(adjusted, sector_map)
        deviations = sector_weight_deviation(current_sector_weights, benchmark_sector_weights)
        overweight_sector = None
        overweight_value = 0.0
        for sector, deviation in deviations.items():
            if deviation > max_sector_deviation + tolerance and deviation > overweight_value:
                overweight_sector = sector
                overweight_value = float(deviation)
        if overweight_sector is None:
            break

        members = [ticker for ticker in adjusted.index if sector_map.get(ticker, "Unknown") == overweight_sector]
        sector_total = float(adjusted.reindex(members).sum())
        upper_bound = float(benchmark_sector_weights.get(overweight_sector, 0.0) + max_sector_deviation)
        trim_amount = max(sector_total - upper_bound, 0.0)
        if trim_amount <= tolerance:
            break

        adjusted.loc[members] *= max(sector_total - trim_amount, 0.0) / max(sector_total, tolerance)
        residual = gross_target - float(adjusted.sum())
        if residual <= tolerance:
            continue

        recipients = _build_sector_recipient_allocation(
            weights=adjusted,
            benchmark_sector_weights=benchmark_sector_weights,
            sector_map=sector_map,
            candidate_ranking=candidate_ranking,
            max_sector_deviation=max_sector_deviation,
        )
        if recipients.empty:
            break
        adjusted = adjusted.add(recipients * residual, fill_value=0.0)

    return _clip_negative(adjusted)


def _build_sector_recipient_allocation(
    *,
    weights: pd.Series,
    benchmark_sector_weights: Mapping[str, float],
    sector_map: Mapping[str, str],
    candidate_ranking: list[str],
    max_sector_deviation: float,
) -> pd.Series:
    current_sector_weights = compute_sector_weights(weights, sector_map)
    allocations = pd.Series(dtype=float)
    tolerance = 1e-12

    for sector, benchmark_weight in benchmark_sector_weights.items():
        sector_limit = float(benchmark_weight + max_sector_deviation)
        current_weight = float(current_sector_weights.get(sector, 0.0))
        sector_capacity = max(sector_limit - current_weight, 0.0)
        if sector_capacity <= tolerance:
            continue

        tickers = [ticker for ticker in weights.index if sector_map.get(ticker, "Unknown") == sector]
        if not tickers:
            for ticker in candidate_ranking:
                if sector_map.get(ticker, "Unknown") == sector:
                    tickers = [ticker]
                    break
        if not tickers:
            continue

        base = weights.reindex(tickers).fillna(0.0)
        if float(base.sum()) <= tolerance:
            base = pd.Series(1.0, index=pd.Index(tickers, dtype=object), dtype=float)
        else:
            base = base / float(base.sum())
        allocations = allocations.add(base * sector_capacity, fill_value=0.0)

    total = float(allocations.sum())
    if total <= tolerance:
        return pd.Series(dtype=float)
    return allocations / total


def _adjust_beta_exposure(
    *,
    weights: pd.Series,
    beta_lookup: Mapping[str, float] | None,
    target_bounds: tuple[float, float],
    max_weight: float,
) -> pd.Series:
    if not beta_lookup:
        return weights.copy()

    adjusted = weights.copy()
    tolerance = 1e-12
    for _ in range(200):
        portfolio_beta = compute_portfolio_beta(adjusted, beta_lookup)
        if portfolio_beta is None:
            break
        if target_bounds[0] <= portfolio_beta <= target_bounds[1]:
            break

        beta_series = pd.Series(beta_lookup, dtype=float).reindex(adjusted.index).dropna()
        if beta_series.empty:
            break

        if portfolio_beta < target_bounds[0]:
            donor = beta_series.sort_values().index[0]
            receiver_candidates = [ticker for ticker in beta_series.sort_values(ascending=False).index if adjusted.get(ticker, 0.0) < max_weight - tolerance]
        else:
            donor = beta_series.sort_values(ascending=False).index[0]
            receiver_candidates = [ticker for ticker in beta_series.sort_values().index if adjusted.get(ticker, 0.0) < max_weight - tolerance]

        receiver = next((ticker for ticker in receiver_candidates if ticker != donor), None)
        if receiver is None:
            break

        shift = min(0.01, float(adjusted.get(donor, 0.0)), max_weight - float(adjusted.get(receiver, 0.0)))
        if shift <= tolerance:
            break
        adjusted.loc[donor] = float(adjusted.get(donor, 0.0)) - shift
        adjusted.loc[receiver] = float(adjusted.get(receiver, 0.0)) + shift
        adjusted = _clip_negative(adjusted)

    return adjusted


def _deduplicate_correlated_positions(
    *,
    weights: pd.Series,
    return_history: pd.DataFrame | None,
    ranking: list[str],
    threshold: float,
    max_weight: float,
) -> pd.Series:
    pairs = identify_high_correlation_pairs(return_history=return_history, tickers=weights.index, threshold=threshold)
    if not pairs:
        return weights.copy()

    adjusted = weights.copy()
    ranking_order = {ticker: index for index, ticker in enumerate(ranking)}
    removed_weight = 0.0

    for left, right, _ in pairs:
        if left not in adjusted.index or right not in adjusted.index:
            continue
        left_rank = ranking_order.get(left, len(ranking_order))
        right_rank = ranking_order.get(right, len(ranking_order))
        if left_rank == right_rank:
            loser = left if adjusted[left] <= adjusted[right] else right
        else:
            loser = left if left_rank > right_rank else right
        removed_weight += float(adjusted.pop(loser))

    if removed_weight <= 0.0 or adjusted.empty:
        return _clip_negative(adjusted)

    base = adjusted / float(adjusted.sum())
    adjusted = adjusted.add(base * removed_weight, fill_value=0.0)
    adjusted = _cap_position_weights(adjusted, max_weight=max_weight)
    return adjusted


def _apply_turnover_cap(
    *,
    target_weights: pd.Series,
    current_weights: pd.Series,
    turnover_cap: float,
) -> pd.Series:
    turnover = compute_turnover(target_weights, current_weights)
    if turnover <= turnover_cap:
        return target_weights.copy()

    tickers = sorted(set(target_weights.index) | set(current_weights.index))
    target = target_weights.reindex(tickers).fillna(0.0)
    current = current_weights.reindex(tickers).fillna(0.0)
    scale = float(turnover_cap / max(turnover, 1e-12))
    adjusted = current + (target - current) * scale
    return _clip_negative(adjusted)


def _ensure_minimum_holdings(
    *,
    weights: pd.Series,
    candidate_ranking: list[str],
    benchmark_weights: pd.Series,
    min_holdings: int,
    max_weight: float,
) -> pd.Series:
    adjusted = weights.copy()
    active = [ticker for ticker, weight in adjusted.items() if weight > 1e-12]
    if len(active) >= min_holdings:
        return adjusted

    candidates = [ticker for ticker in candidate_ranking if ticker not in adjusted.index]
    if not candidates:
        candidates = [ticker for ticker in benchmark_weights.index if ticker not in adjusted.index]
    needed = max(int(min_holdings) - len(active), 0)
    additions = candidates[:needed]
    if not additions:
        return adjusted

    gross_target = float(adjusted.sum())
    if gross_target <= 0.0:
        return adjusted

    seed_total = min(max(0.01 * len(additions), gross_target * 0.20), gross_target * 0.50)
    adjusted *= max(gross_target - seed_total, 0.0) / gross_target

    seed_weights = benchmark_weights.reindex(additions).fillna(0.0)
    if float(seed_weights.sum()) <= 1e-12:
        seed_weights = pd.Series(1.0, index=pd.Index(additions, dtype=object), dtype=float)
    else:
        seed_weights = seed_weights / float(seed_weights.sum())
    adjusted = adjusted.add(seed_weights * seed_total, fill_value=0.0)
    return _cap_position_weights(adjusted, max_weight=max_weight)


def _resolve_beta_map(
    *,
    beta_map: Mapping[str, float] | None,
    return_history: pd.DataFrame | None,
    spy_returns: pd.Series | None,
) -> dict[str, float] | None:
    if beta_map:
        return {
            str(ticker).upper(): float(beta)
            for ticker, beta in beta_map.items()
            if beta is not None and np.isfinite(beta)
        }

    if return_history is None or return_history.empty or spy_returns is None or pd.Series(spy_returns).empty:
        return None

    history = return_history.copy()
    history.columns = history.columns.astype(str).str.upper()
    benchmark = pd.Series(spy_returns, dtype=float).rename("SPY").replace([np.inf, -np.inf], np.nan).dropna()
    if benchmark.empty:
        return None

    joined = history.join(benchmark, how="inner")
    joined = joined.tail(252)
    if joined.empty or joined["SPY"].var() <= 1e-12:
        return None

    beta_lookup: dict[str, float] = {}
    for ticker in history.columns:
        aligned = joined[[ticker, "SPY"]].dropna()
        if len(aligned) < 60:
            continue
        clipped_ticker = np.asarray(
            mstats.winsorize(aligned[ticker].to_numpy(dtype=float), limits=[0.01, 0.01]),
            dtype=float,
        )
        clipped_spy = np.asarray(
            mstats.winsorize(aligned["SPY"].to_numpy(dtype=float), limits=[0.01, 0.01]),
            dtype=float,
        )
        benchmark_var = float(np.var(clipped_spy, ddof=1)) if len(clipped_spy) > 1 else float("nan")
        if not np.isfinite(benchmark_var) or benchmark_var <= 1e-12:
            continue
        covariance = float(np.cov(clipped_ticker, clipped_spy)[0, 1])
        beta = covariance / benchmark_var if benchmark_var > 1e-12 else np.nan
        if np.isfinite(beta) and abs(beta) > 5.0:
            continue
        beta_lookup[ticker] = beta

    clean = {
        ticker: float(beta)
        for ticker, beta in beta_lookup.items()
        if beta is not None and np.isfinite(beta)
    }
    return clean or None


def _coerce_weights(
    weights: Mapping[str, float] | pd.Series | None,
    *,
    normalize_to: float | None,
) -> pd.Series:
    if weights is None:
        return pd.Series(dtype=float)

    series = pd.Series(weights, dtype=float)
    if series.empty:
        return pd.Series(dtype=float)
    series.index = series.index.astype(str).str.upper()
    series = series.groupby(level=0).sum()
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    series = series.loc[series > 0.0].sort_values(ascending=False)
    if series.empty:
        return pd.Series(dtype=float)

    if normalize_to is None:
        total = float(series.sum())
        if total > 1.0 + 1e-6:
            return series / total
        return series

    total = float(series.sum())
    if total <= 0.0:
        return pd.Series(dtype=float)
    return (series / total) * float(normalize_to)


def _normalize_sector_map(sector_map: Mapping[str, str]) -> dict[str, str]:
    return {
        str(ticker).upper(): str(sector) if sector not in (None, "") else "Unknown"
        for ticker, sector in sector_map.items()
    }


def _normalize_ranking(candidate_ranking: Sequence[str] | pd.Index | None, *, fallback: list[str]) -> list[str]:
    if candidate_ranking is None:
        return [str(ticker).upper() for ticker in fallback]
    return [str(ticker).upper() for ticker in candidate_ranking]


def _cap_position_weights(weights: pd.Series, *, max_weight: float) -> pd.Series:
    adjusted = _clip_negative(weights)
    tolerance = 1e-12
    if adjusted.empty:
        return adjusted

    while bool((adjusted > max_weight + tolerance).any()):
        over = adjusted.loc[adjusted > max_weight + tolerance]
        excess = float((over - max_weight).sum())
        adjusted.loc[over.index] = max_weight

        under_index = adjusted.index[adjusted < max_weight - tolerance]
        if excess <= tolerance or len(under_index) == 0:
            break

        base = adjusted.loc[under_index]
        if float(base.sum()) <= tolerance:
            adjusted.loc[under_index] += excess / len(under_index)
        else:
            adjusted.loc[under_index] += excess * (base / float(base.sum()))

        adjusted = _clip_negative(adjusted)

    return adjusted


def _clip_negative(weights: pd.Series) -> pd.Series:
    cleaned = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cleaned = cleaned.loc[cleaned > 1e-12].sort_values(ascending=False)
    return cleaned


def _max_abs_value(values: Mapping[str, float]) -> float:
    if not values:
        return 0.0
    return float(max(abs(float(value)) for value in values.values()))
