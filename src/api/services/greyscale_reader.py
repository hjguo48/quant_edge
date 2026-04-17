from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from loguru import logger

_WEEK_PATTERN = re.compile(r"week_(\d+)\.json$")
_MODEL_NAMES = ("ridge", "xgboost", "lightgbm")


class GreyscaleReader:
    """Read weekly greyscale reports from disk and expose API-ready views."""

    def __init__(
        self,
        report_dir: Path | str,
        g3_gate_results_path: Path | str | None = None,
        quintile_expected_returns_path: Path | str | None = None,
    ) -> None:
        self._report_dir = Path(report_dir)
        self._g3_gate_results_path = (
            Path(g3_gate_results_path)
            if g3_gate_results_path is not None
            else Path("data/reports/g3_gate_phase_e_v2.json")
        )
        self._quintile_expected_returns_path = (
            Path(quintile_expected_returns_path)
            if quintile_expected_returns_path is not None
            else Path("data/reports/quintile_expected_returns.json")
        )
        self._cache: dict[int, dict[str, Any]] = {}
        self._snapshot: tuple[tuple[str, int, int], ...] | None = None
        self._bootstrap_cache: dict[str, Any] | None = None
        self._bootstrap_snapshot: tuple[int, int] | None = None
        self._quintile_cache: dict[str, Any] | None = None
        self._quintile_snapshot: tuple[int, int] | None = None

    def _build_snapshot(self) -> tuple[tuple[str, int, int], ...]:
        if not self._report_dir.is_dir():
            return ()

        snapshot: list[tuple[str, int, int]] = []
        for path in sorted(self._report_dir.glob("week_*.json")):
            try:
                stat = path.stat()
            except OSError as exc:
                logger.warning("failed to stat {}: {}", path, exc)
                continue
            snapshot.append((path.name, stat.st_mtime_ns, stat.st_size))
        return tuple(snapshot)

    def _load_all(self) -> dict[int, dict[str, Any]]:
        snapshot = self._build_snapshot()
        if self._snapshot == snapshot and self._cache:
            return self._cache

        reports: dict[int, dict[str, Any]] = {}
        if not self._report_dir.is_dir():
            self._cache = reports
            self._snapshot = snapshot
            return reports

        for path in sorted(self._report_dir.glob("week_*.json")):
            match = _WEEK_PATTERN.search(path.name)
            if not match:
                continue

            week_num = int(match.group(1))
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("failed to load {}: {}", path, exc)
                continue
            reports[week_num] = data

        self._cache = reports
        self._snapshot = snapshot
        return reports

    @staticmethod
    def _build_file_snapshot(path: Path) -> tuple[int, int] | None:
        if not path.is_file():
            return None

        try:
            stat = path.stat()
        except OSError as exc:
            logger.warning("failed to stat {}: {}", path, exc)
            return None
        return (stat.st_mtime_ns, stat.st_size)

    @staticmethod
    def _load_json_file(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None

        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("failed to load {}: {}", path, exc)
            return None

    def _load_g3_gate_results(self) -> dict[str, Any] | None:
        snapshot = self._build_file_snapshot(self._g3_gate_results_path)
        if snapshot is None:
            self._bootstrap_cache = None
            self._bootstrap_snapshot = None
            return None

        if self._bootstrap_snapshot == snapshot and self._bootstrap_cache is not None:
            return self._bootstrap_cache

        report = self._load_json_file(self._g3_gate_results_path)
        self._bootstrap_cache = report
        self._bootstrap_snapshot = snapshot
        return report

    def _load_quintile_expected_returns(self) -> dict[str, Any] | None:
        snapshot = self._build_file_snapshot(self._quintile_expected_returns_path)
        if snapshot is None:
            self._quintile_cache = None
            self._quintile_snapshot = None
            return None

        if self._quintile_snapshot == snapshot and self._quintile_cache is not None:
            return self._quintile_cache

        report = self._load_json_file(self._quintile_expected_returns_path)
        self._quintile_cache = report
        self._quintile_snapshot = snapshot
        return report

    def invalidate_cache(self) -> None:
        self._cache.clear()
        self._snapshot = None
        self._bootstrap_cache = None
        self._bootstrap_snapshot = None
        self._quintile_cache = None
        self._quintile_snapshot = None

    def get_latest_report(self) -> dict[str, Any] | None:
        reports = self._load_all()
        if not reports:
            return None
        return reports[max(reports)]

    def get_reports(self) -> dict[int, dict[str, Any]]:
        return dict(self._load_all())

    def get_all_fusion_scores(self) -> list[dict[str, Any]]:
        report = self.get_latest_report()
        if report is None:
            return []

        ranked = self._rank_scores(report.get("score_vectors", {}).get("fusion", {}))
        total = len(ranked)
        return [
            {
                "ticker": ticker,
                "score": score,
                "rank": index + 1,
                "percentile": round((1 - index / max(total, 1)) * 100, 1),
            }
            for index, (ticker, score) in enumerate(ranked)
        ]

    def get_fusion_scores_for_tickers(self, tickers: list[str]) -> list[dict[str, Any]]:
        if not tickers:
            return []

        requested = {ticker.strip().upper() for ticker in tickers if ticker.strip()}
        if not requested:
            return []

        return [
            item
            for item in self.get_all_fusion_scores()
            if item["ticker"].upper() in requested
        ]

    def get_ticker_detail(self, ticker: str) -> dict[str, Any] | None:
        report = self.get_latest_report()
        if report is None:
            return None

        normalized_ticker = ticker.upper()
        fusion_scores = report.get("score_vectors", {}).get("fusion", {})
        if normalized_ticker not in fusion_scores:
            return None

        ranked = self._rank_scores(fusion_scores)
        total = len(ranked)
        rank = next((index + 1 for index, (name, _) in enumerate(ranked) if name == normalized_ticker), None)
        model_scores = {
            model_name: float(report.get("score_vectors", {}).get(model_name, {})[normalized_ticker])
            for model_name in _MODEL_NAMES
            if normalized_ticker in report.get("score_vectors", {}).get(model_name, {})
        }
        weights = report.get("live_outputs", {}).get("target_weights_after_risk", {})

        return {
            "ticker": normalized_ticker,
            "fusion_score": float(fusion_scores[normalized_ticker]),
            "rank": rank,
            "total": total,
            "percentile": round((1 - (rank - 1) / max(total, 1)) * 100, 1) if rank is not None else None,
            "model_scores": model_scores,
            "weight": self._maybe_float(weights.get(normalized_ticker)),
            "signal_date": report.get("live_outputs", {}).get("signal_date"),
        }

    def get_ticker_confidence(self, ticker: str) -> dict[str, Any] | None:
        report = self.get_latest_report()
        if report is None:
            return None

        normalized_ticker = ticker.upper()
        score_vectors = report.get("score_vectors", {})
        model_scores = [
            float(score_vectors[model][normalized_ticker])
            for model in _MODEL_NAMES
            if normalized_ticker in score_vectors.get(model, {})
        ]
        if len(model_scores) < 2:
            return None

        spread = statistics.pstdev(model_scores)

        # Rank this ticker's spread against all tickers to determine confidence.
        # Lower spread = higher model agreement = higher confidence.
        all_spreads: list[float] = []
        all_tickers = score_vectors.get("fusion", {})
        for t in all_tickers:
            scores = [
                float(score_vectors[m][t])
                for m in _MODEL_NAMES
                if t in score_vectors.get(m, {})
            ]
            if len(scores) >= 2:
                all_spreads.append(statistics.pstdev(scores))

        if all_spreads:
            all_spreads.sort()
            rank = sum(1 for s in all_spreads if s <= spread)
            percentile = rank / len(all_spreads)
        else:
            percentile = 0.5

        if percentile <= 0.33:
            confidence = "high"
        elif percentile <= 0.66:
            confidence = "medium"
        else:
            confidence = "low"

        positive_count = sum(1 for s in model_scores if s > 0)
        negative_count = sum(1 for s in model_scores if s < 0)
        agreement = max(positive_count, negative_count) / len(model_scores)

        return {
            "confidence": confidence,
            "model_spread": spread,
            "model_agreement": agreement,
        }

    def get_portfolio_holdings(self) -> list[dict[str, Any]]:
        report = self.get_latest_report()
        if report is None:
            return []

        weights = report.get("live_outputs", {}).get("target_weights_after_risk", {})
        fusion_scores = report.get("score_vectors", {}).get("fusion", {})
        sorted_holdings = sorted(
            ((str(ticker), float(weight)) for ticker, weight in weights.items()),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            {
                "ticker": ticker,
                "weight": weight,
                "score": self._maybe_float(fusion_scores.get(ticker)),
            }
            for ticker, weight in sorted_holdings
        ]

    def get_portfolio_summary(self) -> dict[str, Any] | None:
        report = self.get_latest_report()
        if report is None:
            return None

        metrics = report.get("portfolio_metrics", {})
        risk = report.get("risk_checks", {}).get("layer3_portfolio", {})
        risk_report = risk.get("report", {})
        return {
            "signal_date": report.get("live_outputs", {}).get("signal_date"),
            "week_number": report.get("week_number"),
            "holding_count": self._maybe_int(
                metrics.get("holding_count_after_risk", risk_report.get("holding_count")),
            ),
            "gross_exposure": self._maybe_float(
                metrics.get("gross_exposure_after_risk", risk_report.get("gross_exposure")),
            ),
            "cash_weight": self._maybe_float(metrics.get("cash_weight_after_risk", risk_report.get("cash_weight"))),
            "portfolio_beta": self._maybe_float(risk_report.get("portfolio_beta")),
            "cvar_95": self._maybe_float(risk_report.get("cvar_95")),
            "turnover": self._maybe_float(metrics.get("turnover_vs_previous")),
            "risk_pass": risk.get("pass"),
        }

    def get_signal_history(self, ticker: str) -> list[dict[str, Any]]:
        normalized_ticker = ticker.upper()
        reports = self._load_all()
        history: list[dict[str, Any]] = []

        for week_num in sorted(reports):
            report = reports[week_num]
            fusion_scores = report.get("score_vectors", {}).get("fusion", {})
            if normalized_ticker not in fusion_scores:
                continue

            ranked = self._rank_scores(fusion_scores)
            total = len(ranked)
            rank = next((index + 1 for index, (name, _) in enumerate(ranked) if name == normalized_ticker), None)
            history.append(
                {
                    "week": week_num,
                    "signal_date": report.get("live_outputs", {}).get("signal_date"),
                    "score": float(fusion_scores[normalized_ticker]),
                    "rank": rank,
                    "total": total,
                },
            )

        return history

    def get_report(self, week: int) -> dict[str, Any] | None:
        return self._load_all().get(week)

    def get_bootstrap_confidence(self) -> dict[str, Any] | None:
        report = self._load_g3_gate_results()
        if report is None:
            return None

        bootstrap = report.get("checks", {}).get("bootstrap_ci_positive")
        if not isinstance(bootstrap, dict):
            return None

        return {
            "annualized_excess_ci_lower": self._maybe_float(bootstrap.get("annualized_excess_ci_lower")),
            "annualized_excess_ci_upper": self._maybe_float(bootstrap.get("annualized_excess_ci_upper")),
            "annualized_excess_estimate": self._maybe_float(bootstrap.get("annualized_excess_estimate")),
            "sharpe_ci_lower": self._maybe_float(bootstrap.get("sharpe_ci_lower")),
            "sharpe_ci_upper": self._maybe_float(bootstrap.get("sharpe_ci_upper")),
            "sharpe_estimate": self._maybe_float(bootstrap.get("sharpe_estimate")),
            "n_bootstrap": self._maybe_int(bootstrap.get("n_bootstrap")),
            "ci_level": self._maybe_float(bootstrap.get("ci_level")),
        }

    def get_expected_returns(self, quintile: int | None = None) -> dict[str, Any] | None:
        if quintile is not None:
            report = self._load_quintile_expected_returns()
            if report is not None:
                quintile_payload = report.get("quintiles", {}).get(str(int(quintile)))
                if isinstance(quintile_payload, dict):
                    return {
                        "data_source": str(
                            report.get("data_source", "walk_forward_quintile_bootstrap"),
                        ),
                        "ci_level": self._maybe_float(report.get("ci_level")),
                        "n_observations": self._maybe_int(quintile_payload.get("n_observations")),
                        "annualized_excess": {
                            "estimate": float(quintile_payload["annualized_excess"]["estimate"]),
                            "ci_lower": float(quintile_payload["annualized_excess"]["ci_lower"]),
                            "ci_upper": float(quintile_payload["annualized_excess"]["ci_upper"]),
                        },
                        "sharpe": {
                            "estimate": float(quintile_payload["sharpe"]["estimate"]),
                            "ci_lower": float(quintile_payload["sharpe"]["ci_lower"]),
                            "ci_upper": float(quintile_payload["sharpe"]["ci_upper"]),
                        },
                    }

        report = self._load_g3_gate_results()
        if report is None:
            return None

        bootstrap = report.get("checks", {}).get("bootstrap_ci_positive")
        if not isinstance(bootstrap, dict):
            return None

        return {
            "data_source": "g3_gate_bootstrap",
            "ci_level": self._maybe_float(bootstrap.get("ci_level")),
            "n_observations": self._maybe_int(bootstrap.get("n_observations")),
            "annualized_excess": {
                "estimate": float(bootstrap["annualized_excess_estimate"]),
                "ci_lower": float(bootstrap["annualized_excess_ci_lower"]),
                "ci_upper": float(bootstrap["annualized_excess_ci_upper"]),
            },
            "sharpe": {
                "estimate": float(bootstrap["sharpe_estimate"]),
                "ci_lower": float(bootstrap["sharpe_ci_lower"]),
                "ci_upper": float(bootstrap["sharpe_ci_upper"]),
            },
        }

    @staticmethod
    def _rank_scores(scores: dict[str, Any]) -> list[tuple[str, float]]:
        ranked = [(str(ticker), float(score)) for ticker, score in scores.items()]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _maybe_int(value: Any) -> int | None:
        return None if value is None else int(value)
