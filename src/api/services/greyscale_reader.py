from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

_WEEK_PATTERN = re.compile(r"week_(\d+)\.json$")
_MODEL_NAMES = ("ridge", "xgboost", "lightgbm")


class GreyscaleReader:
    """Read weekly greyscale reports from disk and expose API-ready views."""

    def __init__(self, report_dir: Path | str) -> None:
        self._report_dir = Path(report_dir)
        self._cache: dict[int, dict[str, Any]] = {}
        self._snapshot: tuple[tuple[str, int, int], ...] | None = None

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

    def invalidate_cache(self) -> None:
        self._cache.clear()
        self._snapshot = None

    def get_latest_report(self) -> dict[str, Any] | None:
        reports = self._load_all()
        if not reports:
            return None
        return reports[max(reports)]

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
