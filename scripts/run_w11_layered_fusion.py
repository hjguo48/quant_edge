from __future__ import annotations

"""W11 4-layer multi-horizon fusion (asymmetric control stack design).

Per data/reports/W11_design_2026-04-27.md (Codex deep review):

  60D Core   = decides admissible book (top 35% by 60D rank = envelope)
  20D Adjust = bounded rerank inside envelope: + 0.20 * clip(z20, -2, +2)
  5D Event   = sparse premium/penalty: + 0.10 * sign(z5) * 1{|z5|>=1.5}
                                          * 1{r60 >= 0.50}
  1D Throttle = scales today's trade Δw against adverse z1 signal,
                NOT the long-run book

Variants tested (max 5, multi-testing discipline):
  V0 = 60D only (baseline = W10 champion)
  V1 = 60D + 20D
  V2 = V1 + 5D event
  V3 = V2 + 1D throttle (full 4-layer)
  V2b = 60D + 5D + 1D (contingency, only if V1 fails)

Promotion gate (1.0x cost, gate_off, vs W10 champion 7.34%/0.72/14.0%):
  EITHER  net >= 7.84% AND IR >= 0.67
  OR      net >= 7.09% AND IR >= 0.67 AND DD <= 11.0%
  + Robustness at 1.25x: net > 5% AND <= 25 bps below 60D
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_horizon_fusion import (  # noqa: E402
    extract_ridge_alpha,
    parse_horizon_days,
    prepare_horizon_artifacts,
    rebuild_ridge_predictions,
    select_report_windows,
    slice_all_splits,
)
from scripts.run_ic_screening import write_json_atomic  # noqa: E402
from scripts.run_turnover_optimization import simulate_score_weighted_controlled  # noqa: E402
from scripts.run_walkforward_comparison import (  # noqa: E402
    BENCHMARK_TICKER,
    DEFAULT_ALL_FEATURES_PATH,
    LABEL_BUFFER_DAYS,
    REBALANCE_WEEKDAY,
    json_safe,
    parse_date,
)
from src.backtest.cost_model import AlmgrenChrissCostModel  # noqa: E402
from src.backtest.execution import prepare_execution_price_frame  # noqa: E402
from src.data.db.pit import get_prices_pit  # noqa: E402

# Default inputs
HORIZON_REPORTS = {
    "1d": "data/reports/walkforward_v9full9y_1d_ridge_13w.json",
    "5d": "data/reports/walkforward_v9full9y_5d_ridge_13w.json",
    "20d": "data/reports/walkforward_v9full9y_20d_ridge_13w.json",
    "60d": "data/reports/walkforward_v9full9y_60d_ridge_13w.json",
}
HORIZON_FEATURE_MATRICES = {
    "1d": "data/features/walkforward_v9full9y_fm_1d.parquet",
    "5d": "data/features/walkforward_v9full9y_fm_5d.parquet",
    "20d": "data/features/walkforward_v9full9y_fm_20d.parquet",
    "60d": "data/features/walkforward_v9full9y_fm_60d.parquet",
}
HORIZON_LABELS = {
    "1d": "data/labels/forward_returns_1d_v9full9y.parquet",
    "5d": "data/labels/forward_returns_5d_v9full9y.parquet",
    "20d": "data/labels/forward_returns_20d_v9full9y.parquet",
    "60d": "data/labels/forward_returns_60d_v9full9y.parquet",
}
DEFAULT_OUTPUT = "data/reports/w11_layered_fusion.json"
DEFAULT_PERIODS_OUTPUT = "data/reports/w11_layered_fusion_periods.parquet"
DEFAULT_DEBUG_OUTPUT = "data/reports/w11_layered_fusion_debug.parquet"

DEFAULT_ETA = 0.426
DEFAULT_GAMMA = 0.942
COST_MULTIPLIERS = (0.75, 1.0, 1.25)

# Fusion hyper-parameters (Codex prior)
ENVELOPE_RANK_60D = 0.65   # top 35% by 60D rank
LAMBDA_20D = 0.20          # bounded 20D adjust coefficient
LAMBDA_20D_CLIP = 2.0      # |z20| clip
DELTA_5D = 0.10            # 5D event premium/penalty
TAU_5D = 1.5               # |z5| event trigger
EVENT_R60_GATE = 0.50      # 5D event only fires when r60 >= 0.50
TAU_1D = 1.0               # |z1| throttle trigger
ADVERSE_SCALE = 0.5        # Δw scale factor on adverse 1D signal

# Champion params (W10 score_weighted_buffered)
PORTFOLIO_PARAMS = {
    "selection_pct": 0.20,
    "sell_buffer_pct": 0.25,
    "min_trade_weight": 0.01,
    "max_weight": 0.05,
    "min_holdings": 20,
    "weight_shrinkage": 0.0,
    "no_trade_zone": 0.0,
    "turnover_penalty_lambda": 0.0,
}

# W10 champion baseline for comparison
W10_BASELINE_NET = 0.0734
W10_BASELINE_IR = 0.7156
W10_BASELINE_DD = 0.1404

# Pass gate (Codex revised)
PROMOTE_NET_HIGH = 0.0784  # 7.84%
PROMOTE_IR_MIN = 0.67
PROMOTE_NET_MID = 0.0709   # 7.09%
PROMOTE_DD_MAX = 0.110     # 11.0%
ROBUST_NET_MIN = 0.05
ROBUST_LOSS_VS_60D_MAX = 0.0025  # 25 bps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    # Variant selection
    variants_to_run = args.variants.split(",") if args.variants else ["V0", "V1", "V2", "V3"]
    variants_to_run = [v.strip() for v in variants_to_run]
    logger.info("Variants: {}", variants_to_run)

    # Stage 1: rebuild predictions for each horizon
    logger.info("Stage 1: rebuilding predictions for {} horizons", len(HORIZON_REPORTS))
    predictions_by_horizon = {}
    aligned_payload = None
    for h in ("60d", "20d", "5d", "1d"):
        report_path = REPO_ROOT / HORIZON_REPORTS[h]
        feature_matrix_path = REPO_ROOT / HORIZON_FEATURE_MATRICES[h]
        label_path = REPO_ROOT / HORIZON_LABELS[h]
        payload = json.loads(report_path.read_text())
        horizon_days = parse_horizon_days(payload)
        windows = select_report_windows(payload, limit=args.window_limit)
        if aligned_payload is None:
            aligned_payload = payload
            as_of = parse_date(str(payload["as_of"]))
            benchmark_ticker = str(payload.get("split_config", {}).get("calendar_ticker", BENCHMARK_TICKER)).upper()
            rebalance_weekday = int(payload.get("split_config", {}).get("rebalance_weekday", REBALANCE_WEEKDAY))
        artifacts = prepare_horizon_artifacts(
            label=f"{horizon_days}D",
            horizon_days=horizon_days,
            report_path=report_path,
            report_payload=payload,
            windows=windows,
            all_features_path=REPO_ROOT / args.all_features_path,
            feature_matrix_cache_path=feature_matrix_path,
            label_cache_path=label_path,
            as_of=as_of,
            label_buffer_days=args.label_buffer_days,
            benchmark_ticker=benchmark_ticker,
            rebalance_weekday=rebalance_weekday,
        )
        preds = build_predictions(
            windows=windows,
            artifacts=artifacts,
            rebalance_weekday=rebalance_weekday,
        )
        predictions_by_horizon[h] = preds
        logger.info("  {}: {} rows, {} dates", h, len(preds),
                    preds.index.get_level_values("trade_date").nunique())

    # Stage 2: align + cross-sectional z-score
    logger.info("Stage 2: aligning + z-scoring 4 horizon predictions")
    aligned_z = align_and_zscore(predictions_by_horizon)
    logger.info("  aligned frame: {} rows ({} dates × tickers)",
                len(aligned_z),
                aligned_z["trade_date"].nunique())

    # Stage 3: load PIT prices
    logger.info("Stage 3: loading PIT prices")
    signal_dates = pd.DatetimeIndex(aligned_z["trade_date"].unique()).sort_values()
    tickers = sorted(set(aligned_z["ticker"].astype(str).tolist()) | {benchmark_ticker})
    price_start = pd.Timestamp(signal_dates.min()).date()
    price_end = (pd.Timestamp(signal_dates.max()) + pd.Timedelta(days=args.price_buffer_days)).date()
    prices = get_prices_pit(tickers=tickers, start_date=price_start, end_date=price_end, as_of=as_of)
    if prices.empty:
        raise RuntimeError("No PIT prices returned.")
    execution_frame = prepare_execution_price_frame(prices)

    # Stage 4: build fused scores per variant
    logger.info("Stage 4: building fused scores for {} variants", len(variants_to_run))
    fused_by_variant: dict[str, pd.Series] = {}
    throttle_by_variant: dict[str, pd.Series | None] = {}
    debug_frames: list[pd.DataFrame] = []
    for variant in variants_to_run:
        fused_score, throttle_signal, debug_df = build_variant_fusion(aligned_z, variant)
        fused_by_variant[variant] = fused_score
        throttle_by_variant[variant] = throttle_signal
        debug_df["variant"] = variant
        debug_frames.append(debug_df)
        logger.info("  {}: {} (date,ticker) pairs in fused_score, throttle={}",
                    variant, len(fused_score), "yes" if throttle_signal is not None else "no")

    # Stage 5: simulate each variant at 1.0x cost
    logger.info("Stage 5: simulating variants at 1.0x cost")
    rows: list[dict[str, Any]] = []
    period_frames: list[pd.DataFrame] = []
    cost_model_1x = AlmgrenChrissCostModel(eta=DEFAULT_ETA, gamma=DEFAULT_GAMMA)
    for variant in variants_to_run:
        logger.info("  variant={} cost=1.0x", variant)
        portfolio = simulate_score_weighted_controlled(
            predictions=fused_by_variant[variant],
            execution=execution_frame,
            cost_model=cost_model_1x,
            benchmark_ticker=benchmark_ticker,
            directional_throttle_signal=throttle_by_variant[variant],
            adverse_buy_threshold=TAU_1D,
            adverse_sell_threshold=TAU_1D,
            adverse_trade_scale=ADVERSE_SCALE,
            **PORTFOLIO_PARAMS,
        )
        if portfolio.periods:
            row, periods_df = aggregate_variant(portfolio, variant=variant, cost_mult=1.0)
            rows.append(row)
            period_frames.append(periods_df)

    # Stage 6: cost robustness on best variant
    logger.info("Stage 6: cost robustness check on best variant")
    if rows:
        best_v0_row = next((r for r in rows if r["variant"] == "V0"), None)
        if best_v0_row is None:
            best_v0_row = rows[0]
        # Find best non-V0 variant by net_ann_excess
        non_v0_rows = [r for r in rows if r["variant"] != "V0"]
        if non_v0_rows:
            best_variant_row = max(non_v0_rows, key=lambda r: r["net_ann_excess"])
            best_variant = best_variant_row["variant"]
            logger.info("  best non-V0 variant: {}", best_variant)
            for cost_mult in COST_MULTIPLIERS:
                if cost_mult == 1.0:
                    continue
                logger.info("    cost_mult={:.2f}", cost_mult)
                cost_model_alt = AlmgrenChrissCostModel(
                    eta=DEFAULT_ETA * cost_mult,
                    gamma=DEFAULT_GAMMA * cost_mult,
                )
                portfolio_alt = simulate_score_weighted_controlled(
                    predictions=fused_by_variant[best_variant],
                    execution=execution_frame,
                    cost_model=cost_model_alt,
                    benchmark_ticker=benchmark_ticker,
                    directional_throttle_signal=throttle_by_variant[best_variant],
                    adverse_buy_threshold=TAU_1D,
                    adverse_sell_threshold=TAU_1D,
                    adverse_trade_scale=ADVERSE_SCALE,
                    **PORTFOLIO_PARAMS,
                )
                if portfolio_alt.periods:
                    row_alt, periods_alt = aggregate_variant(
                        portfolio_alt, variant=best_variant, cost_mult=cost_mult
                    )
                    rows.append(row_alt)
                    period_frames.append(periods_alt)
            # Also rerun V0 for cost robustness comparison
            v0_throttle = throttle_by_variant.get("V0")
            for cost_mult in COST_MULTIPLIERS:
                if cost_mult == 1.0:
                    continue
                cost_model_alt = AlmgrenChrissCostModel(
                    eta=DEFAULT_ETA * cost_mult,
                    gamma=DEFAULT_GAMMA * cost_mult,
                )
                portfolio_v0 = simulate_score_weighted_controlled(
                    predictions=fused_by_variant["V0"],
                    execution=execution_frame,
                    cost_model=cost_model_alt,
                    benchmark_ticker=benchmark_ticker,
                    directional_throttle_signal=v0_throttle,
                    adverse_buy_threshold=TAU_1D,
                    adverse_sell_threshold=TAU_1D,
                    adverse_trade_scale=ADVERSE_SCALE,
                    **PORTFOLIO_PARAMS,
                )
                if portfolio_v0.periods:
                    row_v0, periods_v0 = aggregate_variant(
                        portfolio_v0, variant="V0", cost_mult=cost_mult
                    )
                    rows.append(row_v0)
                    period_frames.append(periods_v0)

    # Stage 7: verdict
    verdict = build_w11_verdict(rows)

    # Stage 8: persist
    output = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "as_of": as_of.isoformat(),
        "fusion_hyperparams": {
            "envelope_rank_60d": ENVELOPE_RANK_60D,
            "lambda_20d": LAMBDA_20D,
            "lambda_20d_clip": LAMBDA_20D_CLIP,
            "delta_5d": DELTA_5D,
            "tau_5d": TAU_5D,
            "event_r60_gate": EVENT_R60_GATE,
            "tau_1d": TAU_1D,
            "adverse_scale": ADVERSE_SCALE,
        },
        "portfolio_params": PORTFOLIO_PARAMS,
        "cost_model_base": {"eta": DEFAULT_ETA, "gamma": DEFAULT_GAMMA},
        "cost_multipliers": list(COST_MULTIPLIERS),
        "w10_baseline": {
            "net_ann_excess": W10_BASELINE_NET,
            "ir": W10_BASELINE_IR,
            "max_drawdown": W10_BASELINE_DD,
        },
        "promotion_gate": {
            "net_high": PROMOTE_NET_HIGH,
            "ir_min": PROMOTE_IR_MIN,
            "net_mid": PROMOTE_NET_MID,
            "dd_max": PROMOTE_DD_MAX,
            "robust_net_min": ROBUST_NET_MIN,
            "robust_loss_vs_60d_max": ROBUST_LOSS_VS_60D_MAX,
        },
        "variants_run": variants_to_run,
        "rows": rows,
        "verdict": verdict,
    }
    output_path = REPO_ROOT / args.output
    write_json_atomic(output_path, json_safe(output))
    logger.info("saved truth table to {}", output_path)

    if period_frames:
        periods_combined = pd.concat(period_frames, ignore_index=True)
        for col in ("selected_tickers", "cost_breakdown"):
            if col in periods_combined.columns:
                periods_combined = periods_combined.drop(columns=[col])
        periods_path = REPO_ROOT / args.periods_output
        periods_path.parent.mkdir(parents=True, exist_ok=True)
        periods_combined.to_parquet(periods_path, index=False)
        logger.info("saved periods parquet to {} ({} rows)", periods_path, len(periods_combined))

    if debug_frames:
        debug_combined = pd.concat(debug_frames, ignore_index=True)
        debug_path = REPO_ROOT / args.debug_output
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_combined.to_parquet(debug_path, index=False)
        logger.info("saved fusion debug parquet to {} ({} rows)", debug_path, len(debug_combined))

    print_summary(rows, verdict)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--periods-output", default=DEFAULT_PERIODS_OUTPUT)
    p.add_argument("--debug-output", default=DEFAULT_DEBUG_OUTPUT)
    p.add_argument("--variants", help="comma-separated variant list (e.g., V0,V1,V2,V3)")
    p.add_argument("--window-limit", type=int)
    p.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    p.add_argument("--price-buffer-days", type=int, default=7)
    return p.parse_args(argv)


def configure_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")


def build_predictions(*, windows, artifacts, rebalance_weekday):
    parts: list[pd.Series] = []
    for window in windows:
        date_keys = ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end")
        dates = {key: parse_date(str(window["dates"][key])) for key in date_keys}
        split = slice_all_splits(
            X=artifacts.feature_matrix,
            y=artifacts.labels,
            dates=dates,
            rebalance_weekday=rebalance_weekday,
        )
        alpha = extract_ridge_alpha(window)
        _, test_pred = rebuild_ridge_predictions(
            train_X=split["train_X"],
            train_y=split["train_y"],
            validation_X=split["validation_X"],
            validation_y=split["validation_y"],
            test_X=split["test_X"],
            alpha=alpha,
        )
        parts.append(test_pred.rename("score"))
    return pd.concat(parts).sort_index()


def align_and_zscore(predictions_by_horizon: dict[str, pd.Series]) -> pd.DataFrame:
    """Cross-sectional z-score per (trade_date, horizon), then align on common (date, ticker).

    Also keeps raw 60D Ridge predictions for V0 baseline (must match W10 exactly).
    """
    z_frames = {}
    for h, preds in predictions_by_horizon.items():
        df = preds.reset_index()
        df.columns = ["trade_date", "ticker", "raw"]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["ticker"] = df["ticker"].astype(str)
        df[f"z{h}"] = df.groupby("trade_date")["raw"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )
        if h == "60d":
            # Keep raw 60D for V0 baseline
            df = df.rename(columns={"raw": "raw60"})
            z_frames[h] = df[["trade_date", "ticker", "raw60", f"z{h}"]]
        else:
            z_frames[h] = df[["trade_date", "ticker", f"z{h}"]]

    aligned = z_frames["60d"]
    for h in ("20d", "5d", "1d"):
        aligned = aligned.merge(z_frames[h], on=["trade_date", "ticker"], how="inner")

    aligned["r60"] = aligned.groupby("trade_date")["z60d"].rank(pct=True)
    return aligned


def build_variant_fusion(aligned: pd.DataFrame, variant: str) -> tuple[pd.Series, pd.Series | None, pd.DataFrame]:
    """Build fused_score Series + optional throttle signal + debug frame for a variant.

    V0 is the W10 60D baseline — uses z60 directly across the FULL universe
    (no envelope, no fusion). This must reproduce W10 champion 7.34% net.

    V1+ applies the asymmetric fusion stack:
      - envelope (top 35% by r60)
      - + 20D bounded adjust
      - + 5D sparse event (V2+)
      - + 1D throttle (V3+)
    """
    df = aligned.copy()
    debug = df[["trade_date", "ticker", "z60d", "z20d", "z5d", "z1d", "r60"]].copy()

    if variant == "V0":
        # W10 baseline: RAW 60D Ridge predictions across full universe.
        # Must use raw scores (not z-scored) because score_weighted_buffered's
        # _build_score_weights filters pos_scores>0 → z-scores produce different
        # weight distribution than raw Ridge predictions.
        debug["envelope"] = 1
        debug["event_flag"] = 0
        debug["latent"] = df["raw60"]
        debug["fused_score"] = df["raw60"]
        debug["throttle_z1"] = np.nan
        fused = df.set_index(["trade_date", "ticker"])["raw60"].rename("fused_score")
        return fused, None, debug

    # V1, V2, V3, V2b — fusion stack
    envelope_mask = df["r60"] >= ENVELOPE_RANK_60D
    debug["envelope"] = envelope_mask.astype(int)

    latent = df["z60d"].copy()

    # 20D bounded adjust (V1, V2, V3 — but NOT V2b which is 60+5+1 only)
    if variant in ("V1", "V2", "V3"):
        z20_clipped = df["z20d"].clip(-LAMBDA_20D_CLIP, LAMBDA_20D_CLIP)
        latent = latent + LAMBDA_20D * z20_clipped

    # 5D sparse event (V2, V3, V2b)
    if variant in ("V2", "V3", "V2b"):
        event_trigger = (df["z5d"].abs() >= TAU_5D) & (df["r60"] >= EVENT_R60_GATE)
        event_premium = DELTA_5D * np.sign(df["z5d"]) * event_trigger.astype(float)
        latent = latent + event_premium
        debug["event_flag"] = event_trigger.astype(int)
    else:
        debug["event_flag"] = 0

    debug["latent"] = latent

    # Apply envelope: NaN outside, percentile rank inside
    df["latent"] = latent
    df.loc[~envelope_mask, "latent"] = np.nan
    df["fused_score"] = df.groupby("trade_date")["latent"].rank(pct=True)
    debug["fused_score"] = df["fused_score"]

    fused = df.dropna(subset=["fused_score"]).set_index(["trade_date", "ticker"])["fused_score"]

    # 1D throttle (V3, V2b)
    if variant in ("V3", "V2b"):
        throttle = df.set_index(["trade_date", "ticker"])["z1d"]
        debug["throttle_z1"] = df["z1d"]
    else:
        throttle = None
        debug["throttle_z1"] = np.nan

    return fused, throttle, debug


def aggregate_variant(portfolio, *, variant: str, cost_mult: float) -> tuple[dict[str, Any], pd.DataFrame]:
    periods = pd.DataFrame([p.to_dict() for p in portfolio.periods])
    periods["execution_date"] = pd.to_datetime(periods["execution_date"])
    periods["exit_date"] = pd.to_datetime(periods["exit_date"])
    periods["signal_date"] = pd.to_datetime(periods["signal_date"])
    periods = periods.sort_values("execution_date").reset_index(drop=True)

    n_periods = len(periods)
    total_days = max((periods["exit_date"].iloc[-1] - periods["execution_date"].iloc[0]).days, 1)
    periods_per_year = float(n_periods * 365.25 / total_days)

    cum_gross = float((1.0 + periods["gross_return"]).prod() - 1.0)
    cum_net = float((1.0 + periods["net_return"]).prod() - 1.0)
    cum_bench = float((1.0 + periods["benchmark_return"]).prod() - 1.0)

    def _annualize(r):
        b = 1.0 + r
        return -1.0 if b <= 0 else b ** (365.25 / max(total_days, 1)) - 1.0

    ann_gross = _annualize(cum_gross)
    ann_net = _annualize(cum_net)
    ann_bench = _annualize(cum_bench)

    net_excess = periods["net_return"] - periods["benchmark_return"]
    if n_periods > 1 and net_excess.std() > 0:
        ir = float(net_excess.mean() / net_excess.std() * math.sqrt(periods_per_year))
        sharpe = float(periods["net_return"].mean() / periods["net_return"].std() * math.sqrt(periods_per_year))
    else:
        ir = float("nan")
        sharpe = float("nan")

    excess_wealth = (1.0 + net_excess).cumprod()
    running_peak = excess_wealth.cummax()
    safe_peak = running_peak.where(running_peak > 0, np.nan)
    drawdown = ((safe_peak - excess_wealth) / safe_peak).fillna(0.0)
    max_dd = float(drawdown.max())

    row = {
        "variant": variant,
        "cost_mult": float(cost_mult),
        "n_periods": n_periods,
        "gross_ann_excess": float(ann_gross - ann_bench),
        "net_ann_excess": float(ann_net - ann_bench),
        "cost_drag_ann": float(ann_gross - ann_net),
        "avg_turnover": float(periods["turnover"].mean()),
        "sharpe": sharpe,
        "ir": ir,
        "max_drawdown": max_dd,
    }
    periods_with_id = periods.copy()
    periods_with_id["variant"] = variant
    periods_with_id["cost_mult"] = float(cost_mult)
    return row, periods_with_id


def build_w11_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"passed": False, "reason": "no rows"}

    # Filter to 1.0x cost first for promotion check
    primary_rows = [r for r in rows if r["cost_mult"] == 1.0]
    v0_row = next((r for r in primary_rows if r["variant"] == "V0"), None)
    non_v0 = [r for r in primary_rows if r["variant"] != "V0"]

    if not non_v0:
        return {"passed": False, "reason": "no non-V0 variants run"}

    promotion_passes = []
    for r in non_v0:
        net = r["net_ann_excess"]
        ir = r["ir"]
        dd = r["max_drawdown"]

        gate_a = (net >= PROMOTE_NET_HIGH) and (ir >= PROMOTE_IR_MIN)
        gate_b = (net >= PROMOTE_NET_MID) and (ir >= PROMOTE_IR_MIN) and (dd <= PROMOTE_DD_MAX)
        if gate_a or gate_b:
            promotion_passes.append({**r, "gate": "A" if gate_a else "B"})

    if not promotion_passes:
        return {
            "passed": False,
            "reason": "no variant passes promotion gate at 1.0x",
            "v0_baseline": v0_row,
            "best_attempt": max(non_v0, key=lambda r: r["net_ann_excess"]) if non_v0 else None,
        }

    # Pick best by net
    champion = max(promotion_passes, key=lambda r: r["net_ann_excess"])

    # Robustness check at 1.25x
    champion_125x = next(
        (r for r in rows if r["variant"] == champion["variant"] and r["cost_mult"] == 1.25),
        None,
    )
    v0_125x = next((r for r in rows if r["variant"] == "V0" and r["cost_mult"] == 1.25), None)
    robust = False
    if champion_125x and v0_125x:
        robust = (
            champion_125x["net_ann_excess"] > ROBUST_NET_MIN
            and (v0_125x["net_ann_excess"] - champion_125x["net_ann_excess"]) <= ROBUST_LOSS_VS_60D_MAX
        )

    return {
        "passed": bool(robust),
        "champion_variant": champion["variant"],
        "champion_at_1x": champion,
        "champion_at_125x": champion_125x,
        "v0_at_125x": v0_125x,
        "robust_at_125x": bool(robust),
        "n_promotion_passes": len(promotion_passes),
    }


def print_summary(rows: list[dict[str, Any]], verdict: dict[str, Any]) -> None:
    print()
    print("=" * 90)
    print("W11 Layered Fusion — Truth Table")
    print("=" * 90)
    if not rows:
        print("(no rows)")
        return
    df = pd.DataFrame(rows).sort_values(["variant", "cost_mult"])
    cols = ["variant", "cost_mult", "n_periods", "gross_ann_excess", "net_ann_excess",
            "cost_drag_ann", "avg_turnover", "sharpe", "ir", "max_drawdown"]
    display = df[cols].copy()
    display["cost_mult"] = display["cost_mult"].map(lambda x: f"{x:.2f}")
    for c in ("gross_ann_excess", "net_ann_excess", "cost_drag_ann",
              "avg_turnover", "sharpe", "ir", "max_drawdown"):
        display[c] = display[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    print(display.to_string(index=False))
    print()
    print(f"W10 baseline (60D-only champion): net={W10_BASELINE_NET:.4f}, "
          f"IR={W10_BASELINE_IR:.4f}, DD={W10_BASELINE_DD:.4f}")
    print()
    status = "PASS — promote fusion" if verdict.get("passed") else "FAIL — keep 60D"
    print(f"=== W11 Verdict: {status} ===")
    if verdict.get("passed"):
        c = verdict["champion_at_1x"]
        print(f"  Champion: {c['variant']} (gate={c.get('gate')})")
        print(f"    1.0x: net={c['net_ann_excess']:.4f}, IR={c['ir']:.4f}, DD={c['max_drawdown']:.4f}")
        c125 = verdict.get("champion_at_125x")
        if c125:
            print(f"    1.25x: net={c125['net_ann_excess']:.4f} (robust={'yes' if verdict['robust_at_125x'] else 'no'})")
    else:
        print(f"  Reason: {verdict.get('reason')}")
        if verdict.get("best_attempt"):
            b = verdict["best_attempt"]
            print(f"  Best attempt: {b['variant']} net={b['net_ann_excess']:.4f} "
                  f"IR={b['ir']:.4f} DD={b['max_drawdown']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
