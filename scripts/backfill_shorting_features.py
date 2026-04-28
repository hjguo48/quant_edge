from __future__ import annotations

"""W12: Backfill FINRA shorting features into feature_store.

Computes short_sale_ratio_{1d,5d,accel} and abnormal_off_exchange_shorting
+ their is_missing_* flags for a list of tickers × recent dates, and writes
to feature_store via the standard pipeline save path.

This unblocks the W12-3 BundleValidator schema check (champion's
short_sale_ratio_5d feature was absent from feature_store because the live
FeaturePipeline did not previously include the shorting family).

Usage:
    python scripts/backfill_shorting_features.py --tickers AAPL,MSFT --as-of 2026-04-17
    python scripts/backfill_shorting_features.py --use-dynamic-universe --recent-fridays 4
    python scripts/backfill_shorting_features.py --use-frozen-universe --recent-fridays 4
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import uuid

from loguru import logger
import pandas as pd
import exchange_calendars as xcals

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.pipeline import FeaturePipeline, prepare_feature_export_frame  # noqa: E402
from src.features.shorting import SHORTING_FEATURE_NAMES, compute_shorting_features_batch  # noqa: E402
from src.data.db.pit import get_prices_pit  # noqa: E402
from src.universe.active import resolve_active_universe  # noqa: E402

DEFAULT_FROZEN_UNIVERSE = "data/features/frozen_universe_503.json"
DEFAULT_MIN_CONTINUOUS_PRICE_DAYS = 200
DEFAULT_CONTINUITY_WINDOW_DAYS = 365

XNYS = xcals.get_calendar("XNYS")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    output_dates = resolve_output_dates(args)
    tickers = resolve_tickers(args, output_dates=output_dates)

    logger.info("output dates: {}", output_dates)
    logger.info("tickers: {} ({} total)", tickers[:5] + (["..."] if len(tickers) > 5 else []), len(tickers))

    # Compute shorting features
    shorting = compute_shorting_features_batch(
        tickers=tickers,
        output_dates=output_dates,
    )
    logger.info("shorting features computed: {} rows", len(shorting))

    # Add is_missing flags
    flagged_rows = []
    for fname in SHORTING_FEATURE_NAMES:
        rows = shorting.loc[shorting["feature_name"] == fname].copy()
        rows["is_missing_value"] = rows["feature_value"].isna().astype(float)
        flag_rows = rows.assign(
            feature_name=f"is_missing_{fname}",
            feature_value=rows["is_missing_value"],
        )[["ticker", "trade_date", "feature_name", "feature_value"]]
        flagged_rows.append(flag_rows)
    flags = pd.concat(flagged_rows, ignore_index=True)
    logger.info("is_missing flags: {} rows", len(flags))

    # Combine shorting + flags, fill NaN feature_values with 0 for is_missing flag and pass-through
    combined = pd.concat([
        shorting[["ticker", "trade_date", "feature_name", "feature_value"]],
        flags,
    ], ignore_index=True)
    combined["is_filled"] = False  # raw computed values, not imputed

    if args.dry_run:
        print()
        print(f"Dry-run: would write {len(combined)} rows to feature_store")
        print(f"Universe size: {len(tickers)}")
        print(f"Contains MRSH: {'MRSH' in set(tickers)} | Contains PSKY: {'PSKY' in set(tickers)}")
        print(combined.head(20).to_string(index=False))
        print()
        print("Feature names produced:")
        for n in sorted(combined["feature_name"].unique()):
            cnt = (combined["feature_name"] == n).sum()
            non_null = combined.loc[combined["feature_name"] == n, "feature_value"].notna().sum()
            print(f"  {n}: {cnt} rows, {non_null} non-null")
        return 0

    # Write to feature_store
    pipeline = FeaturePipeline()
    # batch_id is VARCHAR(36) (UUID-shaped). Use a UUID5 for stability + brevity.
    batch_id = str(uuid.uuid4())
    rows_saved = pipeline.save_to_store(combined, batch_id=batch_id)
    logger.info("wrote {} rows to feature_store (batch_id={})", rows_saved, batch_id)
    print()
    print(f"Saved {rows_saved} rows | batch_id={batch_id}")
    print(f"Feature names: {sorted(combined['feature_name'].unique())}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tickers", help="Comma-separated tickers")
    p.add_argument("--use-dynamic-universe", action="store_true",
                   help="Resolve tickers from historical active universe_membership for each output date "
                        "(default when --tickers/--use-frozen-universe are absent).")
    p.add_argument("--use-frozen-universe", action="store_true",
                   help="Load tickers from frozen_universe_503.json")
    p.add_argument("--frozen-universe", default=DEFAULT_FROZEN_UNIVERSE)
    p.add_argument("--as-of", help="Reference date (YYYY-MM-DD), defaults to today")
    p.add_argument("--recent-fridays", type=int, default=0,
                   help="Backfill last N Fridays ending at --as-of (default 0 = single date)")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print but don't write to feature_store")
    return p.parse_args(argv)


def configure_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")


def resolve_output_dates(args: argparse.Namespace) -> list[date]:
    if args.recent_fridays:
        end = (
            pd.Timestamp(args.as_of).normalize()
            if args.as_of
            else pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
        )
        sessions = XNYS.sessions_in_range(end - pd.Timedelta(days=120), end)
        fridays = [s for s in sessions if s.weekday() == 4]
        return [s.date() for s in fridays[-args.recent_fridays:]]
    as_of = (
        pd.Timestamp(args.as_of).normalize()
        if args.as_of
        else pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    )
    return [as_of.date()]


def resolve_tickers(args: argparse.Namespace, *, output_dates: list[date]) -> list[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.use_frozen_universe:
        path = REPO_ROOT / args.frozen_universe
        payload = json.loads(path.read_text())
        tickers = [str(ticker).upper() for ticker in payload.get("tickers", [])]
        logger.info("loaded {} tickers from frozen universe {}", len(tickers), path.name)
        return tickers

    as_of_cutoff = resolve_runtime_as_of(args)
    source_counts: dict[str, int] = {}
    per_date_candidates: dict[date, list[str]] = {}
    for output_date in output_dates:
        tickers, source = resolve_active_universe(
            output_date,
            as_of=as_of_cutoff,
        )
        source_counts[source] = source_counts.get(source, 0) + 1
        per_date_candidates[output_date] = sorted({str(ticker).upper() for ticker in tickers})

    resolved_union = sorted({ticker for tickers in per_date_candidates.values() for ticker in tickers})
    continuity_counts = resolve_price_continuity_counts(
        tickers=resolved_union,
        output_dates=output_dates,
        as_of=as_of_cutoff,
        continuity_window_days=DEFAULT_CONTINUITY_WINDOW_DAYS,
    )
    filtered_union: set[str] = set()
    rejected_by_continuity: dict[date, int] = {}
    for output_date, tickers in per_date_candidates.items():
        kept = [
            ticker
            for ticker in tickers
            if continuity_counts.get(output_date, {}).get(ticker, 0) >= DEFAULT_MIN_CONTINUOUS_PRICE_DAYS
        ]
        filtered_union.update(kept)
        rejected_by_continuity[output_date] = len(tickers) - len(kept)

    tickers = sorted(filtered_union)
    logger.info(
        "resolved {} unique tickers from historical active universe across {} output dates (sources={}, continuity_rejections={}, threshold={} in {}d)",
        len(tickers),
        len(output_dates),
        source_counts,
        {d.isoformat(): n for d, n in rejected_by_continuity.items()},
        DEFAULT_MIN_CONTINUOUS_PRICE_DAYS,
        DEFAULT_CONTINUITY_WINDOW_DAYS,
    )
    return tickers


def resolve_runtime_as_of(args: argparse.Namespace) -> datetime:
    if args.as_of:
        timestamp = pd.Timestamp(args.as_of)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.to_pydatetime()
    return datetime.now(timezone.utc)


def resolve_price_continuity_counts(
    *,
    tickers: list[str],
    output_dates: list[date],
    as_of: datetime,
    continuity_window_days: int,
) -> dict[date, dict[str, int]]:
    if not tickers or not output_dates:
        return {}

    earliest = min(output_dates) - timedelta(days=continuity_window_days)
    latest = max(output_dates)
    prices = get_prices_pit(
        tickers=tickers,
        start_date=earliest,
        end_date=latest,
        as_of=as_of,
    )
    if prices.empty:
        return {output_date: {} for output_date in output_dates}

    frame = prices.loc[:, ["ticker", "trade_date"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    counts_by_date: dict[date, dict[str, int]] = {}
    for output_date in sorted(output_dates):
        start = output_date - timedelta(days=continuity_window_days)
        window = frame.loc[
            (frame["trade_date"] >= start) & (frame["trade_date"] <= output_date)
        ]
        counts = (
            window.groupby("ticker")["trade_date"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        counts_by_date[output_date] = counts
    return counts_by_date


if __name__ == "__main__":
    raise SystemExit(main())
