#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid

from loguru import logger
import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import (
    build_batch_specs,
    build_or_load_feature_cache,
    install_runtime_optimizations,
    load_universe_tickers,
    write_json_atomic,
)
from src.data.db.session import get_engine
from src.features.pipeline import prepare_feature_export_frame

try:
    import pyarrow.dataset as ds
except ImportError:  # pragma: no cover
    ds = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical feature exporter that writes the same feature slice to parquet and feature_store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output-path", default="data/features/all_features.parquet")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-workers", type=int, default=min(8, max(1, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--sync-feature-store", action="store_true")
    parser.add_argument("--clear-feature-store-range", action="store_true")
    parser.add_argument("--metadata-output", default="")
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def latest_pit_trade_date(as_of: datetime) -> date | None:
    query = text(
        """
        select max(trade_date) filter (where knowledge_time <= :as_of) as latest_pit_trade_date
        from stock_prices
        """,
    )
    with get_engine().connect() as conn:
        row = conn.execute(query, {"as_of": as_of}).mappings().one()
    return row["latest_pit_trade_date"]


def clear_feature_store_range(*, start_date: date, end_date: date) -> int:
    query = text(
        "delete from feature_store where calc_date >= :start_date and calc_date <= :end_date"
    )
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(query, {"start_date": start_date, "end_date": end_date})
    return int(result.rowcount or 0)


def bulk_copy_feature_store_batch(batch_path: Path, *, batch_id: str, csv_batch_rows: int = 100_000) -> int:
    if ds is None:
        raise RuntimeError("pyarrow is required to sync feature_store from parquet batches.")
    engine = get_engine()
    dataset = ds.dataset(batch_path, format="parquet")
    total_rows = 0
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        scanner = dataset.scanner(columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled"], batch_size=csv_batch_rows)
        for record_batch in scanner.to_batches():
            frame = record_batch.to_pandas()
            prepared = prepare_feature_export_frame(frame)
            if prepared.empty:
                continue
            export = prepared.rename(columns={"trade_date": "calc_date"}).copy()
            export["batch_id"] = batch_id
            export["calc_date"] = pd.to_datetime(export["calc_date"]).dt.strftime("%Y-%m-%d")
            export["feature_value"] = export["feature_value"].map(lambda value: "" if pd.isna(value) else repr(float(value)))
            export["is_filled"] = export["is_filled"].map(lambda value: "true" if bool(value) else "false")
            buffer = io.StringIO()
            export[["ticker", "calc_date", "feature_name", "feature_value", "is_filled", "batch_id"]].to_csv(
                buffer,
                index=False,
                header=False,
                na_rep="",
            )
            buffer.seek(0)
            cursor.copy_expert(
                "COPY feature_store (ticker, calc_date, feature_name, feature_value, is_filled, batch_id) FROM STDIN WITH (FORMAT CSV)",
                buffer,
            )
            total_rows += int(len(export))
        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise
    finally:
        raw_conn.close()
    return total_rows


def sync_feature_store_from_batches(*, batch_specs: list[dict[str, Any]], start_date: date, end_date: date, clear_store_range_flag: bool) -> dict[str, Any]:
    deleted_rows = 0
    if clear_store_range_flag:
        deleted_rows = clear_feature_store_range(start_date=start_date, end_date=end_date)

    saved_rows = 0
    for batch_index, batch_spec in enumerate(batch_specs, start=1):
        batch_id = str(uuid.uuid4())
        logger.info(
            "syncing feature_store batch {}/{} from {}",
            batch_index,
            len(batch_specs),
            batch_spec["path"],
        )
        saved_rows += bulk_copy_feature_store_batch(batch_spec["path"], batch_id=batch_id)
    return {
        "feature_store_rows_deleted": deleted_rows,
        "feature_store_rows_saved": saved_rows,
    }


def export_feature_panel(
    *,
    start_date: date,
    end_date: date,
    as_of: datetime,
    output_path: Path,
    batch_size: int,
    max_workers: int,
    progress_interval: int,
    sync_feature_store: bool,
    clear_store_range_flag: bool,
) -> dict[str, Any]:
    install_runtime_optimizations()
    tickers = load_universe_tickers()
    effective_end = min(end_date, latest_pit_trade_date(as_of) or end_date)
    if effective_end < start_date:
        raise RuntimeError(
            f"No PIT-visible price dates are available in requested window {start_date} -> {end_date}.",
        )

    batch_dir = output_path.parent / f"{output_path.stem}_batches"
    manifest_path = batch_dir / "manifest.json"

    if output_path.exists():
        output_path.unlink()
    if batch_dir.exists():
        shutil.rmtree(batch_dir)

    feature_summary = build_or_load_feature_cache(
        tickers=tickers,
        feature_start=start_date,
        feature_end=effective_end,
        as_of=as_of.date(),
        batch_size=batch_size,
        max_workers=max_workers,
        progress_interval=progress_interval,
        feature_output_path=output_path,
        batch_dir=batch_dir,
        manifest_path=manifest_path,
    )
    batch_specs = build_batch_specs(tickers=tickers, batch_size=batch_size, batch_dir=batch_dir)

    store_summary = {
        "feature_store_rows_deleted": 0,
        "feature_store_rows_saved": 0,
    }
    if sync_feature_store:
        store_summary = sync_feature_store_from_batches(
            batch_specs=batch_specs,
            start_date=start_date,
            end_date=effective_end,
            clear_store_range_flag=clear_store_range_flag,
        )

    return {
        "start_date": start_date.isoformat(),
        "end_date": effective_end.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "as_of": as_of.isoformat(),
        "output_path": str(output_path),
        "batch_dir": str(batch_dir),
        "manifest_path": str(manifest_path),
        "ticker_count": len(tickers),
        "batch_size": batch_size,
        "max_workers": max_workers,
        **feature_summary,
        **store_summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    as_of = datetime.fromisoformat(args.as_of)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    summary = export_feature_panel(
        start_date=start_date,
        end_date=end_date,
        as_of=as_of,
        output_path=REPO_ROOT / args.output_path,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        progress_interval=args.progress_interval,
        sync_feature_store=bool(args.sync_feature_store),
        clear_store_range_flag=bool(args.clear_feature_store_range),
    )
    if args.metadata_output:
        write_json_atomic(REPO_ROOT / args.metadata_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
